"""マイク入力の雑音抑制（なし / WebRTC / RNNoise）。音声入力ワーカーが AEC と入力ゲインの後にかける。

ASR と MaAI（ターンテイキング・相槌・頷き）は同じ音を使うので、両方に効く。既定はなし
（VAP のモデルは雑音抑制をかけていない音で学習されているため、かけると出方が変わりうる）。

- WebRTC：LiveKit の音声処理モジュール（WebRTC APM）の雑音抑制。AEC とは別のインスタンスで、
  雑音抑制だけを有効にする（AEC の処理には手を入れない）。10 ms 単位。
- RNNoise：Xiph の RNNoise（BSD-3-Clause）。配布物は pyrnnoise（Apache-2.0）の同梱ライブラリを ctypes で直接
  呼ぶ（`import pyrnnoise` はファイル入出力のために PyAV などを読み込むので使わない）。RNNoise は 48 kHz・
  10 ms（480 サンプル）単位なので、soxr で入力のサンプルレートと 48 kHz の間を連続して変換する。

process() は入力と同じ長さを必ず返す（処理で生じる遅れの分は、最初に無音を詰めて出す）。
WAV の 2 ch 目（MaAI の相手側）など、同じ長さで並べて送る音との対応をずらさないため。
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
from typing import Callable

import numpy as np

MODES = ("off", "webrtc", "rnnoise")


class _FixedLengthOutput:
    """処理済みの音をため、入力と同じ長さずつ取り出す（足りない分は先頭を無音にする）。"""

    def __init__(self) -> None:
        self._buf = np.empty(0, dtype=np.float32)

    def push(self, x: np.ndarray) -> None:
        if x.size:
            self._buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])

    def pop(self, n: int) -> np.ndarray:
        if self._buf.size >= n:
            out, self._buf = self._buf[:n], self._buf[n:]
            return out
        out = np.zeros(n, dtype=np.float32)
        if self._buf.size:
            out[n - self._buf.size:] = self._buf
            self._buf = np.empty(0, dtype=np.float32)
        return out


class WebRtcNoiseSuppressor:
    def __init__(self, sample_rate: int) -> None:
        from livekit.rtc import AudioFrame, AudioProcessingModule

        self._frame_cls = AudioFrame
        self._apm = AudioProcessingModule(
            echo_cancellation=False, noise_suppression=True, high_pass_filter=False, auto_gain_control=False,
        )
        self._rate = int(sample_rate)
        self._frame = self._rate // 100  # 10 ms
        self._pending = np.empty(0, dtype=np.float32)
        self._out = _FixedLengthOutput()

    def process(self, x: np.ndarray) -> np.ndarray:
        n_in = int(x.size)
        if n_in == 0:
            return x
        buf = np.concatenate([self._pending, x.astype(np.float32, copy=False)])
        n_frames = buf.size // self._frame
        for i in range(n_frames):
            seg = buf[i * self._frame:(i + 1) * self._frame]
            pcm = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)
            frame = self._frame_cls(data=pcm.tobytes(), sample_rate=self._rate,
                                    num_channels=1, samples_per_channel=self._frame)
            self._apm.process_stream(frame)
            self._out.push(np.frombuffer(bytes(frame.data), dtype=np.int16).astype(np.float32) / 32768.0)
        self._pending = buf[n_frames * self._frame:].copy()
        return self._out.pop(n_in)


def _rnnoise_library_path() -> str:
    spec = importlib.util.find_spec("pyrnnoise")  # パッケージ自体は import しない（__init__ が PyAV を読み込む）
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pyrnnoise が入っていません（uv sync で入ります）")
    name = {"Windows": "rnnoise.dll", "Darwin": "librnnoise.dylib"}.get(platform.system(), "librnnoise.so")
    path = os.path.join(list(spec.submodule_search_locations)[0], name)
    if not os.path.exists(path):
        raise RuntimeError(f"RNNoise のライブラリが見つかりません: {path}")
    return path


class RNNoiseSuppressor:
    _RATE = 48000

    def __init__(self, sample_rate: int) -> None:
        import soxr

        lib = ctypes.CDLL(_rnnoise_library_path())
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_get_frame_size.restype = ctypes.c_int
        lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        lib.rnnoise_process_frame.restype = ctypes.c_float
        self._lib = lib
        self._state = lib.rnnoise_create(None)
        self._frame = int(lib.rnnoise_get_frame_size())  # 480（48 kHz の 10 ms）
        self._rate = int(sample_rate)
        self._up = soxr.ResampleStream(self._rate, self._RATE, 1, dtype="float32")
        self._down = soxr.ResampleStream(self._RATE, self._rate, 1, dtype="float32")
        self._pending = np.empty(0, dtype=np.float32)
        self._out = _FixedLengthOutput()

    def __del__(self) -> None:
        try:
            if getattr(self, "_state", None):
                self._lib.rnnoise_destroy(self._state)
                self._state = None
        except Exception:  # noqa: BLE001
            pass

    def process(self, x: np.ndarray) -> np.ndarray:
        n_in = int(x.size)
        if n_in == 0:
            return x
        up = self._up.resample_chunk(x.astype(np.float32, copy=False))
        buf = np.concatenate([self._pending, up]) if self._pending.size else up
        n_frames = buf.size // self._frame
        done: list[np.ndarray] = []
        for i in range(n_frames):
            # RNNoise は 16 bit の値の範囲の float を受け取る
            seg = np.ascontiguousarray(buf[i * self._frame:(i + 1) * self._frame] * 32768.0, dtype=np.float32)
            ptr = seg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._lib.rnnoise_process_frame(self._state, ptr, ptr)
            done.append(seg / 32768.0)
        self._pending = buf[n_frames * self._frame:].copy()
        if done:
            self._out.push(self._down.resample_chunk(np.concatenate(done)))
        return self._out.pop(n_in)


def make_noise_suppressor(
    mode: str, sample_rate: int, log: Callable[[str], None] | None = None,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """設定の値（off / webrtc / rnnoise）から処理を作る。使えなければ None（雑音抑制なしで続ける）。"""
    mode = str(mode or "off").strip().lower()
    try:
        if mode == "webrtc":
            proc = WebRtcNoiseSuppressor(sample_rate)
        elif mode == "rnnoise":
            proc = RNNoiseSuppressor(sample_rate)
        else:
            return None
    except Exception as ex:  # noqa: BLE001 - 入っていない・読み込めないときは雑音抑制なしで動かす
        if log:
            log(f"雑音抑制（{mode}）を使えないため、なしで続けます: {ex}")
        return None
    if log:
        log(f"雑音抑制: {mode}")
    return proc.process
