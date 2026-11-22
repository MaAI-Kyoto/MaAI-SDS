from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np


def resample_pcm16le(raw_pcm16le: bytes, source_rate: int, target_rate: int = 16000) -> bytes:
    pcm = raw_pcm16le if (len(raw_pcm16le) % 2 == 0) else raw_pcm16le[:-1]
    if not pcm:
        return b""
    src_rate = int(source_rate)
    dst_rate = int(target_rate)
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return pcm
    src = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if src.size == 0:
        return b""
    target_len = max(1, int(round(src.size * (float(dst_rate) / float(src_rate)))))
    x_old = np.linspace(0.0, 1.0, num=src.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    dst = np.interp(x_new, x_old, src).astype(np.float32)
    dst = np.clip(dst, -32768.0, 32767.0).astype("<i2")
    return dst.tobytes()


# 16kHz int16 で絶対値がこれ以下の先頭は無音とみなす。実測時と同じしきい値。
_LEADING_SILENCE_THRESHOLD = 200
_LEADING_SILENCE_MAX_SEC = 2.0


def strip_leading_silence_pcm16be(
    pcm: bytes,
    *,
    threshold: int = _LEADING_SILENCE_THRESHOLD,
    sample_rate: int = 16000,
    max_drop_sec: float = _LEADING_SILENCE_MAX_SEC,
) -> bytes:
    """PCM16 ビッグエンディアンの先頭無音を落とす。無音が max_drop_sec を超えたらそこまでしか削らない。"""
    if not pcm:
        return b""
    even = len(pcm) - (len(pcm) % 2)
    if even < 2:
        return b""
    samples = np.frombuffer(pcm[:even], dtype=">i2")
    hot = np.flatnonzero(np.abs(samples.astype(np.int32)) > int(threshold))
    max_drop = max(0, int(float(sample_rate) * float(max_drop_sec)))
    if hot.size == 0:
        if samples.size <= max_drop:
            return b""
        return pcm[max_drop * 2 : even]
    start = min(int(hot[0]), max_drop)
    if start <= 0:
        return pcm[:even]
    return pcm[start * 2 : even]


class Pcm16BeLeadingSilenceGate:
    """チャンク分割された PCM の、発話先頭の無音だけを捨てる。"""

    def __init__(
        self,
        *,
        threshold: int = _LEADING_SILENCE_THRESHOLD,
        sample_rate: int = 16000,
        max_drop_sec: float = _LEADING_SILENCE_MAX_SEC,
    ) -> None:
        self._threshold = int(threshold)
        self._rate = int(sample_rate)
        self._max_samples = max(0, int(self._rate * float(max_drop_sec)))
        self._pending = bytearray()
        self._passed = False

    def reset(self) -> None:
        self._pending.clear()
        self._passed = False

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        self._pending.extend(chunk)
        even = len(self._pending) - (len(self._pending) % 2)
        if even < 2:
            return b""
        if self._passed:
            rest = bytes(self._pending[:even])
            del self._pending[:even]
            return rest
        samples = np.frombuffer(self._pending[:even], dtype=">i2")
        hot = np.flatnonzero(np.abs(samples.astype(np.int32)) > self._threshold)
        if hot.size:
            start = min(int(hot[0]), self._max_samples)
            rest = bytes(self._pending[start * 2 : even])
            del self._pending[:even]
            self._passed = True
            return rest
        if samples.size >= self._max_samples:
            rest = bytes(self._pending[self._max_samples * 2 : even])
            del self._pending[:even]
            self._passed = True
            return rest
        return b""

    def flush(self) -> bytes:
        """合成終了時に、まだ無音とみなしている残りを捨てる。"""
        if self._passed:
            rest = bytes(self._pending)
            self._pending.clear()
            return rest if len(rest) % 2 == 0 else rest[:-1]
        self._pending.clear()
        self._passed = True
        return b""


def swap_pcm16_endian(raw_pcm16: bytes) -> bytes:
    pcm = raw_pcm16 if (len(raw_pcm16) % 2 == 0) else raw_pcm16[:-1]
    if not pcm:
        return b""
    swapped = bytearray(len(pcm))
    swapped[0::2] = pcm[1::2]
    swapped[1::2] = pcm[0::2]
    return bytes(swapped)


def pcmf32le_to_pcm16le(raw_pcm_f32le: bytes) -> bytes:
    if len(raw_pcm_f32le) < 4:
        return b""
    aligned = raw_pcm_f32le[: (len(raw_pcm_f32le) // 4) * 4]
    if not aligned:
        return b""
    samples = np.frombuffer(aligned, dtype="<f4")
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class StreamingPcm16Resampler:
    """少しずつ届く 16 bit PCM（LE）を、つなぎ目で途切れないように別のサンプリング周波数へ直す。

    `resample_pcm16le` はかたまりごとに独立して直すので、ストリーミングの各チャンクにそのまま使うと
    チャンクの境目で位相がずれて小さな雑音が出る。ここでは直前のチャンクの最後の 1 サンプルと、
    次に出すサンプルの位置（端数）を持ち越して、全体を 1 本の音声として線形補間する（補間の方法は
    `resample_pcm16le` と同じ線形）。奇数バイトで届いたときの残りの 1 バイトも持ち越す。
    """

    def __init__(self, source_rate: int, target_rate: int = 16000) -> None:
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._step = float(self.source_rate) / float(self.target_rate) if self.target_rate > 0 else 1.0
        self._pos = 0.0  # 次に出すサンプルの位置（いまの作業列の先頭からの、入力サンプル単位）
        self._prev: np.ndarray | None = None  # 直前のチャンクの最後のサンプル
        self._odd = b""

    def process(self, raw_pcm16le: bytes) -> bytes:
        data = self._odd + bytes(raw_pcm16le or b"")
        cut = len(data) - (len(data) % 2)
        data, self._odd = data[:cut], data[cut:]
        if not data:
            return b""
        if self.source_rate <= 0 or self.target_rate <= 0 or self.source_rate == self.target_rate:
            return data
        x = np.frombuffer(data, dtype="<i2").astype(np.float32)
        if self._prev is not None:
            x = np.concatenate([self._prev, x])
        last = float(x.size - 1)
        if self._pos > last:
            # 入力が短く、まだ次の出力位置に届かない
            self._prev = x[-1:]
            self._pos -= last
            return b""
        count = int(np.floor((last - self._pos) / self._step)) + 1
        positions = self._pos + self._step * np.arange(count, dtype=np.float64)
        out = np.interp(positions, np.arange(x.size, dtype=np.float64), x)
        self._pos = float(positions[-1] + self._step - last)
        self._prev = x[-1:]
        return np.clip(out, -32768.0, 32767.0).astype("<i2").tobytes()


def pcm16le_to_pcm16be_16k(raw_pcm16le: bytes, sample_rate: int) -> bytes:
    pcm16le = resample_pcm16le(raw_pcm16le, source_rate=int(sample_rate), target_rate=16000)
    return swap_pcm16_endian(pcm16le)


def wav_bytes_to_pcm16be_16k_mono(wav_bytes: bytes) -> bytes:
    if not wav_bytes:
        raise RuntimeError("音声データが空です")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = int(wf.getnchannels())
            sample_width = int(wf.getsampwidth())
            sample_rate = int(wf.getframerate())
            frames = wf.readframes(wf.getnframes())
    except Exception as ex:
        raise RuntimeError(f"WAV解析失敗: {ex}") from ex
    if sample_width != 2:
        raise RuntimeError(f"WAVフォーマット不一致: sampwidth={sample_width}")
    if not frames:
        raise RuntimeError("音声データが空です")
    pcm16 = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        pcm16 = pcm16.reshape(-1, channels).mean(axis=1).astype("<i2")
    elif channels != 1:
        raise RuntimeError(f"WAVフォーマット不一致: channels={channels}")
    return pcm16le_to_pcm16be_16k(pcm16.tobytes(), sample_rate=sample_rate)


def audio_any_to_pcm16be_16k_mono(audio_obj: Any, sample_rate: int) -> bytes:
    if audio_obj is None:
        return b""
    source = audio_obj
    if hasattr(source, "detach") and callable(getattr(source, "detach", None)):
        try:
            source = source.detach().cpu().numpy()
        except Exception:
            return b""
    arr = np.asarray(source)
    if arr.size <= 0:
        return b""
    arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        pcm16le = np.clip(arr.astype(np.int64), -32768, 32767).astype("<i2").tobytes()
    else:
        f32 = np.clip(arr.astype(np.float32), -1.0, 1.0)
        pcm16le = (f32 * 32767.0).astype("<i2").tobytes()
    return pcm16le_to_pcm16be_16k(pcm16le, sample_rate=int(sample_rate))


# ---------------------------------------------------------------------------
# int16 <-> float32（numpy 配列レベル）
#
# **float32 -> int16 は 2 通りある。片方に寄せてはいけない。**
#   pcm16_from_f32_unit  : clip(x, -1, 1) * 32767   … 振幅 1.0 を int16 最大値に対応
#   pcm16_from_f32_full  : clip(x * 32768, i16 範囲) … LSB を素直に使う
# x=0.5 なら前者は 16383、後者は 16384 で値が違う。既存の呼び出し箇所が
# どちらを使っているかには意味があるので、名前で区別して残している。
# ---------------------------------------------------------------------------


def f32_from_pcm16_bytes(raw_pcm16le: bytes) -> np.ndarray:
    """int16 リトルエンディアンのバイト列 -> float32 配列（/ 32768.0）。"""
    return np.frombuffer(raw_pcm16le, dtype="<i2").astype(np.float32) / 32768.0


def f32_from_pcm16_array(samples: Any) -> np.ndarray:
    """int16 の配列（list / ndarray / array.array）-> float32 配列（/ 32768.0）。"""
    return np.array(samples, dtype=np.int16).astype(np.float32) / 32768.0


def pcm16_from_f32_unit(samples: Any) -> np.ndarray:
    """float32 -> int16。**先に [-1,1] へ clip してから 32767 倍**する方式。"""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def pcm16_from_f32_full(samples: Any) -> np.ndarray:
    """float32 -> int16。**32768 倍してから int16 の範囲へ clip** する方式。"""
    return np.clip(np.asarray(samples, dtype=np.float32) * 32768.0, -32768, 32767).astype(np.int16)
