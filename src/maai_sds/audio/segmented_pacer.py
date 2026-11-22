"""16kHz/mono PCM をバースト＋実時間で送出する共通ペーサ（MMD / ローカル / TCP 共通）。"""

from __future__ import annotations

import base64
import threading
import time
from collections import deque
from typing import Callable


class SegmentedAudioPacer:
    """PCM を 640B フレームに分割し、バースト後は実時間で送出する。"""

    PCM_BYTES_PER_SEC = 32000
    FRAME_BYTES = 640

    def __init__(
        self,
        *,
        emit_fn: Callable[[dict[str, object]], None],
        burst_snd_frames: int = 20,
        is_connected: Callable[[], bool] | None = None,
        send_frame: Callable[[bytes], None] | None = None,
        reset_output: Callable[[], None] | None = None,
        segment_boundary: Callable[[], bool] | None = None,
    ) -> None:
        self._emit = emit_fn
        self._is_connected = is_connected or (lambda: True)
        self._send_frame_raw = send_frame or (lambda _b: None)
        self._reset_output = reset_output or (lambda: None)
        # 発話途中の区切り（complete_kind="segment"）で出力を区切らず続けるかを決める。
        # True を返したら reset_output もバーストの張り直しもせず、ストリームを継続する。
        self._segment_boundary = segment_boundary
        self._resume_without_catchup = False
        self._burst_snd_frames = max(0, min(500, int(burst_snd_frames)))
        self._dq: deque[tuple] = deque()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        # clear_and_reset() で進める世代カウンタ（_cv 保護）。_send_frames は開始時（pop 時）の
        # 世代を保持し、途中で世代が変わったら（バージイン等）即時退出する。
        # 旧方式（プロデューサ側で解除するブール Event）は、退出チェック前に enqueue_pcm が
        # 割り込むと中断指示を取りこぼす競合があった。
        self._epoch = 0
        self._next_snd_at = 0.0
        self._have_schedule = False
        self._arm_next_segment_burst = True
        self._burst_frames_left = 0
        self._segment_burst_bytes_acc = 0
        # inject_only（SND 無効＋MaAI 注入）は実時間スリープを伴うため、SND 再生をペーシングする
        # _run 消費スレッドとは別スレッドで処理する（長い注入が後続 SND 再生を待たせない）。
        self._inject_dq: deque[bytes] = deque()
        self._inject_cv = threading.Condition()
        self._inject_epoch = 0  # inject_only 用世代（_inject_cv 保護）
        self._thread = threading.Thread(target=self._run, daemon=True, name="segmented-audio-pacer")
        self._inject_thread = threading.Thread(
            target=self._inject_only_run, daemon=True, name="segmented-maai-inject-only"
        )
        self._thread.start()
        self._inject_thread.start()

    def set_burst_snd_frames(self, n: int) -> None:
        with self._cv:
            self._burst_snd_frames = max(0, min(500, int(n)))
            self._cv.notify()

    def enqueue_pcm(self, data: bytes, *, inject_maai: bool = False, snd_enabled: bool = True) -> None:
        if not data:
            return
        if not snd_enabled:
            # SND 無効: MaAI 注入のみ専用スレッドで実時間送出（注入不要なら破棄）。
            if inject_maai:
                with self._inject_cv:
                    self._inject_dq.append(bytes(data))
                    self._inject_cv.notify()
            return
        with self._cv:
            self._dq.append(("pcm", bytes(data), bool(inject_maai), True))
            self._cv.notify()

    def enqueue_break(
        self,
        generation: int | None,
        *,
        complete_kind: str | None = None,
        segment_index: int | None = None,
    ) -> None:
        meta: dict[str, object] | None = None
        if complete_kind is not None or segment_index is not None:
            meta = {}
            if complete_kind is not None:
                meta["complete_kind"] = str(complete_kind)
            if segment_index is not None:
                meta["segment_index"] = int(segment_index)
        with self._cv:
            if meta is None:
                self._dq.append(("break", generation))
            else:
                self._dq.append(("break", generation, meta))
            self._cv.notify()

    def clear_and_reset(self) -> None:
        # ロックのネストを避けるため _inject_cv → _cv の順に別々に取得する。
        with self._inject_cv:
            self._inject_dq.clear()
            self._inject_epoch += 1
            self._inject_cv.notify()
        with self._cv:
            self._dq.clear()
            self._reset_output()
            self._arm_next_segment_burst = True
            self._burst_frames_left = 0
            self._segment_burst_bytes_acc = 0
            self._have_schedule = False
            self._next_snd_at = 0.0
            self._resume_without_catchup = False
            self._epoch += 1
            self._cv.notify()

    def close(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify()
        with self._inject_cv:
            self._inject_cv.notify()
        self._thread.join(timeout=5.0)
        self._inject_thread.join(timeout=5.0)

    def _emit_maai_frame(self, frame: bytes) -> None:
        try:
            self._emit(
                {
                    "type": "tts_maai_pcm_frame",
                    "audio_b64": base64.b64encode(frame).decode("ascii"),
                }
            )
        except Exception:
            pass

    def _inject_only_run(self) -> None:
        """SND 無効＋MaAI 注入を実時間ペーシングで送出する専用スレッド。"""
        while not self._stop.is_set():
            with self._inject_cv:
                while not self._inject_dq and not self._stop.is_set():
                    self._inject_cv.wait(timeout=0.2)
                if self._stop.is_set():
                    break
                if not self._inject_dq:
                    continue
                chunk = self._inject_dq.popleft()
                epoch = self._inject_epoch
            self._send_maai_timed(chunk, epoch)

    def _send_maai_timed(self, chunk_l16le: bytes, epoch: int) -> None:
        # SND 無し（inject_only）でも MaAI へは 16k/mono 20ms 相当の間隔で送る。
        # 無間隔で一括 emit すると受信側バッファが瞬間充填され、可視化・後段でぶつ切りになりやすい。
        frame_dur = self.FRAME_BYTES / float(self.PCM_BYTES_PER_SEC)
        next_at = time.monotonic()
        pos = 0
        while pos < len(chunk_l16le):
            if self._stop.is_set() or self._inject_epoch != epoch:
                return
            frame = chunk_l16le[pos : pos + self.FRAME_BYTES]
            pos += len(frame)
            if not frame:
                continue
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
            self._emit_maai_frame(frame)
            next_at += frame_dur

    def _send_frames(self, chunk_l16le: bytes, *, inject_maai: bool = False, epoch: int = 0) -> None:
        pos = 0
        connected = self._is_connected()
        while pos < len(chunk_l16le):
            if self._epoch != epoch:
                return
            frame = chunk_l16le[pos : pos + self.FRAME_BYTES]
            pos += len(frame)
            if not frame:
                continue
            if not connected:
                if inject_maai:
                    self._emit_maai_frame(frame)
                continue
            if self._burst_frames_left > 0:
                self._send_frame_raw(frame)
                self._burst_frames_left -= 1
                self._segment_burst_bytes_acc += len(frame)
                if inject_maai:
                    self._emit_maai_frame(frame)
                continue
            if not self._have_schedule:
                self._next_snd_at = time.monotonic()
                self._have_schedule = True
            elif self._resume_without_catchup:
                # 区切りをまたいで続けたストリームで次の音声が遅れて来たら、間を詰めて一気に
                # 送らず、今から実時間で送り直す。
                self._next_snd_at = max(self._next_snd_at, time.monotonic())
            self._resume_without_catchup = False
            now = time.monotonic()
            frame_dur = len(frame) / float(self.PCM_BYTES_PER_SEC)
            # 長時間ストール後はスケジュールが大きく過去になり catch-up バースト（無音間引き無しの
            # 一気送出）になるため、大幅に遅延していたら現在時刻へ再アンカーする。
            if now - self._next_snd_at > max(0.5, frame_dur * 4.0):
                self._next_snd_at = now
            if now < self._next_snd_at:
                time.sleep(self._next_snd_at - now)
            self._send_frame_raw(frame)
            self._next_snd_at += frame_dur
            if inject_maai:
                self._emit_maai_frame(frame)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._dq and not self._stop.is_set():
                    self._cv.wait(timeout=0.2)
                if self._stop.is_set():
                    break
                if not self._dq:
                    continue
                item = self._dq.popleft()
                my_epoch = self._epoch
            kind = item[0]
            break_meta: dict[str, object] | None = None
            if kind == "break" and len(item) > 2 and isinstance(item[2], dict):
                break_meta = item[2]
            if kind == "pcm":
                if self._arm_next_segment_burst:
                    self._burst_frames_left = self._burst_snd_frames
                    self._segment_burst_bytes_acc = 0
                    self._arm_next_segment_burst = False
                pcm_data = item[1]
                inject_maai = bool(item[2])
                self._send_frames(pcm_data, inject_maai=inject_maai, epoch=my_epoch)
            elif kind == "break":
                gen = item[1]
                burst_acc = int(self._segment_burst_bytes_acc)
                continuous = False
                if (
                    self._segment_boundary is not None
                    and break_meta
                    and str(break_meta.get("complete_kind") or "") == "segment"
                ):
                    with self._cv:
                        try:
                            continuous = bool(self._segment_boundary())
                        except Exception:
                            continuous = False
                if continuous:
                    # 発話の頭のバーストで先行した分は再生が終わるまで残るので、補正もそのまま使う。
                    self._resume_without_catchup = True
                else:
                    with self._cv:
                        self._reset_output()
                    self._burst_frames_left = 0
                    self._segment_burst_bytes_acc = 0
                    self._have_schedule = False
                    self._next_snd_at = 0.0
                    self._arm_next_segment_burst = True
                if gen is not None:
                    try:
                        burst_ms = max(
                            0,
                            int(round(burst_acc / float(self.PCM_BYTES_PER_SEC) * 1000.0)),
                        )
                        payload: dict[str, object] = {
                            "type": "tts_mmd_send_complete",
                            "generation": int(gen),
                            "burst_compensation_ms": burst_ms,
                        }
                        if break_meta:
                            payload.update(break_meta)
                        self._emit(payload)
                    except Exception:
                        pass
