"""MMDAgent 向け SND フレーム送出（audio/segmented_pacer の薄いラッパ）。"""

from __future__ import annotations

from typing import Callable

from .segmented_pacer import SegmentedAudioPacer
from ..mmd.remote_bridge import MmdRemoteBridge


class _MmdBridgeFrameSender:
    # 口パクは MMDAgent-EX が受け取った SND 音声から自分で作る（TCP / WebSocket とも同じ経路）。
    def __init__(self, bridge: MmdRemoteBridge) -> None:
        self._bridge = bridge
        self._mmd_started = False

    def send_frame(self, frame: bytes) -> None:
        if not frame:
            return
        if not self._bridge.is_connected():
            return
        self._bridge.ensure_snd_segment_preamble()
        header = f"SND{len(frame):04d}".encode("ascii")
        try:
            self._bridge.send_binary(header + frame)
            self._mmd_started = True
        except Exception:
            pass

    def reset(self) -> None:
        if self._mmd_started and self._bridge.is_connected():
            try:
                self._bridge.send_text("SNDBRKS\n")
            except Exception:
                pass
        self._mmd_started = False
        self._bridge.mark_snd_segment_ended()

    def segment_boundary(self) -> bool:
        """発話途中の区切り。続けてよければ True（SNDBRKS を送らずストリームを続ける）。

        MMDAgent-EX は SNDBRKS のあと __AV_SETMODEL を送り直さないと口パクを受け付けず、
        __AV_SETMODEL は顔や頭の状態を初期化して区切りごとにガクッとする。
        TCP の MMDAgent-EX では発話の最後まで SND を区切らず、WebSocket 経路は
        SNDBRKS で再生予定を区切る。
        """
        return self._bridge.mode != MmdRemoteBridge.MODE_WS


class MmdAudioPacer:
    """MMDAgent 向け PCM を 16kHz/mono 実時間相当で送出し、break 後に完了を emit する。"""

    def __init__(self, bridge: MmdRemoteBridge, emit_fn: Callable[[dict[str, object]], None], *, burst_snd_frames: int = 20) -> None:
        self._bridge = bridge
        self._sender = _MmdBridgeFrameSender(bridge)
        self._pacer = SegmentedAudioPacer(
            emit_fn=emit_fn,
            burst_snd_frames=burst_snd_frames,
            is_connected=lambda: self._bridge.is_connected(),
            send_frame=self._sender.send_frame,
            reset_output=self._sender.reset,
            segment_boundary=self._sender.segment_boundary,
        )

    def set_burst_snd_frames(self, n: int) -> None:
        self._pacer.set_burst_snd_frames(n)

    def enqueue_pcm(self, data: bytes, *, inject_maai: bool = False, snd_enabled: bool = True) -> None:
        self._pacer.enqueue_pcm(data, inject_maai=inject_maai, snd_enabled=snd_enabled)

    def enqueue_break(
        self,
        generation: int | None,
        *,
        complete_kind: str | None = None,
        segment_index: int | None = None,
    ) -> None:
        self._pacer.enqueue_break(
            generation,
            complete_kind=complete_kind,
            segment_index=segment_index,
        )

    def clear_and_reset(self) -> None:
        self._pacer.clear_and_reset()

    def close(self) -> None:
        self._pacer.close()
