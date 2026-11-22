"""Audio Output Worker: MMD / ローカル / TCP へバースト＋実時間で PCM を送出し、MaAI 用フレームを emit する。"""

from __future__ import annotations

import argparse
import base64
import socket
import sys
import threading
import time
from collections import deque
from ..ipc.jsonline import JsonLineDuplexClient
from ..ipc.worker_runtime import connect_worker_client, finish_worker, parse_json_object
from .segmented_pacer import SegmentedAudioPacer
from .mmd_pacer import MmdAudioPacer
from ..mmd.blink import (
    DEFAULT_BLINK_INTERVAL_MEAN_SEC,
    AutoBlinker,
    blink_should_run,
    normalize_blink_enable,
    normalize_blink_interval_mean_sec,
    normalize_blink_mode,
    sanitize_blink_morph,
)
from ..mmd.remote_bridge import MmdRemoteBridge


class _LocalCallbackAdapter:
    """sounddevice コールバックモードで PCM を再生する。
    SegmentedAudioPacer と同じ public インターフェースを持つ。
    sleep ベースのペーシングを排除してオーディオデバイスのクロックに委ねるため
    バッファアンダーランによるぶつぶつが発生しない。
    inject_maai 時は再生バイトと同じストリームを 640B 単位で tts_maai_pcm_frame 送出する
    （SegmentedAudioPacer と同様。enqueue 単位でフレーム境界をリセット）。
    """

    _PCM_BYTES_PER_SEC = 32000
    _FRAME_BYTES = 640

    def __init__(self, emit_fn):
        self._emit = emit_fn
        self._queue: deque[tuple[bytes, bool]] = deque()
        self._lock = threading.Lock()
        self._bytes_enqueued = 0
        self._bytes_played = 0
        self._maai_carry: bytes = b""
        self._break_events: deque[tuple[int, int | None, dict | None]] = deque()
        self._break_lock = threading.Lock()
        self._break_notify = threading.Event()
        self._stop = threading.Event()
        self._inject_abort = threading.Event()
        self._inject_only_cv = threading.Condition()
        self._inject_only_dq: deque[bytes] = deque()
        # 再生中（snd 有効）の inject_maai フレームは、デバイスクロックに同期した realtime
        # コールバック内では送らず、ここへ積んで非 realtime スレッドから emit する。
        # コールバック内でソケット書き込みをすると相手の遅延がアンダーラン（ぶつぶつ）を招くため。
        self._maai_emit_cv = threading.Condition()
        self._maai_emit_dq: deque[bytes] = deque()
        self._stream: object = None
        self._watcher = threading.Thread(
            target=self._break_watcher, daemon=True, name="local-cb-break-watcher"
        )
        self._inject_worker = threading.Thread(
            target=self._inject_only_worker, daemon=True, name="local-maai-inject-only"
        )
        self._maai_emit_worker_thread = threading.Thread(
            target=self._maai_emit_worker, daemon=True, name="local-maai-emit"
        )
        self._watcher.start()
        self._inject_worker.start()
        self._maai_emit_worker_thread.start()

    def _maai_emit_worker(self) -> None:
        """realtime コールバックから積まれた MaAI フレームを非 realtime で順次 emit する。"""
        while not self._stop.is_set():
            with self._maai_emit_cv:
                while not self._maai_emit_dq and not self._stop.is_set():
                    self._maai_emit_cv.wait(timeout=0.2)
                if self._stop.is_set() and not self._maai_emit_dq:
                    break
                frames = list(self._maai_emit_dq)
                self._maai_emit_dq.clear()
            for frame in frames:
                self._emit_maai_frame(frame)

    def _emit_maai_frame(self, frame: bytes) -> None:
        if not frame:
            return
        try:
            self._emit(
                {
                    "type": "tts_maai_pcm_frame",
                    "audio_b64": base64.b64encode(frame).decode("ascii"),
                }
            )
        except Exception:
            pass

    def _emit_maai_timed(self, chunk_l16le: bytes) -> None:
        """SND 無し（inject_only）。SegmentedAudioPacer._send_frames と同様に実時間相当で送出。"""
        frame_dur = self._FRAME_BYTES / float(self._PCM_BYTES_PER_SEC)
        next_at = time.monotonic()
        pos = 0
        while pos < len(chunk_l16le):
            if self._inject_abort.is_set() or self._stop.is_set():
                return
            frame = chunk_l16le[pos : pos + self._FRAME_BYTES]
            pos += len(frame)
            if not frame:
                continue
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
            self._emit_maai_frame(frame)
            next_at += frame_dur

    def _inject_only_worker(self) -> None:
        while not self._stop.is_set():
            with self._inject_only_cv:
                while not self._inject_only_dq and not self._stop.is_set():
                    self._inject_only_cv.wait(timeout=0.2)
                if self._stop.is_set():
                    break
                chunk = self._inject_only_dq.popleft()
                # 新チャンク再生開始時に abort をクリア（前チャンクの中断指示を持ち越さない）。
                # clear_and_reset 側の set と同じ _inject_only_cv 下で行うことで、popleft と
                # clear の間に割り込んだ set を取りこぼさない（バージイン中断を確実に観測）。
                self._inject_abort.clear()
            self._emit_maai_timed(chunk)

    def start(self) -> None:
        import sounddevice as sd  # noqa: PLC0415
        try:
            self._stream = sd.RawOutputStream(
                samplerate=16000, channels=1, dtype="int16",
                blocksize=320, latency=0.06,
                callback=self._callback,
            )
        except Exception:
            self._stream = sd.RawOutputStream(
                samplerate=16000, channels=1, dtype="int16",
                blocksize=0, latency=0.1,
                callback=self._callback,
            )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        needed = frames * 2  # int16 mono: 2 bytes/sample
        buf = bytearray(needed)
        pos = 0
        pending_maai: list[bytes] = []
        with self._lock:
            while pos < needed and self._queue:
                chunk, inject_maai = self._queue[0]
                take = min(len(chunk), needed - pos)
                slice_b = chunk[:take]
                buf[pos : pos + take] = slice_b
                pos += take
                if inject_maai:
                    self._maai_carry += slice_b
                    while len(self._maai_carry) >= self._FRAME_BYTES:
                        frame = self._maai_carry[: self._FRAME_BYTES]
                        self._maai_carry = self._maai_carry[self._FRAME_BYTES :]
                        pending_maai.append(frame)
                else:
                    self._maai_carry = b""
                if take == len(chunk):
                    if inject_maai and self._maai_carry:
                        pending_maai.append(self._maai_carry)
                        self._maai_carry = b""
                    elif not inject_maai:
                        self._maai_carry = b""
                    self._queue.popleft()
                else:
                    self._queue[0] = (chunk[take:], inject_maai)
            self._bytes_played += pos
        memoryview(outdata).cast("B")[:needed] = buf
        if pending_maai:
            # realtime コールバック内では emit せず、専用スレッドへ渡す（下記 deque 操作のみ）。
            with self._maai_emit_cv:
                self._maai_emit_dq.extend(pending_maai)
                self._maai_emit_cv.notify()
        if self._break_events:
            self._break_notify.set()

    def _break_watcher(self) -> None:
        while not self._stop.is_set():
            self._break_notify.wait(timeout=0.05)
            self._break_notify.clear()
            while True:
                with self._break_lock:
                    if not self._break_events:
                        break
                    threshold, gen, meta = self._break_events[0]
                with self._lock:
                    played = self._bytes_played
                if played < threshold:
                    break
                with self._break_lock:
                    # クリア/別スレッドの pop と競合しないよう取り出し直前に再確認する。
                    if not self._break_events or self._break_events[0][0] != threshold:
                        continue
                    self._break_events.popleft()
                if gen is not None:
                    payload: dict[str, object] = {
                        "type": "tts_mmd_send_complete",
                        "generation": gen,
                        "burst_compensation_ms": 0,
                    }
                    if meta:
                        payload.update(meta)
                    try:
                        self._emit(payload)
                    except Exception:
                        pass

    def enqueue_pcm(self, data: bytes, *, inject_maai: bool = False, snd_enabled: bool = True) -> None:
        if not data:
            return
        if not snd_enabled:
            if inject_maai:
                with self._inject_only_cv:
                    self._inject_only_dq.append(bytes(data))
                    self._inject_only_cv.notify()
            return
        with self._lock:
            self._queue.append((bytes(data), bool(inject_maai)))
            self._bytes_enqueued += len(data)

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
        with self._lock:
            threshold = self._bytes_enqueued
        with self._break_lock:
            self._break_events.append((threshold, generation, meta))
        self._break_notify.set()

    def clear_and_reset(self) -> None:
        with self._inject_only_cv:
            # abort の set と dq.clear を _inject_only_cv 下で行い、_inject_only_worker の
            # popleft→abort.clear と相互排他にする。これにより worker の clear がここでの
            # set を上書きして中断指示を失うレース（バージイン取りこぼし）を防ぐ。
            self._inject_abort.set()
            self._inject_only_dq.clear()
            self._inject_only_cv.notify()
        with self._lock:
            self._queue.clear()
            self._bytes_enqueued = self._bytes_played
            self._maai_carry = b""
        with self._maai_emit_cv:
            self._maai_emit_dq.clear()
        with self._break_lock:
            self._break_events.clear()
        # 注意: _inject_abort はここでクリアしない。即時クリアすると、フレーム単位でしか
        # abort を確認しない _emit_maai_timed が set→clear を取りこぼし、バージイン後も
        # 古い inject チャンクを再生し続ける。abort は次チャンク再生開始時にクリアする。

    def set_burst_snd_frames(self, n: int) -> None:
        pass

    def close(self) -> None:
        self._stop.set()
        self._inject_abort.set()
        with self._inject_only_cv:
            self._inject_only_cv.notify()
        with self._maai_emit_cv:
            self._maai_emit_cv.notify()
        self._break_notify.set()
        self._watcher.join(timeout=2.0)
        self._inject_worker.join(timeout=5.0)
        self._maai_emit_worker_thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio output worker process")
    parser.add_argument("--host", required=True, type=str)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--config-json", required=True, type=str)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        config = parse_json_object(args.config_json, worker_name="AudioOutputWorker")
    except ValueError as ex:
        print(str(ex), file=sys.stderr, flush=True)
        return 2

    output_mode = str(config.get("output_mode", "mmd")).strip().lower() or "mmd"
    burst_default = 20
    try:
        burst_default = int(config.get("mmd_burst_snd_frames", 20) or 20)
    except (TypeError, ValueError):
        pass
    burst_default = max(0, min(500, burst_default))
    blink_enable = normalize_blink_enable(config.get("mmd_blink_enable"), True)
    blink_interval_mean_sec = normalize_blink_interval_mean_sec(
        config.get("mmd_blink_interval_mean_sec", DEFAULT_BLINK_INTERVAL_MEAN_SEC)
    )
    blink_morph = sanitize_blink_morph(config.get("mmd_blink_morph", "まばたき"))
    blink_mode = normalize_blink_mode(config.get("mmd_blink_mode"))
    # 対話中かどうかは Orchestrator が update_settings で知らせる。
    blink_dialog_active = bool(config.get("mmd_blink_dialog_active", False))

    client = JsonLineDuplexClient(args.host, args.port)
    if not connect_worker_client(client, worker_name="AudioOutputWorker"):
        return 2

    def emit(payload: dict[str, object]) -> None:
        try:
            client.send(payload)
        except Exception as ex:
            if str(payload.get("type", "")).strip().lower() == "ready":
                print(
                    f"[AudioOutputWorker][FATAL] IPC send failed (ready): {ex}",
                    file=sys.stderr,
                    flush=True,
                )

    stop_event = threading.Event()

    pacer: object | None = None
    tcp_sock: socket.socket | None = None
    # ペーサ未生成（output_mode=mmd で MMD 接続前など）の間に届いた再生操作を順序通りに保持する。
    # ("chunk", pcm, inject_maai, inject_only) または ("break", gen, complete_kind, segment_index)。
    pending_tts_chunks: list[tuple] = []
    pending_tts_lock = threading.Lock()
    mmd_bridge: object | None = None
    mmd_blinker: AutoBlinker | None = None

    def _mmd_worker_emit(pl: dict[str, object]) -> None:
        emit(pl)

    def _dispose_mmd_pacer_only() -> None:
        nonlocal pacer
        if pacer is not None:
            try:
                pacer.close()
            except Exception:
                pass
            pacer = None

    def _ensure_mmd_pacer(burst_arg: int) -> None:
        nonlocal pacer, mmd_bridge
        _dispose_mmd_pacer_only()
        if mmd_bridge is None:
            return
        try:
            connected = bool(mmd_bridge.is_connected())
        except Exception:
            connected = False
        if not connected:
            return
        try:
            bf = max(0, min(500, int(burst_arg)))
        except (TypeError, ValueError):
            bf = burst_default
        try:
            pacer = MmdAudioPacer(
                mmd_bridge,
                _mmd_worker_emit,
                burst_snd_frames=bf,
            )
        except Exception:
            pacer = None
        # MMD 接続前に積まれた再生操作（chunk/break）を順序通りに流す。
        # local モードと異なり、これを呼ばないと最初の発話が再生されない。
        _flush_pending_tts()

    def _on_mmd_connected() -> None:
        nonlocal burst_default
        try:
            emit(
                {
                    "type": "remote_status",
                    "_remote_bridge_origin": "audio_output",
                    "connected": True,
                    "mode": str(mmd_bridge.mode) if mmd_bridge else "",
                }
            )
        except Exception:
            pass
        _ensure_mmd_pacer(burst_default)

    def _on_mmd_disconnected() -> None:
        _dispose_mmd_pacer_only()
        try:
            emit(
                {
                    "type": "remote_status",
                    "_remote_bridge_origin": "audio_output",
                    "connected": False,
                    "mode": str(mmd_bridge.mode) if mmd_bridge else "",
                }
            )
        except Exception:
            pass

    def _flush_pending_tts() -> None:
        nonlocal pending_tts_chunks
        if pacer is None:
            return
        with pending_tts_lock:
            if not pending_tts_chunks:
                return
            ops = pending_tts_chunks
            pending_tts_chunks = []
        for op in ops:
            try:
                if op[0] == "chunk":
                    _, pcm, inject_maai, inject_only = op
                    pacer.enqueue_pcm(pcm, inject_maai=inject_maai, snd_enabled=not inject_only)
                elif op[0] == "break":
                    _, gen, ck_s, si_i = op
                    pacer.enqueue_break(gen, complete_kind=ck_s, segment_index=si_i)
            except Exception:
                pass

    def _close_tcp() -> None:
        nonlocal tcp_sock
        if tcp_sock is not None:
            try:
                tcp_sock.close()
            except Exception:
                pass
            tcp_sock = None

    def _stop_mmd_blinker() -> None:
        nonlocal mmd_blinker
        if mmd_blinker is not None:
            try:
                mmd_blinker.stop()
            except Exception:
                pass
            mmd_blinker = None

    def _apply_mmd_blink_settings() -> None:
        nonlocal mmd_blinker
        if mmd_bridge is None:
            return
        run = blink_should_run(blink_enable, blink_mode, blink_dialog_active)
        if run:
            # まばたきのモーフを KeyValue に結び付ける（変わっていなければ送らない）
            mmd_bridge.set_blink_binding(blink_morph)
        if mmd_blinker is None:
            if not run:
                return
            mmd_blinker = AutoBlinker(
                mmd_bridge.send_text,
                enable=True,
                interval_mean_sec=blink_interval_mean_sec,
                morph=blink_morph,
            )
            mmd_blinker.start()
            return
        mmd_blinker.update(
            enable=run,
            interval_mean_sec=blink_interval_mean_sec,
            morph=blink_morph,
        )

    def _stop_mmd_bridge() -> None:
        nonlocal mmd_bridge
        # 先にまばたきを止め、bridge が生きている間にモーフを必ず 0 へ戻す。
        _stop_mmd_blinker()
        if mmd_bridge is not None:
            try:
                mmd_bridge.stop()
            except Exception:
                pass
            mmd_bridge = None

    # burst は受け取らない。各モードが使わないため（tcp は相手を詰まらせないよう
    # burst_snd_frames=0 固定、local はバースト概念なし、mmd は
    # remote_plugin_apply_settings 側で configure される）。
    def _rebuild_pacer() -> None:
        nonlocal pacer, tcp_sock, mmd_bridge
        if pacer is not None:
            try:
                pacer.close()
            except Exception:
                pass
            pacer = None
        _close_tcp()
        _stop_mmd_bridge()

        if output_mode == "mmd":
            # MmdRemoteBridge + MmdAudioPacer は remote_plugin_apply_settings 初回で生成・configure される。
            return
        if output_mode == "local":
            # コールバックモードを使用することで sounddevice のデバイスクロックがペーシングを担い、
            # sleep ベース送出によるバッファアンダーラン（ぶつぶつ）を回避する。
            adapter = _LocalCallbackAdapter(emit)
            try:
                adapter.start()
            except Exception as ex:
                emit({"type": "fatal", "message": f"local output stream failed: {ex}"})
                return
            pacer = adapter
            _flush_pending_tts()
            return
        if output_mode == "tcp":
            host = str(config.get("tcp_output_host", "127.0.0.1")).strip() or "127.0.0.1"
            try:
                port = int(config.get("tcp_output_port", 15601) or 15601)
            except (TypeError, ValueError):
                port = 15601
            port = max(1, min(65535, port))
            try:
                tcp_sock = socket.create_connection((host, port), timeout=5.0)
                tcp_sock.settimeout(None)
            except Exception as ex:
                emit({"type": "fatal", "message": f"tcp output connect failed: {ex}"})
                return

            def _tcp_send(frame: bytes) -> None:
                if tcp_sock is not None:
                    try:
                        tcp_sock.sendall(frame)
                    except Exception:
                        pass

            # TCP 先が等間隔受信を想定している場合、先頭バーストは相手側の詰まりやすさに繋がるため実時間のみ。
            pacer = SegmentedAudioPacer(
                emit_fn=emit,
                burst_snd_frames=0,
                is_connected=lambda: tcp_sock is not None,
                send_frame=_tcp_send,
                reset_output=lambda: None,
            )
            _flush_pending_tts()
            return
        emit({"type": "fatal", "message": f"unknown output_mode: {output_mode}"})

    def _handle(payload: dict[str, object]) -> None:
        nonlocal pacer, mmd_bridge, burst_default
        nonlocal blink_enable, blink_interval_mean_sec, blink_morph, blink_mode, blink_dialog_active
        cmd = str(payload.get("type", "")).strip().lower()
        if cmd == "ping":
            emit({"type": "pong", "ts": time.time()})
            return
        if cmd == "remote_plugin_apply_settings":
            remote_settings = payload.get("settings")
            if not isinstance(remote_settings, dict):
                return
            if output_mode != "mmd":
                return
            if mmd_bridge is None:
                mmd_bridge = MmdRemoteBridge(
                    log_fn=lambda msg: emit({"type": "log", "message": msg}),
                    on_connected=_on_mmd_connected,
                    on_disconnected=_on_mmd_disconnected,
                )
            try:
                mmd_bridge.configure(remote_settings)
            except Exception:
                pass
            # まばたきは再設定をまたいで動かし続ける（作り直すと間隔が引き直しになる）。
            # 切断中の send_text は失敗を返すだけで、MMDAgent-EX は切断ごとにモーフを既定へ戻す。
            _apply_mmd_blink_settings()
            _flush_pending_tts()
            return
        if cmd == "remote_plugin_stop":
            with pending_tts_lock:
                pending_tts_chunks.clear()
            if pacer is not None:
                try:
                    pacer.clear_and_reset()
                except Exception:
                    pass
            _stop_mmd_bridge()
            _dispose_mmd_pacer_only()
            try:
                emit(
                    {
                        "type": "remote_status",
                        "_remote_bridge_origin": "audio_output",
                        "connected": False,
                        "mode": "",
                    }
                )
            except Exception:
                pass
            return
        if cmd in {"mmd_audio_flush", "flush", "interrupt"}:
            # ペーサ未生成（output_mode=mmd など）でも、起動前に積まれた pending_tts_chunks を必ず捨てる。
            with pending_tts_lock:
                pending_tts_chunks.clear()
            if pacer is not None:
                try:
                    pacer.clear_and_reset()
                except Exception:
                    pass
            return
        if cmd == "update_settings":
            settings_obj = payload.get("settings")
            if not isinstance(settings_obj, dict):
                return
            try:
                bf = int(settings_obj.get("mmd_burst_snd_frames", burst_default) or burst_default)
            except (TypeError, ValueError):
                bf = burst_default
            bf = max(0, min(500, bf))
            burst_default = bf
            if "mmd_blink_enable" in settings_obj:
                blink_enable = normalize_blink_enable(settings_obj.get("mmd_blink_enable"), True)
            if "mmd_blink_interval_mean_sec" in settings_obj:
                blink_interval_mean_sec = normalize_blink_interval_mean_sec(
                    settings_obj.get("mmd_blink_interval_mean_sec")
                )
            if "mmd_blink_morph" in settings_obj:
                blink_morph = sanitize_blink_morph(settings_obj.get("mmd_blink_morph"))
            if "mmd_blink_mode" in settings_obj:
                blink_mode = normalize_blink_mode(settings_obj.get("mmd_blink_mode"))
            if "mmd_blink_dialog_active" in settings_obj:
                blink_dialog_active = bool(settings_obj.get("mmd_blink_dialog_active"))
            _apply_mmd_blink_settings()
            if pacer is not None:
                if output_mode == "mmd":
                    pacer.set_burst_snd_frames(bf)
                else:
                    # local/tcp はバースト無効（UI の mmd_burst は無視）
                    pacer.set_burst_snd_frames(0)
            return
        if cmd == "mmd_send_text":
            text = str(payload.get("text", ""))
            if text and mmd_bridge is not None:
                try:
                    mmd_bridge.send_text(text)
                except Exception:
                    pass
            return
        if cmd == "mmd_bind_head":
            if mmd_bridge is not None:
                try:
                    mmd_bridge.set_head_binding(
                        str(payload.get("bone", "") or ""), float(payload.get("eye_scale", 0.0) or 0.0),
                    )
                except Exception:
                    pass
            return
        if cmd == "mmd_av_preamble_once":
            ok = False
            if mmd_bridge is not None:
                try:
                    ok = bool(mmd_bridge.ensure_snd_segment_preamble())
                except Exception:
                    ok = False
            emit({"type": "mmd_av_preamble_done", "ok": ok})
            return
        if cmd == "mmd_mark_snd_segment_ended":
            if mmd_bridge is not None:
                try:
                    mmd_bridge.mark_snd_segment_ended()
                except Exception:
                    pass
            return
        if cmd == "tts_play_chunk":
            data_b64 = str(payload.get("audio_b64", "")).strip()
            if not data_b64:
                return
            try:
                pcm = base64.b64decode(data_b64)
            except Exception:
                return
            inject_maai = bool(payload.get("inject_maai", False))
            inject_only = bool(payload.get("inject_only", False))
            if pacer is None:
                with pending_tts_lock:
                    pending_tts_chunks.append(("chunk", pcm, inject_maai, inject_only))
                return
            try:
                pacer.enqueue_pcm(pcm, inject_maai=inject_maai, snd_enabled=not inject_only)
            except Exception:
                pass
            return
        if cmd == "tts_play_break":
            if "generation" in payload:
                try:
                    gen = int(payload.get("generation", 0) or 0)
                except (TypeError, ValueError):
                    gen = None
            else:
                gen = None
            if "complete_kind" in payload or "segment_index" in payload:
                ck = payload.get("complete_kind")
                si = payload.get("segment_index")
                try:
                    si_i = int(si) if si is not None else None
                except (TypeError, ValueError):
                    si_i = None
                ck_s = str(ck).strip() if ck is not None else None
            else:
                ck_s = None
                si_i = None
            if pacer is None:
                # MMD 接続前は chunk と順序を保って break もバッファし、接続時にまとめて流す。
                with pending_tts_lock:
                    pending_tts_chunks.append(("break", gen, ck_s, si_i))
                return
            pacer.enqueue_break(gen, complete_kind=ck_s, segment_index=si_i)
            return

    def _on_event(payload: dict[str, object]) -> None:
        _handle(payload)

    def _stdin_watcher() -> None:
        try:
            for raw in sys.stdin:
                if raw.strip().lower() == "stop":
                    stop_event.set()
                    return
        except Exception:
            pass
        finally:
            stop_event.set()

    threading.Thread(target=_stdin_watcher, daemon=True, name="audio-output-stdin").start()
    client.start_reader(_on_event, on_disconnected=lambda: stop_event.set())

    # ready は重い出力初期化より先に送る（sounddevice / TCP 先などでブロックしても Supervisor が待ち続けない）。
    emit({"type": "ready", "worker": "audio_output", "output_mode": output_mode})
    _rebuild_pacer()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if pacer is not None:
            try:
                pacer.close()
            except Exception:
                pass
        _stop_mmd_bridge()
        try:
            _close_tcp()
        except BaseException:
            pass
        finish_worker(client, lambda: emit({"type": "stopped", "worker": "audio_output"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
