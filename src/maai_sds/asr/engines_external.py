"""external-asr-tcp（外部プロセスと TCP で音声/結果をやり取りする）。

`asr/service.ASRService` から**そのまま移した**もの（mixin）。
本文は 1 文字も変えていない。エンジンごとにファイルを分けたのは、
1 エンジンを読む/直すときに他のエンジンを読み飛ばさなくて済むようにするため。
"""

from __future__ import annotations

import json
import socket
import time

from .frame_types import AudioFrame, SegmentLoopHooks
from ..utils import to_float_or_none


class AsrExternalEngineMixin:
    """external-asr-tcp（外部プロセスと TCP で音声/結果をやり取りする）。 `ASRService` に mixin する。"""

    def _init_engine_external(self) -> None:
        self._faster_whisper_model = None
        self._sherpa_recognizer = None

    def _external_asr_tcp_loop(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.external_asr_host, self.external_asr_port))
            server.listen(5)
            server.settimeout(0.5)
            self._external_server_socket = server
            self._log(f"External ASR TCP listening on {self.external_asr_host}:{self.external_asr_port}")
            while not self._stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                self._log(f"External ASR TCP client connected: {addr[0]}:{addr[1]}")
                try:
                    self._external_asr_read_connection(conn)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    self._log(f"External ASR TCP client disconnected: {addr[0]}:{addr[1]}")
        except Exception as ex:
            if not self._stop_event.is_set():
                self._error(f"External ASR TCP受信でエラー: {ex}")
        finally:
            self._external_server_socket = None
            try:
                server.close()
            except Exception:
                pass

    def _external_asr_audio_send_loop(self) -> None:
        sock: socket.socket | None = None
        self._set_force_finalize_speaking(False)
        try:
            self._ensure_vad_backend_initialized()
            segmentation_enabled = True
        except Exception as ex:
            self._error(f"External ASR セグメント境界推定を初期化できません: {ex}")
            segmentation_enabled = False
        while not self._stop_event.is_set():
            if sock is None:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2.0)
                    sock.connect((self.external_asr_audio_host, self.external_asr_audio_port))
                    sock.settimeout(None)
                    self._external_audio_socket = sock
                    self._log(
                        f"External ASR audio connected: {self.external_asr_audio_host}:{self.external_asr_audio_port}"
                    )
                except Exception as ex:
                    self._error(f"External ASR audio接続に失敗: {ex}")
                    if sock is not None:
                        try:
                            sock.close()
                        except Exception:
                            pass
                    sock = None
                    self._external_audio_socket = None
                    time.sleep(1.0)
                    continue

            try:
                def _on_frame(frame: AudioFrame) -> None:
                    assert sock is not None
                    sock.sendall(self._to_pcm16le_bytes(frame.audio))

                def _on_segment_start(frame: AudioFrame) -> None:
                    with self._external_segment_lock:
                        self._external_active_segment_start_ts = float(frame.ts_start)
                        self._external_last_segment_start_ts = float(frame.ts_start)

                def _on_segment_end(end_ts: float) -> None:
                    with self._external_segment_lock:
                        self._external_last_segment_end_ts = float(end_ts)
                        self._external_active_segment_start_ts = None

                self._run_segment_loop(
                    source_queue=self.raw_audio_queue,
                    hooks=SegmentLoopHooks(
                        emit_to_inference=False,
                        emit_to_smart_turn_on_noninference=False,
                        forward_to_turn_queue=True,
                        on_frame=_on_frame,
                        on_segment_start=(_on_segment_start if segmentation_enabled else None),
                        on_segment_end=(_on_segment_end if segmentation_enabled else None),
                    ),
                )
            except Exception as ex:
                self._error(f"External ASR audio送信に失敗: {ex}")
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
                self._external_audio_socket = None
                time.sleep(0.5)
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        self._external_audio_socket = None

    def _external_asr_handle_line(self, line: str) -> None:
        text_line = line.strip()
        if not text_line:
            return
        now = time.time()
        try:
            payload = json.loads(text_line)
        except json.JSONDecodeError:
            self._error("external-asr-tcp: JSON形式ではない行を受信しました。is_final/is_streaming必須です。")
            return
        parsed = self._validate_external_asr_payload(payload)
        if parsed is None:
            return
        text = str(parsed["text"])
        engine = str(parsed["engine"])
        is_final_raw = bool(parsed["is_final"])
        is_streaming_raw = bool(parsed["is_streaming"])
        payload_start_ts = to_float_or_none(parsed.get("payload_start_ts"))
        payload_end_ts = to_float_or_none(parsed.get("payload_end_ts"))
        with self._external_segment_lock:
            segment_start_hint = (
                payload_start_ts
                if payload_start_ts is not None
                else (self._external_active_segment_start_ts or self._external_last_segment_start_ts)
            )
            if bool(is_final_raw) and self._external_last_segment_end_ts is not None:
                segment_end_hint = self._external_last_segment_end_ts
            else:
                segment_end_hint = payload_end_ts if payload_end_ts is not None else now
        sample_count_hint = int(self.sample_rate * 0.2)
        if segment_start_hint is not None and segment_end_hint is not None:
            try:
                sample_count_hint = max(
                    1,
                    self._ts_to_sample_index(float(segment_end_hint))
                    - self._ts_to_sample_index(float(segment_start_hint)),
                )
            except Exception:
                sample_count_hint = int(self.sample_rate * 0.2)
        segment_start_ts, segment_end_ts = self._resolve_segment_bounds(
            segment_start_hint,
            segment_end_hint,
            sample_count_hint,
        )
        self._publish_result(
            text=text,
            end_ts=segment_end_ts,
            engine=engine,
            is_final=is_final_raw,
            is_streaming=is_streaming_raw,
            segment_start_ts=segment_start_ts,
            segment_end_ts=segment_end_ts,
        )

    def _external_asr_read_connection(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        buffer = bytearray()
        while not self._stop_event.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                lf_pos = buffer.find(b"\n")
                if lf_pos < 0:
                    break
                raw_line = bytes(buffer[:lf_pos])
                del buffer[: lf_pos + 1]
                self._external_asr_handle_line(raw_line.decode("utf-8", errors="replace"))
        if buffer:
            self._external_asr_handle_line(buffer.decode("utf-8", errors="replace"))
