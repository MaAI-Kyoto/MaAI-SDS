"""Qwen 系 ASR エンジン（vLLM/WebSocket の qwen3-asr と ONNX の qwen3-asr-onnx）。

`asr/service.ASRService` に mixin する。公開 int8 モデルはエンジン管理の
`asr-qwen3-onnx` 適用で取得し、設定パスが欠けていれば展開済みディレクトリを探す。
"""

from __future__ import annotations

import base64
import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import engine_manager
from .frame_types import AudioFrame, SegmentLoopHooks


class AsrQwenEnginesMixin:
    """Qwen 系 ASR エンジン（vLLM/WebSocket の qwen3-asr と ONNX の qwen3-asr-onnx）。 `ASRService` に mixin する。"""

    def _qwen3_onnx_layout_from_dir(self, root: Path) -> dict[str, str] | None:
        """k2-fsa 配布物（int8）または自前 fp32 のディレクトリから 4 ファイルを拾う。"""
        conv = root / "conv_frontend.onnx"
        if not conv.is_file():
            return None
        encoder = None
        decoder = None
        for name in ("encoder.int8.onnx", "encoder.onnx"):
            cand = root / name
            if cand.is_file():
                encoder = cand
                break
        for name in ("decoder.int8.onnx", "decoder.onnx"):
            cand = root / name
            if cand.is_file():
                decoder = cand
                break
        tokenizer = root / "tokenizer"
        if encoder is None or decoder is None or not tokenizer.exists():
            return None
        return {
            "conv_frontend": str(conv),
            "encoder": str(encoder),
            "decoder": str(decoder),
            "tokenizer": str(tokenizer),
        }

    def _discover_qwen3_onnx_paths(self) -> None:
        """設定パスが空／古い fp32 名で欠けているとき、展開済みディレクトリを探す。"""
        def _ok(path: str) -> bool:
            return bool(path) and Path(path).exists()

        if all(_ok(p) for p in (
            self.qwen3_onnx_conv_frontend_path,
            self.qwen3_onnx_encoder_path,
            self.qwen3_onnx_decoder_path,
            self.qwen3_onnx_tokenizer_path,
        )):
            return

        roots: list[Path] = []
        for configured in (
            self.qwen3_onnx_conv_frontend_path,
            self.qwen3_onnx_encoder_path,
            self.qwen3_onnx_decoder_path,
            self.qwen3_onnx_tokenizer_path,
        ):
            if not configured:
                continue
            p = Path(configured)
            d = p if p.is_dir() else p.parent
            if d.is_dir() and d not in roots:
                roots.append(d)
        repo = Path(self._project_root())
        for d in sorted(repo.glob("sherpa-onnx-qwen3-asr-*")):
            if d.is_dir() and d not in roots:
                roots.append(d)

        for root in roots:
            found = self._qwen3_onnx_layout_from_dir(root)
            if found is None:
                continue
            self.qwen3_onnx_conv_frontend_path = found["conv_frontend"]
            self.qwen3_onnx_encoder_path = found["encoder"]
            self.qwen3_onnx_decoder_path = found["decoder"]
            self.qwen3_onnx_tokenizer_path = found["tokenizer"]
            self._log(f"qwen3-asr-onnx: モデルを自動検出しました dir={root.name}")
            return

    def _init_engine_qwen3_onnx(self) -> None:
        """sherpa-onnx の Qwen3-ASR onnx（オフライン）。既存 sherpa と同じデコード経路を使う。"""
        engine_manager.require_module_installed("sherpa_onnx", "sherpa-onnx")
        try:
            import sherpa_onnx
        except Exception as ex:
            raise RuntimeError(
                f"sherpa-onnx をインポートできません: {ex}"
            ) from ex
        self._discover_qwen3_onnx_paths()
        required_paths = [
            ("conv_frontend", self.qwen3_onnx_conv_frontend_path),
            ("encoder", self.qwen3_onnx_encoder_path),
            ("decoder", self.qwen3_onnx_decoder_path),
            ("tokenizer", self.qwen3_onnx_tokenizer_path),
        ]
        missing = [name for name, path in required_paths if not path or not Path(path).exists()]
        if missing:
            raise RuntimeError(
                f"Qwen3-ASR onnx モデルファイルが不足しています: {', '.join(missing)}。"
                " エンジン管理で「Qwen3-ASR ONNX (ASR)」にチェックを入れて適用するか、"
                " k2-fsa の int8 配布物（sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25）を展開してください。"
            )
        self._log(
            "qwen3-asr-onnx: loading sherpa-onnx "
            f"{self._sherpa_onnx_version_str()} provider={self.qwen3_onnx_provider} "
            f"threads={self.qwen3_onnx_num_threads} "
            f"encoder={self.qwen3_onnx_encoder_path}"
        )
        onnx_paths_l = (
            f"{self.qwen3_onnx_encoder_path} {self.qwen3_onnx_decoder_path}".lower()
        )
        if self.qwen3_onnx_provider == "cuda" and "int8" in onnx_paths_l:
            self._log(
                "qwen3-asr-onnx: int8 を CUDA に載せると MatMulInteger が CPU に落ち、"
                "KV コピーの分だけ遅くなることが多いです。同じモデルなら CPU + 多スレッド推奨。"
            )
        if self.nonstream_pseudo_enabled and self.nonstream_pseudo_chunk_sec < 1.0:
            self._log(
                "qwen3-asr-onnx: Pseudo streaming が "
                f"{self.nonstream_pseudo_chunk_sec:.2f}s 間隔です。"
                " オフライン Qwen3 は発話全体を毎回再推論するため、"
                "1.0s 以上にするか無効化すると体感が速くなります。"
            )
        self._sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=self.qwen3_onnx_conv_frontend_path,
            encoder=self.qwen3_onnx_encoder_path,
            decoder=self.qwen3_onnx_decoder_path,
            tokenizer=self.qwen3_onnx_tokenizer_path,
            num_threads=self.qwen3_onnx_num_threads,
            sample_rate=self.sherpa_sample_rate,
            feature_dim=self.qwen3_onnx_feature_dim,
            decoding_method="greedy_search",
            provider=self.qwen3_onnx_provider,
        )
        self._faster_whisper_model = None
        self._nemotron_onnx_recognizer = None
        self._sherpa_needs_lead_silence = False

    def _init_engine_qwen(self) -> None:
        self._faster_whisper_model = None
        self._sherpa_recognizer = None
        self._qwen_ws_endpoint = f"ws://{self.qwen_host}:{self.qwen_port}{self.qwen_ws_path}"

    def _qwen_stream_loop(self) -> None:
        if self.qwen_transport == "http":
            self._error("Qwen ASR HTTP transportは削除されました。qwen_transport=ws を使用してください。")
            return
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            try:
                self._qwen_stream_ws_loop()
                return
            except Exception as ex:
                self._error(f"Qwen ASR WS streaming failed (attempt={attempt}): {ex}")
                if self._stop_event.is_set():
                    return
                time.sleep(min(2.0, 0.2 * max(1, attempt)))

    def _qwen_stream_ws_loop(self) -> None:
        try:
            import websocket  # type: ignore
        except Exception as ex:
            raise RuntimeError("websocket-client が見つかりません。`pip install websocket-client` を実行してください。") from ex

        if self._silero_model is None and self._maai_instance is None:
            self._ensure_vad_backend_initialized()

        emit_samples = max(1, int(self.sample_rate * max(0.0, self.qwen_chunk_sec)))
        frame_hz = max(1.0, float(self.sample_rate) / float(max(1, self.block_size)))
        activation_frames = max(1, int(math.ceil(self.vad_start_hold_sec * frame_hz)))
        deactivation_frames = max(1, int(math.ceil(self.vad_end_hold_sec * frame_hz)))
        pre_roll_frames = max(1, int(math.ceil(max(0.0, self.pre_roll_buffer_sec) * frame_hz)))
        pre_roll_buffer: deque[AudioFrame] = deque(maxlen=pre_roll_frames)

        @dataclass
        class QwenLoopState:
            buffered: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
            buffered_start_sample: int | None = None
            last_text: str = ""
            session_seq: int = 0
            active_session_id: str | None = None
            speech_active: bool = False
            activation_count: int = 0
            deactivation_count: int = 0
            active_segment_start_ts: float | None = None
            active_segment_end_ts: float | None = None

        @dataclass
        class SmartTurnLoopState:
            pre_buffer: deque[AudioFrame]
            speech_active: bool = False
            activation_count: int = 0
            deactivation_count: int = 0
            end_of_utterance_ts: float | None = None
            segment_start_ts: float | None = None
            current_segment_frames: list[AudioFrame] = field(default_factory=list)

        qwen_state = QwenLoopState()
        smart_turn_state = SmartTurnLoopState(pre_buffer=deque(maxlen=activation_frames))
        last_suppressed_resp_type = ""
        ws = None
        ws_msg_incoming: queue.Queue[dict] = queue.Queue(maxsize=512)
        sync_reply_lock = threading.Lock()
        pending_chunk_end_ts: deque[float] = deque()
        sync_reply_target: list[queue.Queue[dict] | None] = [None]
        recv_stop = threading.Event()
        recv_fatal: list[BaseException | None] = [None]
        recv_thread: threading.Thread | None = None
        # Silero/MaAI は内部バッファが溜まるまで None を返す。None を 0.0 とみなすと発話開始カウンタが毎フレームリセットされる。
        qwen_last_vad_score: list[float | None] = [None]

        def _ws_rpc(payload: dict, *, timeout: float | None = None) -> dict | None:
            assert ws is not None
            reply_q: queue.Queue[dict] = queue.Queue(maxsize=1)
            with sync_reply_lock:
                sync_reply_target[0] = reply_q
            try:
                ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception:
                with sync_reply_lock:
                    sync_reply_target[0] = None
                raise
            try:
                return reply_q.get(timeout=timeout if timeout is not None else float(self.qwen_timeout_sec))
            except queue.Empty:
                return None
            finally:
                with sync_reply_lock:
                    sync_reply_target[0] = None

        def _publish_response(
            resp: dict,
            end_ts: float | None = None,
            segment_start_ts: float | None = None,
            segment_end_ts: float | None = None,
        ) -> str:
            nonlocal last_suppressed_resp_type
            resp_type = str(resp.get("type", "")).strip().lower()
            if resp_type in {"partial_suppressed_by_min_audio", "final_suppressed_by_min_audio"}:
                if resp_type != last_suppressed_resp_type:
                    received_sec = float(resp.get("received_audio_sec", 0.0) or 0.0)
                    min_sec = float(resp.get("min_return_audio_sec", 0.0) or 0.0)
                    self._log(
                        "Qwen ASR WS result suppressed by min audio: "
                        f"received={received_sec:.3f}s < min={min_sec:.3f}s"
                    )
                    last_suppressed_resp_type = resp_type
                return resp_type
            last_suppressed_resp_type = ""
            text = self._clean_qwen_text(str(resp.get("text", "")).strip())
            is_final = resp_type in {"final", "final_by_no_text_flush", "final_suppressed_by_min_audio"}
            should_emit = bool(text) and (is_final or text != qwen_state.last_text)
            if should_emit:
                qwen_state.last_text = text
                self._publish_result(
                    text=text,
                    end_ts=end_ts or time.time(),
                    engine="qwen3-asr",
                    is_final=is_final,
                    is_streaming=True,
                    segment_start_ts=segment_start_ts,
                    segment_end_ts=segment_end_ts,
                )
            return resp_type

        def _handle_streaming_response(data: dict) -> None:
            end_ts = pending_chunk_end_ts.popleft() if pending_chunk_end_ts else None
            now = float(end_ts) if end_ts is not None else time.time()
            qwen_state.active_segment_end_ts = float(now)
            resp_type = _publish_response(
                data,
                end_ts=now,
                segment_start_ts=qwen_state.active_segment_start_ts,
                segment_end_ts=now,
            )
            if resp_type in {
                "final",
                "final_by_no_text_flush",
                "final_suppressed_by_min_audio",
            }:
                if resp_type == "final_by_no_text_flush":
                    self._log(
                        f"Qwen ASR WS session closed by no_text_flush: session_id={qwen_state.active_session_id}"
                    )
                qwen_state.active_session_id = None
                qwen_state.active_segment_start_ts = None
                qwen_state.activation_count = 0
                qwen_state.deactivation_count = 0
                qwen_state.speech_active = False
                qwen_state.buffered = np.zeros(0, dtype=np.float32)
                qwen_state.buffered_start_sample = None

        def _drain_qwen_ws_messages() -> None:
            while True:
                try:
                    item = ws_msg_incoming.get_nowait()
                except queue.Empty:
                    break
                _handle_streaming_response(item)

        def _qwen_recv_loop() -> None:
            assert ws is not None
            while not recv_stop.is_set() and not self._stop_event.is_set():
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as ex:
                    recv_fatal[0] = ex
                    break
                try:
                    data = json.loads(raw)
                except Exception as ex:
                    recv_fatal[0] = ex
                    break
                if not isinstance(data, dict):
                    recv_fatal[0] = RuntimeError("websocket response is not a JSON object")
                    break
                if str(data.get("type", "")) == "error":
                    recv_fatal[0] = RuntimeError(str(data.get("message", "unknown websocket error")))
                    break

                with sync_reply_lock:
                    st = sync_reply_target[0]
                if st is not None:
                    try:
                        st.put_nowait(data)
                    except queue.Full:
                        pass
                    continue

                try:
                    ws_msg_incoming.put_nowait(data)
                except queue.Full:
                    self._log("Qwen ASR WS incoming queue full (dropping server message)")

        def _start_session(session_id: str, start_ts: float | None = None) -> None:
            self._log(f"Qwen ASR WS session start: session_id={session_id}")
            qwen_state.active_segment_start_ts = float(start_ts) if start_ts is not None else time.time()
            ack = _ws_rpc(
                {
                    "type": "start_session",
                    "session_id": session_id,
                    "language": self.qwen_language,
                    "chunk_size_sec": self.qwen_chunk_sec,
                    "unfixed_chunk_num": 2,
                    "unfixed_token_num": 5,
                    "no_text_flush_sec": self.qwen_no_text_flush_sec,
                    "min_return_audio_sec": self.qwen_min_return_audio_sec,
                },
                timeout=max(float(self.qwen_timeout_sec), 90.0),
            )
            if ack is None:
                raise RuntimeError("Qwen ASR WS start_session timed out waiting for server reply")

        def _send_audio_chunk(
            session_id: str, audio: np.ndarray, bootstrap: bool, chunk_end_ts: float | None = None
        ) -> None:
            if qwen_state.active_session_id != session_id:
                return
            now = float(chunk_end_ts) if chunk_end_ts is not None else time.time()
            qwen_state.active_segment_end_ts = float(now)
            pending_chunk_end_ts.append(now)
            assert ws is not None
            ws.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "session_id": session_id,
                        "audio_b64": base64.b64encode(self._to_pcm16le_bytes(audio)).decode("ascii"),
                        "bootstrap": bool(bootstrap),
                    },
                    ensure_ascii=False,
                )
            )

        def _close_active_session() -> None:
            sid = qwen_state.active_session_id
            if sid is None:
                return
            self._log(f"Qwen ASR WS session finish request: session_id={sid}")
            flush_audio: np.ndarray | None = None
            flush_end_ts: float | None = None
            if qwen_state.buffered.shape[0] > 0:
                flush_end_ts = (
                    self._sample_index_to_ts(qwen_state.buffered_start_sample + int(qwen_state.buffered.shape[0]))
                    if qwen_state.buffered_start_sample is not None
                    else None
                )
                flush_audio = np.asarray(qwen_state.buffered, dtype=np.float32).copy()
                qwen_state.buffered = np.zeros(0, dtype=np.float32)
                qwen_state.buffered_start_sample = None

            try:
                if flush_audio is not None and flush_audio.shape[0] > 0:
                    _send_audio_chunk(sid, flush_audio, bootstrap=False, chunk_end_ts=flush_end_ts)
                    _drain_qwen_ws_messages()

                resp = _ws_rpc(
                    {"type": "finish_session", "session_id": sid},
                    timeout=max(90.0, float(self.qwen_timeout_sec)),
                ) or {}

                final_end_ts = (
                    float(qwen_state.active_segment_end_ts)
                    if qwen_state.active_segment_end_ts is not None
                    else time.time()
                )
                _publish_response(
                    resp,
                    end_ts=final_end_ts,
                    segment_start_ts=qwen_state.active_segment_start_ts,
                    segment_end_ts=final_end_ts,
                )
                self._log(f"Qwen ASR WS session finished: session_id={sid}")

                qwen_state.active_session_id = None
                qwen_state.speech_active = False
                qwen_state.activation_count = 0
                qwen_state.deactivation_count = 0
                qwen_state.last_text = ""
                qwen_state.active_segment_start_ts = None
                qwen_state.active_segment_end_ts = None
                qwen_state.buffered = np.zeros(0, dtype=np.float32)
                qwen_state.buffered_start_sample = None

            except Exception:
                try:
                    assert ws is not None
                    ws.send(json.dumps({"type": "cancel_session", "session_id": sid}, ensure_ascii=False))
                    self._log(f"Qwen ASR WS session cancelled: session_id={sid}")
                except Exception:
                    pass
                qwen_state.active_session_id = None
                qwen_state.speech_active = False
                qwen_state.activation_count = 0
                qwen_state.deactivation_count = 0
                qwen_state.last_text = ""
                qwen_state.active_segment_start_ts = None
                qwen_state.active_segment_end_ts = None
                qwen_state.buffered = np.zeros(0, dtype=np.float32)
                qwen_state.buffered_start_sample = None

        try:
            ws = websocket.create_connection(self._qwen_ws_endpoint, timeout=self.qwen_timeout_sec)
            # 初回ウォームアップ時に推論応答が遅いケースがあるため、read timeout は余裕を持たせる。
            ws.settimeout(max(float(self.qwen_timeout_sec), 90.0))
            recv_thread = threading.Thread(target=_qwen_recv_loop, daemon=True, name="asr-qwen-ws-recv")
            recv_thread.start()

            def _on_frame(frame: AudioFrame) -> None:
                _drain_qwen_ws_messages()
                x = frame.audio.astype(np.float32)
                pre_roll_buffer.append(frame)
                smart_turn_state.pre_buffer.append(frame)
                vad_raw = self._get_vad_score(x, frame)
                if vad_raw is not None:
                    qwen_last_vad_score[0] = float(vad_raw)
                effective_vad = float(vad_raw) if vad_raw is not None else qwen_last_vad_score[0]
                is_vad_on = bool(
                    effective_vad is not None and effective_vad > self.vad_start_threshold
                )

                if self.turn_backend == "smart-turn":
                    if not smart_turn_state.speech_active:
                        if is_vad_on:
                            smart_turn_state.activation_count += 1
                            if smart_turn_state.activation_count >= activation_frames:
                                smart_turn_state.speech_active = True
                                smart_turn_state.deactivation_count = 0
                                smart_turn_state.end_of_utterance_ts = None
                                smart_turn_state.current_segment_frames = list(smart_turn_state.pre_buffer)
                                smart_turn_state.segment_start_ts = (
                                    float(smart_turn_state.current_segment_frames[0].ts_start)
                                    if smart_turn_state.current_segment_frames
                                    else float(frame.ts_start)
                                )
                        elif vad_raw is not None:
                            smart_turn_state.activation_count = 0
                            smart_turn_state.pre_buffer.clear()
                    else:
                        smart_turn_state.current_segment_frames.append(frame)
                        if vad_raw is not None and vad_raw < self.vad_end_threshold:
                            if smart_turn_state.deactivation_count == 0:
                                smart_turn_state.end_of_utterance_ts = float(frame.ts_end)
                            smart_turn_state.deactivation_count += 1
                            if smart_turn_state.deactivation_count >= deactivation_frames:
                                if smart_turn_state.current_segment_frames:
                                    segment_audio = np.concatenate(
                                        [f.audio for f in smart_turn_state.current_segment_frames], axis=0
                                    )
                                    min_samples = int(self.sample_rate * self.min_segment_sec)
                                    if len(segment_audio) >= min_samples:
                                        payload = {
                                            "audio": segment_audio,
                                            "start_ts": smart_turn_state.segment_start_ts,
                                            "end_ts": smart_turn_state.end_of_utterance_ts
                                            or float(smart_turn_state.current_segment_frames[-1].ts_end),
                                        }
                                        self._queue_put(
                                            self.smart_turn_queue, payload, policy=self.QUEUE_POLICY_DROP_OLDEST
                                        )
                                smart_turn_state.speech_active = False
                                smart_turn_state.activation_count = 0
                                smart_turn_state.deactivation_count = 0
                                smart_turn_state.end_of_utterance_ts = None
                                smart_turn_state.segment_start_ts = None
                                smart_turn_state.current_segment_frames = []
                                smart_turn_state.pre_buffer.clear()
                        elif vad_raw is not None:
                            smart_turn_state.deactivation_count = 0
                            smart_turn_state.end_of_utterance_ts = None

                if not qwen_state.speech_active:
                    if qwen_state.active_session_id is not None:
                        # サーバー主導終端(no_text_flush)を有効にするため、
                        # 非発話中でも開いているセッションには通常チャンクを送り続ける。
                        if qwen_state.buffered_start_sample is None:
                            qwen_state.buffered_start_sample = int(frame.sample_start)
                        qwen_state.buffered = np.concatenate([qwen_state.buffered, x], axis=0)
                        while qwen_state.buffered.shape[0] >= emit_samples:
                            seg = qwen_state.buffered[:emit_samples]
                            qwen_state.buffered = qwen_state.buffered[emit_samples:]
                            seg_end_ts = (
                                self._sample_index_to_ts(qwen_state.buffered_start_sample + emit_samples)
                                if qwen_state.buffered_start_sample is not None
                                else float(frame.ts_end)
                            )
                            if qwen_state.buffered_start_sample is not None:
                                qwen_state.buffered_start_sample += emit_samples
                            if qwen_state.buffered.shape[0] == 0:
                                qwen_state.buffered_start_sample = None
                            _send_audio_chunk(
                                qwen_state.active_session_id, seg, bootstrap=False, chunk_end_ts=seg_end_ts
                            )
                            _drain_qwen_ws_messages()
                            if qwen_state.active_session_id is None:
                                break
                        if qwen_state.active_session_id is not None:
                            qwen_state.activation_count = 0
                        return

                    if is_vad_on:
                        qwen_state.activation_count += 1
                        if qwen_state.activation_count >= activation_frames:
                            if qwen_state.active_session_id is not None:
                                _close_active_session()
                            qwen_state.session_seq += 1
                            qwen_state.active_session_id = (
                                f"maai-{int(time.time() * 1000)}-{threading.get_ident()}-{qwen_state.session_seq}"
                            )
                            bootstrap_end_sample = int(frame.sample_end)
                            bootstrap_start_sample = max(0, bootstrap_end_sample - int(emit_samples))
                            bootstrap_start_ts = self._sample_index_to_ts(bootstrap_start_sample)
                            bootstrap_end_ts = self._sample_index_to_ts(bootstrap_end_sample)
                            _start_session(qwen_state.active_session_id, start_ts=float(bootstrap_start_ts))
                            if pre_roll_buffer:
                                bootstrap_audio = np.concatenate([f.audio for f in pre_roll_buffer], axis=0)
                            else:
                                bootstrap_audio = np.zeros(0, dtype=np.float32)
                            if bootstrap_audio.shape[0] >= emit_samples:
                                bootstrap_audio = bootstrap_audio[-emit_samples:]
                            else:
                                pad = np.zeros(emit_samples - bootstrap_audio.shape[0], dtype=np.float32)
                                bootstrap_audio = np.concatenate([pad, bootstrap_audio], axis=0)
                            qwen_state.activation_count = 0
                            qwen_state.deactivation_count = 0
                            qwen_state.speech_active = True
                            qwen_state.last_text = ""
                            _send_audio_chunk(
                                qwen_state.active_session_id,
                                bootstrap_audio,
                                bootstrap=True,
                                chunk_end_ts=float(bootstrap_end_ts),
                            )
                            _drain_qwen_ws_messages()
                            if qwen_state.active_session_id is None:
                                qwen_state.active_segment_start_ts = None
                                qwen_state.speech_active = False
                                qwen_state.deactivation_count = 0
                                qwen_state.buffered = np.zeros(0, dtype=np.float32)
                                qwen_state.buffered_start_sample = None
                    elif vad_raw is not None:
                        qwen_state.activation_count = 0
                    return

                if qwen_state.active_session_id is None:
                    qwen_state.speech_active = False
                    qwen_state.activation_count = 0
                    qwen_state.deactivation_count = 0
                    qwen_state.buffered = np.zeros(0, dtype=np.float32)
                    qwen_state.buffered_start_sample = None
                    return

                if qwen_state.buffered_start_sample is None:
                    qwen_state.buffered_start_sample = int(frame.sample_start)
                qwen_state.buffered = np.concatenate([qwen_state.buffered, x], axis=0)
                while qwen_state.buffered.shape[0] >= emit_samples:
                    seg = qwen_state.buffered[:emit_samples]
                    qwen_state.buffered = qwen_state.buffered[emit_samples:]
                    seg_end_ts = (
                        self._sample_index_to_ts(qwen_state.buffered_start_sample + emit_samples)
                        if qwen_state.buffered_start_sample is not None
                        else float(frame.ts_end)
                    )
                    if qwen_state.buffered_start_sample is not None:
                        qwen_state.buffered_start_sample += emit_samples
                    if qwen_state.buffered.shape[0] == 0:
                        qwen_state.buffered_start_sample = None
                    _send_audio_chunk(qwen_state.active_session_id, seg, bootstrap=False, chunk_end_ts=seg_end_ts)
                    _drain_qwen_ws_messages()
                    if qwen_state.active_session_id is None:
                        break

                if not qwen_state.speech_active:
                    return

                if vad_raw is not None:
                    if vad_raw < self.vad_end_threshold:
                        qwen_state.deactivation_count += 1
                        if qwen_state.deactivation_count >= deactivation_frames:
                            if self.qwen_finish_on_vad_end:
                                # VAD終端時に明示的にセッションを閉じる。
                                _close_active_session()
                            else:
                                # 終端はクライアントVADで閉じず、サーバー no_text_flush に委ねる。
                                qwen_state.speech_active = False
                                qwen_state.deactivation_count = 0
                                qwen_state.activation_count = 0
                    else:
                        qwen_state.deactivation_count = 0

            self._run_segment_loop(
                source_queue=self.raw_audio_queue,
                hooks=SegmentLoopHooks(
                    emit_to_inference=False,
                    emit_to_smart_turn_on_noninference=False,
                    forward_to_turn_queue=True,
                    process_only_frames=True,
                    on_frame=_on_frame,
                    # VAD end 後も finish_on_vad_end=False ならサーバ主導終了まで session を開いたまま
                    # 音声を送り続ける。その間 speech_active は False のため raw バックログ破棄で
                    # 発話末尾が欠落しうる(#9)。open session 中も発話中扱いにする。
                    is_in_active_segment=lambda: qwen_state.speech_active
                    or qwen_state.active_session_id is not None,
                ),
            )

            _drain_qwen_ws_messages()

            if recv_fatal[0] is not None:
                raise recv_fatal[0]

            if qwen_state.active_session_id is not None:
                _close_active_session()
        finally:
            recv_stop.set()
            if recv_thread is not None and recv_thread.is_alive():
                recv_thread.join(timeout=3.0)
            if ws is not None:
                try:
                    sid_cancel = qwen_state.active_session_id
                    if sid_cancel is not None:
                        ws.send(
                            json.dumps({"type": "cancel_session", "session_id": sid_cancel}, ensure_ascii=False)
                        )
                except Exception:
                    pass
                try:
                    ws.close()
                except Exception:
                    pass
