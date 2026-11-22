from __future__ import annotations

from . import ort_windows_preload

ort_windows_preload.ensure_for_asr_engine(ort_windows_preload.peek_asr_engine_from_argv())

import argparse
import queue
import sys
import threading
import time
import numpy as np

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

from ..ipc.jsonline import JsonLineEventClient
from .service import ASRService
from .frame_types import AudioFrame
from ..ipc.queue_policy import QUEUE_POLICY_DROP_OLDEST, queue_put
from ..ipc.worker_runtime import connect_worker_client, finish_worker, parse_json_object, safe_emit, to_jsonable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASR worker process")
    parser.add_argument("--host", required=True, type=str)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--config-json", required=True, type=str)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        config = parse_json_object(args.config_json, worker_name="ASRWorker")
    except ValueError as ex:
        print(str(ex), file=sys.stderr, flush=True)
        return 2
    # 廃止キー（旧: 別ソケットの audio_ingress）。Orchestrator 経路では --host/--port の1本のみ。
    config.pop("audio_ingress_host", None)
    config.pop("audio_ingress_port", None)

    client = JsonLineEventClient(args.host, args.port)
    if not connect_worker_client(client, worker_name="ASRWorker"):
        return 2

    def emit(event_type: str, **data: object) -> None:
        safe_emit(client.send, event_type, **data)

    stop_event = threading.Event()
    force_finalize_request_seq = 0
    high_freq_event_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=64)
    # audio/input_worker の送信単位（320）と一致させ、ASR 内で Silero/MaAI 必要長へ再チャンクする。
    target_chunk_size = 320
    input_buffer = np.empty(0, dtype=np.float32)
    input_sample_start: int | None = None
    input_ts_start: float | None = None
    audio_accepting = False

    def _handle_audio_payload(obj: dict[str, object]) -> None:
        nonlocal input_buffer, input_sample_start, input_ts_start, audio_accepting
        if not audio_accepting:
            # Drop pre-init audio to avoid stale backlog after heavy model startup.
            input_buffer = np.empty(0, dtype=np.float32)
            input_sample_start = None
            input_ts_start = None
            return
        audio = np.asarray(obj.get("audio", []), dtype=np.float32).reshape(-1)
        if audio.size <= 0:
            return
        sample_start = int(obj.get("sample_start", 0) or 0)
        ts_start = float(obj.get("ts_start", time.time()) or time.time())
        # 送出側の壁時計時刻（端から端までの遅延計測用）。無ければ受信時刻で代替する。
        _emitted_raw = obj.get("emitted_at")
        origin_ts = float(_emitted_raw) if isinstance(_emitted_raw, (int, float)) else time.time()
        # 上流のファンアウト送信側がバックログ解消のため遅延フレームをドロップすると、
        # 受信ペイロードの sample_start が不連続（ジャンプ）する。input_ts_start を
        # サンプル数で自走させたままだと、ドロップした分だけクロックが過去へズレて
        # VAD/プロットの定常ズレが復活する。ギャップ検出時はペイロードが持つ絶対
        # sample_start / ts_start（audio/input_worker の壁時計アンカー）へ再同期する。
        expected_next = (
            input_sample_start + int(input_buffer.size)
            if input_sample_start is not None
            else None
        )
        if (
            input_sample_start is None
            or input_ts_start is None
            or expected_next is None
            or abs(sample_start - expected_next) > 1
        ):
            input_buffer = np.empty(0, dtype=np.float32)
            input_sample_start = sample_start
            input_ts_start = ts_start
        input_buffer = np.concatenate([input_buffer, audio]) if input_buffer.size > 0 else audio.copy()
        while input_buffer.size >= target_chunk_size:
            chunk = input_buffer[:target_chunk_size].copy()
            input_buffer = input_buffer[target_chunk_size:]
            chunk_sample_start = int(input_sample_start if input_sample_start is not None else sample_start)
            chunk_sample_end = chunk_sample_start + int(target_chunk_size)
            chunk_ts_start = float(input_ts_start if input_ts_start is not None else ts_start)
            sr = max(1, int(config.get("sample_rate", 16000) or 16000))
            chunk_ts_end = chunk_ts_start + (float(target_chunk_size) / float(sr))
            frame = AudioFrame(
                audio=chunk,
                sample_start=chunk_sample_start,
                sample_end=chunk_sample_end,
                ts_start=chunk_ts_start,
                ts_end=chunk_ts_end,
                # 再チャンクした断片は、いま処理しているペイロードが届いた時点で
                # 揃ったことになるので、そのペイロードの送出時刻を引き継ぐ。
                origin_ts=origin_ts,
            )
            queue_put(service.raw_audio_queue, frame, policy=QUEUE_POLICY_DROP_OLDEST)
            input_sample_start = chunk_sample_end
            input_ts_start = chunk_ts_end

    def _stdin_watcher() -> None:
        nonlocal force_finalize_request_seq
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if line.lower() == "stop":
                    stop_event.set()
                    return
                if line.lower() == "force_finalize":
                    try:
                        force_finalize_request_seq += 1
                        request_id = int(force_finalize_request_seq)
                        speaking = bool(service.request_force_finalize_current_utterance(request_id=request_id))
                        emit(
                            "force_finalize_status",
                            is_speaking=speaking,
                            request_id=request_id,
                            ts=time.time(),
                        )
                        if not speaking:
                            emit(
                                "force_finalize_outcome",
                                status="not_speaking",
                                request_id=request_id,
                                ts=time.time(),
                            )
                    except Exception:
                        pass
                    continue
            stop_event.set()
        except Exception:
            stop_event.set()

    threading.Thread(target=_stdin_watcher, daemon=True, name="asr-worker-stdin").start()

    def _on_audio_event(payload: dict[str, object]) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "system_audio_frame":
            if not audio_accepting:
                return
            audio_obj = payload.get("audio")
            audio_f32 = np.asarray(audio_obj, dtype=np.float32).reshape(-1)
            if audio_f32.size > 0:
                service.inject_system_audio(audio_f32)
            return
        if event_type == "audio_frame":
            _handle_audio_payload(payload)

    def _enqueue_high_freq_event(event_type: str, **data: object) -> None:
        payload: dict[str, object] = {"type": str(event_type)}
        payload.update(data)
        queue_put(high_freq_event_queue, payload, policy=QUEUE_POLICY_DROP_OLDEST)

    def _on_maai_result_from_service(result: dict[str, object]) -> None:
        _enqueue_high_freq_event("maai_result", result=to_jsonable(result))

    def _high_freq_sender_loop() -> None:
        while not stop_event.is_set():
            try:
                payload = high_freq_event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not isinstance(payload, dict):
                continue
            event_type = str(payload.get("type", "")).strip()
            if not event_type:
                continue
            event_data = {k: v for k, v in payload.items() if k != "type"}
            try:
                emit(event_type, **event_data)
            except Exception:
                pass

    service = ASRService(
        **config,
        on_log=lambda message: emit("log", message=message),
        on_error=lambda message: emit("error", message=message),
        on_result=lambda result: emit(
            "result",
            text=result.text,
            engine=result.engine,
            is_final=bool(getattr(result, "is_final", True)),
            is_streaming=bool(getattr(result, "is_streaming", False)),
            latency_sec=result.latency_sec,
            utterance_index=int(getattr(result, "utterance_index", 0)),
            end_of_utterance_ts=result.end_of_utterance_ts,
            segment_start_ts=getattr(result, "segment_start_ts", None),
            segment_end_ts=getattr(result, "segment_end_ts", None),
        ),
        on_turn_probability=lambda probability, ts: _enqueue_high_freq_event(
            "turn",
            probability=probability,
            ts=ts,
        ),
        on_maai_result=_on_maai_result_from_service,
        on_inference_state=lambda busy: emit("inference_state", busy=bool(busy)),
        on_force_finalize_outcome=lambda payload: emit("force_finalize_outcome", **to_jsonable(payload)),
    )

    threading.Thread(target=_high_freq_sender_loop, daemon=True, name="asr-worker-high-freq-sender").start()

    try:
        service.start()
    except Exception as ex:
        emit("fatal", message=str(ex))
        client.close()
        return 1

    client.start_reader(_on_audio_event, on_disconnected=lambda: stop_event.set())

    audio_accepting = True

    emit("ready")

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        try:
            service.stop()
        except Exception as ex:
            emit("error", message=f"ASR停止時にエラー: {ex}")
        finish_worker(client, lambda: emit("stopped"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

