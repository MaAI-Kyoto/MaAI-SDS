from __future__ import annotations

from . import ort_windows_preload

# compare の top/bottom いずれかが sherpa 系なら同梱 ORT を先にロードする
# （asr/worker と同じ方針。汎用 ensure() だと venv ORT 1.25 を先に import して
#  sherpa-onnx 同梱 1.17 が API version 不一致で推論失敗する）。
ort_windows_preload.ensure_for_asr_engines(ort_windows_preload.peek_compare_engines_from_argv())

import argparse
import base64
import json
import queue
import sys
import threading
import time
import numpy as np

from ..ipc.jsonline import JsonLineEventClient
from .service import ASRService
from .frame_types import AudioFrame
from ..ipc.queue_policy import QUEUE_POLICY_DROP_OLDEST, queue_put
from ..ipc.worker_runtime import connect_worker_client, finish_worker, parse_json_object, safe_emit, to_jsonable
from ..audio.convert import f32_from_pcm16_bytes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASR compare worker process")
    parser.add_argument("--host", required=True, type=str)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--config-json", required=True, type=str)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        config = parse_json_object(args.config_json, worker_name="ASRCompareWorker")
    except ValueError as ex:
        print(str(ex), file=sys.stderr, flush=True)
        return 2
    config.pop("audio_ingress_host", None)
    config.pop("audio_ingress_port", None)

    top_engine = str(config.get("compare_top_engine", "")).strip()
    bottom_engine = str(config.get("compare_bottom_engine", "")).strip()
    if not top_engine or not bottom_engine:
        print("[ASRCompareWorker][FATAL] compare_top_engine/bottom_engine are required.", file=sys.stderr, flush=True)
        return 2

    client = JsonLineEventClient(args.host, args.port)
    if not connect_worker_client(client, worker_name="ASRCompareWorker"):
        return 2

    def emit(event_type: str, **data: object) -> None:
        safe_emit(client.send, event_type, **data)

    stop_event = threading.Event()
    _services: list[ASRService] = []
    force_finalize_request_seq = {"top": 0, "bottom": 0}
    _sr_cfg = float(config.get("sample_rate", 16000) or 16000)
    _fr_cfg = float(config.get("frame_rate", 20) or 20)
    if not (_fr_cfg > 0.0) or _fr_cfg != _fr_cfg:
        _fr_cfg = 20.0
    target_chunk_size = (
        512
        if str(config.get("vad_backend", "maai")).strip().lower() == "silero"
        else max(1, int(round(_sr_cfg / max(1e-9, _fr_cfg))))
    )
    input_buffer = np.empty(0, dtype=np.float32)
    input_sample_start: int | None = None
    input_ts_start: float | None = None
    audio_accepting = False

    top_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=16)
    bottom_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=16)

    def _handle_audio_payload(obj: dict[str, object]) -> None:
        nonlocal input_buffer, input_sample_start, input_ts_start, audio_accepting
        if not audio_accepting:
            input_buffer = np.empty(0, dtype=np.float32)
            input_sample_start = None
            input_ts_start = None
            return
        audio = np.asarray(obj.get("audio", []), dtype=np.float32).reshape(-1)
        if audio.size <= 0:
            return
        sample_start = int(obj.get("sample_start", 0) or 0)
        ts_start = float(obj.get("ts_start", time.time()) or time.time())
        _emitted_raw = obj.get("emitted_at")
        origin_ts = float(_emitted_raw) if isinstance(_emitted_raw, (int, float)) else time.time()
        # 上流フレームドロップで sample_start が不連続になった場合、ペイロードの絶対
        # sample_start/ts_start へ再同期する（asr/worker と同じ。自走させるとクロックがズレる）。
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
            chunk_ts_end = chunk_ts_start + (float(target_chunk_size) / 16000.0)
            frame = AudioFrame(
                audio=chunk,
                sample_start=chunk_sample_start,
                sample_end=chunk_sample_end,
                ts_start=chunk_ts_start,
                ts_end=chunk_ts_end,
                origin_ts=origin_ts,
            )
            queue_put(top_queue, frame, policy=QUEUE_POLICY_DROP_OLDEST)
            queue_put(bottom_queue, frame, policy=QUEUE_POLICY_DROP_OLDEST)
            input_sample_start = chunk_sample_end
            input_ts_start = chunk_ts_end

    def _stdin_watcher() -> None:
        nonlocal input_buffer, input_sample_start, input_ts_start
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if line.lower() == "stop":
                    stop_event.set()
                    return
                if line.startswith("system_audio:"):
                    try:
                        audio_bytes = base64.b64decode(line[13:])
                        audio_f32 = f32_from_pcm16_bytes(audio_bytes)
                        for svc in _services:
                            svc.inject_system_audio(audio_f32)
                    except Exception:
                        pass
                    continue
                if line.startswith("system_audio_frame:"):
                    try:
                        obj = json.loads(line[len("system_audio_frame:") :])
                        audio_f32 = np.asarray(obj.get("audio", []), dtype=np.float32).reshape(-1)
                        if audio_f32.size > 0:
                            for svc in _services:
                                svc.inject_system_audio(audio_f32)
                    except Exception:
                        pass
                    continue
                if line.startswith("audio_frame:"):
                    try:
                        obj = json.loads(line[len("audio_frame:") :])
                        _handle_audio_payload(obj)
                    except Exception:
                        pass
                    continue
                if line.lower() == "force_finalize":
                    try:
                        for idx, svc in enumerate(_services):
                            slot = "top" if idx == 0 else "bottom"
                            force_finalize_request_seq[slot] = int(force_finalize_request_seq[slot]) + 1
                            request_id = int(force_finalize_request_seq[slot])
                            speaking = bool(
                                svc.request_force_finalize_current_utterance(request_id=request_id)
                            )
                            emit(
                                "force_finalize_status",
                                compare_slot=slot,
                                is_speaking=speaking,
                                request_id=request_id,
                                ts=time.time(),
                            )
                            if not speaking:
                                emit(
                                    "force_finalize_outcome",
                                    compare_slot=slot,
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

    threading.Thread(target=_stdin_watcher, daemon=True, name="asr-compare-worker-stdin").start()

    service_base = dict(config)
    service_base.pop("asr_engine", None)
    service_base.pop("compare_mode_enabled", None)
    service_base.pop("compare_top_engine", None)
    service_base.pop("compare_bottom_engine", None)
    maai_output_gui_plot = bool(service_base.pop("maai_output_gui_plot", False))

    service_top = ASRService(
        **service_base,
        asr_engine=top_engine,
        maai_output_gui_plot=maai_output_gui_plot,
        external_raw_audio_queue=top_queue,
        on_log=lambda message: emit("log", message=f"[top:{top_engine}] {message}"),
        on_error=lambda message: emit("error", message=f"[top:{top_engine}] {message}"),
        on_result=lambda result: emit(
            "result",
            compare_slot="top",
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
        on_turn_probability=lambda probability, ts: emit("turn", compare_slot="top", probability=probability, ts=ts),
        on_inference_state=lambda busy: emit("inference_state", compare_slot="top", busy=bool(busy)),
        on_force_finalize_outcome=lambda payload: emit(
            "force_finalize_outcome",
            compare_slot="top",
            **to_jsonable(payload),
        ),
        on_maai_result=lambda result: emit("maai_result", compare_slot="top", result=to_jsonable(result)),
    )
    service_bottom = ASRService(
        **service_base,
        asr_engine=bottom_engine,
        maai_output_gui_plot=False,
        external_raw_audio_queue=bottom_queue,
        on_log=lambda message: emit("log", message=f"[bottom:{bottom_engine}] {message}"),
        on_error=lambda message: emit("error", message=f"[bottom:{bottom_engine}] {message}"),
        on_result=lambda result: emit(
            "result",
            compare_slot="bottom",
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
        on_turn_probability=lambda probability, ts: emit(
            "turn", compare_slot="bottom", probability=probability, ts=ts
        ),
        on_inference_state=lambda busy: emit("inference_state", compare_slot="bottom", busy=bool(busy)),
        on_force_finalize_outcome=lambda payload: emit(
            "force_finalize_outcome",
            compare_slot="bottom",
            **to_jsonable(payload),
        ),
        on_maai_result=lambda result: emit("maai_result", compare_slot="bottom", result=to_jsonable(result)),
    )

    _services.extend([service_top, service_bottom])
    try:
        service_top.start()
        service_bottom.start()
    except Exception as ex:
        emit("fatal", message=str(ex))
        client.close()
        return 1

    def _on_audio_event(payload: dict[str, object]) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "system_audio_frame":
            if not audio_accepting:
                return
            audio_f32 = np.asarray(payload.get("audio", []), dtype=np.float32).reshape(-1)
            if audio_f32.size > 0:
                for svc in _services:
                    svc.inject_system_audio(audio_f32)
            return
        if event_type == "audio_frame":
            _handle_audio_payload(payload)

    client.start_reader(_on_audio_event, on_disconnected=lambda: stop_event.set())

    audio_accepting = True

    emit("ready")
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except Exception as ex:
        emit("error", message=f"比較モード実行中にエラー: {ex}")
    finally:
        try:
            service_top.stop()
        except Exception as ex:
            emit("error", message=f"上段ASR停止時にエラー: {ex}")
        try:
            service_bottom.stop()
        except Exception as ex:
            emit("error", message=f"下段ASR停止時にエラー: {ex}")
        finish_worker(client, lambda: emit("stopped"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

