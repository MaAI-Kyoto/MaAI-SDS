from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import multiprocessing
import os
import queue
import re
import site
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..ipc.jsonline import JsonLineDuplexClient
from ..audio.convert import (
    audio_any_to_pcm16be_16k_mono,
    pcm16le_to_pcm16be_16k,
    pcmf32le_to_pcm16le,
    wav_bytes_to_pcm16be_16k_mono,
)
from ..ipc.worker_runtime import connect_worker_client, parse_json_object, safe_emit

from .. import paths as _paths
from .engines import (
    AivisSpeechEngine,
    CartesiaRealtimeEngine,
    make_irodori_engine,
    Qwen3TTSCppEngine,
    RealtimeTTSKokoroEngine,
    SupertonicEngine,
    TextStreamAdapterSession,
    TsukasaEngine,
    VoicevoxCoreEngine,
)


def _lfm2_audio_engine(cfg: dict[str, Any]) -> Any:
    from .engines_audiocpp import Lfm2AudioEngine

    return Lfm2AudioEngine(cfg)


ENGINE_FACTORIES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "Qwen3-TTS-CPP": Qwen3TTSCppEngine,
    "VOICEVOX-Core": VoicevoxCoreEngine,
    "Kokoro-RealtimeTTS": RealtimeTTSKokoroEngine,
    "Cartesia-Realtime": CartesiaRealtimeEngine,
    "AivisSpeech": AivisSpeechEngine,
    "Supertonic-3": SupertonicEngine,
    "Irodori-v4": make_irodori_engine,
    "Tsukasa": TsukasaEngine,
    "LFM2.5-Audio": _lfm2_audio_engine,
}

# `tts/playback_service` と同一の集合を 2 箇所で定義していた（片方だけ直すと食い違う。
# 向こうの docstring も「tts/worker の BINARY_STREAM_ENGINES と対応」と書いていた）。
# 参照専用なので単一ソースから import する（Phase 4）。
from .playback_service import BINARY_STREAM_ENGINES
from ..audio.convert import Pcm16BeLeadingSilenceGate, strip_leading_silence_pcm16be
TEXT_STREAM_ENGINES = {
    "Qwen3-TTS-CPP",
    "VOICEVOX-Core",
    "Kokoro-RealtimeTTS",
    "Cartesia-Realtime",
    "AivisSpeech",
    "Supertonic-3",
    "Irodori-v4",
    "Tsukasa",
    "LFM2.5-Audio",
}
NATIVE_TEXT_STREAM_ENGINES = {"Kokoro-RealtimeTTS"}


def _build_engine(engine_name: str, config: dict[str, Any]) -> Any:
    factory = ENGINE_FACTORIES.get(engine_name)
    if factory is None:
        raise RuntimeError(f"未対応のTTSエンジンです: {engine_name}")
    return factory(config)


def _new_audio_emitter(emit: Callable[..., None], *, request_id: str) -> Callable[[bytes], None]:
    seq = 0

    def _emit_chunk(chunk: bytes) -> None:
        nonlocal seq
        emit(
            "audio",
            request_id=request_id,
            seq=seq,
            data_b64=base64.b64encode(chunk).decode("ascii"),
        )
        seq += 1

    return _emit_chunk


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS worker process")
    parser.add_argument("--host", required=True, type=str)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--engine", required=True, type=str)
    parser.add_argument("--config-json", required=True, type=str)
    return parser


def _timed_audio_emitter(emit, *, request_id: str):
    """`_new_audio_emitter` を包んで「最初のチャンクを出した時刻」を記録する。

    体感遅延に効くのは合計時間ではなく最初の音が出るまでなので、
    そこだけは別に測れるようにしておく。
    """
    inner = _new_audio_emitter(emit, request_id=request_id)
    first: list[float] = []

    def _emit_chunk(payload: bytes) -> None:
        if not first:
            first.append(time.perf_counter())
        inner(payload)

    return _emit_chunk, first


def _emit_synth_timing(
    emit,
    *,
    engine_name: str,
    text: str,
    total_bytes: int,
    synth_sec: float,
    first_chunk_at: list[float],
    cmd_started: float,
) -> None:
    """合成にかかった時間をログに出す（エンジン本体 / 初音 / 全体を分ける）。

    「対話中に 2 秒かかる」がエンジンの合成なのか送出側なのかを
    ログだけで切り分けられるようにするため。
    """
    total_sec = time.perf_counter() - cmd_started
    audio_sec = total_bytes / 2.0 / 16000.0 if total_bytes > 0 else 0.0
    first_sec = (first_chunk_at[0] - cmd_started) if first_chunk_at else float("nan")
    rtf = (synth_sec / audio_sec) if audio_sec > 0 else float("nan")
    emit(
        "log",
        message=(
            f"{engine_name} synth={synth_sec:.3f}s first={first_sec:.3f}s "
            f"total={total_sec:.3f}s audio={audio_sec:.2f}s RTF={rtf:.3f} "
            f"chars={len(text)}"
        ),
    )


def main() -> int:
    args = _build_parser().parse_args()
    engine_name = (args.engine or "").strip()
    try:
        config = parse_json_object(args.config_json, worker_name="TTSWorker")
    except ValueError as ex:
        print(str(ex), file=sys.stderr, flush=True)
        return 2

    client = JsonLineDuplexClient(args.host, args.port)
    if not connect_worker_client(client, worker_name="TTSWorker"):
        return 2

    stop_event = threading.Event()
    command_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    stream_request_inputs: dict[str, dict[str, Any]] = {}
    adapter_sessions: dict[str, TextStreamAdapterSession] = {}

    def emit(event_type: str, **data: object) -> None:
        safe_emit(client.send, event_type, **data)

    def _on_cmd(payload: dict[str, object]) -> None:
        command_queue.put(dict(payload))

    def _on_disconnect() -> None:
        stop_event.set()

    client.start_reader(on_event=_on_cmd, on_disconnected=_on_disconnect)

    def _stdin_watcher() -> None:
        try:
            for raw in sys.stdin:
                if raw.strip().lower() == "stop":
                    stop_event.set()
                    return
            stop_event.set()
        except Exception:
            stop_event.set()

    try:
        engine = _build_engine(engine_name, config)
        emit("log", message=f"{engine_name} model loading...")
        started = time.perf_counter()
        engine.load()
        emit("log", message=f"{engine_name} model loaded in {(time.perf_counter()-started):.1f}s")
    except Exception as ex:
        emit("fatal", message=str(ex))
        client.close()
        return 1

    emit("ready")
    threading.Thread(target=_stdin_watcher, daemon=True, name="tts-worker-stdin").start()

    while not stop_event.is_set():
        try:
            cmd = command_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        cmd_type = str(cmd.get("type", "")).strip()
        request_id = str(cmd.get("request_id", "")).strip()
        inputs = cmd.get("inputs")
        is_native_text_stream = engine_name in NATIVE_TEXT_STREAM_ENGINES

        if engine_name in TEXT_STREAM_ENGINES:
            if not request_id:
                continue
            if cmd_type == "start_stream":
                if not isinstance(inputs, dict):
                    emit("error", request_id=request_id, message="inputs must be object")
                    continue
                try:
                    stream_request_inputs[request_id] = dict(inputs)
                    if is_native_text_stream:
                        emit_chunk = _new_audio_emitter(emit, request_id=request_id)

                        # request_id はループ変数。コールバックはワーカースレッドから
                        # 後で非同期発火するため、定義時の値をデフォルト引数で束縛する
                        # （束縛しないと次コマンド到来後に別IDへ done/error を誤配送する）。
                        def _emit_done(total_bytes: int, resolved_voice: str, _rid: str = request_id) -> None:
                            stream_request_inputs.pop(_rid, None)
                            emit("done", request_id=_rid, voice=resolved_voice, bytes=total_bytes)

                        def _emit_error(message: str, _rid: str = request_id) -> None:
                            stream_request_inputs.pop(_rid, None)
                            emit("error", request_id=_rid, message=message)

                        resolved_voice = engine.start_stream(
                            request_id=request_id,
                            req=inputs,
                            on_chunk=emit_chunk,
                            on_done=_emit_done,
                            on_error=_emit_error,
                        )
                        emit("started", request_id=request_id, voice=resolved_voice)
                    else:
                        emit_chunk = _new_audio_emitter(emit, request_id=request_id)

                        # request_id をデフォルト引数で定義時の値に束縛（上の native 経路と同様）。
                        def _emit_done(total_bytes: int, resolved_voice: str, _rid: str = request_id) -> None:
                            adapter_sessions.pop(_rid, None)
                            stream_request_inputs.pop(_rid, None)
                            emit("done", request_id=_rid, voice=resolved_voice, bytes=total_bytes)

                        def _emit_error(message: str, _rid: str = request_id) -> None:
                            adapter_sessions.pop(_rid, None)
                            stream_request_inputs.pop(_rid, None)
                            emit("error", request_id=_rid, message=message)

                        session = TextStreamAdapterSession(
                            engine_name=engine_name,
                            engine=engine,
                            request_id=request_id,
                            base_inputs=dict(inputs),
                            emit_audio_chunk=emit_chunk,
                            emit_done=_emit_done,
                            emit_error=_emit_error,
                        )
                        adapter_sessions[request_id] = session
                        session.start()
                        emit("started", request_id=request_id, voice=str(inputs.get("voice", "")))
                except Exception as ex:
                    adapter_sessions.pop(request_id, None)
                    stream_request_inputs.pop(request_id, None)
                    emit("error", request_id=request_id, message=str(ex))
                continue
            if cmd_type == "push_text":
                text = str(cmd.get("text", ""))
                try:
                    if is_native_text_stream:
                        if text:
                            engine.push_text(request_id=request_id, text=text)
                    else:
                        session = adapter_sessions.get(request_id)
                        if session is None:
                            raise RuntimeError("stream session not found")
                        session.push_text(text)
                except Exception as ex:
                    emit("error", request_id=request_id, message=str(ex))
                continue
            if cmd_type == "end_stream":
                try:
                    if is_native_text_stream:
                        engine.end_stream(request_id=request_id)
                    else:
                        session = adapter_sessions.get(request_id)
                        if session is not None:
                            session.end_input()
                except Exception as ex:
                    emit("error", request_id=request_id, message=str(ex))
                finally:
                    if is_native_text_stream:
                        stream_request_inputs.pop(request_id, None)
                continue
            if cmd_type == "cancel_stream":
                try:
                    if is_native_text_stream:
                        engine.cancel_stream(request_id=request_id)
                    else:
                        session = adapter_sessions.pop(request_id, None)
                        if session is not None:
                            session.cancel()
                except Exception:
                    pass
                stream_request_inputs.pop(request_id, None)
                if is_native_text_stream:
                    emit("done", request_id=request_id, voice="", bytes=0)
                continue

        if cmd_type != "synthesize":
            continue
        if not request_id:
            continue
        if not isinstance(inputs, dict):
            emit("error", request_id=request_id, message="inputs must be object")
            continue
        cmd_started = time.perf_counter()
        synth_sec = 0.0
        try:
            text_chunks = [str(inputs.get("text", "") or "")]
            if engine_name in BINARY_STREAM_ENGINES:
                emit_chunk, first_chunk_at = _timed_audio_emitter(emit, request_id=request_id)
                total_bytes = 0
                resolved_voice = ""
                for chunk_text in text_chunks:
                    chunk_inputs = dict(inputs)
                    chunk_inputs["text"] = chunk_text
                    t0 = time.perf_counter()
                    gate = Pcm16BeLeadingSilenceGate()

                    def _emit_without_lead(chunk: bytes, _gate: Pcm16BeLeadingSilenceGate = gate) -> None:
                        audible = _gate.feed(chunk)
                        if audible:
                            emit_chunk(audible)

                    _chunk_bytes, chunk_voice = engine.synthesize_stream(chunk_inputs, _emit_without_lead)
                    tail = gate.flush()
                    if tail:
                        emit_chunk(tail)
                    synth_sec += time.perf_counter() - t0
                    total_bytes += int(_chunk_bytes)
                    if (not resolved_voice) and chunk_voice:
                        resolved_voice = str(chunk_voice)
                emit("done", request_id=request_id, voice=resolved_voice, bytes=total_bytes)
            else:
                chunk_size = 24 * 1024
                emit_chunk, first_chunk_at = _timed_audio_emitter(emit, request_id=request_id)
                resolved_voice = ""
                total_bytes = 0
                for chunk_text in text_chunks:
                    chunk_inputs = dict(inputs)
                    chunk_inputs["text"] = chunk_text
                    t0 = time.perf_counter()
                    audio_l16be, chunk_voice = engine.synthesize(chunk_inputs)
                    synth_sec += time.perf_counter() - t0
                    audio_l16be = strip_leading_silence_pcm16be(audio_l16be)
                    for pos in range(0, len(audio_l16be), chunk_size):
                        emit_chunk(audio_l16be[pos : pos + chunk_size])
                    total_bytes += int(len(audio_l16be))
                    if (not resolved_voice) and chunk_voice:
                        resolved_voice = str(chunk_voice)
                emit("done", request_id=request_id, voice=resolved_voice, bytes=total_bytes)
            _emit_synth_timing(
                emit,
                engine_name=engine_name,
                text="".join(text_chunks),
                total_bytes=total_bytes,
                synth_sec=synth_sec,
                first_chunk_at=first_chunk_at,
                cmd_started=cmd_started,
            )
        except Exception as ex:
            message = str(ex).strip() or type(ex).__name__
            emit("error", request_id=request_id, message=message)

    try:
        emit("stopped")
    except BaseException:
        pass
    try:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    except BaseException:
        pass
    try:
        client.close()
    except BaseException:
        pass
    import os as _os
    _os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
