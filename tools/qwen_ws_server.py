from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from qwen_asr import Qwen3ASRModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-ASR WebSocket streaming server")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--served-model-name", default="Qwen3-ASR-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--session-timeout-sec", type=float, default=60.0)
    parser.add_argument("--default-chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--default-unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--default-unfixed-token-num", type=int, default=5)
    parser.add_argument("--default-no-text-flush-sec", type=float, default=0.0)
    parser.add_argument("--default-min-return-audio-sec", type=float, default=0.0)
    return parser.parse_args()


@dataclass
class SessionState:
    state: Any
    last_update_ts: float
    language: str
    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    no_text_flush_chunks: int
    no_text_unchanged_chunks: int
    last_text: str
    min_return_audio_sec: float
    min_return_audio_samples: int
    total_audio_samples: int


class QwenStreamingRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            max_new_tokens=32,
        )
        self.sessions: dict[str, SessionState] = {}
        self.lock = asyncio.Lock()
        self._cleaner_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._cleaner_task is None:
            self._cleaner_task = asyncio.create_task(self._cleanup_loop(), name="qwen-ws-cleaner")

    async def stop(self) -> None:
        if self._cleaner_task is not None:
            self._cleaner_task.cancel()
            try:
                await self._cleaner_task
            except asyncio.CancelledError:
                pass
            self._cleaner_task = None
        async with self.lock:
            self.sessions.clear()

    async def _cleanup_loop(self) -> None:
        timeout_sec = max(5.0, float(self.args.session_timeout_sec))
        while True:
            await asyncio.sleep(2.0)
            now = time.time()
            async with self.lock:
                stale = [sid for sid, sess in self.sessions.items() if (now - sess.last_update_ts) > timeout_sec]
                for sid in stale:
                    self.sessions.pop(sid, None)

    async def start_session(
        self,
        session_id: str,
        language: str | None,
        chunk_size_sec: float | None,
        unfixed_chunk_num: int | None,
        unfixed_token_num: int | None,
        no_text_flush_sec: float | None,
        min_return_audio_sec: float | None,
    ) -> dict[str, Any]:
        lang = self._normalize_language(language)
        chunk_sec = float(chunk_size_sec or self.args.default_chunk_size_sec)
        uc = int(unfixed_chunk_num or self.args.default_unfixed_chunk_num)
        ut = int(unfixed_token_num or self.args.default_unfixed_token_num)
        flush_sec = float(
            self.args.default_no_text_flush_sec if no_text_flush_sec is None else no_text_flush_sec
        )
        min_ret_sec = float(
            self.args.default_min_return_audio_sec if min_return_audio_sec is None else min_return_audio_sec
        )
        if min_ret_sec < 0:
            min_ret_sec = 0.0
        flush_chunks = 0
        if flush_sec > 0 and chunk_sec > 0:
            flush_chunks = int(max(1, math.ceil(flush_sec / chunk_sec)))
        min_ret_samples = int(max(0, math.ceil(min_ret_sec * 16000.0)))
        state = self.model.init_streaming_state(
            language=lang,
            chunk_size_sec=chunk_sec,
            unfixed_chunk_num=uc,
            unfixed_token_num=ut,
        )
        now = time.time()
        async with self.lock:
            self.sessions[session_id] = SessionState(
                state=state,
                last_update_ts=now,
                language=(lang or ""),
                chunk_size_sec=chunk_sec,
                unfixed_chunk_num=uc,
                unfixed_token_num=ut,
                no_text_flush_chunks=flush_chunks,
                no_text_unchanged_chunks=0,
                last_text="",
                min_return_audio_sec=min_ret_sec,
                min_return_audio_samples=min_ret_samples,
                total_audio_samples=0,
            )
        return {"type": "session_started", "session_id": session_id}

    async def push_chunk(self, session_id: str, pcm16le_b64: str, bootstrap: bool | None = None) -> dict[str, Any]:
        raw = base64.b64decode(pcm16le_b64.encode("ascii"))
        if len(raw) % 2 != 0:
            raise ValueError("audio chunk must be 16bit PCM (even bytes).")
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        _ = bool(bootstrap)  # protocol互換用: 現時点では振る舞いに影響させない
        async with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(f"session not found: {session_id}")
            self.model.streaming_transcribe(x, session.state)
            session.total_audio_samples += int(x.shape[0])
            now = time.time()
            text = str(getattr(session.state, "text", "") or "")
            if text != session.last_text:
                session.last_text = text
                session.no_text_unchanged_chunks = 0
            else:
                session.no_text_unchanged_chunks += 1
            session.last_update_ts = now
            if session.no_text_flush_chunks > 0 and session.no_text_unchanged_chunks >= session.no_text_flush_chunks:
                self.model.finish_streaming_transcribe(session.state)
                final_text = str(getattr(session.state, "text", "") or "")
                final_language = str(getattr(session.state, "language", "") or "")
                if session.total_audio_samples < session.min_return_audio_samples:
                    self.sessions.pop(session_id, None)
                    return {
                        "type": "final_suppressed_by_min_audio",
                        "session_id": session_id,
                        "reason": "audio_too_short",
                        "received_audio_sec": float(session.total_audio_samples) / 16000.0,
                        "min_return_audio_sec": session.min_return_audio_sec,
                    }
                print(
                    f"[QwenWS] final_by_no_text_flush: session_id={session_id} text={final_text}",
                    flush=True,
                )
                self.sessions.pop(session_id, None)
                return {
                    "type": "final_by_no_text_flush",
                    "session_id": session_id,
                    "language": final_language,
                    "text": final_text,
                }
            if session.total_audio_samples < session.min_return_audio_samples:
                return {
                    "type": "partial_suppressed_by_min_audio",
                    "session_id": session_id,
                    "reason": "audio_too_short",
                    "received_audio_sec": float(session.total_audio_samples) / 16000.0,
                    "min_return_audio_sec": session.min_return_audio_sec,
                }
            return {
                "type": "partial",
                "session_id": session_id,
                "language": str(getattr(session.state, "language", "") or ""),
                "text": text,
            }

    async def finish_session(self, session_id: str) -> dict[str, Any]:
        async with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(f"session not found: {session_id}")
            self.model.finish_streaming_transcribe(session.state)
            if session.total_audio_samples < session.min_return_audio_samples:
                result = {
                    "type": "final_suppressed_by_min_audio",
                    "session_id": session_id,
                    "reason": "audio_too_short",
                    "received_audio_sec": float(session.total_audio_samples) / 16000.0,
                    "min_return_audio_sec": session.min_return_audio_sec,
                }
                self.sessions.pop(session_id, None)
                return result
            result = {
                "type": "final",
                "session_id": session_id,
                "language": str(getattr(session.state, "language", "") or ""),
                "text": str(getattr(session.state, "text", "") or ""),
            }
            self.sessions.pop(session_id, None)
            return result

    async def cancel_session(self, session_id: str) -> dict[str, Any]:
        async with self.lock:
            self.sessions.pop(session_id, None)
        return {"type": "session_cancelled", "session_id": session_id}

    @staticmethod
    def _normalize_language(language: str | None) -> str | None:
        raw = (language or "").strip()
        if not raw:
            return None
        lower = raw.lower().replace("_", "-")
        aliases = {
            "ja": "Japanese",
            "ja-jp": "Japanese",
            "jp": "Japanese",
            "japanese": "Japanese",
            "en": "English",
            "en-us": "English",
            "english": "English",
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "chinese": "Chinese",
            "yue": "Cantonese",
            "cantonese": "Cantonese",
        }
        return aliases.get(lower, raw)


def create_app(runtime: QwenStreamingRuntime) -> FastAPI:
    app = FastAPI(title="Qwen3-ASR WS Streaming Server")

    @app.on_event("startup")
    async def _startup() -> None:
        await runtime.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await runtime.stop()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "data": [
                {
                    "id": runtime.args.served_model_name,
                    "object": "model",
                    "owned_by": "qwen-asr",
                }
            ]
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("message must be JSON object")
                    mtype = str(msg.get("type", "")).strip()
                    session_id = str(msg.get("session_id", "")).strip()
                    if not session_id:
                        raise ValueError("session_id is required")
                    if mtype == "start_session":
                        out = await runtime.start_session(
                            session_id=session_id,
                            language=msg.get("language"),
                            chunk_size_sec=msg.get("chunk_size_sec"),
                            unfixed_chunk_num=msg.get("unfixed_chunk_num"),
                            unfixed_token_num=msg.get("unfixed_token_num"),
                            no_text_flush_sec=msg.get("no_text_flush_sec"),
                            min_return_audio_sec=msg.get("min_return_audio_sec"),
                        )
                    elif mtype == "audio_chunk":
                        out = await runtime.push_chunk(
                            session_id=session_id,
                            pcm16le_b64=str(msg.get("audio_b64", "")),
                            bootstrap=msg.get("bootstrap"),
                        )
                    elif mtype == "finish_session":
                        out = await runtime.finish_session(session_id=session_id)
                    elif mtype == "cancel_session":
                        out = await runtime.cancel_session(session_id=session_id)
                    else:
                        raise ValueError(f"unknown type: {mtype}")
                    await ws.send_text(json.dumps(out, ensure_ascii=False))
                except Exception as ex:
                    err = {"type": "error", "message": str(ex)}
                    await ws.send_text(json.dumps(err, ensure_ascii=False))
        except WebSocketDisconnect:
            return

    return app


def main() -> int:
    args = _parse_args()
    runtime = QwenStreamingRuntime(args)
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
