"""TTS 合成 RPC とPCM ルーティングを集約する。

`OrchestratorSupervisor` を経由した TTS ワーカー / TTS Bridge への
synthesize 呼び出し、L16BE→L16LE 変換、`send_audio_output` / MaAI 注入への振り分け、
ウォームアップ、待機状態 (`_orch_tts_pending`) 管理を担う。
挙動・ログ文言・スレッド名は移管前と完全互換。
"""
from __future__ import annotations

import base64
import threading
import time
from typing import Any, Callable

from . import playback_service as tps


class TtsRouter:
    """TTS 合成と PCM ルーティングのまとめ役。

    `OrchestratorSupervisor.send_audio_output`（audio/output_worker）を主経路とし、
    失敗時のみ MaAI への直接注入にフォールバックする。
    """

    def __init__(
        self,
        *,
        emit: Callable[[dict[str, object]], None],
        supervisor: "Any",
        vd_state: dict[str, object],
        on_maai_system_inject_activity: Callable[[], None] | None = None,
    ) -> None:
        self._emit = emit
        self._supervisor = supervisor
        self._vd_state = vd_state
        self._on_maai_system_inject_activity = on_maai_system_inject_activity

        self._tts_worker_engine = tps.is_worker_engine
        self._tts_synth_timeout_sec = tps.default_synth_timeout_sec

        # TTS synthesize RPC（supervisor → TTSワーカー／bridge）の完了待ち
        self._pending_lock = threading.Lock()
        self._pending: dict[str, dict[str, object]] = {}

    # -- pending state (TTS RPC) ---------------------------------------------

    def complete_from_event(self, payload: dict) -> None:
        et = str(payload.get("type", "")).strip().lower()
        if et in {"ready", "log", "fatal", "bridge_status", "bridge_launch_result", "bridge_stop_result"}:
            return
        if et == "bridge_http_chunk" or et == "bridge_template_stream_done":
            return  # テンプレストリームは別経路
        rid = str(payload.get("request_id", "")).strip()
        if et == "error" and not rid:
            return
        if not rid:
            return
        with self._pending_lock:
            state = self._pending.get(rid)
        if state is None:
            return
        if et == "started":
            with self._pending_lock:
                state["voice"] = str(payload.get("voice", "")).strip()
                ev = state.get("started")
            if isinstance(ev, threading.Event):
                ev.set()
            return
        if et == "audio":
            data_b64 = str(payload.get("data_b64", "")).strip()
            if not data_b64:
                return
            try:
                chunk = base64.b64decode(data_b64, validate=True)
            except Exception:
                return
            with self._pending_lock:
                if bool(state.get("collect_audio", True)):
                    chunks = state.setdefault("chunks", [])
                    if isinstance(chunks, list):
                        chunks.append(chunk)
                callback = state.get("on_chunk")
            if callable(callback):
                try:
                    callback(chunk, payload)
                except Exception as ex:
                    with self._pending_lock:
                        state["error"] = str(ex).strip() or "audio chunk callback failed"
                    done_ev = state.get("done")
                    if isinstance(done_ev, threading.Event):
                        done_ev.set()
            return
        if et == "done":
            with self._pending_lock:
                state["voice"] = str(payload.get("voice", "")).strip()
            done_ev = state.get("done")
            if isinstance(done_ev, threading.Event):
                done_ev.set()
            return
        if et == "error":
            with self._pending_lock:
                state["error"] = str(payload.get("message", "")).strip() or "不明なエラー"
            done_ev = state.get("done")
            if isinstance(done_ev, threading.Event):
                done_ev.set()

    # -- synthesize -----------------------------------------------------------

    def synthesize_l16be(
        self,
        engine: str,
        base_inputs: dict,
        utterance: str,
        *,
        on_chunk: Callable[[bytes, dict], None] | None = None,
    ) -> tuple[bytes, str]:
        chunk_inputs = dict(base_inputs or {})
        chunk_inputs["text"] = str(utterance or "")
        timeout_sec = self._tts_synth_timeout_sec(engine)

        streaming = on_chunk is not None
        rid = f"orch-tts-{int(time.time() * 1000)}-{threading.get_ident()}"
        wait_state: dict[str, object] = {
            "chunks": [],
            "error": "",
            "voice": "",
            "done": threading.Event(),
            "collect_audio": not streaming,
            "on_chunk": on_chunk,
        }
        with self._pending_lock:
            self._pending[rid] = wait_state
        try:
            if self._tts_worker_engine(engine):
                if not self._supervisor.send_tts(
                    {
                        "type": "synthesize",
                        "request_id": rid,
                        "engine": engine,
                        "inputs": chunk_inputs,
                    }
                ):
                    raise RuntimeError("TTS synthesize が送信できません（未接続）。")
            elif engine == "VOICEVOX":
                vv = str(chunk_inputs.get("voice", "1")).strip() or "1"
                try:
                    speaker = int(vv)
                except ValueError as ex:
                    raise RuntimeError("VOICEVOX の話者IDは整数で指定してください。") from ex
                self._supervisor.handle_tts_bridge_command(
                    {
                        "op": "synthesize",
                        "request_id": rid,
                        "method": "voicevox",
                        "host": str(chunk_inputs.get("host", "127.0.0.1")),
                        "port": int(chunk_inputs.get("port", 50021)),
                        "speaker": speaker,
                        "utterance": chunk_inputs["text"],
                        "voice_text": vv,
                        "timeout": float(max(5.0, min(600.0, 10.0))),
                    }
                )
            elif engine == "TEMPLATE":
                template = chunk_inputs.get("template")
                if not isinstance(template, dict):
                    raise RuntimeError("TEMPLATE は template 設定が orchestrator に渡されていません。")
                self._supervisor.handle_tts_bridge_command(
                    {
                        "op": "synthesize",
                        "request_id": rid,
                        "method": "template",
                        "timeout": float(max(5.0, min(600.0, 10.0))),
                        "template": template,
                        "inputs": chunk_inputs,
                        "voice": str(chunk_inputs.get("voice", "")),
                    }
                )
            else:
                raise RuntimeError(f"未対応の TTS エンジン: {engine}")

            done_ev = wait_state["done"]
            if not isinstance(done_ev, threading.Event):
                raise RuntimeError("internal: bad wait state")
            if not done_ev.wait(timeout=max(1.0, timeout_sec)):
                raise RuntimeError(f"{engine} IPC応答がタイムアウトしました。")
            err = str(wait_state.get("error", "") or "")
            if err:
                raise RuntimeError(err)
            if streaming:
                return b"", str(wait_state.get("voice") or "")
            raw_chunks = wait_state.get("chunks") or []
            audio = b"".join(bytes(c) for c in raw_chunks if c)
            if not audio:
                raise RuntimeError(f"{engine} から空の音声が返されました。")
            return audio, str(wait_state.get("voice") or "")
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)

    def warmup_tts_worker_synthesis(
        self,
        engine: str,
        config: dict[str, object],
        inputs: dict[str, object],
        timeout_sec: float,
    ) -> None:
        """TTS ワーカー起動後のウォームアップ（合成結果は破棄）。Controller からの OpenAI 的経路を避ける。"""
        if not self._tts_worker_engine(engine):
            raise RuntimeError(f"TTS ワーカーウォームアップは worker エンジンのみ: {engine}")
        self._supervisor.ensure_tts(engine=engine, config=config, timeout_sec=180.0)
        chunk_inputs = dict(inputs or {})
        rid = f"orch-tts-wu-{int(time.time() * 1000)}-{threading.get_ident()}"
        wait_state: dict[str, object] = {
            "chunks": [],
            "error": "",
            "voice": "",
            "done": threading.Event(),
            "collect_audio": False,
            "on_chunk": None,
        }
        with self._pending_lock:
            self._pending[rid] = wait_state
        try:
            if not self._supervisor.send_tts(
                {
                    "type": "synthesize",
                    "request_id": rid,
                    "engine": engine,
                    "inputs": chunk_inputs,
                }
            ):
                raise RuntimeError("TTS synthesize が送信できません（未接続）。")
            done_ev = wait_state["done"]
            if not isinstance(done_ev, threading.Event):
                raise RuntimeError("internal: bad wait state")
            if not done_ev.wait(timeout=max(1.0, float(timeout_sec))):
                raise RuntimeError(f"{engine} ウォームアップ IPC応答がタイムアウトしました。")
            err = str(wait_state.get("error", "") or "")
            if err:
                raise RuntimeError(err)
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)

    # -- routing --------------------------------------------------------------

    def route_pcm_l16le(
        self,
        chunk_l16le: bytes,
        *,
        inject_maai: bool,
        inject_only: bool = False,
        playback_tracker: dict[str, int] | None = None,
    ) -> None:
        if not chunk_l16le:
            return
        if playback_tracker is not None:
            playback_tracker["bytes"] = int(playback_tracker.get("bytes", 0)) + len(chunk_l16le)
        try:
            self._vd_state["playback_pcm_bytes"] = int(
                self._vd_state.get("playback_pcm_bytes", 0) or 0,
            ) + len(chunk_l16le)
        except Exception:
            pass
        try:
            pl: dict[str, object] = {
                "type": "tts_play_chunk",
                "audio_b64": base64.b64encode(chunk_l16le).decode("ascii"),
            }
            if inject_maai:
                pl["inject_maai"] = True
            if inject_only:
                pl["inject_only"] = True
            if self._supervisor.send_audio_output(pl):
                return
        except Exception:
            pass
        if inject_maai or inject_only:
            frame_bytes = 640
            pos = 0
            while pos < len(chunk_l16le):
                frame = chunk_l16le[pos : pos + frame_bytes]
                pos += len(frame)
                if not frame:
                    continue
                # Worker 内で audio/input_worker に直接 system_audio を注入する
                # （ヘッドレス時にも MaAI へ system が届く）。
                self.inject_maai_system_audio(frame)

    def inject_maai_system_audio(self, frame: bytes) -> None:
        """audio/input_worker へ `system_audio:` で直接注入する（Controller 経由の再注入はしない）。"""
        if not frame:
            return
        try:
            data_b64 = base64.b64encode(frame).decode("ascii")
        except Exception:
            return
        try:
            self._supervisor.send_audio_input_command(f"system_audio:{data_b64}")
        except Exception:
            pass
        else:
            fn = self._on_maai_system_inject_activity
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def enqueue_break(self, payload: dict[str, object]) -> None:
        try:
            self._supervisor.send_audio_output(dict(payload))
        except Exception:
            pass
