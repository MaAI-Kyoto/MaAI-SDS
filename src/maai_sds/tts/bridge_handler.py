"""軽量 TTS Bridge コマンド処理。

`OrchestratorSupervisor.handle_tts_bridge_command` から委譲される。
公開するイベント (`type` フィールド・ペイロード) は移管前と完全互換。
"""
from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import Any, Callable

from ..audio.convert import strip_leading_silence_pcm16be
from .service import TTSBridge


class TtsBridgeHandler:
    """TTSBridge を保持し、コマンド payload を捌くハンドラ。

    Supervisor 側からは `handle(payload)` だけ呼べばよい。
    `_ensure_bridge()` を遅延初期化することで、起動コストを必要なときだけ払う。
    """

    def __init__(
        self,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._on_event = on_event
        self._log = log_fn or (lambda msg: None)
        self._bridge: TTSBridge | None = None
        self._bridge_lock = threading.Lock()

    @property
    def bridge_or_none(self) -> TTSBridge | None:
        return self._bridge

    def _ensure_bridge(self) -> TTSBridge:
        with self._bridge_lock:
            if self._bridge is None:
                self._bridge = TTSBridge(Path(""), None)
            return self._bridge

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._on_event:
            try:
                self._on_event(payload)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Supervisor の `stop_all` から呼ばれる。Bridge とサブプロセスを停止する。"""
        with self._bridge_lock:
            if self._bridge is not None:
                try:
                    self._bridge.stop_server()
                    self._bridge.stop_voicevox()
                    self._bridge.stop_aivisspeech()
                except Exception:
                    pass
                self._bridge = None

    def handle(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        op = str(payload.get("op", "")).strip().lower()
        if op == "configure":
            self._bridge_configure(payload)
            return
        if op == "status":
            self._bridge_status(payload)
            return
        if op == "synthesize":
            threading.Thread(
                target=self._bridge_synthesize_thread,
                args=(dict(payload),),
                daemon=True,
                name="tts-bridge-synth",
            ).start()
            return
        if op == "synthesize_template_stream":
            threading.Thread(
                target=self._bridge_synthesize_template_stream_thread,
                args=(dict(payload),),
                daemon=True,
                name="tts-bridge-tmpl-stream",
            ).start()
            return
        if op == "launch_voicevox":
            threading.Thread(
                target=self._bridge_launch_voicevox_thread,
                args=(dict(payload),),
                daemon=True,
                name="tts-bridge-vvox",
            ).start()
            return
        if op == "launch_aivisspeech":
            threading.Thread(target=self._bridge_launch_aivisspeech_thread, args=(dict(payload),), daemon=True, name="tts-bridge-aivis").start()
            return
        if op == "stop_voicevox":
            threading.Thread(
                target=self._bridge_stop_voicevox_thread,
                args=(dict(payload),),
                daemon=True,
                name="tts-bridge-vvox-stop",
            ).start()
            return
        if op == "stop_aivisspeech":
            threading.Thread(target=self._bridge_stop_aivisspeech_thread, args=(dict(payload),), daemon=True, name="tts-bridge-aivis-stop").start()
            return
        if op == "stream_start":
            self._bridge_stream_start(payload)
            return
        if op == "stream_push":
            self._bridge_stream_push(payload)
            return
        if op == "stream_end":
            self._bridge_stream_end(payload)
            return
        if op == "stream_cancel":
            self._bridge_stream_cancel(payload)
            return
        if op == "stop_server":
            self._bridge_stop_server()
            return

    def _bridge_configure(self, payload: dict[str, Any]) -> None:
        bridge = self._ensure_bridge()
        vv = str(payload.get("voicevox_exe_path", "") or "").strip()
        aivis = str(payload.get("aivisspeech_exe_path", "") or "").strip()
        bridge.voicevox_exe_path = Path(vv) if vv else None
        bridge.aivisspeech_exe_path = Path(aivis) if aivis else None

    def _bridge_status(self, payload: dict[str, Any]) -> None:
        bridge = self._ensure_bridge()
        # 別スレッドの launch/stop と競合して proc が None 化しても落ちないようスナップショットで判定する。
        _vp = bridge.voicevox_proc
        vv = _vp is not None and _vp.poll() is None
        self._emit(
            {
                "type": "bridge_status",
                "request_id": str(payload.get("request_id", "") or ""),
                "voicevox_running": bool(vv),
            }
        )

    def _bridge_synthesize_thread(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        if not request_id:
            return
        method = str(payload.get("method", "") or "").strip().lower()
        try:
            bridge = self._ensure_bridge()
            voice = ""
            audio: bytes = b""
            if method == "voicevox":
                audio = bridge.synthesize_voicevox(
                    host=str(payload.get("host", "127.0.0.1")),
                    port=int(payload.get("port", 50021)),
                    speaker=int(payload.get("speaker", 1)),
                    utterance=str(payload.get("utterance", "")),
                    timeout=float(payload.get("timeout", 10.0) or 10.0),
                )
                voice = str(payload.get("voice_text", payload.get("speaker", "")))
            elif method == "template":
                audio = bridge.synthesize_by_template(
                    template=dict(payload.get("template") or {}),
                    inputs=dict(payload.get("inputs") or {}),
                    timeout=float(payload.get("timeout", 10.0) or 10.0),
                )
                voice = str(payload.get("voice", ""))
            else:
                raise RuntimeError(f"不明な bridge synthesize method: {method}")
            audio = strip_leading_silence_pcm16be(audio)
            seq = 0
            chunk_size = 24 * 1024
            for pos in range(0, len(audio), chunk_size):
                sub = audio[pos : pos + chunk_size]
                self._emit(
                    {
                        "type": "audio",
                        "request_id": request_id,
                        "seq": seq,
                        "data_b64": base64.b64encode(sub).decode("ascii"),
                    }
                )
                seq += 1
            self._emit(
                {
                    "type": "done",
                    "request_id": request_id,
                    "voice": voice,
                    "bytes": len(audio),
                }
            )
        except Exception as ex:
            self._emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": str(ex),
                }
            )

    def _bridge_synthesize_template_stream_thread(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        if not request_id:
            return
        seq = [0]

        def on_chunk(chunk: bytes, meta: dict):
            slim_meta: dict[str, Any] = {}
            if isinstance(meta, dict):
                if "is_chunked" in meta:
                    slim_meta["is_chunked"] = meta.get("is_chunked")
                if "transfer_encoding" in meta:
                    slim_meta["transfer_encoding"] = str(meta.get("transfer_encoding", ""))
                te = meta.get("url")
                if te is not None:
                    slim_meta["url"] = str(te)
            self._emit(
                {
                    "type": "bridge_http_chunk",
                    "request_id": request_id,
                    "seq": seq[0],
                    "data_b64": base64.b64encode(chunk).decode("ascii"),
                    "meta": slim_meta,
                }
            )
            seq[0] += 1

        try:
            bridge = self._ensure_bridge()
            bridge.synthesize_by_template(
                template=dict(payload.get("template") or {}),
                inputs=dict(payload.get("inputs") or {}),
                timeout=float(payload.get("timeout", 60.0) or 60.0),
                on_stream_chunk=on_chunk,
                stream_only_if_chunked=bool(payload.get("stream_only_if_chunked", False)),
            )
            self._emit(
                {"type": "bridge_template_stream_done", "request_id": request_id}
            )
        except Exception as ex:
            self._emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": str(ex),
                }
            )

    def launch_voicevox_blocking(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50021,
        exe_path: str = "",
    ) -> None:
        bridge = self._ensure_bridge()
        path = str(exe_path or "").strip()
        if path:
            bridge.voicevox_exe_path = Path(path)
        bridge.launch_voicevox(host=host, port=port)

    def _bridge_launch_voicevox_thread(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        host = str(payload.get("host") or "127.0.0.1")
        try:
            port = int(payload.get("port") or 50021)
        except (TypeError, ValueError):
            port = 50021
        try:
            bridge = self._ensure_bridge()
            exe = str(payload.get("voicevox_exe_path") or "").strip()
            if exe:
                bridge.voicevox_exe_path = Path(exe)
            proc = bridge.launch_voicevox(host=host, port=port)
            pid = int(proc.pid) if proc is not None and proc.pid else 0
            self._log(f"[TTSBridge] VOICEVOX ready host={host}:{port} pid={pid or '-'}")
            self._emit(
                {
                    "type": "bridge_launch_result",
                    "request_id": request_id,
                    "kind": "voicevox",
                    "ok": True,
                    "pid": pid,
                }
            )
        except Exception as ex:
            self._emit(
                {
                    "type": "bridge_launch_result",
                    "request_id": request_id,
                    "kind": "voicevox",
                    "ok": False,
                    "message": str(ex),
                }
            )

    def _bridge_stop_voicevox_thread(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        try:
            bridge = self._ensure_bridge()
            bridge.stop_voicevox()
            self._emit(
                {
                    "type": "bridge_stop_result",
                    "request_id": request_id,
                    "kind": "voicevox",
                    "ok": True,
                }
            )
        except Exception as ex:
            self._emit(
                {
                    "type": "bridge_stop_result",
                    "request_id": request_id,
                    "kind": "voicevox",
                    "ok": False,
                    "message": str(ex),
                }
            )

    def _bridge_launch_aivisspeech_thread(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        try:
            bridge = self._ensure_bridge()
            proc = bridge.launch_aivisspeech(
                str(payload.get("host", "127.0.0.1")), int(payload.get("port", 10104)),
                use_gpu=bool(payload.get("use_gpu", True)),
            )
            self._emit({"type": "bridge_launch_result", "request_id": request_id, "ok": True, "pid": proc.pid if proc else "", "owned": bool(proc)})
        except Exception as ex:
            self._emit({"type": "bridge_launch_result", "request_id": request_id, "ok": False, "message": str(ex)})

    def _bridge_stop_aivisspeech_thread(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        try:
            self._ensure_bridge().stop_aivisspeech()
            self._emit({"type": "bridge_stop_result", "request_id": request_id, "ok": True})
        except Exception as ex:
            self._emit({"type": "bridge_stop_result", "request_id": request_id, "ok": False, "message": str(ex)})

    def _bridge_stream_start(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        if not request_id:
            return
        engine = str(payload.get("engine", "") or "")
        base_inputs = dict(payload.get("base_inputs") or {})
        timeout = float(payload.get("timeout", 30.0) or 30.0)
        bridge = self._ensure_bridge()
        seq = [0]

        def on_audio_chunk(chunk_l16be: bytes, _meta: dict) -> None:
            chunk_l16be = strip_leading_silence_pcm16be(chunk_l16be)
            if not chunk_l16be:
                return
            self._emit(
                {
                    "type": "audio",
                    "request_id": request_id,
                    "seq": seq[0],
                    "data_b64": base64.b64encode(chunk_l16be).decode("ascii"),
                }
            )
            seq[0] += 1

        def on_done(resolved_voice: str) -> None:
            self._emit(
                {
                    "type": "done",
                    "request_id": request_id,
                    "voice": str(resolved_voice),
                    "bytes": 0,
                }
            )

        def on_error(message: str) -> None:
            self._emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": str(message).strip() or "不明なエラー",
                }
            )

        bridge.start_stream_session(
            request_id=request_id,
            engine=engine,
            base_inputs=base_inputs,
            timeout=max(1.0, timeout),
            on_audio_chunk=on_audio_chunk,
            on_done=on_done,
            on_error=on_error,
        )
        self._emit(
            {
                "type": "started",
                "request_id": request_id,
                "voice": "",
            }
        )

    def _bridge_stream_push(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        text = str(payload.get("text", "") or "")
        if not request_id:
            return
        bridge = self._ensure_bridge()
        bridge.push_stream_text(request_id=request_id, text=text)

    def _bridge_stream_end(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        if not request_id:
            return
        bridge = self._ensure_bridge()
        bridge.end_stream_session(request_id=request_id)

    def _bridge_stream_cancel(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "") or "").strip()
        if not request_id:
            return
        bridge = self._ensure_bridge()
        bridge.cancel_stream_session(request_id=request_id)

    def _bridge_stop_server(self) -> None:
        bridge = self._ensure_bridge()
        bridge.stop_server()
