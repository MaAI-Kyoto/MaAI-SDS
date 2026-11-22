from __future__ import annotations

import base64
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib import parse as urllib_parse
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..audio.convert import swap_pcm16_endian, wav_bytes_to_pcm16be_16k_mono
from ..llm.common import normalize_docker_loopback_host, running_in_docker


class TTSBridge:
    def __init__(
        self,
        server_exe_path: Path,
        voicevox_exe_path: Path | None = None,
        aivisspeech_exe_path: Path | None = None,
    ) -> None:
        self.server_exe_path = server_exe_path
        self.voicevox_exe_path = voicevox_exe_path
        self.aivisspeech_exe_path = aivisspeech_exe_path
        self.server_proc: subprocess.Popen | None = None
        self.voicevox_proc: subprocess.Popen | None = None
        self.aivisspeech_proc: subprocess.Popen | None = None
        self._stream_sessions_lock = threading.Lock()
        # 外部プロセス（VOICEVOX / AivisSpeech）の起動・停止を直列化する。
        # launch_* は「HTTP で生きているか見る → 自分の Popen を見る → 起動する」
        # という check-then-act で、判定に HTTP プローブを挟むので窓が広い。
        # TTS テストの launch RPC と voice_dialog の ensure は別スレッドなので、
        # 素のままだと 2 つ起動しうる。
        self._proc_launch_lock = threading.RLock()
        self._stream_sessions: dict[str, dict[str, Any]] = {}

    def stop_server(self) -> None:
        if self.server_proc is None:
            return
        if self.server_proc.poll() is None:
            try:
                self.server_proc.terminate()
            except Exception:
                pass

    def launch_voicevox(
        self,
        host: str = "127.0.0.1",
        port: int = 50021,
    ) -> subprocess.Popen | None:
        with self._proc_launch_lock:
            host = normalize_docker_loopback_host(host or "127.0.0.1")
            try:
                port = int(port or 50021)
            except (TypeError, ValueError):
                port = 50021
            # 既にホスト側で VOICEVOX Engine が動いていれば所有しない（AivisSpeech と同じ）。
            if self._http_alive(host, port):
                return self.voicevox_proc
            if self.voicevox_proc is not None and self.voicevox_proc.poll() is None:
                return self.voicevox_proc
            if running_in_docker():
                raise RuntimeError(
                    "Docker からは VOICEVOX.exe を起動できません。"
                    f" Windows 側で VOICEVOX を起動し http://{host}:{port}/version が応答するのを待つか、"
                    "TTS エンジンを VOICEVOX-Core にしてください。"
                )
            if self.voicevox_exe_path is None:
                raise RuntimeError("VOICEVOX.exe path is not configured")
            if not self.voicevox_exe_path.exists():
                raise FileNotFoundError(str(self.voicevox_exe_path))
            # PIPE のまま読み取らないと stderr/stdout が詰まり、子プロセスがブロックして GUI が固まる。
            self.voicevox_proc = subprocess.Popen(
                [str(self.voicevox_exe_path)],
                cwd=str(self.voicevox_exe_path.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # 起動直後に合成を送るテスト再生でも競合しないよう、HTTP API が応答可能に
            # なるまで待つ。VOICEVOX.exe の標準 API ポートは 50021。
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                if self.voicevox_proc.poll() is not None:
                    raise RuntimeError(
                        f"VOICEVOX が終了しました (exit {self.voicevox_proc.returncode})"
                    )
                if self._http_alive(host, port):
                    return self.voicevox_proc
                time.sleep(0.2)
            raise RuntimeError("VOICEVOX の起動がタイムアウトしました")

    def stop_voicevox(self) -> None:
        with self._proc_launch_lock:
            if self.voicevox_proc is None:
                return
            if self.voicevox_proc.poll() is None:
                try:
                    self.voicevox_proc.terminate()
                except Exception:
                    pass

    @staticmethod
    def _http_alive(host: str, port: int) -> bool:
        try:
            with urllib_request.urlopen(f"http://{host}:{port}/version", timeout=1.5):
                return True
        except Exception:
            return False

    def launch_aivisspeech(self, host: str, port: int, *, use_gpu: bool = True) -> subprocess.Popen | None:
        # 既にユーザーが起動した Engine は所有しない。停止操作でも落とさない。
        with self._proc_launch_lock:
            if self._http_alive(host, port):
                return None
            if self.aivisspeech_proc is not None and self.aivisspeech_proc.poll() is None:
                return self.aivisspeech_proc
            if self.aivisspeech_exe_path is None:
                raise RuntimeError("AivisSpeech Engine の run.exe パスが未設定です")
            if not self.aivisspeech_exe_path.exists():
                raise FileNotFoundError(str(self.aivisspeech_exe_path))
            args = [str(self.aivisspeech_exe_path), "--host", host, "--port", str(port)]
            if use_gpu:
                args.append("--use_gpu")
            self.aivisspeech_proc = subprocess.Popen(
                args, cwd=str(self.aivisspeech_exe_path.parent), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                if self.aivisspeech_proc.poll() is not None:
                    raise RuntimeError(f"AivisSpeech Engine が終了しました (exit {self.aivisspeech_proc.returncode})")
                if self._http_alive(host, port):
                    return self.aivisspeech_proc
                time.sleep(0.2)
            raise RuntimeError("AivisSpeech Engine の起動がタイムアウトしました")

    def stop_aivisspeech(self) -> None:
        with self._proc_launch_lock:
            if self.aivisspeech_proc is not None and self.aivisspeech_proc.poll() is None:
                try:
                    self.aivisspeech_proc.terminate()
                except Exception:
                    pass

    def synthesize_voicevox(
        self,
        host: str,
        port: int,
        speaker: int,
        utterance: str,
        timeout: float = 10.0,
    ) -> bytes:
        template = {
            "base_url": "http://{{input.host}}:{{input.port}}",
            "auth": {"type": "none"},
            "steps": [
                {
                    "name": "audio_query",
                    "method": "POST",
                    "path": "/audio_query",
                    "query": {"text": "{{input.text}}", "speaker": "{{input.voice}}"},
                    "request_mode": "none",
                    "response_extract": {"type": "json", "store_as": "audio_query"},
                },
                {
                    "name": "synthesis",
                    "method": "POST",
                    "path": "/synthesis",
                    "query": {"speaker": "{{input.voice}}"},
                    "request_mode": "json",
                    "body_template": "{{step.audio_query}}",
                    "body_mutations": {
                        "outputSamplingRate": 16000,
                        "outputStereo": False,
                        "prePhonemeLength": 0.0,
                    },
                    "headers": {"Content-Type": "application/json"},
                    "response_extract": {"type": "bytes", "store_as": "audio_wav"},
                },
            ],
            "final_response": {"type": "wav_to_l16be", "from": "step.audio_wav"},
        }
        resolved_inputs: dict[str, Any] = {
            "engine": "VOICEVOX",
            "host": host,
            "port": port,
            "voice": speaker,
            "text": utterance,
        }
        return self.synthesize_by_template(template=template, inputs=resolved_inputs, timeout=timeout)

    def synthesize_by_template(
        self,
        template: dict,
        inputs: dict,
        timeout: float = 10.0,
        on_stream_chunk: Callable[[bytes, dict], None] | None = None,
        stream_only_if_chunked: bool = False,
    ) -> bytes:
        return self._synthesize_by_template_single_pass(
            template=template,
            inputs=inputs,
            timeout=timeout,
            on_stream_chunk=on_stream_chunk,
            stream_only_if_chunked=stream_only_if_chunked,
        )

    def _synthesize_by_template_single_pass(
        self,
        *,
        template: dict,
        inputs: dict,
        timeout: float,
        on_stream_chunk: Callable[[bytes, dict], None] | None,
        stream_only_if_chunked: bool,
    ) -> bytes:
        self._validate_template(template)
        context = {"input": dict(inputs), "step": {}, "last": {}}
        base_url = str(self._resolve_value(template.get("base_url", ""), context)).rstrip("/")
        auth_cfg = self._resolve_value(template.get("auth", {}), context)
        default_headers = self._resolve_value(template.get("headers", {}), context)
        self._execute_template_steps(
            template=template,
            context=context,
            base_url=base_url,
            auth_cfg=auth_cfg,
            default_headers=default_headers,
            timeout=timeout,
            on_stream_chunk=on_stream_chunk,
            stream_only_if_chunked=stream_only_if_chunked,
        )

        final_cfg = template.get("final_response", {"type": "bytes", "from": "last.extract"})
        return self._finalize_audio(final_cfg, context)

    def start_stream_session(
        self,
        *,
        request_id: str,
        engine: str,
        base_inputs: dict[str, Any],
        timeout: float,
        on_audio_chunk: Callable[[bytes, dict[str, Any]], None],
        on_done: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> None:
        engine_name = str(engine or "").strip()
        if engine_name not in {"VOICEVOX", "TEMPLATE"}:
            raise RuntimeError(f"TTSBridge stream未対応エンジンです: {engine_name}")
        rid = str(request_id or "").strip()
        if not rid:
            raise RuntimeError("request_id が空です。")
        task_queue: queue.Queue[str | None] = queue.Queue()
        session: dict[str, Any] = {
            "request_id": rid,
            "engine": engine_name,
            "base_inputs": dict(base_inputs or {}),
            "timeout": float(timeout),
            "queue": task_queue,
            "cancelled": False,
            "voice": "",
            "done_sent": False,
            "done_lock": threading.Lock(),
            "on_audio_chunk": on_audio_chunk,
            "on_done": on_done,
            "on_error": on_error,
        }

        def _emit_done_once() -> None:
            with session["done_lock"]:
                if session["done_sent"]:
                    return
                session["done_sent"] = True
            session["on_done"](str(session.get("voice", "")))

        def _emit_error_once(message: str) -> None:
            with session["done_lock"]:
                if session["done_sent"]:
                    return
                session["done_sent"] = True
            session["on_error"](str(message).strip() or "不明なエラー")

        def _run() -> None:
            try:
                while True:
                    item = task_queue.get()
                    if item is None:
                        break
                    if bool(session.get("cancelled", False)):
                        break
                    text = str(item or "")
                    if not text.strip():
                        continue
                    one_inputs = dict(session["base_inputs"])
                    one_inputs["engine"] = engine_name
                    one_inputs["text"] = text
                    audio, resolved_voice = self._synthesize_stream_chunk(
                        engine_name=engine_name,
                        inputs=one_inputs,
                        timeout=float(session["timeout"]),
                    )
                    if resolved_voice and (not session.get("voice")):
                        session["voice"] = str(resolved_voice)
                    if audio:
                        session["on_audio_chunk"](audio, {"request_id": rid, "engine": engine_name})
                if bool(session.get("cancelled", False)):
                    # キャンセルされたセッションでは done を出さない（正常完了と区別不能になるため）。
                    return
                _emit_done_once()
            except Exception as ex:
                _emit_error_once(str(ex))
            finally:
                with self._stream_sessions_lock:
                    self._stream_sessions.pop(rid, None)

        with self._stream_sessions_lock:
            if rid in self._stream_sessions:
                raise RuntimeError("stream session already exists")
            self._stream_sessions[rid] = session
        worker = threading.Thread(target=_run, daemon=True, name=f"tts-bridge-stream-{rid}")
        session["worker"] = worker
        worker.start()

    def push_stream_text(self, *, request_id: str, text: str) -> None:
        rid = str(request_id or "").strip()
        if not rid:
            return
        with self._stream_sessions_lock:
            session = self._stream_sessions.get(rid)
        if session is None:
            raise RuntimeError("stream session not found")
        if bool(session.get("cancelled", False)):
            return
        session["queue"].put(str(text or ""))

    def end_stream_session(self, request_id: str) -> None:
        rid = str(request_id or "").strip()
        if not rid:
            return
        with self._stream_sessions_lock:
            session = self._stream_sessions.get(rid)
        if session is None:
            return
        session["queue"].put(None)

    def cancel_stream_session(self, request_id: str) -> None:
        rid = str(request_id or "").strip()
        if not rid:
            return
        with self._stream_sessions_lock:
            session = self._stream_sessions.get(rid)
        if session is None:
            return
        session["cancelled"] = True
        session["queue"].put(None)

    def _synthesize_stream_chunk(
        self,
        *,
        engine_name: str,
        inputs: dict[str, Any],
        timeout: float,
    ) -> tuple[bytes, str]:
        text = str(inputs.get("text", "") or "")
        if not text.strip():
            return b"", str(inputs.get("voice", ""))
        if engine_name == "VOICEVOX":
            speaker_text = str(inputs.get("voice", "1")).strip() or "1"
            try:
                speaker = int(speaker_text)
            except Exception as ex:
                raise RuntimeError("VOICEVOX の話者IDは整数で指定してください。") from ex
            audio = self.synthesize_voicevox(
                host=str(inputs.get("host", "127.0.0.1")),
                port=int(inputs.get("port", 50021)),
                speaker=speaker,
                utterance=text,
                timeout=float(timeout),
            )
            return audio, speaker_text
        if engine_name == "TEMPLATE":
            template = inputs.get("template")
            if not isinstance(template, dict):
                raise RuntimeError("TEMPLATE streamには template が必要です。")
            audio = self.synthesize_by_template(
                template=template,
                inputs=dict(inputs),
                timeout=float(timeout),
            )
            return audio, str(inputs.get("voice", ""))
        raise RuntimeError(f"TTSBridge stream未対応エンジンです: {engine_name}")

    def _validate_template(self, template: dict) -> None:
        if not isinstance(template, dict):
            raise RuntimeError("TTSテンプレートがobjectではありません")
        steps = template.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise RuntimeError("TTSテンプレート steps が未定義です")
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise RuntimeError(f"TTSテンプレート step[{idx}] が不正です")
            method = str(step.get("method", "POST")).strip().upper()
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                raise RuntimeError(f"TTSテンプレート step[{idx}] method が不正です: {method}")

    def _execute_template_steps(
        self,
        *,
        template: dict,
        context: dict[str, Any],
        base_url: str,
        auth_cfg: object,
        default_headers: object,
        timeout: float,
        on_stream_chunk: Callable[[bytes, dict], None] | None,
        stream_only_if_chunked: bool,
    ) -> None:
        steps = template.get("steps", [])
        for step in steps:
            method = str(self._resolve_value(step.get("method", "POST"), context)).upper()
            path = str(self._resolve_value(step.get("path", ""), context))
            query = self._resolve_value(step.get("query", {}), context)
            headers = {}
            if isinstance(default_headers, dict):
                headers.update({str(k): str(v) for k, v in default_headers.items() if v is not None})
            step_headers = self._resolve_value(step.get("headers", {}), context)
            if isinstance(step_headers, dict):
                headers.update({str(k): str(v) for k, v in step_headers.items() if v is not None})
            self._apply_auth(headers, auth_cfg)
            url = self._build_url(base_url, path, query)
            request_mode = str(self._resolve_value(step.get("request_mode", "none"), context)).lower()
            body = self._build_request_body(request_mode, step, context)
            content = self._perform_step_request(
                method=method,
                url=url,
                headers=headers,
                body=body,
                step=step,
                context=context,
                timeout=timeout,
                on_stream_chunk=on_stream_chunk,
                stream_only_if_chunked=stream_only_if_chunked,
            )
            extract_cfg = step.get("response_extract", {"type": "bytes"})
            extracted = self._extract_response(content, extract_cfg)
            store_as = str(extract_cfg.get("store_as", "")).strip()
            if store_as:
                context["step"][store_as] = extracted
            context["last"]["bytes"] = content
            context["last"]["extract"] = extracted

    def _perform_step_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        step: dict[str, Any],
        context: dict[str, Any],
        timeout: float,
        on_stream_chunk: Callable[[bytes, dict], None] | None,
        stream_only_if_chunked: bool,
    ) -> bytes:
        req = urllib_request.Request(url=url, data=body, headers=headers, method=method)
        try:
            with urllib_request.urlopen(req, timeout=timeout) as res:
                status = res.getcode()
                if status != 200:
                    raise RuntimeError(f"TTS HTTP {status} {method} {url}")
                stream_response = bool(self._resolve_value(step.get("stream_response", False), context))
                if not stream_response:
                    return res.read()
                transfer_encoding = str(res.headers.get("Transfer-Encoding", "")).strip().lower()
                is_chunked = "chunked" in transfer_encoding
                allow_stream_emit = not (bool(stream_only_if_chunked) and (not is_chunked))
                chunk_size_raw = self._resolve_value(step.get("stream_chunk_size", 4096), context)
                try:
                    chunk_size = int(chunk_size_raw)
                except Exception:
                    chunk_size = 4096
                if chunk_size <= 0:
                    chunk_size = 4096
                chunks: list[bytes] = []
                while True:
                    chunk = res.read(chunk_size)
                    if not chunk:
                        break
                    if on_stream_chunk is not None and allow_stream_emit:
                        on_stream_chunk(
                            chunk,
                            {
                                "step": step,
                                "url": url,
                                "is_chunked": is_chunked,
                                "transfer_encoding": transfer_encoding,
                            },
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except urllib_error.HTTPError as ex:
            body_text = ex.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS HTTP {ex.code} {method} {url}: {body_text[:400]}") from ex

    def _build_request_body(self, request_mode: str, step: dict, context: dict) -> bytes | None:
        if request_mode == "none":
            return None
        resolved = self._resolve_value(step.get("body_template", ""), context)
        if request_mode == "json":
            if isinstance(resolved, (str, int, float, bool)) or resolved is None:
                resolved = {"value": resolved}
            if not isinstance(resolved, dict):
                raise RuntimeError("JSON body_template は object である必要があります")
            mutations = self._resolve_value(step.get("body_mutations", {}), context)
            if isinstance(mutations, dict):
                resolved = dict(resolved)
                for key, value in mutations.items():
                    resolved[str(key)] = value
            return json.dumps(resolved, ensure_ascii=False).encode("utf-8")
        if request_mode in ("text", "xml_ssml"):
            return str(resolved).encode("utf-8")
        raise RuntimeError(f"未対応の request_mode: {request_mode}")

    def _extract_response(self, content: bytes, extract_cfg: dict) -> object:
        kind = str(extract_cfg.get("type", "bytes")).lower()
        if kind == "bytes":
            return content
        if kind == "json":
            return json.loads(content.decode("utf-8"))
        if kind == "text":
            return content.decode("utf-8")
        raise RuntimeError(f"未対応の response_extract.type: {kind}")

    def _finalize_audio(self, final_cfg: dict, context: dict) -> bytes:
        output_type = str(final_cfg.get("type", "bytes")).lower()
        source_ref = str(final_cfg.get("from", "last.extract"))
        source = self._resolve_ref(source_ref, context)
        if output_type == "bytes":
            if not isinstance(source, (bytes, bytearray)):
                raise RuntimeError("final_response bytes の入力が不正です")
            return bytes(source)
        if output_type == "wav_to_l16be":
            if not isinstance(source, (bytes, bytearray)):
                raise RuntimeError("final_response wav_to_l16be の入力が不正です")
            return wav_bytes_to_pcm16be_16k_mono(bytes(source))
        if output_type == "raw_le_to_be":
            if not isinstance(source, (bytes, bytearray)):
                raise RuntimeError("final_response raw_le_to_be の入力が不正です")
            return swap_pcm16_endian(bytes(source))
        if output_type == "json_base64":
            if not isinstance(source, dict):
                raise RuntimeError("final_response json_base64 の入力が不正です")
            json_path = str(final_cfg.get("json_path", "audioContent"))
            raw = self._json_path_get(source, json_path)
            if not isinstance(raw, str):
                raise RuntimeError(f"json_path が文字列ではありません: {json_path}")
            return base64.b64decode(raw)
        if output_type == "json_base64_wav_to_l16be":
            if not isinstance(source, dict):
                raise RuntimeError("final_response json_base64_wav_to_l16be の入力が不正です")
            json_path = str(final_cfg.get("json_path", "audioContent"))
            raw = self._json_path_get(source, json_path)
            if not isinstance(raw, str):
                raise RuntimeError(f"json_path が文字列ではありません: {json_path}")
            return wav_bytes_to_pcm16be_16k_mono(base64.b64decode(raw))
        raise RuntimeError(f"未対応の final_response.type: {output_type}")

    def _apply_auth(self, headers: dict[str, str], auth_cfg: object) -> None:
        if not isinstance(auth_cfg, dict):
            return
        auth_type = str(auth_cfg.get("type", "none")).strip().lower()
        if auth_type in ("", "none"):
            return
        env_name = str(auth_cfg.get("env", "")).strip()
        if not env_name:
            return
        token = str(os.environ.get(env_name, "")).strip()
        if not token:
            return
        if auth_type == "bearer_env":
            prefix = str(auth_cfg.get("prefix", "Bearer "))
            if not prefix:
                prefix = "Bearer "
            # "Bearer" のように末尾スペースが欠けていても有効な Authorization ヘッダを組み立てる。
            if not prefix[-1].isspace():
                prefix = f"{prefix} "
            headers["Authorization"] = f"{prefix}{token}"
            return
        if auth_type == "header_env":
            header_name = str(auth_cfg.get("header", "X-API-Key")).strip() or "X-API-Key"
            headers[header_name] = token

    def _build_url(self, base_url: str, path: str, query: object) -> str:
        url = base_url
        if path:
            if not path.startswith("/"):
                path = f"/{path}"
            url = f"{url}{path}"
        if isinstance(query, dict) and query:
            query_text = urllib_parse.urlencode({str(k): str(v) for k, v in query.items()})
            if query_text:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}{query_text}"
        return url

    def _resolve_value(self, value: object, context: dict) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{{") and stripped.endswith("}}") and stripped.count("{{") == 1 and stripped.count("}}") == 1:
                ref = stripped[2:-2].strip()
                return self._resolve_ref(ref, context)
            parts = []
            cursor = 0
            while True:
                start = value.find("{{", cursor)
                if start < 0:
                    parts.append(value[cursor:])
                    break
                end = value.find("}}", start + 2)
                if end < 0:
                    parts.append(value[cursor:])
                    break
                parts.append(value[cursor:start])
                ref = value[start + 2 : end].strip()
                resolved = self._resolve_ref(ref, context)
                parts.append("" if resolved is None else str(resolved))
                cursor = end + 2
            return "".join(parts)
        if isinstance(value, list):
            return [self._resolve_value(item, context) for item in value]
        if isinstance(value, dict):
            return {str(k): self._resolve_value(v, context) for k, v in value.items()}
        return value

    def _resolve_ref(self, ref: str, context: dict) -> object:
        parts = [p for p in ref.split(".") if p]
        if not parts:
            return ""
        cur: object = context
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return ""
        return cur

    def _json_path_get(self, data: dict, path: str) -> object:
        cur: object = data
        for part in [p for p in path.split(".") if p]:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise RuntimeError(f"json_path が存在しません: {path}")
        return cur
