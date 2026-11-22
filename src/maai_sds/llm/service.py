from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import httpx

from .common import normalize_lmstudio_base_url as _shared_normalize_lmstudio_base_url
from .common import resolve_openai_api_key as _shared_resolve_openai_api_key

_DROP = object()  # パラメータを外す
_MOVE = object()  # max_tokens を max_completion_tokens に付け替える

ChunkCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]
LogCallback = Callable[[str], None]

@dataclass
class LLMChatResult:
    assistant_text: str
    ttft_sec: float
    total_sec: float

class LLMService:
    def __init__(self, on_log: LogCallback | None = None) -> None:
        self._on_log = on_log
        self._client_cache_lock = threading.Lock()
        self._client_cache: dict[str, object] = {}
        self._warmed_keys: set[str] = set()
        self._warmup_inflight: set[str] = set()
        # モデルに断られたパラメータの直し方（(backend, model) ごと）。一度断られたら次から最初からこう送る。
        # 値は _DROP（外す）/ _MOVE（max_tokens → max_completion_tokens）/ 置き換える値。
        self._param_overrides: dict[tuple[str, str], dict[str, object]] = {}

    @staticmethod
    def _token_limit_param(backend: str) -> str:
        """トークン上限のパラメータ名。

        OpenAI 本家は新しいモデル（gpt-5 系・o 系・luna など）で `max_tokens` を受け付けず、
        `max_completion_tokens` は旧モデルでも通る。モデル名で振り分けると新モデルが出るたびに
        漏れるので、OpenAI では常に後者にする。LM Studio / Custom（OpenAI 互換サーバー）は `max_tokens`。
        """
        if backend in ("LM Studio", "Custom"):
            return "max_tokens"
        return "max_completion_tokens"

    @staticmethod
    def _reasoning_param(backend: str) -> dict[str, object]:
        """思考を切る指定。OpenAI は思考の既定がモデルで違い（gpt-5-nano は既定で考えて 4 秒以上かかる）、
        指定できる最も軽い値もモデルで違う（none / minimal）。まず none を送り、断られたら
        _create_completion がエラーに書かれた使える値から最も軽いものに直す（旧モデルは外す）。"""
        if backend in ("LM Studio", "Custom"):
            return {}
        return {"reasoning_effort": "none"}

    _RETRYABLE_PARAMS = ("temperature", "max_tokens", "reasoning_effort")
    _EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh")

    @staticmethod
    def _rejected_param(ex: Exception) -> tuple[str, list[str]]:
        """400 で断られたパラメータ名と、エラー本文に書かれていた使える値の一覧を返す。

        新しいモデル: code=unsupported_parameter / unsupported_value、param に名前、本文に
        "Supported values are: 'none', 'low', ..."。古いモデル: code・param は空で、本文が
        "Unrecognized request argument supplied: reasoning_effort"。
        """
        if getattr(ex, "status_code", None) != 400:
            return "", []
        message = str(getattr(ex, "message", "") or ex)
        code = str(getattr(ex, "code", "") or "")
        param = str(getattr(ex, "param", "") or "")
        if not param:
            found = re.search(r"Unrecognized request argument supplied: (\w+)", message)
            if found:
                param, code = found.group(1), "unsupported_parameter"
        if code not in ("unsupported_parameter", "unsupported_value") or param not in LLMService._RETRYABLE_PARAMS:
            return "", []
        supported: list[str] = []
        if code == "unsupported_value":
            found = re.search(r"Supported values are:([^.]*)", message)
            if found:
                supported = re.findall(r"'([^']+)'", found.group(1))
        return param, supported

    @staticmethod
    def _close_stream(stream_iter: object) -> None:
        """ストリーミング応答を、読み終えても途中で抜けても、その場で必ず閉じる。

        取り消しで途中で抜けたまま放置すると、後で GC が任意のスレッドで後始末（接続を閉じる）を走らせる。
        それが別の要求の送信中（httpcore の接続プールのロックを持った状態）に来ると、同じロックを
        取りに行って自分で固まり、以降の全要求がそのロック待ちで LLM が返ってこなくなる（2026-11-01、
        py-spy で確認）。閉じればサーバー側も切断に気づいて生成を止める。
        """
        close = getattr(stream_iter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _apply_overrides(params: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
        sent = dict(params)
        for name, how in overrides.items():
            if name not in sent:
                continue
            if how is _DROP:
                sent.pop(name)
            elif how is _MOVE:
                sent["max_completion_tokens"] = sent.pop(name)
            else:
                sent[name] = how
        return sent

    def _create_completion(self, client, backend: str, params: dict[str, object]):
        """chat.completions.create。使えないパラメータ（値）で断られたら直して送り直す。

        直し方はモデルごとに覚え、次からは最初からその形で送る（毎回 400 を往復しない）。
        """
        model_key = (backend, str(params.get("model", "")))
        with self._client_cache_lock:
            overrides = dict(self._param_overrides.get(model_key, {}))
        sent = self._apply_overrides(params, overrides)
        for _attempt in range(len(self._RETRYABLE_PARAMS) + 1):
            try:
                return client.chat.completions.create(**sent)
            except Exception as ex:
                bad, supported = self._rejected_param(ex)
                if not bad or bad not in sent:
                    raise
                value = sent[bad]
                lighter = ""
                if bad == "reasoning_effort" and supported:
                    lighter = next((e for e in self._EFFORT_ORDER if e in supported), "")
                if lighter and lighter != value:
                    how: object = lighter
                    note = f"{bad}={lighter} に変えて"
                elif bad == "max_tokens":
                    how, note = _MOVE, "max_tokens を max_completion_tokens に変えて"
                else:
                    how, note = _DROP, f"{bad} を外して"
                overrides[bad] = how
                with self._client_cache_lock:
                    self._param_overrides[model_key] = dict(overrides)
                self._log(f"LLM: {model_key[1]} が {bad}={value!r} を受け付けないため、{note}再送します")
                sent = self._apply_overrides(params, overrides)
        return client.chat.completions.create(**sent)

    def resolve_openai_api_key(self, api_key: str, api_key_env: str) -> str:
        """`llm/common` に集約（Phase 4）。"""
        return _shared_resolve_openai_api_key(api_key, api_key_env)

    @staticmethod
    def normalize_lmstudio_base_url(host: str) -> str:
        """`llm/common` に集約（Phase 4）。"""
        return _shared_normalize_lmstudio_base_url(host)

    def llm_client_cache_key(
        self,
        *,
        backend: str,
        host: str,
        resolved_api_key: str,
        api_key_env: str,
    ) -> str:
        if backend == "LM Studio":
            base = self.normalize_lmstudio_base_url(host)
            return f"lmstudio::{base}"
        if backend == "Custom":
            base = self.normalize_lmstudio_base_url(host or "http://127.0.0.1:8000")
            if resolved_api_key:
                return f"custom::{base}::key::{len(resolved_api_key)}::{resolved_api_key[:8]}"
            env_name = (api_key_env or "").strip()
            return f"custom::{base}::env::{env_name}"
        if resolved_api_key:
            return f"openai::key::{len(resolved_api_key)}::{resolved_api_key[:8]}"
        env_name = (api_key_env or "").strip()
        return f"openai::env::{env_name}"

    def get_or_create_client(
        self,
        *,
        backend: str,
        host: str,
        resolved_api_key: str,
        api_key_env: str,
    ):
        try:
            from openai import OpenAI
        except Exception as ex:
            raise RuntimeError(f"openai ライブラリの読み込みに失敗しました: {ex}") from ex
        cache_key = self.llm_client_cache_key(
            backend=backend,
            host=host,
            resolved_api_key=resolved_api_key,
            api_key_env=api_key_env,
        )
        with self._client_cache_lock:
            cached = self._client_cache.get(cache_key)
        if cached is not None:
            return cached, cache_key

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
        if backend == "LM Studio":
            base = self.normalize_lmstudio_base_url(host or "http://127.0.0.1:1234")
            created = OpenAI(
                base_url=f"{base}/v1",
                api_key="lmstudio",
                timeout=timeout,
                max_retries=0,
            )
        elif backend == "Custom":
            base = self.normalize_lmstudio_base_url(host or "http://127.0.0.1:8000")
            key = (resolved_api_key or "").strip() or "custom"
            created = OpenAI(
                base_url=f"{base}/v1",
                api_key=key,
                timeout=timeout,
                max_retries=0,
            )
        else:
            if not resolved_api_key:
                raise RuntimeError("OpenAI API Key を入力するか、API Key Env の環境変数を設定してください。")
            created = OpenAI(
                api_key=resolved_api_key,
                timeout=timeout,
                max_retries=1,
            )

        with self._client_cache_lock:
            already = self._client_cache.get(cache_key)
            if already is not None:
                return already, cache_key
            self._client_cache[cache_key] = created
        return created, cache_key

    @staticmethod
    def _extra_body_for_backend(backend: str, custom_extra_body_thinking: bool) -> dict[str, object] | None:
        if backend == "Custom" and custom_extra_body_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if backend == "LM Studio":
            # LM Studio の OpenAI 互換 API は chat_template_kwargs を無視するが reasoning_effort は効く。
            # 思考モデル（Gemma 4 12B など）は既定で考えてから答えるため、これがないと返事まで十数秒かかる。
            # 思考を持たないモデルに付けても無視される。
            return {"reasoning_effort": "none"}
        return None

    def warmup_client_once(
        self,
        *,
        backend: str,
        host: str,
        model: str,
        api_key: str,
        api_key_env: str,
        force: bool = False,
        send_temperature: bool | None = None,
        send_max_tokens: bool | None = None,
        temperature: float | None = None,
        custom_extra_body_thinking: bool = False,
        messages: list[dict[str, str]] | None = None,
        stream: bool = True,
    ) -> bool:
        """軽い chat を 1 回投げて接続・モデルロード・ストリーミング経路を温める。

        音声対話の本番は stream=True なので、既定もストリームにして初回 JIT を起動側へ前倒しする。
        戻り値: 成功（スキップ含む）なら True、失敗なら False。
        """
        model_name = (model or "").strip()
        if not model_name:
            return True
        resolved_api_key = self.resolve_openai_api_key(api_key, api_key_env)
        try:
            client, cache_key = self.get_or_create_client(
                backend=backend,
                host=host,
                resolved_api_key=resolved_api_key,
                api_key_env=api_key_env,
            )
        except Exception:
            return False
        warmup_key = f"{cache_key}::{model_name}"
        with self._client_cache_lock:
            if (not force) and (warmup_key in self._warmed_keys or warmup_key in self._warmup_inflight):
                return True
            self._warmup_inflight.add(warmup_key)
        ok = False
        try:
            warmup_messages = messages if messages else [{"role": "user", "content": "hi"}]
            warmup_params: dict[str, object] = {
                "model": model_name,
                "messages": warmup_messages,
                "stream": bool(stream),
            }
            # 推論モデル（gpt-5 系など）は上限 1 だと考える途中で切れてエラーになるので 16 にする。
            token_param = self._token_limit_param(backend)
            warmup_tokens = 16 if token_param == "max_completion_tokens" else 1
            if send_temperature is None and send_max_tokens is None:
                warmup_params[token_param] = warmup_tokens
                if not model_name.startswith("gpt-5"):
                    warmup_params["temperature"] = 0.0
            else:
                if send_max_tokens is not False:
                    warmup_params[token_param] = warmup_tokens
                if send_temperature is not False and not model_name.startswith("gpt-5"):
                    warmup_params["temperature"] = (
                        float(temperature) if temperature is not None else 0.0
                    )
            extra_body = self._extra_body_for_backend(backend, custom_extra_body_thinking)
            if extra_body:
                warmup_params["extra_body"] = extra_body
            warmup_params.update(self._reasoning_param(backend))
            response = self._create_completion(client, backend, warmup_params)
            if stream:
                try:
                    for _chunk in response:
                        pass
                except TypeError:
                    pass
                finally:
                    self._close_stream(response)
            with self._client_cache_lock:
                self._warmed_keys.add(warmup_key)
            self._log(f"LLM warmup完了: backend={backend}, model={model_name}")
            ok = True
        except Exception as ex:
            message = str(ex)
            if "max_tokens or model output limit was reached" in message:
                with self._client_cache_lock:
                    self._warmed_keys.add(warmup_key)
                self._log(f"LLM warmup完了(上限到達扱い): backend={backend}, model={model_name}")
                ok = True
            else:
                self._log(f"LLM warmup失敗: backend={backend}, model={model_name}, error={ex}")
        finally:
            with self._client_cache_lock:
                self._warmup_inflight.discard(warmup_key)
        return ok

    def run_chat(
        self,
        *,
        backend: str,
        host: str,
        model: str,
        api_key: str,
        api_key_env: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        stream: bool,
        on_chunk: ChunkCallback | None = None,
        is_cancelled: CancelCallback | None = None,
        send_temperature: bool | None = None,
        send_max_tokens: bool | None = None,
        temperature: float | None = None,
        custom_extra_body_thinking: bool = False,
    ) -> LLMChatResult:
        resolved_key = self.resolve_openai_api_key(api_key, api_key_env)
        client, _cache_key = self.get_or_create_client(
            backend=backend,
            host=host,
            resolved_api_key=resolved_key,
            api_key_env=api_key_env,
        )
        self.warmup_client_once(
            backend=backend,
            host=host,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            send_temperature=send_temperature,
            send_max_tokens=send_max_tokens,
            temperature=temperature,
            custom_extra_body_thinking=custom_extra_body_thinking,
        )
        params: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": bool(stream),
        }
        max_tokens = max(1, int(max_tokens))
        token_param = self._token_limit_param(backend)
        if send_temperature is None and send_max_tokens is None:
            params[token_param] = max_tokens
            if not model.startswith("gpt-5"):
                params["temperature"] = 0.0
        else:
            if send_max_tokens is not False:
                params[token_param] = max_tokens
            if send_temperature is not False:
                params["temperature"] = float(temperature) if temperature is not None else 0.7

        extra_body = self._extra_body_for_backend(backend, custom_extra_body_thinking)
        if extra_body:
            params["extra_body"] = extra_body
        params.update(self._reasoning_param(backend))

        assistant_text = ""
        ttft_sec: float | None = None
        api_started_at = time.perf_counter()

        if stream:
            stream_iter = self._create_completion(client, backend, params)
            try:
                for chunk in stream_iter:
                    if is_cancelled is not None and is_cancelled():
                        break
                    text = self.chunk_content_to_text(self._extract_chunk_content(chunk))
                    if not text:
                        continue
                    if ttft_sec is None:
                        ttft_sec = time.perf_counter() - api_started_at
                    assistant_text += text
                    if on_chunk is not None:
                        on_chunk(text)
            finally:
                self._close_stream(stream_iter)
        else:
            response = self._create_completion(client, backend, params)
            assistant_text = self.chunk_content_to_text(self._extract_message_content(response))
            if assistant_text and on_chunk is not None:
                on_chunk(assistant_text)

        total_sec = time.perf_counter() - api_started_at
        if ttft_sec is None:
            ttft_sec = total_sec
        return LLMChatResult(
            assistant_text=assistant_text,
            ttft_sec=float(ttft_sec),
            total_sec=float(total_sec),
        )

    @staticmethod
    def _extract_chunk_content(chunk) -> object:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return None
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return None
        return getattr(delta, "content", None)

    @staticmethod
    def _extract_message_content(response) -> object:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
        return getattr(message, "content", None)

    @staticmethod
    def chunk_content_to_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return str(content)

    def _log(self, message: str) -> None:
        if self._on_log is None:
            return
        try:
            self._on_log(message)
        except Exception:
            pass
