"""LLMService が送るトークン上限・温度・思考のパラメータと、断られたときの再送のテスト。

OpenAI の新しいモデル（luna など）は `max_tokens` を受け付けず 400 を返す。思考の指定
（reasoning_effort）は、使える最も軽い値がモデルで違い（none / minimal）、旧モデルは引数ごと知らない
（2026-11-01、実 API で確認したエラー本文を再現している）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from maai_sds.llm.service import LLMService  # noqa: E402


class _BadRequest(Exception):
    def __init__(self, message: str, *, param: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = 400
        self.param = param
        self.code = code


def unsupported_parameter(name: str) -> _BadRequest:
    return _BadRequest(f"Unsupported parameter: '{name}' is not supported with this model.",
                       param=name, code="unsupported_parameter")


def unsupported_value(name: str, value: str, supported: list[str]) -> _BadRequest:
    listed = ", ".join(f"'{v}'" for v in supported)
    return _BadRequest(f"Unsupported value: '{name}' does not support '{value}' with this model. "
                       f"Supported values are: {listed}.", param=name, code="unsupported_value")


def unrecognized(name: str) -> _BadRequest:
    return _BadRequest(f"Unrecognized request argument supplied: {name}")


class _FakeClient:
    """chat.completions.create を記録し、reject(params) が例外を返したらそれを投げる。"""

    def __init__(self, reject=None) -> None:
        self.calls: list[dict] = []
        self.reject = reject or (lambda params: None)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.calls.append(dict(params))
        err = self.reject(params)
        if err is not None:
            raise err
        message = SimpleNamespace(content="こんにちは")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _chat(svc: LLMService, backend: str, model: str, **kw) -> None:
    result = svc.run_chat(backend=backend, host="", model=model, api_key="x", api_key_env="",
                          messages=[{"role": "user", "content": "hi"}], max_tokens=64, stream=False, **kw)
    assert result.assistant_text == "こんにちは"


def _service(client: _FakeClient) -> LLMService:
    svc = LLMService()
    svc.get_or_create_client = lambda **_: (client, "k")
    svc.warmup_client_once = lambda **_: True
    return svc


class ClientTransportTests(unittest.TestCase):
    def test_clients_use_explicit_timeouts_and_backend_retry_limits(self) -> None:
        calls: list[dict] = []

        def _recording_openai(**kwargs):
            calls.append(dict(kwargs))
            return SimpleNamespace()

        svc = LLMService()
        with patch("openai.OpenAI", side_effect=_recording_openai):
            svc.get_or_create_client(
                backend="LM Studio",
                host="http://127.0.0.1:1234",
                resolved_api_key="",
                api_key_env="",
            )
            svc.get_or_create_client(
                backend="Custom",
                host="http://127.0.0.1:8000",
                resolved_api_key="custom-key",
                api_key_env="",
            )
            svc.get_or_create_client(
                backend="OpenAI",
                host="",
                resolved_api_key="openai-key",
                api_key_env="",
            )

        self.assertEqual([call["max_retries"] for call in calls], [0, 0, 1])
        for call in calls:
            timeout = call["timeout"]
            self.assertEqual(timeout.connect, 5.0)
            self.assertEqual(timeout.read, 30.0)
            self.assertEqual(timeout.write, 10.0)
            self.assertEqual(timeout.pool, 5.0)


class TokenParamTests(unittest.TestCase):
    def test_openai_always_uses_max_completion_tokens(self) -> None:
        for model in ("luna", "gpt-4o-mini", "gpt-5-mini"):
            client = _FakeClient()
            _chat(_service(client), "OpenAI", model)
            self.assertEqual(client.calls[0].get("max_completion_tokens"), 64, model)
            self.assertNotIn("max_tokens", client.calls[0], model)

    def test_local_servers_keep_max_tokens(self) -> None:
        for backend in ("LM Studio", "Custom"):
            client = _FakeClient()
            _chat(_service(client), backend, "gemma")
            self.assertEqual(client.calls[0].get("max_tokens"), 64, backend)
            self.assertNotIn("max_completion_tokens", client.calls[0], backend)

    def test_rejected_temperature_is_dropped_and_remembered(self) -> None:
        client = _FakeClient(lambda p: unsupported_value("temperature", "0.7", ["1"]) if "temperature" in p else None)
        svc = _service(client)
        kw = dict(send_temperature=True, send_max_tokens=True, temperature=0.7)
        _chat(svc, "OpenAI", "luna", **kw)
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("temperature", client.calls[1])
        client.calls.clear()
        _chat(svc, "OpenAI", "luna", **kw)  # 2 回目以降は最初から送らない
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("temperature", client.calls[0])

    def test_rejected_max_tokens_moves_to_max_completion_tokens(self) -> None:
        client = _FakeClient(lambda p: unsupported_parameter("max_tokens") if "max_tokens" in p else None)
        _chat(_service(client), "Custom", "some-model")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1].get("max_completion_tokens"), 64)
        self.assertNotIn("max_tokens", client.calls[1])

    def test_other_errors_are_not_retried(self) -> None:
        client = _FakeClient(lambda p: _BadRequest("bad messages", param="messages", code="invalid_value"))
        with self.assertRaises(_BadRequest):
            _chat(_service(client), "OpenAI", "luna")
        self.assertEqual(len(client.calls), 1)


class ReasoningEffortTests(unittest.TestCase):
    def test_openai_asks_for_no_reasoning(self) -> None:
        client = _FakeClient()
        _chat(_service(client), "OpenAI", "gpt-5.4-nano")
        self.assertEqual(client.calls[0].get("reasoning_effort"), "none")

    def test_local_servers_get_no_top_level_reasoning_effort(self) -> None:
        for backend in ("LM Studio", "Custom"):
            client = _FakeClient()
            _chat(_service(client), backend, "gemma")
            self.assertNotIn("reasoning_effort", client.calls[0], backend)

    def test_none_rejected_switches_to_lightest_supported_value(self) -> None:
        # gpt-5-nano: none は使えず minimal が最も軽い
        def reject(p):
            if p.get("reasoning_effort") == "none":
                return unsupported_value("reasoning_effort", "none", ["minimal", "low", "medium", "high"])
            return None
        client = _FakeClient(reject)
        svc = _service(client)
        _chat(svc, "OpenAI", "gpt-5-nano")
        self.assertEqual([c.get("reasoning_effort") for c in client.calls], ["none", "minimal"])
        client.calls.clear()
        _chat(svc, "OpenAI", "gpt-5-nano")
        self.assertEqual([c.get("reasoning_effort") for c in client.calls], ["minimal"])

    def test_old_model_without_reasoning_drops_the_argument(self) -> None:
        # gpt-4o-mini: code・param が空で本文だけ
        client = _FakeClient(lambda p: unrecognized("reasoning_effort") if "reasoning_effort" in p else None)
        _chat(_service(client), "OpenAI", "gpt-4o-mini")
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("reasoning_effort", client.calls[1])

    def test_several_rejections_in_a_row_are_all_fixed(self) -> None:
        def reject(p):
            if "reasoning_effort" in p:
                return unrecognized("reasoning_effort")
            if "temperature" in p:
                return unsupported_parameter("temperature")
            return None
        client = _FakeClient(reject)
        _chat(_service(client), "OpenAI", "odd-model", send_temperature=True, send_max_tokens=True, temperature=0.5)
        self.assertEqual(len(client.calls), 3)
        self.assertNotIn("reasoning_effort", client.calls[2])
        self.assertNotIn("temperature", client.calls[2])



class _FakeStream:
    """openai.Stream の代わり。チャンクを返し、close されたかを記録する。"""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.closed = False

    def __iter__(self):
        for t in self.texts:
            delta = SimpleNamespace(content=t)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    def close(self) -> None:
        self.closed = True


class StreamCloseTests(unittest.TestCase):
    """ストリームは読み終えても途中で抜けてもその場で閉じる（放置すると GC が別スレッドの送信中に
    後始末を走らせ、httpcore の接続プールのロックで固まって LLM が返ってこなくなる）。"""

    def _run(self, stream: _FakeStream, **kw) -> None:
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: stream)))
        svc = _service(client)
        svc.run_chat(backend="LM Studio", host="", model="gemma", api_key="", api_key_env="",
                     messages=[{"role": "user", "content": "hi"}], max_tokens=64, stream=True, **kw)

    def test_closed_after_full_read(self) -> None:
        s = _FakeStream(["こん", "にちは"])
        self._run(s)
        self.assertTrue(s.closed)

    def test_closed_when_cancelled_midway(self) -> None:
        s = _FakeStream(["こん", "にちは", "！"])
        seen: list[str] = []
        self._run(s, on_chunk=seen.append, is_cancelled=lambda: len(seen) >= 1)
        self.assertEqual(seen, ["こん"])
        self.assertTrue(s.closed)

    def test_closed_when_callback_raises(self) -> None:
        s = _FakeStream(["こん", "にちは"])

        def boom(_text: str) -> None:
            raise RuntimeError("ipc down")

        with self.assertRaises(RuntimeError):
            self._run(s, on_chunk=boom)
        self.assertTrue(s.closed)


if __name__ == "__main__":
    unittest.main()
