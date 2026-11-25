"""LM Studio の「No models loaded」を、何をすればよいかが分かる知らせにする。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from maai_sds.llm import common
from maai_sds.llm.common import lmstudio_model_error_hint

_NO_MODELS = ("Error code: 400 - {'error': {'message': \"No models loaded. Please load a model in the "
              "developer page or use the 'lms load' command.\"}}")
_KEYS = ["google/gemma-4-12b", "gemma-4-e4b-it-japanese@q4_k_m", "qwen3.5-9b"]


class HintTest(unittest.TestCase):
    def test_model_that_is_not_downloaded_says_how_to_get_it(self) -> None:
        hint = lmstudio_model_error_hint("http://127.0.0.1:1234", "google/gemma-4-e4b", _NO_MODELS, keys=_KEYS)
        self.assertIn("入っていません", hint)
        self.assertIn("lms get google/gemma-4-e4b", hint)

    def test_downloaded_model_points_at_just_in_time_loading(self) -> None:
        for model in ("google/gemma-4-12b", "gemma-4-e4b-it-japanese@q4_k_m", "gemma-4-e4b-it-japanese"):
            hint = lmstudio_model_error_hint("", model, _NO_MODELS, keys=_KEYS)
            self.assertIn("Just-in-Time", hint, model)

    def test_unrelated_errors_and_unreachable_lm_studio_are_left_alone(self) -> None:
        self.assertIsNone(lmstudio_model_error_hint("", "google/gemma-4-e4b", "Connection refused", keys=_KEYS))
        with mock.patch.object(common, "lmstudio_model_keys", return_value=None):
            self.assertIsNone(lmstudio_model_error_hint("", "google/gemma-4-e4b", _NO_MODELS))


class ServiceTest(unittest.TestCase):
    def test_chat_error_carries_the_hint(self) -> None:
        from maai_sds.llm.service import LLMService

        class _Completions:
            def create(self, **_kw):
                raise RuntimeError(_NO_MODELS)

        client = SimpleNamespace(base_url="http://127.0.0.1:1234/v1/",
                                 chat=SimpleNamespace(completions=_Completions()))
        svc = LLMService()
        with mock.patch("maai_sds.llm.common.lmstudio_model_keys", return_value=_KEYS):
            with self.assertRaises(RuntimeError) as cm:
                svc._create_completion(client, "LM Studio", {"model": "google/gemma-4-e4b", "messages": []})
        self.assertIn("lms get google/gemma-4-e4b", str(cm.exception))
        # LM Studio 以外では元のエラーのまま
        with self.assertRaises(RuntimeError) as cm2:
            svc._create_completion(client, "OpenAI", {"model": "gpt-5-nano", "messages": []})
        self.assertNotIn("lms get", str(cm2.exception))


if __name__ == "__main__":
    unittest.main()
