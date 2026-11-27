"""ヘッドレス（headless_main.HeadlessClient）が対話のイベントをどう表示するか。"""

from __future__ import annotations

import contextlib
import io
import unittest

from maai_sds.headless_main import HeadlessClient


def _printed(client: HeadlessClient, event: dict) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        client._on_event(event)
    return buf.getvalue()


class HeadlessEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = HeadlessClient("127.0.0.1", 9)  # つながない（イベント処理だけを見る）

    def test_dialog_llm_reply_is_printed_from_assistant_text(self) -> None:
        # dialog/runtime.py の llm_done は返答を assistant_text で送る
        out = _printed(self.client, {"type": "llm_done", "ok": True, "assistant_text": "本当にそうですね。", "request_id": "llm-1"})
        self.assertIn("[llm] 本当にそうですね。", out)
        self.assertEqual(self.client._llm_text, "本当にそうですね。")
        self.assertTrue(self.client._llm_done.is_set())

    def test_llm_reply_in_text_is_still_printed(self) -> None:
        out = _printed(self.client, {"type": "llm_done", "text": "はい。", "request_id": "llm-2"})
        self.assertIn("[llm] はい。", out)

    def test_empty_reply_prints_nothing(self) -> None:
        self.assertNotIn("[llm]", _printed(self.client, {"type": "llm_done", "ok": False, "error": "x", "request_id": "r"}))

    def test_only_final_asr_results_are_printed(self) -> None:
        self.assertEqual(_printed(self.client, {"type": "asr_result", "text": "こんに", "is_final": False}), "")
        self.assertIn("[asr] こんにちは", _printed(self.client, {"type": "asr_result", "text": "こんにちは", "is_final": True}))


if __name__ == "__main__":
    unittest.main()
