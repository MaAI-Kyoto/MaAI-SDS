"""先読みは、前の返事のあとにユーザーの発話が入っているときだけ始める。

疑似ストリーミングの ASR をオフにすると、話している途中の文字起こしが履歴に入らない。以前は履歴に
ユーザーの発話が 1 つでもあれば先読みし、前のターンの発話に答え直した返事をそのまま使っていた
（毎回「こんにちは、今日はどんなことを…」に戻る）。
"""

from __future__ import annotations

import unittest

from maai_sds.dialog.orchestrator import DialogOrchestrator


def _prefetched(events: list[dict]) -> bool:
    return any(ev.get("type") == "trigger_llm" and ev.get("prefetch") for ev in events)


class PrefetchNeedsNewUserTextTest(unittest.TestCase):
    def _orch(self, messages: list[dict]) -> DialogOrchestrator:
        orch = DialogOrchestrator()
        orch.start({"maai_turn_threshold": 0.5, "maai_turn_condition": "p_now_and_future"})
        orch.state.opening_tts_pending = False
        orch.state.messages.extend(messages)
        return orch

    def _cross_future(self, orch: DialogOrchestrator) -> bool:
        orch.on_turn_probability(0.2, p_future=0.3)
        return _prefetched(orch.on_turn_probability(0.3, p_future=0.6))

    def test_prefetches_when_the_last_message_is_the_users(self) -> None:
        orch = self._orch([{"role": "user", "content": "こんにちは"}])
        self.assertTrue(self._cross_future(orch))

    def test_no_prefetch_after_an_answer_without_new_user_text(self) -> None:
        orch = self._orch([
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは。今日はどんなことをお手伝いしましょうか？"},
        ])
        self.assertFalse(self._cross_future(orch))

    def test_empty_assistant_rows_are_skipped(self) -> None:
        orch = self._orch([
            {"role": "assistant", "content": "どうぞ"},
            {"role": "user", "content": "趣味の話をしよう"},
            {"role": "assistant", "content": ""},
        ])
        self.assertTrue(self._cross_future(orch))


class PrefetchFinalizesAsrFirstTest(unittest.TestCase):
    """疑似ストリーミングでない ASR では、先読みの前に ASR へ確定を指示し、結果を履歴に入れてから先読みする。"""

    def _orch(self) -> DialogOrchestrator:
        orch = DialogOrchestrator()
        orch.start({
            "maai_turn_threshold": 0.5,
            "maai_turn_condition": "p_now_and_future",
            "asr_nonstream_pseudo_enabled": False,
        })
        orch.state.opening_tts_pending = False
        orch.state.messages.extend([
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは。今日はどんなことをお手伝いしましょうか？"},
        ])
        return orch

    def _cross_future(self, orch: DialogOrchestrator) -> list[dict]:
        orch.on_turn_probability(0.2, p_future=0.3)
        return orch.on_turn_probability(0.3, p_future=0.6)

    @staticmethod
    def _final(orch: DialogOrchestrator, text: str) -> list[dict]:
        return orch.on_asr_result({"text": text, "is_final": True, "is_streaming": False})

    def test_requests_finalize_then_prefetches_with_the_new_text(self) -> None:
        orch = self._orch()
        events = self._cross_future(orch)
        self.assertTrue(any(ev.get("type") == "request_force_finalize" for ev in events))
        self.assertFalse(_prefetched(events))
        events = self._final(orch, "趣味の話をしよう")
        self.assertTrue(_prefetched(events))
        self.assertEqual(orch.state.messages[-1], {"role": "user", "content": "趣味の話をしよう"})
        self.assertEqual(orch.state.pending_texts, [])

    def test_pending_text_is_used_when_the_user_is_not_speaking(self) -> None:
        orch = self._orch()
        self._final(orch, "映画が好きです")  # ターンを取る前に確定して保留中
        self._cross_future(orch)
        events = orch.on_force_finalize_status(is_speaking=False, request_id=1, ts=0.0)
        self.assertTrue(_prefetched(events))
        self.assertEqual(orch.state.messages[-1], {"role": "user", "content": "映画が好きです"})

    def test_no_text_means_no_prefetch(self) -> None:
        orch = self._orch()
        self._cross_future(orch)
        events = orch.on_force_finalize_outcome(status="empty_text", request_id=1, ts=0.0)
        self.assertFalse(_prefetched(events))
        self.assertIsNone(orch.state.prefetch_wait_final)

    def test_p_future_falling_drops_the_wait(self) -> None:
        orch = self._orch()
        self._cross_future(orch)
        orch.on_turn_probability(0.3, p_future=0.3)
        self.assertIsNone(orch.state.prefetch_wait_final)
        events = self._final(orch, "やっぱりなんでもない")
        self.assertFalse(_prefetched(events))
        self.assertEqual(orch.state.pending_texts, ["やっぱりなんでもない"])


if __name__ == "__main__":
    unittest.main()
