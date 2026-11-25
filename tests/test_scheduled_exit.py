"""終了時の発話（voice_dialog_scheduled_exit_*）は、開始時の発話と同じく最後まで流す。"""

from __future__ import annotations

import time
import unittest

from maai_sds.dialog.orchestrator import DialogOrchestrator

_EXIT = "時間になりました。ありがとうございました！"


def _types(events: list[dict]) -> list[str]:
    return [str(ev.get("type")) for ev in events]


class ScheduledExitTest(unittest.TestCase):
    def _orch(self) -> DialogOrchestrator:
        orch = DialogOrchestrator()
        orch.start({
            "maai_turn_threshold": 0.5,
            "voice_dialog_scheduled_exit_sec": 1.0,
            "voice_dialog_scheduled_exit_utterance": _EXIT,
            "interrupt_on_user_speech": True,
            "interrupt_vad_threshold": 0.5,
            "interrupt_p_now_threshold": 0.5,
            "voice_dialog_idle_take_turn_sec": 0.1,
        })
        orch.state.voice_dialog_start_wall_ts = time.time() - 10.0  # 終了までの時間を過ぎている
        orch.state.messages.append({"role": "user", "content": "こんにちは"})
        return orch

    def _say_exit(self, orch: DialogOrchestrator) -> None:
        orch.on_turn_probability(0.2, p_future=0.2)
        events = orch.on_turn_probability(0.7, p_future=0.7)
        self.assertIn("request_scheduled_exit_tts", _types(events))
        self.assertTrue(orch.state.scheduled_exit_speaking)

    def test_user_speech_does_not_interrupt_the_exit_utterance(self) -> None:
        orch = self._orch()
        self._say_exit(orch)
        # 相手が話している（VAD が高く p_now が低い）フレームでも、割り込みを求めない
        events = orch.on_maai_result({"vad": [0.99, 0.0], "p_now": 0.1})
        self.assertNotIn("request_interrupt", _types(events))

    def test_no_turn_and_no_second_exit_while_it_plays(self) -> None:
        orch = self._orch()
        self._say_exit(orch)
        orch.on_maai_result({"vad": [0.99, 0.0], "p_now": 0.1})  # ユーザー発話でターンロックが外れても
        for p in (0.2, 0.8, 0.2, 0.9):
            self.assertEqual(orch.on_turn_probability(p, p_future=p), [])
        orch.state.last_user_speech_detect_wall_ts = time.time() - 5.0
        self.assertNotIn("turn", _types(orch.on_tick()))

    def test_user_speech_still_interrupts_before_the_exit(self) -> None:
        orch = self._orch()
        events = orch.on_maai_result({"vad": [0.99, 0.0], "p_now": 0.1})
        self.assertIn("request_interrupt", _types(events))


if __name__ == "__main__":
    unittest.main()
