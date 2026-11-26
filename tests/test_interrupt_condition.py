"""割り込む条件（voice_dialog_interrupt_condition：p_now / p_now_and_future）。"""

from __future__ import annotations

import unittest

from maai_sds.dialog import settings as vds
from maai_sds.dialog.orchestrator import DialogOrchestrator


def _interrupted(events: list[dict]) -> bool:
    return any(str(ev.get("type")) == "request_interrupt" for ev in events)


class InterruptConditionTest(unittest.TestCase):
    def _orch(self, condition: str | None) -> DialogOrchestrator:
        orch = DialogOrchestrator()
        settings: dict[str, object] = {
            "interrupt_on_user_speech": True,
            "interrupt_vad_threshold": 0.5,
            "interrupt_p_now_threshold": 0.5,
        }
        if condition is not None:
            settings["interrupt_condition"] = condition
        orch.start(settings)
        orch.state.opening_tts_pending = False
        return orch

    def _frame(self, orch: DialogOrchestrator, vad: float, p_now: float, p_future: float | None) -> bool:
        result: dict[str, object] = {"silero_vad_score": vad, "p_now": [1.0 - p_now, p_now]}
        if p_future is not None:
            result["p_future"] = [1.0 - p_future, p_future]
        return _interrupted(orch.on_maai_result(result))

    def test_default_is_p_now_and_ignores_p_future(self) -> None:
        orch = self._orch(None)
        self.assertTrue(self._frame(orch, 0.9, 0.2, 0.8))

    def test_and_mode_needs_p_future_below_the_threshold(self) -> None:
        orch = self._orch("p_now_and_future")
        self.assertFalse(self._frame(orch, 0.9, 0.2, 0.8))
        self.assertTrue(self._frame(orch, 0.9, 0.2, 0.3))

    def test_and_mode_still_needs_vad_and_p_now(self) -> None:
        orch = self._orch("p_now_and_future")
        self.assertFalse(self._frame(orch, 0.1, 0.2, 0.3))
        self.assertFalse(self._frame(orch, 0.9, 0.7, 0.3))

    def test_and_mode_falls_back_to_p_now_without_p_future(self) -> None:
        orch = self._orch("p_now_and_future")
        self.assertTrue(self._frame(orch, 0.9, 0.2, None))

    def test_setting_reaches_the_orchestrator(self) -> None:
        build = vds.build_dialog_orchestrator_settings
        self.assertEqual(build({})["interrupt_condition"], "p_now")
        self.assertEqual(
            build({"voice_dialog_interrupt_condition": "p_now_and_future"})["interrupt_condition"], "p_now_and_future"
        )
        self.assertEqual(build({"voice_dialog_interrupt_condition": "nonsense"})["interrupt_condition"], "p_now")


if __name__ == "__main__":
    unittest.main()
