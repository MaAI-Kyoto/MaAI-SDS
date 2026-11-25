"""ターンを取る条件（asr_maai_turn_condition：p_now / p_now_and_future）。"""

from __future__ import annotations

import unittest

from maai_sds.dialog import settings as vds
from maai_sds.dialog.orchestrator import DialogOrchestrator


def _took_turn(events: list[dict]) -> bool:
    return any(str(ev.get("type")) == "turn" for ev in events)


class TurnConditionTest(unittest.TestCase):
    def _orch(self, condition: str | None) -> DialogOrchestrator:
        orch = DialogOrchestrator()
        settings: dict[str, object] = {"maai_turn_threshold": 0.5}
        if condition is not None:
            settings["maai_turn_condition"] = condition
        orch.start(settings)
        return orch

    def _feed(self, orch: DialogOrchestrator, frames: list[tuple[float, float | None]]) -> list[bool]:
        return [_took_turn(orch.on_turn_probability(p, p_future=f)) for p, f in frames]

    def test_p_now_mode_ignores_p_future(self) -> None:
        orch = self._orch("p_now")
        self.assertEqual(self._feed(orch, [(0.2, 0.1), (0.7, 0.1)]), [False, True])

    def test_default_is_p_now_and_future(self) -> None:
        orch = self._orch(None)
        self.assertEqual(self._feed(orch, [(0.2, 0.1), (0.7, 0.1)]), [False, False])
        orch = self._orch(None)
        self.assertEqual(self._feed(orch, [(0.2, 0.1), (0.7, 0.7)]), [False, True])

    def test_and_mode_takes_turn_when_p_future_is_above_at_the_crossing(self) -> None:
        orch = self._orch("p_now_and_future")
        # p_future はひとつ前のフレームで既に上にまたいでいる（このとき先読みが始まる）
        self.assertEqual(self._feed(orch, [(0.2, 0.3), (0.3, 0.6), (0.7, 0.7)]), [False, False, True])

    def test_and_mode_skips_when_p_future_is_below_at_the_crossing(self) -> None:
        orch = self._orch("p_now_and_future")
        events = []
        for p, f in [(0.2, 0.1), (0.7, 0.3)]:
            events = orch.on_turn_probability(p, p_future=f)
        self.assertFalse(_took_turn(events))
        self.assertTrue(any("p_future_below_threshold" in str(ev.get("message", "")) for ev in events))
        # p_now がしきい値の上にいる間に p_future が上がっても、またぎ直すまでは取らない
        self.assertEqual(self._feed(orch, [(0.8, 0.9)]), [False])
        # いったん下回ってから再びまたぎ、そのとき p_future も上なら取る
        self.assertEqual(self._feed(orch, [(0.3, 0.9), (0.7, 0.9)]), [False, True])

    def test_and_mode_falls_back_to_p_now_without_p_future(self) -> None:
        orch = self._orch("p_now_and_future")
        self.assertEqual(self._feed(orch, [(0.2, None), (0.7, None)]), [False, True])

    def _orch_with_prefetch(self, condition: str) -> tuple[DialogOrchestrator, dict[str, int]]:
        calls = {"commit": 0, "cancel": 0}
        orch = DialogOrchestrator(
            on_commit_prefetch=lambda: calls.__setitem__("commit", calls["commit"] + 1),
            on_cancel_prefetch=lambda: calls.__setitem__("cancel", calls["cancel"] + 1),
        )
        orch.start({"maai_turn_threshold": 0.5, "maai_turn_condition": condition})
        orch.state.messages.append({"role": "user", "content": "こんにちは"})  # 先読みには直前のユーザー発話が要る
        return orch, calls

    def test_and_mode_commits_the_prefetch(self) -> None:
        orch, calls = self._orch_with_prefetch("p_now_and_future")
        # p_future の上抜けで先読み → p_now の上抜け（p_future も上）で先読みを確定
        self.assertEqual(self._feed(orch, [(0.2, 0.3), (0.3, 0.6), (0.7, 0.7)]), [False, False, True])
        self.assertEqual(calls, {"commit": 1, "cancel": 0})
        self.assertFalse(orch.state.prefetch_active)

    def test_and_mode_cancelled_prefetch_does_not_take_the_turn(self) -> None:
        orch, calls = self._orch_with_prefetch("p_now_and_future")
        # p_future が下がって先読みを取り消したあと、p_future が下のまま p_now が超えても取らない
        self.assertEqual(self._feed(orch, [(0.2, 0.3), (0.3, 0.6), (0.3, 0.4), (0.7, 0.4)]), [False] * 4)
        self.assertEqual(calls, {"commit": 0, "cancel": 1})
        self.assertFalse(orch.state.prefetch_active)

    def test_p_now_mode_takes_the_turn_when_prefetch_is_cancelled_in_the_same_frame(self) -> None:
        orch, calls = self._orch_with_prefetch("p_now")
        # 先読み中に、同じフレームで p_future が下回り p_now が超えた：先読みは取り消し、ターンは取る
        self.assertEqual(self._feed(orch, [(0.2, 0.3), (0.3, 0.6), (0.7, 0.4)]), [False, False, True])
        self.assertEqual(calls, {"commit": 0, "cancel": 1})
        self.assertFalse(orch.state.prefetch_active)

    def test_and_mode_does_not_take_the_turn_when_prefetch_is_cancelled_in_the_same_frame(self) -> None:
        orch, calls = self._orch_with_prefetch("p_now_and_future")
        self.assertEqual(self._feed(orch, [(0.2, 0.3), (0.3, 0.6), (0.7, 0.4)]), [False, False, False])
        self.assertEqual(calls, {"commit": 0, "cancel": 1})
        self.assertFalse(orch.state.prefetch_active)

    def _orch_hold(self, hold: int, condition: str = "p_now") -> DialogOrchestrator:
        orch = DialogOrchestrator()
        orch.start({"maai_turn_threshold": 0.5, "maai_turn_condition": condition, "maai_turn_hold_frames": hold})
        return orch

    def test_hold_one_frame_is_the_crossing(self) -> None:
        orch = self._orch_hold(1)
        self.assertEqual(self._feed(orch, [(0.2, 0.1), (0.7, 0.1), (0.8, 0.1)]), [False, True, False])

    def test_hold_three_frames_waits_and_ignores_short_spikes(self) -> None:
        orch = self._orch_hold(3)
        # 2 フレームだけ超えて下がる（話の途中の間）→ 取らない
        self.assertEqual(self._feed(orch, [(0.2, 0.1), (0.7, 0.1), (0.7, 0.1), (0.3, 0.1)]), [False] * 4)
        # 3 フレーム続けて超えたら、その 3 フレーム目で取る（その後は取り直さない）
        self.assertEqual(self._feed(orch, [(0.7, 0.1), (0.7, 0.1), (0.7, 0.1), (0.7, 0.1)]),
                         [False, False, True, False])

    def test_hold_does_not_start_from_the_first_frame(self) -> None:
        orch = self._orch_hold(2)
        # 始めから上にいるときは、上にまたいだことにならないので取らない（1 フレームのときと同じ）
        self.assertEqual(self._feed(orch, [(0.7, 0.1), (0.7, 0.1), (0.7, 0.1)]), [False] * 3)

    def test_hold_with_and_condition_checks_p_future_at_the_taking_frame(self) -> None:
        orch = self._orch_hold(2, "p_now_and_future")
        self.assertEqual(self._feed(orch, [(0.2, 0.1), (0.7, 0.4), (0.7, 0.4)]), [False, False, False])
        orch = self._orch_hold(2, "p_now_and_future")
        self.assertEqual(self._feed(orch, [(0.2, 0.1), (0.7, 0.4), (0.7, 0.6)]), [False, False, True])

    def test_hold_commits_the_prefetch_started_in_between(self) -> None:
        orch, calls = self._orch_with_prefetch("p_now")
        orch.settings["maai_turn_hold_frames"] = 3
        # p_now と p_future が同じフレームで上にまたぐ：ターンはまだ取らないので先読みを始め、3 フレーム目で確定
        self.assertEqual(self._feed(orch, [(0.2, 0.2), (0.7, 0.7), (0.7, 0.7), (0.7, 0.7)]),
                         [False, False, False, True])
        self.assertEqual(calls, {"commit": 1, "cancel": 0})

    def test_hold_setting_reaches_the_orchestrator(self) -> None:
        build = vds.build_dialog_orchestrator_settings
        self.assertEqual(build({})["maai_turn_hold_frames"], 1)
        self.assertEqual(build({"asr_maai_turn_hold_frames": "3"})["maai_turn_hold_frames"], 3)
        self.assertEqual(build({"asr_maai_turn_hold_frames": "0"})["maai_turn_hold_frames"], 1)

    def test_setting_reaches_the_orchestrator(self) -> None:
        build = vds.build_dialog_orchestrator_settings
        self.assertEqual(build({})["maai_turn_condition"], "p_now_and_future")
        self.assertEqual(build({"asr_maai_turn_condition": "p_now_and_future"})["maai_turn_condition"], "p_now_and_future")
        self.assertEqual(build({"asr_maai_turn_condition": "nonsense"})["maai_turn_condition"], "p_now_and_future")
        self.assertEqual(build({"asr_maai_turn_condition": "p_now"})["maai_turn_condition"], "p_now")


if __name__ == "__main__":
    unittest.main()
