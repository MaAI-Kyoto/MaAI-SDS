"""頷きと相槌は、出したあと確率がしきい値を一度下回るまで次を出さない。

頷きは以前、しきい値ではなく固定の 0.5 で「下がった」を見ていた。相槌には条件が無く、
確率がしきい値より上に居続けると、最短間隔ごとに相槌を打ち続けていた。
"""

from __future__ import annotations

import unittest

from maai_sds.dialog.orchestrator import DialogOrchestrator

_NOD = {"nod_repetitions": [0.6, 0.3, 0.1], "nod_swing_up": 0.2, "nod_range": 0.1, "nod_speed": 0.2}


class RearmTest(unittest.TestCase):
    def _orch(self, settings: dict) -> tuple[DialogOrchestrator, list, list]:
        nods: list = []
        bcs: list = []
        orch = DialogOrchestrator(
            on_nod_motion=nods.append,
            on_request_nonstream_tts=lambda text, src: bcs.append(text),
        )
        orch.start(settings)
        return orch, nods, bcs

    def _nod_seq(self, orch: DialogOrchestrator, nods: list, probs: list[float]) -> list[bool]:
        out = []
        for p in probs:
            before = len(nods)
            orch.on_maai_nod_result({**_NOD, "p_nod": p})
            out.append(len(nods) > before)
        return out

    def _bc_seq(self, orch: DialogOrchestrator, bcs: list, probs: list[float]) -> list[bool]:
        out = []
        for p in probs:
            before = len(bcs)
            orch.on_maai_bc_result({"p_bc": p})
            out.append(len(bcs) > before)
        return out

    def test_nod_waits_until_p_nod_falls_below_its_threshold(self) -> None:
        orch, nods, _ = self._orch({
            "nod_para_enabled": True, "nod_para_prob_threshold": 0.4,
            "nod_para_min_consecutive_frames": 1, "nod_para_min_interval_ms": 0,
        })
        # 0.45 は 0.5 未満だが、しきい値 0.4 は下回っていないので、次の頷きはしない
        self.assertEqual(self._nod_seq(orch, nods, [0.9, 0.45, 0.9, 0.45, 0.9]), [True, False, False, False, False])
        # しきい値を下回ってから超えたら、また頷く
        self.assertEqual(self._nod_seq(orch, nods, [0.3, 0.9]), [False, True])

    def test_nod_with_a_threshold_above_one_half(self) -> None:
        orch, nods, _ = self._orch({
            "nod_para_enabled": True, "nod_para_prob_threshold": 0.6,
            "nod_para_min_consecutive_frames": 1, "nod_para_min_interval_ms": 0,
        })
        # 0.55 は 0.5 以上だが、しきい値 0.6 を下回ったので、次の頷きを許す
        self.assertEqual(self._nod_seq(orch, nods, [0.9, 0.55, 0.9]), [True, False, True])

    def test_backchannel_waits_until_p_bc_falls_below_its_threshold(self) -> None:
        orch, _, bcs = self._orch({
            "maai_backchannel_enabled": True, "maai_backchannel_prob_threshold": 0.4,
            "maai_backchannel_consecutive_frames": 1, "maai_backchannel_min_interval_sec": 0,
        })
        # 上に居続ける間は 1 回だけ
        self.assertEqual(self._bc_seq(orch, bcs, [0.8, 0.8, 0.8, 0.8]), [True, False, False, False])
        # 一度下回ってから超えたら、また打つ
        self.assertEqual(self._bc_seq(orch, bcs, [0.2, 0.8]), [False, True])


if __name__ == "__main__":
    unittest.main()
