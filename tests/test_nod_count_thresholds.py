"""頷きの回数・振り上げの決め方（maai 0.2.27 の nod_repetitions を読み、モデルと同じ規則で決める）。

以前は nod_repetitions を読まず回数が常に 1 回で、しきい値も掛け算（モデルは割り算）だった。
"""

from __future__ import annotations

import unittest

from maai_sds.dialog import settings as vds
from maai_sds.dialog.orchestrator import DialogOrchestrator
from maai_sds.utils import apply_maai_nod_para_prediction_aliases, nod_para_frame_decision

# maai 0.2.27 の VapNodPara が返す 1 フレーム（2 回の確率が最大、モデル自身の判定も 2 回・振り上げあり）
_FRAME = {
    "p_nod": 0.9,
    "nod_repetitions": [0.30, 0.45, 0.25],
    "nod_repetitions_pred": 1,
    "nod_swing_up": 0.45,
    "nod_swing_up_pred": 1,
    "nod_range": 0.1,
    "nod_speed": 0.2,
}


class FrameDecisionTest(unittest.TestCase):
    def test_blank_thresholds_use_the_models_own_decision(self) -> None:
        self.assertEqual(nod_para_frame_decision(_FRAME, None, None), (1, 1, 0.45))

    def test_thresholds_divide_the_probabilities_like_the_model(self) -> None:
        # t0 を小さくすると 1 回が出やすい（0.30/0.5=0.6 > 0.45）
        idx, _, _ = nod_para_frame_decision(_FRAME, {"t0": 0.5, "t1": 1.0, "t2": 1.0}, None)
        self.assertEqual(idx, 0)
        # 3 つとも同じ値なら、いちばん確率の高い回数（公平）
        idx, _, _ = nod_para_frame_decision(_FRAME, {"t0": 0.33, "t1": 0.33, "t2": 0.33}, None)
        self.assertEqual(idx, 1)

    def test_swing_threshold_overrides_the_model(self) -> None:
        self.assertEqual(nod_para_frame_decision(_FRAME, None, 0.5)[1], 0)
        self.assertEqual(nod_para_frame_decision(_FRAME, None, 0.4)[1], 1)

    def test_gui_aliases_read_the_new_key_names(self) -> None:
        r = dict(_FRAME)
        apply_maai_nod_para_prediction_aliases(r)
        self.assertEqual(r["nod_count"], 1)
        self.assertNotIn("nod_repetitions", r)


class SettingsTest(unittest.TestCase):
    def test_partial_or_blank_count_thresholds_fall_back_to_the_model(self) -> None:
        self.assertIsNone(vds._cfg_nod_count_thresholds({}))
        self.assertIsNone(vds._cfg_nod_count_thresholds(
            {"maai_test_nod_count_t0": "0.5", "maai_test_nod_count_t1": "", "maai_test_nod_count_t2": ""}))
        self.assertEqual(
            vds._cfg_nod_count_thresholds(
                {"maai_test_nod_count_t0": "0.5", "maai_test_nod_count_t1": 1, "maai_test_nod_count_t2": "1"}),
            {"t0": 0.5, "t1": 1.0, "t2": 1.0},
        )
        self.assertIsNone(vds._cfg_optional_unit({"maai_test_nod_swing_threshold": ""}, "maai_test_nod_swing_threshold"))


class OrchestratorTest(unittest.TestCase):
    def _fire(self, settings: dict) -> tuple[int, bool]:
        orch = DialogOrchestrator()
        orch.settings.update(settings)
        frames = [orch._parse_nod_frame(dict(_FRAME)) for _ in range(3)]
        out = orch._resolve_series_params_for_fire(
            use_range=[f["frame_range"] for f in frames],
            use_speed=[f["frame_speed"] for f in frames],
            use_swing=[f["frame_swing"] for f in frames],
            use_raw_range=[f["raw_range"] for f in frames],
            use_raw_speed=[f["raw_speed"] for f in frames],
            use_count_idx=[f["frame_count_idx"] for f in frames],
            use_p_nod=[f["p_nod"] for f in frames],
            use_swing_up=[f["frame_swing_up"] for f in frames],
        )
        return out[0], out[3]

    def test_two_nods_reach_the_avatar_with_model_defaults(self) -> None:
        self.assertEqual(self._fire({"nod_para_count_thresholds": None, "nod_para_swing_up_threshold": None}), (2, True))

    def test_settings_built_from_config_drive_the_count(self) -> None:
        cfg = {"maai_test_nod_count_t0": "0.5", "maai_test_nod_count_t1": "1", "maai_test_nod_count_t2": "1",
               "maai_test_nod_swing_threshold": "0.5"}
        settings = {
            "nod_para_count_thresholds": vds._cfg_nod_count_thresholds(cfg),
            "nod_para_swing_up_threshold": vds._cfg_optional_unit(cfg, "maai_test_nod_swing_threshold"),
        }
        self.assertEqual(self._fire(settings), (1, False))


if __name__ == "__main__":
    unittest.main()
