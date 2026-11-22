"""MaAI プロットの既定で隠すパネルが、Qt・Web・設定の見本でそろっていること。"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from maai_sds.maai.plot_window import MAAI_PLOT_HIDDEN_BY_DEFAULT, MAAI_PLOT_VISIBILITY_SPECS

ROOT = Path(__file__).resolve().parents[1]


class MaaiPlotDefaultsTest(unittest.TestCase):
    def test_silero_and_nod_parameters_are_hidden_by_default(self) -> None:
        self.assertTrue({"silero_vad_score", "nod_range", "nod_speed", "nod_count", "nod_swing_up"} <= MAAI_PLOT_HIDDEN_BY_DEFAULT)
        self.assertNotIn("p_nod", MAAI_PLOT_HIDDEN_BY_DEFAULT)
        self.assertTrue(MAAI_PLOT_HIDDEN_BY_DEFAULT <= {pid for pid, _ in MAAI_PLOT_VISIBILITY_SPECS})

    def test_web_rows_match(self) -> None:
        js = (ROOT / "web" / "app" / "plot.js").read_text(encoding="utf-8")
        hidden = set(re.findall(r"\{ id: '(\w+)'[^\n]*hiddenByDefault: true", js))
        self.assertEqual(hidden, set(MAAI_PLOT_HIDDEN_BY_DEFAULT))

    def test_example_configs_match(self) -> None:
        for name in ("controller_config.example.json", "controller_config.docker.example.json"):
            cfg = json.loads((ROOT / name).read_text(encoding="utf-8"))
            for pid, _ in MAAI_PLOT_VISIBILITY_SPECS:
                key = f"maai_plot_show_{pid}"
                if key in cfg:
                    self.assertEqual(bool(cfg[key]), pid not in MAAI_PLOT_HIDDEN_BY_DEFAULT, (name, key))


if __name__ == "__main__":
    unittest.main()
