"""MaAI プロットは、表示するグラフが 9 個までは 1 列、10 個以上で 2 列（Qt と Web で同じ）。"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None

_REPO = Path(__file__).resolve().parents[1]


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class PlotColumnsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_up_to_nine_graphs_stay_in_one_column(self) -> None:
        from maai_sds.maai.plot_window import MAAI_PLOT_VISIBILITY_SPECS, MaaiPlotWindow

        win = MaaiPlotWindow(frame_rate=12.5, sample_rate=16000)
        win.init_full_layout()
        ids = [panel_id for panel_id, _label in MAAI_PLOT_VISIBILITY_SPECS]
        win.configure_maai_plot_panel_visibility({i: False for i in ids})
        seen: dict[int, int] = {}
        for i in ids:
            win.configure_maai_plot_panel_visibility({i: True})
            lay = win.graph.ci.layout
            n = lay.count()
            seen[n] = lay.columnCount()
        self.assertIn(9, seen, seen)
        self.assertIn(10, seen, seen)
        for n, cols in seen.items():
            self.assertEqual(cols, 1 if n <= 9 else 2, f"{n} 個のとき {cols} 列")
        win.close()

    def test_web_uses_the_same_threshold(self) -> None:
        js = (_REPO / "web/app/plot.js").read_text(encoding="utf-8")
        self.assertRegex(js, r"const twoColumns = visibleCount > 9;")
        py = (_REPO / "src/maai_sds/maai/plot_window.py").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"use_two_cols = n_visible > 9", py))


if __name__ == "__main__":
    unittest.main()
