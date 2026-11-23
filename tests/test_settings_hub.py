"""設定・テストを 1 枚にまとめたウィンドウ（gui/settings_hub.py）。"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
except ImportError:  # pragma: no cover - GUI extra が無い環境
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class SettingsHubTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _controller(self) -> QWidget:
        from maai_sds.gui.settings_hub import HUB_PAGES

        ctrl = QWidget()
        for _section, items in HUB_PAGES:
            for attr, label in items:
                win = QWidget(ctrl)
                layout = QVBoxLayout(win)
                layout.addWidget(QLabel(label))
                setattr(ctrl, attr, win)
        return ctrl

    def test_existing_show_calls_open_the_page_in_one_window(self) -> None:
        from maai_sds.gui.settings_hub import SettingsHub

        ctrl = self._controller()
        hub = SettingsHub(ctrl)
        self.assertIn("ASR（音声認識）", hub.page_titles())
        self.assertIn("音声対話の詳細設定", hub.page_titles())
        # 呼び出し側は今までどおり xxx_window.show() / raise_() を呼ぶ。
        ctrl.maai_settings_window.show()
        ctrl.maai_settings_window.raise_()
        self.app.processEvents()
        self.assertTrue(hub.isVisible())
        self.assertEqual(hub.title.text(), "MaAIモデル設定")
        self.assertTrue(ctrl.maai_settings_window.isVisible())
        self.assertFalse(ctrl.asr_test_window.isVisible())
        ctrl.asr_test_window.show()
        self.app.processEvents()
        self.assertEqual(hub.title.text(), "ASR（音声認識）")
        self.assertTrue(ctrl.asr_test_window.isVisible())
        self.assertFalse(ctrl.maai_settings_window.isVisible())
        # 終了処理の xxx_window.close() はページを閉じない（ウィンドウは 1 枚だけ）。
        self.assertTrue(ctrl.asr_test_window.close())
        hub.close()

    def test_nav_selection_switches_page(self) -> None:
        from maai_sds.gui.settings_hub import SettingsHub

        ctrl = self._controller()
        hub = SettingsHub(ctrl)
        for row in range(hub.nav.count()):
            if hub.nav.item(row).text() == "OBS録画設定":
                hub.nav.setCurrentRow(row)
                break
        self.assertEqual(hub.title.text(), "OBS録画設定")
        hub.close()


if __name__ == "__main__":
    unittest.main()
