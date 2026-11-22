"""窓の位置・大きさを覚えて戻す（gui/window_state.py）のテスト。開いていた画面は開き直さない。"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from maai_sds.gui import window_state  # noqa: E402

APP = QApplication.instance() or QApplication([])


def make_controller() -> QWidget:
    c = QWidget()
    c.settings_hub_window = QWidget(None, Qt.WindowType.Window)
    c.console_window = QWidget(None, Qt.WindowType.Window)
    return c


class WindowStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, "state.ini")
        self.factory = lambda: QSettings(path, QSettings.Format.IniFormat)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_geometry_comes_back_but_windows_are_not_reopened(self) -> None:
        c = make_controller()
        c.setGeometry(40, 50, 520, 380)  # 画面の外へはみ出す大きさは Qt が画面に収めるので、収まる大きさで
        c.console_window.setGeometry(60, 70, 400, 300)
        c.console_window.show()
        c.settings_hub_window.show()
        window_state.save(c, self.factory)

        d = make_controller()
        window_state.restore_main(d, self.factory)
        window_state.restore_children(d, self.factory)
        self.assertEqual((d.width(), d.height()), (520, 380))
        self.assertEqual((d.console_window.width(), d.console_window.height()), (400, 300))
        # 閉じたときに開いていても、起動時に勝手に開かない
        self.assertFalse(d.console_window.isVisible())
        self.assertFalse(d.settings_hub_window.isVisible())

    def test_missing_state_is_harmless(self) -> None:
        d = make_controller()
        window_state.restore_main(d, self.factory)  # 何も保存していない
        window_state.restore_children(d, self.factory)
        self.assertFalse(d.console_window.isVisible())


if __name__ == "__main__":
    unittest.main()
