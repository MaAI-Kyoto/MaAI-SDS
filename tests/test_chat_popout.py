"""メイン画面の「会話」の欄を別ウィンドウに出す・戻す（gui/chat_window.py）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget
except ImportError:  # pragma: no cover
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class ChatPopoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        ini = str(Path(self.tmp.name) / "s.ini")
        self.settings = lambda: QSettings(ini, QSettings.Format.IniFormat)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _build(self):
        from maai_sds.gui.chat_window import ChatPopout

        main = QWidget()
        left = QWidget(main)
        box = QGroupBox("会話", left)
        box_layout = QVBoxLayout(box)
        content = QWidget()
        QVBoxLayout(content).addWidget(QLabel("chat"))
        box_layout.addWidget(content)
        button = QPushButton(main)
        popout = ChatPopout(main, content, box_layout, left, toggle_button=button, settings_factory=self.settings)
        main.show()
        return main, left, box, content, button, popout

    def test_toggle_moves_the_chat_out_and_back(self) -> None:
        main, left, box, content, button, popout = self._build()
        self.assertEqual(button.text(), "別ウィンドウ")
        button.click()
        self.app.processEvents()
        self.assertTrue(popout.is_detached())
        self.assertIs(content.window(), popout.window)
        self.assertTrue(popout.window.isVisible())
        self.assertFalse(left.isVisible())
        popout.attach()  # 戻すのは会話の窓の側（ボタンは欄ごと隠れている）
        self.app.processEvents()
        self.assertFalse(popout.is_detached())
        self.assertIs(content.parentWidget(), box)
        self.assertTrue(left.isVisible())
        self.assertFalse(popout.window.isVisible())

    def test_closing_the_chat_window_brings_the_chat_back(self) -> None:
        main, left, box, content, button, popout = self._build()
        popout.detach()
        popout.window.close()
        self.app.processEvents()
        self.assertFalse(popout.is_detached())
        self.assertIs(content.parentWidget(), box)

    def test_detached_state_survives_a_restart(self) -> None:
        main, left, box, content, button, popout = self._build()
        popout.detach()
        popout.shutdown()  # アプリの終了：戻さずに閉じる
        self.assertTrue(popout.is_detached())
        main2, left2, box2, content2, button2, popout2 = self._build()
        popout2.restore()
        self.app.processEvents()
        self.assertTrue(popout2.is_detached())
        self.assertIs(content2.window(), popout2.window)
        popout2.attach()
        main3, *_rest, popout3 = self._build()
        popout3.restore()
        self.assertFalse(popout3.is_detached())


if __name__ == "__main__":
    unittest.main()
