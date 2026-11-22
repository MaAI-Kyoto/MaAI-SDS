"""会話の欄（gui/chat_view.py）が一番下に付いていく。"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from maai_sds.gui.chat_view import ChatView  # noqa: E402

APP = QApplication.instance() or QApplication([])


def _settle() -> None:
    for _ in range(5):
        APP.processEvents()


class ChatViewScrollTest(unittest.TestCase):
    def setUp(self) -> None:
        self.view = ChatView()
        self.view.resize(360, 240)
        self.view.show()
        self.items: list[tuple[str, str]] = []
        _settle()

    def tearDown(self) -> None:
        self.view.close()

    def _add(self, n: int) -> None:
        for i in range(n):
            role = "user" if len(self.items) % 2 == 0 else "assistant"
            self.items.append((role, f"発話 {len(self.items)} " + "あいうえお" * 6))
            self.view.set_items(list(self.items))
            _settle()

    def _bar(self):
        return self.view.verticalScrollBar()

    def test_follows_new_messages_to_the_bottom(self) -> None:
        self._add(30)
        self.assertGreater(self._bar().maximum(), 0)
        self.assertEqual(self._bar().value(), self._bar().maximum())

    def test_stays_put_while_reading_and_follows_again_at_the_bottom(self) -> None:
        self._add(30)
        self._bar().setValue(0)  # 上へスクロールして読んでいる
        _settle()
        self._add(5)
        self.assertEqual(self._bar().value(), 0)
        self._bar().setValue(self._bar().maximum())  # 一番下に戻す
        _settle()
        self._add(5)
        self.assertEqual(self._bar().value(), self._bar().maximum())


if __name__ == "__main__":
    unittest.main()
