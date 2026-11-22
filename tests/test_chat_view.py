"""メイン画面の会話の欄（gui/chat_view.py）。"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class ChatViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_updates_only_changed_bubbles(self) -> None:
        from maai_sds.gui.chat_view import ChatView

        view = ChatView()
        view.resize(600, 400)
        view.set_items([("assistant", "こんにちは"), ("user_partial", "今日は")])
        first = [row.widget for row in view._rows]
        # 認識の途中結果が伸びるだけなら、吹き出しは作り直さない。
        view.set_items([("assistant", "こんにちは"), ("user_partial", "今日は疲れた")])
        self.assertEqual([row.widget for row in view._rows], first)
        self.assertEqual(view.items()[-1], ("user_partial", "今日は疲れた"))
        # 途中結果が確定して役割が変わったら、その吹き出しから作り直す。
        view.set_items([("assistant", "こんにちは"), ("user", "今日は疲れた"), ("assistant_streaming", "それは")])
        self.assertIs(view._rows[0].widget, first[0])
        self.assertEqual([r for r, _t in view.items()], ["assistant", "user", "assistant_streaming"])
        view.set_items([])
        self.assertEqual(view.items(), [])
        self.assertFalse(view._placeholder.isHidden())

    def test_chat_shows_confirmed_only_and_live_box_shows_progress(self) -> None:
        from types import SimpleNamespace

        from maai_sds.gui.controller import MMDAgentController

        fake = SimpleNamespace(
            _voice_dialog_messages=[
                {"role": "system", "content": "あなたは…（プロンプト）"},
                {"role": "user", "content": "やあ"},
                {"role": "assistant", "content": "こんにちは"},
            ],
            _voice_dialog_pending_texts=["確定済みの発話"],
            _voice_dialog_current_partial="認識中の",
            _voice_dialog_streaming_assistant_full="こんにちは",
            _voice_dialog_tts_confirmed_segments=[],
        )
        fake._voice_dialog_streaming_text = lambda: MMDAgentController._voice_dialog_streaming_text(fake)
        # 会話の欄: プロンプト・認識中・生成中は出さない。確定した発話と応答だけ。
        items = MMDAgentController._voice_dialog_chat_items(fake)
        self.assertEqual(items, [("user", "やあ"), ("assistant", "こんにちは"), ("user", "確定済みの発話")])
        # 進行中の欄: 応答が履歴に入ったら生成中は空。
        self.assertEqual(MMDAgentController._voice_dialog_live_texts(fake), ("認識中の", ""))
        fake._voice_dialog_streaming_assistant_full = "次の応答"
        self.assertEqual(MMDAgentController._voice_dialog_live_texts(fake), ("認識中の", "次の応答"))


if __name__ == "__main__":
    unittest.main()
