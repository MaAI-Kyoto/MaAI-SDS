"""メイン画面の「会話」の欄を、別ウィンドウに出したり戻したりする。

会話の欄の中身（吹き出しの ChatView と、認識中・生成中の欄）をそのまま付け替えるだけなので、
会話を更新する処理（controller の voice_dialog_history_view など）は何も変えずに使える。

別ウィンドウにしている間は、メイン画面の左の列を隠して右の操作の列だけにする（会話を別のモニターに
大きく出し、操作の窓は小さくしておける）。別ウィンドウを閉じるか「メイン画面に戻す」で元に戻る。
別ウィンドウにしていたかどうかは QSettings（window_state と同じ保存先）に覚えて、次の起動でも同じにする。
位置・大きさは window_state が "chat" として覚える。
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

_DETACHED_KEY = "chat/detached"


def _default_settings() -> QSettings:
    from .window_state import APPLICATION, ORGANIZATION

    return QSettings(ORGANIZATION, APPLICATION)


class _ChatWindow(QWidget):
    def __init__(self, parent: QWidget, on_close: Callable[[], None]) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._on_close = on_close
        self.shutting_down = False

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt の名前
        # 利用者が閉じたら会話をメイン画面へ戻す（アプリの終了時は戻さず、別ウィンドウだったことを覚えたまま閉じる）
        if not self.shutting_down:
            self._on_close()
        super().closeEvent(event)


class ChatPopout:
    def __init__(
        self,
        controller: QWidget,
        content: QWidget,
        box_layout: QVBoxLayout,
        left_host: QWidget,
        toggle_button: QPushButton | None = None,
        settings_factory: Callable[[], QSettings] = _default_settings,
    ) -> None:
        self._controller = controller
        self._content = content
        self._box_layout = box_layout
        self._left_host = left_host
        self._toggle_button = toggle_button
        self._settings_factory = settings_factory
        self._main_width: int | None = None
        self._detached = False

        self.window = _ChatWindow(controller, on_close=self.attach)
        self.window.setWindowTitle("会話 - MaAI-SDS")
        self.window.resize(560, 760)
        layout = QVBoxLayout(self.window)
        layout.setContentsMargins(10, 10, 10, 10)
        bar = QHBoxLayout()
        bar.addStretch(1)
        back = QPushButton("メイン画面に戻す")
        back.setToolTip("会話の欄をメイン画面の左に戻します（このウィンドウを閉じても戻ります）。")
        back.clicked.connect(self.attach)
        bar.addWidget(back)
        layout.addLayout(bar)
        self._window_layout = layout
        if toggle_button is not None:
            toggle_button.clicked.connect(self.toggle)
        self._update_button()

    def is_detached(self) -> bool:
        return self._detached

    def toggle(self) -> None:
        if self._detached:
            self.attach()
        else:
            self.detach()

    def detach(self) -> None:
        if self._detached:
            self.window.raise_()
            self.window.activateWindow()
            return
        self._detached = True
        self._main_width = self._controller.width()
        self._window_layout.addWidget(self._content, 1)
        self._left_host.setVisible(False)
        self.window.show()
        self.window.raise_()
        self._save(True)
        self._update_button()
        # 右の列だけになった分、メイン画面の幅を詰める（レイアウトが落ち着いてから）
        QTimer.singleShot(0, self._shrink_main)

    def attach(self) -> None:
        if not self._detached:
            return
        self._detached = False
        self._box_layout.addWidget(self._content, 1)
        self._left_host.setVisible(True)
        if self.window.isVisible():
            self.window.hide()
        self._save(False)
        self._update_button()
        if self._main_width:
            width = self._main_width
            QTimer.singleShot(0, lambda: self._controller.resize(width, self._controller.height()))

    def restore(self) -> None:
        """前回の終了時に別ウィンドウだったら、別ウィンドウで開く（起動後に 1 回呼ぶ）。"""
        if self._load():
            self.detach()

    def shutdown(self) -> None:
        """アプリの終了時。別ウィンドウだったことは覚えたまま閉じる。"""
        self.window.shutting_down = True
        self.window.close()

    def _shrink_main(self) -> None:
        hint = self._controller.minimumSizeHint().width()
        if hint > 0:
            self._controller.resize(hint, self._controller.height())

    def _update_button(self) -> None:
        if self._toggle_button is None:
            return
        if self._detached:
            self._toggle_button.setText("会話をメイン画面に戻す")
            self._toggle_button.setToolTip("別ウィンドウにした会話の欄を、メイン画面の左に戻します。")
        else:
            self._toggle_button.setText("会話を別ウィンドウに")
            self._toggle_button.setToolTip("会話の欄を別ウィンドウに出します（別のモニターに大きく出すときなど）。")

    def _save(self, detached: bool) -> None:
        try:
            s = self._settings_factory()
            s.setValue(_DETACHED_KEY, bool(detached))
            s.sync()
        except Exception:  # noqa: BLE001 - 覚えられなくても動作は続ける
            pass

    def _load(self) -> bool:
        try:
            v: Any = self._settings_factory().value(_DETACHED_KEY, False)
        except Exception:  # noqa: BLE001
            return False
        return v is True or str(v).lower() in ("true", "1")
