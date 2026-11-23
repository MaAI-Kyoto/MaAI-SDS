"""設定・テストの各画面を 1 枚のウィンドウにまとめる（左に一覧、右に中身）。

以前は「ASRテスト」「音声入力の設定」などがボタンごとに別ウィンドウで開き、何枚も重なっていた。
各画面のウィジェット（controller.asr_test_window など）はそのまま残し、このウィンドウのページに
入れる。既存のコードは `controller.xxx_window.show()` などを呼んでいるので、ページにした画面の
show / raise_ / activateWindow を「このウィンドウでそのページを開く」に差し替える
（呼び出し側は変えない）。Web 版の画面定義（scripts/extract_web_ui_manifest.py）は各画面の
ウィジェットを直接たどるので、ページにしても変わらない。

コンソール・MaAIプロット・頷きモーション波形・TEMPLATE認証は、横に並べて見たい／小さな
補助画面なので別ウィンドウのまま残す。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .theme import TOKENS

# (見出し, [(controller の属性名, 一覧に出す名前), ...])。並びは使う順。
# エンジン（ASR・LLM・TTS・MaAI）の選択と設定は、各画面の中で試しながら行う。以前は「ASRテスト」
# などの名前で「テスト」の見出しの下にあり、「設定」を探してもエンジンを選ぶ場所が見つからなかった。
HUB_PAGES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("エンジン", (
        ("asr_test_window", "ASR（音声認識）"),
        ("llm_test_window", "LLM（応答生成）"),
        ("tts_test_window", "TTS（音声合成）"),
        ("maai_settings_window", "MaAIモデル設定"),
    )),
    ("設定", (
        ("voice_dialog_settings_window", "音声対話の詳細設定"),
        ("settings_window", "MMDAgent-EX起動設定"),
        ("audio_input_settings_window", "音声入力の設定"),
        ("audio_output_settings_window", "音声出力の設定"),
        ("obs_settings_window", "OBS録画設定"),
    )),
    ("テスト", (
        ("llm_s2s_test_window", "LLM分割テスト"),
        ("maai_test_window", "MaAIテスト"),
    )),
    ("ツール", (
        ("engine_manager_window", "エンジン管理"),
    )),
)

_PAGE_ROLE = Qt.ItemDataRole.UserRole


class SettingsHub(QWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__(controller, Qt.WindowType.Window)
        self.setWindowTitle("設定・テスト")
        self.resize(1040, 720)
        self._pages: dict[int, tuple[int, int, str]] = {}  # id(widget) -> (stack index, nav row, title)

        self.nav = QListWidget()
        self.nav.setFixedWidth(190)
        self.nav.setProperty("role", "hub-nav")
        self.stack = QStackedWidget()
        self.title = QLabel("")
        self.title.setProperty("role", "hub-title")

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)
        right.addWidget(self.title)
        right.addWidget(self.stack, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        layout.addWidget(self.nav)
        layout.addLayout(right, 1)

        caption_font = QFont(self.font())
        caption_font.setPointSize(max(7, caption_font.pointSize() - 1))
        for section, items in HUB_PAGES:
            header = QListWidgetItem(section)
            header.setFlags(Qt.ItemFlag.NoItemFlags)  # 見出し（選べない）
            header.setFont(caption_font)
            header.setForeground(QColor(TOKENS["muted"]))
            self.nav.addItem(header)
            for attr, label in items:
                widget = getattr(controller, attr, None)
                if not isinstance(widget, QWidget):
                    continue
                self._add_page(widget, label)
        self.nav.currentItemChanged.connect(self._on_nav_changed)

    def _add_page(self, widget: QWidget, label: str) -> None:
        widget.setWindowFlags(Qt.WindowType.Widget)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(widget)
        index = self.stack.addWidget(scroll)
        item = QListWidgetItem(label)
        item.setData(_PAGE_ROLE, index)
        self.nav.addItem(item)
        row = self.nav.row(item)
        self._pages[id(widget)] = (index, row, label)
        # 既存の呼び出し（xxx_window.show() / raise_() / activateWindow() / close()）を
        # 「このウィンドウでそのページを開く」に向ける。Python から呼んだときだけ効く。
        widget.show = lambda w=widget: self.open_page(w)  # type: ignore[method-assign]
        widget.raise_ = self.raise_  # type: ignore[method-assign]
        widget.activateWindow = self.activateWindow  # type: ignore[method-assign]
        widget.close = lambda: True  # type: ignore[method-assign]
        widget.hide = lambda: None  # type: ignore[method-assign]

    def open_page(self, widget: QWidget) -> None:
        entry = self._pages.get(id(widget))
        if entry is not None:
            index, row, label = entry
            self.stack.setCurrentIndex(index)
            self.title.setText(label)
            if self.nav.currentRow() != row:
                self.nav.setCurrentRow(row)
        if self.isMinimized():
            self.showNormal()
        else:
            QWidget.show(self)
        self.raise_()
        self.activateWindow()

    def _on_nav_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        index = current.data(_PAGE_ROLE)
        if index is None:
            return
        self.stack.setCurrentIndex(int(index))
        self.title.setText(current.text())

    def page_titles(self) -> list[str]:
        return [label for _i, _r, label in sorted(self._pages.values())]
