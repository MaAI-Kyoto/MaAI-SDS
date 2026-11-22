"""メイン画面の会話の欄（吹き出し）。Web UI の会話の欄と同じ見え方。

- あなた（user）は右、アシスタントは左、システムは中央に小さく。
- 認識中の発話と生成中の応答は、点線の吹き出しで一番下に出す。
- 認識の途中結果は頻繁に届くので、変わった吹き出しだけを書き換える（全部作り直さない）。
- 一番下を見ているあいだは、吹き出しが増えても一番下に付いていく。上へスクロールして読んでいる
  ときは動かさず、一番下に戻すとまた付いていく。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

ROLE_NAMES = {
    "user": "あなた",
    "user_partial": "あなた（認識中）",
    "assistant": "アシスタント",
    "assistant_streaming": "アシスタント（生成中）",
    "system": "システム",
}


@dataclass
class _Row:
    role: str
    text: str
    widget: QWidget
    bubble: QFrame
    label: QLabel
    name: QLabel


class ChatView(QScrollArea):
    BUBBLE_WIDTH_RATIO = 0.78

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setProperty("role", "chat-view")
        self._inner = QWidget()
        self._inner.setProperty("role", "chat-inner")
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(10)
        self._placeholder = QLabel("音声対話を始めると、ここに会話が表示されます。")
        self._placeholder.setProperty("role", "chat-placeholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._placeholder, 1)
        self._layout.addStretch(1)
        self.setWidget(self._inner)
        self._rows: list[_Row] = []
        # 一番下に付いていくか。吹き出しを足した直後は中の高さがまだ反映されておらず、その時点で
        # 一番下へ送っても届かない（以前は QTimer.singleShot(0) で送っていて途中で止まり、次の更新では
        # 「一番下の近く」でなくなって以後付いていかなくなった）。スクロールできる範囲が変わったときに送る。
        self._follow = True
        bar = self.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range_changed)
        bar.valueChanged.connect(self._on_value_changed)

    # -- 公開 API ----------------------------------------------------------------

    def set_items(self, items: list[tuple[str, str]]) -> None:
        """(role, text) の並びで欄を描き直す。変わらない吹き出しはそのまま残す。"""
        keep = 0
        for keep, ((role, text), row) in enumerate(zip(items, self._rows)):
            if row.role != role:
                break
            if row.text != text:
                row.text = text
                row.label.setText(text)
                self._fit_bubble(row)
        else:
            keep = min(len(items), len(self._rows))
        for row in self._rows[keep:]:
            self._layout.removeWidget(row.widget)
            row.widget.deleteLater()
        self._rows = self._rows[:keep]
        for role, text in items[keep:]:
            row = self._make_row(role, text)
            # 末尾のストレッチの直前に入れる。
            self._layout.insertWidget(self._layout.count() - 1, row.widget)
            self._rows.append(row)
        self._placeholder.setVisible(not self._rows)
        self._apply_bubble_width()
        if self._follow:
            self._scroll_to_bottom()

    def items(self) -> list[tuple[str, str]]:
        return [(row.role, row.text) for row in self._rows]

    # -- 内部 ---------------------------------------------------------------------

    def _make_row(self, role: str, text: str) -> _Row:
        widget = QWidget()
        hbox = QHBoxLayout(widget)
        hbox.setContentsMargins(0, 0, 0, 0)
        bubble = QFrame()
        bubble.setProperty("bubble", role)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        vbox = QVBoxLayout(bubble)
        vbox.setContentsMargins(12, 7, 12, 9)
        vbox.setSpacing(2)
        name = QLabel(ROLE_NAMES.get(role, role))
        name.setProperty("role", "bubble-name")
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setProperty("role", "bubble-text")
        if role != "system":
            vbox.addWidget(name)
        vbox.addWidget(label)
        if role.startswith("user"):
            hbox.addStretch(1)
            hbox.addWidget(bubble)
        elif role == "system":
            hbox.addStretch(1)
            hbox.addWidget(bubble)
            hbox.addStretch(1)
        else:
            hbox.addWidget(bubble)
            hbox.addStretch(1)
        return _Row(role=role, text=text, widget=widget, bubble=bubble, label=label, name=name)

    def _max_bubble_width(self) -> int:
        return max(160, int(self.viewport().width() * self.BUBBLE_WIDTH_RATIO))

    def _fit_bubble(self, row: _Row, max_width: int | None = None) -> None:
        """吹き出しの幅を文の長さに合わせる（最大は欄の 78%）。

        折り返しありの QLabel は小さな幅を希望するので、何もしないと短い文でも細長くなる。
        """
        limit = max_width or self._max_bubble_width()
        # スタイルシートの文字サイズを反映してから測る（反映前だと少し狭く出て、最後の 1 文字が折り返す）。
        row.label.ensurePolished()
        row.name.ensurePolished()
        fm = row.label.fontMetrics()
        text_w = max((fm.horizontalAdvance(line) for line in row.text.splitlines() or [""]), default=0)
        name_w = row.name.fontMetrics().horizontalAdvance(row.name.text()) if row.role != "system" else 0
        margins = row.bubble.layout().contentsMargins()
        width = max(text_w, name_w) + margins.left() + margins.right() + 10
        row.bubble.setFixedWidth(min(width, limit))

    def _apply_bubble_width(self) -> None:
        limit = self._max_bubble_width()
        for row in self._rows:
            self._fit_bubble(row, limit)

    def _near_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 40

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._follow:
            self.verticalScrollBar().setValue(maximum)

    def _on_value_changed(self, _value: int) -> None:
        # 上へスクロールしたら付いていくのをやめ、一番下（の近く）に戻したらまた付いていく。
        self._follow = self._near_bottom()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt の名前
        super().resizeEvent(event)
        self._apply_bubble_width()
