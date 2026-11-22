"""たためる「詳細設定」の枠。細かい調整用の項目を、隠さずに一段奥へ置く。

Web の画面定義（scripts/extract_web_ui_manifest.py）はレイアウトをたどって項目を拾うので、
中の項目は親のグループの行としてそのまま Web にも出る（見出しのボタンは拾われない）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    def __init__(self, title: str = "詳細設定", parent: QWidget | None = None, *, expanded: bool = False) -> None:
        super().__init__(parent)
        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setAutoRaise(True)
        self._toggle.setProperty("role", "section-toggle")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._content = QWidget(self)
        self._content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toggle, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._content)
        self._toggle.toggled.connect(self._apply)
        self._apply(expanded)

    def setContentLayout(self, layout: QLayout) -> None:  # noqa: N802 - Qt と同じ命名
        layout.setContentsMargins(0, 0, 0, 0)
        self._content.setLayout(layout)

    def isExpanded(self) -> bool:  # noqa: N802
        return self._toggle.isChecked()

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802
        self._toggle.setChecked(bool(expanded))

    def _apply(self, expanded: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content.setVisible(bool(expanded))
