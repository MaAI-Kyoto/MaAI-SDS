"""PySide6 GUI の暗い配色（Web UI の web/app/app.css と同じ色の決まり）。

色・余白・角の丸みは Web の `:root` の変数と同じ値にそろえる。どちらかを変えるときは
両方を直すこと（docs/ui-guidelines.md）。
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# web/app/app.css の :root と同じ値。
TOKENS: dict[str, str] = {
    "bg": "#14171d",
    "panel": "#1b212b",
    "panel_2": "#232b38",
    "input": "#11161e",
    "line": "#333d4d",
    "line_strong": "#46536a",
    "text": "#e6e9ef",
    "muted": "#97a3b6",
    "accent": "#4f9cf9",
    "accent_hover": "#6aaefb",
    "accent_text": "#0b1220",
    "ok": "#55c98a",
    "warn": "#e6b455",
    "err": "#e2685f",
    "idle": "#5b6577",
}

# モジュール状態のランプ・文字色（停止・起動中・稼働・異常）。
STATE_COLORS: dict[str, str] = {
    "stopped": TOKENS["idle"],
    "starting": TOKENS["warn"],
    "running": TOKENS["ok"],
    "error": TOKENS["err"],
}

# 先頭のフォントが無いと、Qt は代わりを探すために全フォントの別名を作り直し、起動のたびに
# 「Populating font family aliases took ... ms」の警告を出す。macOS には Yu Gothic UI が無いので
# ヒラギノを先頭にする。
FONT_FAMILIES = (
    ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "sans-serif"]
    if sys.platform == "darwin"
    else ["Yu Gothic UI", "Segoe UI", "Meiryo UI", "sans-serif"]
)
FONT_POINT_SIZE = 10  # Web の 13px 相当（96dpi）


def lamp_style(color: str) -> str:
    """モジュール状態ランプ（12px の丸）。"""
    return f"background-color: {color}; border-radius: 6px; border: 1px solid {TOKENS['line_strong']};"


_CHECK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14">'
    '<path d="M3 7.2 5.8 10 11 4.4" fill="none" stroke="{color}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


_ARROW_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12">'
    '<path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="{color}" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _write_icon(name: str, svg: str) -> str:
    """アイコンの SVG を書き出してパスを返す（QSS の url() はファイルしか読めない）。"""
    import tempfile
    from pathlib import Path

    path = Path(tempfile.gettempdir()) / f"maai_sds_theme_{name}.svg"
    try:
        path.write_text(svg, encoding="utf-8")
    except OSError:
        return ""
    return path.as_posix()


def _check_icon_path(color: str) -> str:
    return _write_icon("check", _CHECK_SVG.format(color=color))


def _stylesheet(t: dict[str, str]) -> str:
    check_icon = _check_icon_path(t["accent_text"])
    arrow_icon = _write_icon("arrow", _ARROW_SVG.format(color=t["muted"]))
    return f"""
QWidget {{
    color: {t['text']};
    background-color: {t['bg']};
    selection-background-color: {t['accent']};
    selection-color: {t['accent_text']};
}}
QToolTip {{
    color: {t['text']};
    background-color: {t['panel_2']};
    border: 1px solid {t['line']};
    padding: 4px 6px;
}}
QGroupBox {{
    background-color: {t['panel']};
    border: 1px solid {t['line']};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {t['text']};
    background-color: transparent;
}}
QGroupBox QWidget {{
    background-color: transparent;
    font-weight: normal;
}}
QCheckBox::indicator, QGroupBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {t['line_strong']};
    border-radius: 3px;
    background-color: {t['input']};
}}
QCheckBox::indicator:hover, QGroupBox::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
    background-color: {t['accent']};
    border-color: {t['accent']};
    image: url({check_icon});
}}
QCheckBox::indicator:disabled {{ background-color: #161b23; border-color: #2a323f; }}
QCheckBox::indicator:checked:disabled {{ background-color: #2f4a6e; border-color: #2f4a6e; }}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {t['line_strong']};
    border-radius: 8px;
    background-color: {t['input']};
}}
QRadioButton::indicator:checked {{
    background-color: {t['accent']};
    border: 4px solid {t['input']};
    outline: 1px solid {t['accent']};
}}
QLabel[role="caption"] {{ color: {t['muted']}; font-size: 9pt; }}
QLabel[role="status"] {{ color: {t['muted']}; }}
QPushButton {{
    background-color: {t['panel_2']};
    border: 1px solid {t['line']};
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 18px;
}}
QPushButton:hover {{ border-color: {t['line_strong']}; background-color: #2a3444; }}
QPushButton:pressed {{ background-color: #1d2430; }}
QPushButton:disabled {{ color: #5f6b7d; background-color: #1c222c; border-color: #2a323f; }}
QPushButton[role="primary"] {{
    background-color: {t['accent']};
    border-color: {t['accent']};
    color: {t['accent_text']};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background-color: {t['accent_hover']}; }}
QPushButton[role="primary"]:disabled {{ color: #5f6b7d; background-color: #1c222c; border-color: #2a323f; }}
QPushButton[role="icon"] {{ padding: 0; min-height: 0; }}
QPushButton[size="large"] {{ padding: 9px 18px; font-size: 11pt; }}
QLabel[role="app-title"] {{ font-size: 13pt; font-weight: 700; padding-right: 8px; }}
QToolButton[role="section-toggle"] {{
    color: {t['accent']};
    background: transparent;
    border: none;
    padding: 2px 2px;
    font-weight: 600;
}}
QToolButton[role="section-toggle"]:hover {{ color: {t['accent_hover']}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {t['input']};
    border: 1px solid {t['line']};
    border-radius: 4px;
    padding: 4px 6px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{ border-color: {t['accent']}; }}
QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{ color: #5f6b7d; background-color: #161b23; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({arrow_icon}); width: 12px; height: 12px; }}
QComboBox::down-arrow:disabled {{ image: none; }}
QComboBox QAbstractItemView {{
    background-color: {t['panel_2']};
    border: 1px solid {t['line']};
    selection-background-color: {t['accent']};
    selection-color: {t['accent_text']};
}}
QCheckBox, QRadioButton {{ spacing: 6px; background-color: transparent; }}
QTabWidget::pane {{ border: 1px solid {t['line']}; border-radius: 6px; background-color: {t['panel']}; top: -1px; }}
QTabBar::tab {{
    background-color: {t['bg']};
    border: 1px solid {t['line']};
    border-bottom: none;
    padding: 6px 12px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: {t['muted']};
}}
QTabBar::tab:selected {{ background-color: {t['panel']}; color: {t['text']}; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['line']}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {t['line_strong']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {t['line']}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background-color: {t['line']}; }}
QListWidget[role="hub-nav"] {{
    background-color: {t['panel']};
    border: 1px solid {t['line']};
    border-radius: 8px;
    padding: 6px;
    outline: none;
}}
QListWidget[role="hub-nav"]::item {{ padding: 6px 10px; border-radius: 4px; }}
QListWidget[role="hub-nav"]::item:selected {{ background-color: #1d4c86; color: {t['text']}; }}
QListWidget[role="hub-nav"]::item:hover:!selected {{ background-color: #222a36; }}
QLabel[role="hub-title"] {{ font-size: 12pt; font-weight: 700; padding: 2px 2px 0 2px; }}
QScrollArea[role="chat-view"] {{ background-color: {t['input']}; border: 1px solid {t['line']}; border-radius: 8px; }}
QScrollArea[role="chat-view"] > QWidget > QWidget {{ background-color: {t['input']}; }}
QLabel[role="chat-placeholder"] {{ color: #6b778a; }}
QFrame[bubble] {{ border-radius: 12px; }}
QFrame[bubble="user"] {{ background-color: #1d4c86; border: 1px solid #2a6ab8; }}
QFrame[bubble="user_partial"] {{ background-color: #17304f; border: 1px dashed #2a6ab8; }}
QFrame[bubble="assistant"] {{ background-color: {t['panel_2']}; border: 1px solid {t['line']}; }}
QFrame[bubble="assistant_streaming"] {{ background-color: {t['panel']}; border: 1px dashed {t['line_strong']}; }}
QFrame[bubble="system"] {{ background-color: transparent; border: none; }}
QFrame[bubble] QLabel {{ background-color: transparent; }}
QLabel[role="bubble-name"] {{ color: {t['muted']}; font-size: 8.5pt; }}
QFrame[role="live-panel"] {{ background-color: {t['panel']}; border: 1px dashed {t['line_strong']}; border-radius: 8px; }}
QFrame[role="live-panel"] QLabel {{ background-color: transparent; }}
QLabel[role="live-text"] {{ color: #b9c4d4; }}
QLabel[role="live-text"][empty="true"] {{ color: #5f6b7d; }}
QLabel[role="bubble-text"] {{ font-size: 10.5pt; }}
QFrame[bubble="system"] QLabel[role="bubble-text"] {{ color: {t['muted']}; font-size: 9pt; }}
QFrame[bubble="user_partial"] QLabel[role="bubble-text"],
QFrame[bubble="assistant_streaming"] QLabel[role="bubble-text"] {{ color: #b9c4d4; }}
QMenu {{ background-color: {t['panel_2']}; border: 1px solid {t['line']}; }}
QMenu::item:selected {{ background-color: {t['accent']}; color: {t['accent_text']}; }}
"""


def _palette(t: dict[str, str]) -> QPalette:
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: t["bg"],
        QPalette.ColorRole.WindowText: t["text"],
        QPalette.ColorRole.Base: t["input"],
        QPalette.ColorRole.AlternateBase: t["panel"],
        QPalette.ColorRole.Text: t["text"],
        QPalette.ColorRole.Button: t["panel_2"],
        QPalette.ColorRole.ButtonText: t["text"],
        QPalette.ColorRole.ToolTipBase: t["panel_2"],
        QPalette.ColorRole.ToolTipText: t["text"],
        QPalette.ColorRole.PlaceholderText: "#6b778a",
        QPalette.ColorRole.Highlight: t["accent"],
        QPalette.ColorRole.HighlightedText: t["accent_text"],
        QPalette.ColorRole.Link: t["accent"],
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor("#5f6b7d"))
    return pal


def apply_dark_theme(app: QApplication) -> None:
    """アプリ全体に暗い配色を当てる（Fusion スタイル + パレット + スタイルシート）。"""
    app.setStyle("Fusion")
    app.setPalette(_palette(TOKENS))
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSize(FONT_POINT_SIZE)
    app.setFont(font)
    app.setStyleSheet(_stylesheet(TOKENS))
