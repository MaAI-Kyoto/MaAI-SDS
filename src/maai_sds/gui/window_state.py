"""Qt の窓の位置・大きさを、閉じるときに覚えて、次に開いたときに元に戻す。

GUI からローカル / Docker の対話サーバーを切り替えると、窓は一度閉じて開き直す。そのたびに窓が既定の位置へ
戻らないよう、ここで覚えておく。開いていた画面（「設定・テスト」やコンソール）を開き直すことはしない
（起動のたびに勝手に開くのは邪魔なので）。開いたときに、前と同じ位置・大きさで出るだけ。

保存先は Qt の利用者ごとの設定（QSettings。Windows ではレジストリの HKCU\\Software\\MaAI-SDS\\controller）。
controller_config.json には書かない（窓の位置は PC ごとのものなので、設定ファイルを共有・配布しても混ざらない）。
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QWidget

ORGANIZATION = "MaAI-SDS"
APPLICATION = "controller"

# (保存の名前, controller の属性名)。"" は controller 自身（主窓）
WINDOWS: tuple[tuple[str, str], ...] = (
    ("main", ""),
    ("settings_hub", "settings_hub_window"),
    ("console", "console_window"),
    ("maai_plot", "maai_plot_window"),
    ("nod_motion_chart", "nod_motion_chart_window"),
)


def _settings() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


def _windows(controller: Any) -> list[tuple[str, QWidget]]:
    out = []
    for key, attr in WINDOWS:
        widget = controller if not attr else getattr(controller, attr, None)
        if isinstance(widget, QWidget):
            out.append((key, widget))
    return out


def save(controller: Any, settings_factory: Callable[[], QSettings] = _settings) -> None:
    """閉じる直前に呼ぶ（子の窓を閉じる前）。"""
    s = settings_factory()
    for key, widget in _windows(controller):
        s.setValue(f"{key}/geometry", widget.saveGeometry())
    s.sync()


def restore_main(controller: Any, settings_factory: Callable[[], QSettings] = _settings) -> None:
    """主窓を表示する前に呼ぶ（先に位置を戻しておくと、開いた直後に窓が跳ばない）。"""
    geometry = settings_factory().value("main/geometry")
    if isinstance(geometry, QByteArray) and not geometry.isEmpty():
        controller.restoreGeometry(geometry)


def restore_children(controller: Any, settings_factory: Callable[[], QSettings] = _settings) -> None:
    """主窓を表示した後に呼ぶ。子の窓の位置・大きさだけを戻す（開きはしない）。"""
    s = settings_factory()
    for key, widget in _windows(controller):
        if key == "main":
            continue
        geometry = s.value(f"{key}/geometry")
        if isinstance(geometry, QByteArray) and not geometry.isEmpty():
            widget.restoreGeometry(geometry)
