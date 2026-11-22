"""アプリのアイコン（窓とタスクバー）。

絵は Web の favicon と同じもの（web/app/icon-*.png。元は ロゴの E 案の SVG）を使う。
Windows では、python.exe から起動したアプリはタスクバーで Python としてまとめられ、窓のアイコンを
付けても Python のアイコンが出る。AppUserModelID を先に決めておくと、このアプリのアイコンで別に並ぶ。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

from .. import paths as _paths

APP_USER_MODEL_ID = "MaAI-Kyoto.MaAI-SDS"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def icon_dir() -> Path:
    return _paths.REPO_ROOT / "web" / "app"


def app_icon(directory: Path | None = None) -> QIcon:
    """あるサイズの PNG を全部入れた QIcon（どれも無ければ空の QIcon）。"""
    icon = QIcon()
    base = directory or icon_dir()
    for size in ICON_SIZES:
        path = base / f"icon-{size}.png"
        if path.is_file():
            icon.addFile(str(path))
    return icon


def set_windows_app_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Windows のタスクバーで、このアプリを Python と別に並べる。窓を作る前に呼ぶ。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except (AttributeError, OSError):
        return False
