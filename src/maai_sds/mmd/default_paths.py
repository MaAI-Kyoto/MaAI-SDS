"""MMDAgent-EX の既定の置き場所から EXE / MDF を探す（Qt 版と Web 版で共通）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 置き場所は README の手順のもの。Windows は C:\MMDAgent-EX（公式の zip を展開し、サンプル集を example に clone）、
# macOS はホームの ~/MMDAgent-EX（本体は無く、サンプル集を example に clone してブラウザ版で表示する）。
WINDOWS_BASE_DIR = Path(r"C:\MMDAgent-EX")


def default_base_dir(platform: str | None = None) -> Path:
    """MMDAgent-EX を探す既定の場所（OS ごと）。"""
    if (platform or sys.platform) == "win32":
        return WINDOWS_BASE_DIR
    return Path.home() / "MMDAgent-EX"


# 公式の配布物（GitHub Releases の MMDAgent-EX-x64-*.zip）を C:\MMDAgent-EX に展開すると直下に EXE が来る。
# ソースからビルドすると Release\ の下に来る。
_EXE_CANDIDATES = (
    ("MMDAgent-EX.exe",),
    ("Release", "MMDAgent-EX.exe"),
    ("MMDAgent.exe",),
    ("bin", "MMDAgent.exe"),
)
# 先に見つかったものを使う。最初が公式のサンプル集（C:\MMDAgent-EX\example に git clone --recursive）に入れた
# MaAI-SDS 用の MDF（assets/mmdagent/maai_sds をコピーしたもの。README の手順はこれ）、次がサンプル集の WebSocket の例。
# 配布物の MMDAgent-EX.mdf は Plugin_Remote の設定が無く対話サーバーにつながらないので、最後にする。
_MDF_CANDIDATES = (
    ("example", "maai_sds", "maai_sds.mdf"),
    ("example", "example_websocket", "example_websocket.mdf"),
    ("MMDAgent-EX.mdf",),
    ("MMDAgent.mdf",),
    ("Release", "MMDAgent.mdf"),
)


def _first_existing(base_dir: Path, candidates: tuple[tuple[str, ...], ...]) -> str:
    for parts in candidates:
        path = base_dir.joinpath(*parts)
        if path.exists():
            return str(path)
    return ""


def detect_default_exec_path(base_dir: Path | None = None) -> str:
    return _first_existing(base_dir or default_base_dir(), _EXE_CANDIDATES)


def detect_default_mdf_path(base_dir: Path | None = None) -> str:
    return _first_existing(base_dir or default_base_dir(), _MDF_CANDIDATES)


def fill_default_mmdagent_paths(cfg: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    """exec_path / mdf_path が空か存在しないとき、既定の場所で見つけたものに置き換える。

    Qt 版は起動時にこの場所を探して欄を埋め、設定ファイルのパスが存在するときだけ
    それで上書きする。Web 版も同じ結果になるよう、読み込み時にここを通す。
    見つからなければ元の値のまま（Docker ではコンテナに既定の場所が無いので何もしない）。
    """
    out = dict(cfg)
    for key, detect in (("exec_path", detect_default_exec_path), ("mdf_path", detect_default_mdf_path)):
        current = str(out.get(key) or "").strip()
        if current and Path(current).exists():
            continue
        found = detect(base_dir)
        if found:
            out[key] = found
    return out
