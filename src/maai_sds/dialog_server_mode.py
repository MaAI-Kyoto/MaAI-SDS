"""どちらの対話サーバー（ローカル / Docker）を使うかを選んで、Qt / Web サーバーを開き直す。

設定ファイル・パスの変換・選べるエンジンが起動時のモードで決まるので、切り替えは
「選んだモードで起動し直す」で行う。選んだモードは保存し、次に引数なしで
起動したときも使う（`--remote-orchestrator` や `--dialog-server` を付けたときはそちらが優先）。
Qt と Web で同じ選択を共有する。

保存先はユーザーの設定フォルダーの launch.json（Windows は %APPDATA%\\MaAI-SDS）。
リポジトリの中には書かない。MAAI_SDS_LAUNCH_PREFS で別のファイルにできる。

モードごとの設定ファイルは既定で controller_config.json / controller_config.docker.json。
MAAI_SDS_QT_LOCAL_CONFIG / MAAI_SDS_QT_DOCKER_CONFIG で別のファイルにできる（開き直しても引き継ぐ）。

Docker で開くときは `scripts/start_qt_docker.ps1` と同じ値を環境変数で用意する:
compose ファイル（docker-compose.yml + GPU/CPU 用）、ポート、controller_config.docker.json。
コンテナの起動（docker compose up -d orchestrator）は GUI が開いてから
対話サーバーの ▶ と同じ処理で行う（MAAI_SDS_QT_COMPOSE_UP_ON_START）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODES = ("local", "docker")
MODE_LABELS = {"local": "ローカル", "docker": "Docker"}
DEFAULT_PORT = 8765

_SETTINGS_KEY = "dialog_server"

# 起動したモードに結びつく環境変数。開き直すときはいったん消し、新しい GUI が決め直す。
MODE_ENV_KEYS = (
    "MAAI_SDS_REMOTE_ORCHESTRATOR",
    "MAAI_SDS_CONTROLLER_CONFIG",
    "MAAI_SDS_QT_COMPOSE_UP_ON_START",
)
# Docker の compose の場所。ローカルで開いても消さない（ローカルでは使わないので害はなく、
# 消すと次に Docker へ切り替えたとき、指定していた compose ではなく既定の compose を使ってしまう）。
COMPOSE_ENV_KEYS = (
    "MAAI_SDS_QT_ORCHESTRATOR_PORT",
    "MAAI_SDS_QT_COMPOSE_FILES",
    "MAAI_SDS_QT_COMPOSE_DIR",
    "MAAI_SDS_QT_COMPOSE_PROJECT",
)


def parse_endpoint(value: str) -> tuple[str, int]:
    """``HOST:PORT`` を解釈する（Qt の --remote-orchestrator と同じ規則）。"""
    text = str(value or "").strip()
    host, sep, port_text = text.rpartition(":")
    if not sep or not host.strip() or not port_text.strip():
        raise ValueError(f"remote orchestrator must be HOST:PORT, got {value!r}")
    try:
        port = int(port_text)
    except ValueError as ex:
        raise ValueError(f"remote orchestrator port must be an integer: {port_text!r}") from ex
    if not 1 <= port <= 65535:
        raise ValueError(f"remote orchestrator port is outside 1..65535: {port}")
    return host.strip(), port


def prefs_path(env: dict[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    override = str(source.get("MAAI_SDS_LAUNCH_PREFS", "") or "").strip()
    if override:
        return Path(override)
    if sys.platform == "win32" and source.get("APPDATA"):
        return Path(source["APPDATA"]) / "MaAI-SDS" / "launch.json"
    base = source.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "maai-sds" / "launch.json"


def _read_prefs(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_saved_mode(path: Path | None = None) -> str:
    value = str(_read_prefs(path or prefs_path()).get(_SETTINGS_KEY, "local") or "local").strip().lower()
    return value if value in MODES else "local"


def save_mode(mode: str, path: Path | None = None) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown dialog server mode: {mode}")
    target = path or prefs_path()
    data = _read_prefs(target)
    data[_SETTINGS_KEY] = mode
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def docker_config_path(repo_root: Path, env: dict[str, str] | None = None) -> Path:
    override = str((os.environ if env is None else env).get("MAAI_SDS_QT_DOCKER_CONFIG", "") or "").strip()
    return Path(override) if override else Path(repo_root) / "controller_config.docker.json"


def docker_unavailable_reason(repo_root: Path, *, env: dict[str, str] | None = None, which=shutil.which) -> str:
    """Docker で開けないときの理由（開けるなら空文字）。"""
    root = Path(repo_root)
    source = os.environ if env is None else env
    if which("docker") is None:
        return "docker コマンドが見つかりません（Docker Desktop を入れてください）。"
    if not str(source.get("MAAI_SDS_QT_COMPOSE_FILES", "") or "").strip() and not (root / "docker-compose.yml").is_file():
        return f"{root / 'docker-compose.yml'} がありません。"
    if not docker_config_path(root, source).is_file():
        return (
            "controller_config.docker.json がありません。"
            "controller_config.docker.example.json をコピーして編集してください。"
        )
    return ""


def _configured_port(repo_root: Path, env: dict[str, str]) -> int:
    raw = str(env.get("MAAI_SDS_QT_ORCHESTRATOR_PORT", "") or "").strip()
    if not raw:
        env_file = Path(repo_root) / ".env"
        if env_file.is_file():
            pattern = re.compile(r"""^\s*MAAI_SDS_QT_ORCHESTRATOR_PORT\s*=\s*["']?(\d+)["']?\s*(?:#.*)?$""")
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                match = pattern.match(line)
                if match:
                    raw = match.group(1)
                    break
    try:
        port = int(raw) if raw else DEFAULT_PORT
    except ValueError:
        port = DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


def _device(repo_root: Path) -> str:
    marker = Path(repo_root) / ".device_extra"
    try:
        value = marker.read_text(encoding="utf-8").strip().lower()
    except OSError:
        value = ""
    return "cpu" if value == "cpu" else "gpu"


def docker_launch_env(repo_root: Path, env: dict[str, str]) -> dict[str, str]:
    """Docker で開くための環境変数（start_qt_docker.ps1 と同じ値）。env は書き換えない。"""
    root = Path(repo_root)
    out = dict(env)
    port = _configured_port(root, env)
    if not str(env.get("MAAI_SDS_QT_COMPOSE_FILES", "") or "").strip():
        files = [root / "docker-compose.yml", root / f"docker-compose.{_device(root)}.yml"]
        out["MAAI_SDS_QT_COMPOSE_FILES"] = os.pathsep.join(str(p) for p in files)
        out["MAAI_SDS_QT_COMPOSE_DIR"] = str(root)
    out["MAAI_SDS_QT_ORCHESTRATOR_PORT"] = str(port)
    out["MAAI_SDS_REMOTE_ORCHESTRATOR"] = f"127.0.0.1:{port}"
    out["MAAI_SDS_CONTROLLER_CONFIG"] = str(docker_config_path(root, env))
    out["MAAI_SDS_QT_COMPOSE_UP_ON_START"] = "1"
    return out


def local_launch_env(env: dict[str, str]) -> dict[str, str]:
    """ローカルで開くための環境変数（Docker 用のものを消す）。"""
    drop = ("MAAI_SDS_REMOTE_ORCHESTRATOR", "MAAI_SDS_QT_COMPOSE_UP_ON_START")
    out = {key: value for key, value in env.items() if key not in drop}
    local_config = str(env.get("MAAI_SDS_QT_LOCAL_CONFIG", "") or "").strip()
    if local_config:
        out["MAAI_SDS_CONTROLLER_CONFIG"] = local_config
    return out


def resolve_launch(
    dialog_server_arg: str | None,
    remote_arg: str | None,
    env: dict[str, str],
    repo_root: Path,
    saved_mode: str,
    *,
    which=shutil.which,
) -> tuple[str, dict[str, str], str]:
    """起動するモードを決める。(モード, 使う環境変数, 画面に出すお知らせ) を返す。

    優先順: --remote-orchestrator → --dialog-server → 環境変数 MAAI_SDS_REMOTE_ORCHESTRATOR
    （start_qt_docker.ps1）→ 保存した選択。
    """
    if remote_arg:
        return "docker", dict(env), ""
    if not dialog_server_arg and str(env.get("MAAI_SDS_REMOTE_ORCHESTRATOR", "") or "").strip():
        return "docker", dict(env), ""
    mode = dialog_server_arg if dialog_server_arg in MODES else saved_mode
    notice = ""
    if mode == "docker":
        reason = docker_unavailable_reason(repo_root, env=env, which=which)
        if reason:
            mode = "local"
            notice = f"Docker の対話サーバーを使えないため、ローカルで開きました。\n{reason}"
    launch_env = docker_launch_env(repo_root, env) if mode == "docker" else local_launch_env(env)
    return mode, launch_env, notice


def apply_env(env: dict[str, str]) -> None:
    """os.environ を env と同じにする（消えたキーは消す）。"""
    for key in list(os.environ):
        if key not in env:
            del os.environ[key]
    for key, value in env.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def relaunch_command(mode: str, wait_pid: int, module: str = "maai_sds.main") -> list[str]:
    """module: Qt は maai_sds.main、Web サーバーは maai_sds.web.server。"""
    return [sys.executable, "-m", module, "--dialog-server", mode, "--wait-pid", str(int(wait_pid))]


def relaunch_env(mode: str, env: dict[str, str]) -> dict[str, str]:
    """開き直す GUI の環境変数。モードに結びつく値は新しい GUI が起動時に決め直す。"""
    out = {key: value for key, value in env.items() if key not in MODE_ENV_KEYS}
    # -m maai_sds.main で起動するので、このパッケージの場所を通しておく（editable install でなくても動く）。
    src = str(Path(__file__).resolve().parents[1])
    parts = [p for p in str(out.get("PYTHONPATH", "") or "").split(os.pathsep) if p]
    if src not in parts:
        parts.insert(0, src)
    out["PYTHONPATH"] = os.pathsep.join(parts)
    return out


def spawn_relaunch(mode: str, module: str = "maai_sds.main") -> None:
    """選んだモードで新しく起動する（このプロセスが終わるのを待ってから進む）。"""
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        relaunch_command(mode, os.getpid(), module),
        env=relaunch_env(mode, dict(os.environ)),
        cwd=str(Path.cwd()),
        creationflags=creationflags,
        close_fds=True,
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            return False
        try:
            # WAIT_TIMEOUT (0x102) ならまだ生きている
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid_exit(pid: int, timeout_sec: float = 60.0) -> bool:
    """前の GUI（とその子プロセスのポート）が片付くのを待つ。終わったら True。"""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return False
