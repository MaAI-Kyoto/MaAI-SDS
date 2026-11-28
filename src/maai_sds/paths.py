"""リポジトリ内のパス解決を 1 箇所に集める（Phase 2 の前提条件）。

## なぜ必要か

これまで 10 モジュール・28 箇所が `Path(__file__).resolve().parent` を
「リポジトリルート」として使っていた。`.py` が全部ルート直下に並んでいる今は
たまたま正しいが、`src/maai_sds/` へパッケージ化すると**全部ずれる**。
特に `asr/service` の相対モデルパス解決基準がずれると ASR が起動しなくなる。

## 2 つのアンカーを区別する

| アンカー | 意味 | 例 |
|---|---|---|
| `REPO_ROOT` | データ・設定・モデル・ログの置き場 | `controller_config.json`, `maai_assets/`, `logs/`, `scripts/`, `.venv_nemotron/`, `sherpa-onnx-*/` |
| `PACKAGE_DIR` | `.py` が並んでいる場所 | `asr/worker.py` などワーカースクリプト、`nemotron_asr_server.py` |

**今は両者が同じディレクトリ**だが、パッケージ化すると別になる。
この区別が無いと「設定は repo ルート／ワーカーはパッケージ内」を書き分けられない。

`REPO_ROOT` は `pyproject.toml` を目印に上へ辿って決める（パッケージ化しても同じ場所を指す）。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# `.py` が並んでいるディレクトリ（パッケージ化するとパッケージ内を指す）
PACKAGE_DIR: Path = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    """`pyproject.toml` を目印にリポジトリルートを探す。

    見つからなければ `PACKAGE_DIR` を返す（従来と同じ挙動）。
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


REPO_ROOT: Path = _find_repo_root(PACKAGE_DIR)


def repo_path(*parts: str) -> Path:
    """リポジトリルート基準のパス（データ・モデル・ログなど）。"""
    return REPO_ROOT.joinpath(*parts)


def package_path(*parts: str) -> Path:
    """`.py` が並ぶディレクトリ基準のパス（ワーカースクリプトなど）。"""
    return PACKAGE_DIR.joinpath(*parts)


def worker_script(*parts: str) -> Path:
    """ワーカースクリプトの絶対パス（`worker_script("asr", "worker.py")` 等）。"""
    return package_path(*parts)


def package_module_name(path: Path | str) -> str:
    """`.py` の絶対パスから `-m` に渡すモジュール名を作る。

    `PACKAGE_DIR/asr/worker.py` -> `maai_sds.asr.worker`

    パス指定で起動するとそのファイルが `__main__` になりパッケージ相対 import が
    使えないため、ワーカーは常にモジュール指定で起動する。
    `PACKAGE_DIR` の外を指している場合はファイル名だけで組む（従来の挙動）。
    """
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(PACKAGE_DIR)
    except ValueError:
        return f"maai_sds.{resolved.stem}"
    return ".".join(("maai_sds", *rel.parts[:-1], rel.stem))


def config_path() -> Path:
    """`controller_config.json`（アプリ設定）。"""
    return repo_path("controller_config.json")


def ensure_config_file(path: Path | None = None) -> bool:
    """controller_config.json が無ければ controller_config.example.json を写す。写したら True。

    clone した直後には設定が無く（Git 管理外）、全項目がコードの既定値になる。コードの既定値は
    例の値（動作を確かめて決めた値）と食い違うものがあり、たとえば Mimi の ONNX をローカルの
    ファイルから読もうとして MaAI が起動直後に落ちた。最初の起動で例から始める。
    """
    target = path or config_path()
    example = repo_path("controller_config.example.json")
    if target.exists() or not example.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(example.read_bytes())
    return True


def tts_template_path() -> Path:
    """`tts_template.json`（TEMPLATE TTS の既定テンプレート）。"""
    return repo_path("tts_template.json")


def assets_root() -> Path:
    """`maai_assets/`（MaAI の同梱モデル置き場）。"""
    return repo_path("maai_assets")


def logs_dir() -> Path:
    """`logs/`（ワーカーのログ出力先）。"""
    return repo_path("logs")


def scripts_dir() -> Path:
    """`scripts/`（PowerShell 補助スクリプト）。"""
    return repo_path("scripts")


def running_in_docker() -> bool:
    """コンテナ内かどうか（`/.dockerenv`）。"""
    return Path("/.dockerenv").is_file()


def data_root() -> Path:
    """モデル・ベンダー資産の置き場。

    Docker ではイメージ内の `/app` に巨大モデルをコピーしない。ホストのリポジトリを
    `MAAI_SDS_DATA_ROOT`（既定は compose の `/host_repo`）として見せ、ここへ取得する。
    未設定なら `REPO_ROOT` と同じ（ネイティブ実行）。
    """
    env = os.environ.get("MAAI_SDS_DATA_ROOT", "").strip()
    return Path(env) if env else REPO_ROOT


def resolve_under_repo(path_like: str | Path) -> Path:
    """相対パスをリポジトリ基準で絶対化する。絶対パスはそのまま返す。"""
    p = Path(os.path.expandvars(str(path_like)))
    if p.is_absolute():
        return p
    repo = REPO_ROOT / p
    if repo.exists():
        return repo
    root = data_root()
    data = root / p
    if data.exists() or root != REPO_ROOT:
        return data
    return repo


_VENV_PYTHON_WIN = re.compile(
    r"(?:^|[/\\])(\.venv_[^/\\]+)[/\\]Scripts[/\\]python(?:\.exe)?$",
    re.IGNORECASE,
)


def default_venv_python(venv_dir: str) -> str:
    """専用 venv の Python 相対パス（Windows は Scripts、POSIX は bin）。"""
    name = (venv_dir or "").strip().replace("\\", "/").strip("/")
    if os.name == "nt":
        return f"{name}/Scripts/python.exe"
    return f"{name}/bin/python"


def posixify_venv_python(raw: str, venv_dir: str) -> str:
    """設定の Windows 向け Python パスを、POSIX では `bin/python` に読み替える。"""
    default = default_venv_python(venv_dir)
    text = (raw or "").strip()
    if os.name == "nt":
        return text or default
    if not text:
        return default
    norm = text.replace("\\", "/")
    match = _VENV_PYTHON_WIN.search(norm)
    if match:
        return f"{match.group(1)}/bin/python"
    if "Scripts/python" in norm or norm.endswith(".exe"):
        return default
    return norm


def resolve_venv_python(raw: str, venv_dir: str) -> Path:
    """専用 venv の Python 実行ファイルを解決する。

    専用 venv はコンテナ内の `/app/.venv_*`（名前付き volume）に置く。ホストの
    `MAAI_SDS_DATA_ROOT` 側（Windows の Scripts/python.exe）は使わない。
    """
    relative = posixify_venv_python(raw, venv_dir)
    candidate = Path(relative)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.is_absolute():
        relative = posixify_venv_python(relative, venv_dir)
        candidate = Path(relative)
    default = Path(default_venv_python(venv_dir))
    if not candidate.is_absolute():
        resolved = REPO_ROOT / candidate
        if resolved.exists():
            return resolved
        fallback = REPO_ROOT / default
        if fallback.exists():
            return fallback
        return fallback
    return candidate


_AUDIOCPP_WIN = "audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe"
_AUDIOCPP_LINUX = "audio.cpp/build/linux-cuda-release/bin/audiocpp_server"
# macOS（Apple Silicon）は公式の Metal 版（audio-<版>-bin-macos-arm64-metal.tar.gz）を setup_assets.sh がここに置く。
# フォルダー名の metal で audiocpp_client.backend_for_server が backend を metal にする。
_AUDIOCPP_MACOS = "audio.cpp/build/macos-metal-release/bin/audiocpp_server"


def default_audiocpp_server() -> str:
    """存在する audio.cpp サーバーバイナリの相対パス。"""
    if sys.platform == "darwin":
        return _AUDIOCPP_MACOS
    linux = resolve_under_repo(_AUDIOCPP_LINUX)
    if linux.is_file():
        return _AUDIOCPP_LINUX
    return _AUDIOCPP_WIN


def resolve_audiocpp_server(raw: str) -> Path:
    """設定の EXE パスを解決し、無ければこの OS の置き場（Linux・macOS）を探す。

    設定の既定値は Windows の .exe なので、Linux・macOS ではたいてい置き場のほうが使われる。
    macOS では Linux・Windows のバイナリは動かないので候補にしない（見つからないときのエラーにも macOS の場所が出る）。
    """
    text = (raw or "").strip()
    candidates: list[str] = []
    if text:
        candidates.append(text.replace("\\", "/"))
    if sys.platform == "darwin":
        candidates.append(_AUDIOCPP_MACOS)
    else:
        candidates.append(_AUDIOCPP_LINUX)
        candidates.append(_AUDIOCPP_WIN)
    seen: set[str] = set()
    last = resolve_under_repo(candidates[0])
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        path = Path(item)
        resolved = path if path.is_absolute() else resolve_under_repo(path)
        last = resolved
        if resolved.is_file():
            return resolved
    return last


# macOS（Apple Silicon）の VOICEVOX ENGINE・AivisSpeech Engine。setup_assets.sh の VoicevoxApp・AivisSpeech が置く。
_MACOS_TTS_ENGINE_RUN = {
    "voicevox": "voicevox_engine/run",
    "aivisspeech": "aivisspeech_engine/run",
}


def macos_tts_engine_run(kind: str) -> str | None:
    """macOS で VOICEVOX（kind="voicevox"）・AivisSpeech（"aivisspeech"）のエンジンを置く場所（相対パス）。ほかの OS は None。"""
    if sys.platform != "darwin":
        return None
    return _MACOS_TTS_ENGINE_RUN[kind]


def resolve_tts_engine_exe(configured: Path | str | None, kind: str) -> Path | None:
    """VOICEVOX・AivisSpeech の起動に使う実行ファイル。

    設定のパスがあればそれ。無い・空なら、macOS では setup_assets.sh が置いた run を使う
    （見本の設定は空か Windows の %LOCALAPPDATA%\\...\\run.exe で、macOS では見つからない）。
    どれも無ければ設定の値（空なら None）を返し、呼ぶ側が「見つからない」を出す。
    """
    text = str(configured or "").strip()
    if text and Path(text).exists():
        return Path(text)
    fallback = macos_tts_engine_run(kind)
    if fallback is not None:
        candidate = resolve_under_repo(fallback)
        if candidate.is_file():
            return candidate
    return Path(text) if text else None


_WIN_ABS = re.compile(r"^[A-Za-z]:[/\\]")


def host_uploads_dir() -> Path:
    """ブラウザから上げたホストファイルの置き場。Docker では `/host_repo/host_uploads`。"""
    if running_in_docker():
        return data_root() / "host_uploads"
    return REPO_ROOT / "host_uploads"


def resolve_docker_local_file(raw: str) -> str:
    """コンテナから読めない Windows 絶対パスを空にする。同名が host_uploads にあればそれを使う。"""
    text = str(raw or "").strip()
    if not text:
        return ""
    expanded = Path(os.path.expandvars(text))
    try:
        if expanded.is_file():
            return str(expanded)
    except OSError:
        pass
    name = Path(text.replace("\\", "/")).name
    if name:
        uploaded = host_uploads_dir() / name
        try:
            if uploaded.is_file():
                return str(uploaded.resolve())
        except OSError:
            pass
    if running_in_docker() and (_WIN_ABS.match(text) or text.startswith("\\\\")):
        return ""
    return text


def input_wav_dir() -> Path:
    """WAV 入力の置き場。Docker では `/app/input_wav`（ホストの INPUT_WAV_HOST）。"""
    for candidate in (
        Path("/app/input_wav"),
        data_root() / "input_wav",
        REPO_ROOT / "input_wav",
    ):
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    if running_in_docker():
        return Path("/app/input_wav")
    return REPO_ROOT / "input_wav"


def resolve_runtime_wav_path(raw: str) -> str:
    """設定の WAV パスをそのまま返す。"""
    return str(raw or "").strip()
