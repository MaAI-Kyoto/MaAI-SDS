"""Optional ASR/TTS engine installation management, independent of the GUI."""
from __future__ import annotations
import ast
from dataclasses import dataclass, replace
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Callable
from . import paths as _paths


@dataclass
class EngineInfo:
    id: str
    label: str
    extra: str | None
    probe_module: str | None
    installed: bool = False
    setup_steps: tuple[str, ...] = ()
    # Repository-relative path, so it must include the leading "scripts/" (resolved with
    # paths.resolve_under_repo, same as probe_paths). Keeping one format here is what lets
    # both resolve through that single helper instead of a local special case.
    setup_venv_script: str | None = None
    # For engines with no `extra` (dedicated-venv / external-asset engines: nothing for uv to
    # track), the actual on-disk artifact(s) that prove setup actually ran. All must exist for
    # `installed=True`. Without this, `_is_installed` has no way to distinguish "never set up"
    # from "genuinely external, nothing to check" and used to report every `extra=None` engine
    # as installed unconditionally — wrong for every engine added this session that has a real
    # setup_venv_script/setup_steps but no uv extra (confirmed via a real fresh-clone test).
    probe_paths: tuple[str, ...] = ()
    # 「適用」で取得するモデルなどの大きさ（GB、目安）。Python パッケージは含まない。
    # 値は setup_assets が取る配布物の Content-Length（2026-11-23 確認）。None は取得物が無いか、
    # 初回利用時に取る（Faster-Whisper のモデルなど）もの。
    download_gb: float | None = None
    # 複数のエンジンで共有する実行系（audio.cpp のサーバーなど）。download_gb には含めず、
    # まだ入っていないときだけ 1 回分を足す（SHARED_RUNTIME_GB）。
    shared_runtime: str | None = None


# 共有の実行系の取得量（GB）。audio.cpp は bin と CUDA ランタイムの zip（v0.9.6）。
# macOS は Metal 版の tar.gz だけ（v0.9.6 で 40 MB）。
SHARED_RUNTIME_GB: dict[str, float] = {"audiocpp": 0.04 if sys.platform == "darwin" else 1.1}

_ENGINE_DEFINITIONS = (
    EngineInfo("asr-faster-whisper", "Faster-Whisper", "asr-faster-whisper", "faster_whisper"),
    EngineInfo(
        "asr-sherpa", "ReazonSpeech（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("SherpaModels",),
        download_gb=0.7,
    ),
    EngineInfo(
        "asr-qwen3-onnx", "Qwen3-ASR 0.6B int8（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("Qwen3AsrOnnxModel",),
        probe_paths=("sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx",),
        download_gb=0.9,
    ),
    EngineInfo(
        "asr-qwen3-onnx-fp32", "Qwen3-ASR 0.6B fp32（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("Qwen3AsrOnnxFp32Model",),
        probe_paths=("sherpa-onnx-qwen3-asr-0.6B-fp32/decoder.onnx.data",),
        download_gb=3.6,
    ),
    EngineInfo(
        "asr-qwen3-onnx-1.7b", "Qwen3-ASR 1.7B int8（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("Qwen3AsrOnnx17BModel",),
        probe_paths=("sherpa-onnx-qwen3-asr-1.7B-int8/decoder.int8.onnx",),
        download_gb=2.3,
    ),
    EngineInfo("asr-chirp3", "Chirp 3（Google Cloud）", "asr-chirp3", "google.cloud.speech"),
    EngineInfo("asr-qwen3", "Qwen3-ASR（Docker）", None, None),
    EngineInfo(
        "asr-nemotron", "Nemotron（NeMo専用環境）", None, None,
        setup_venv_script="scripts/setup_nemotron_venv.ps1",
        probe_paths=(".venv_nemotron/Scripts/python.exe",),
        download_gb=8.0,
    ),
    EngineInfo(
        "asr-nemotron-onnx", "Nemotron（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("NemotronOnnxModels",),
        probe_paths=(
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/encoder.int8.onnx",
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11/encoder.int8.onnx",
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/encoder.int8.onnx",
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11/encoder.int8.onnx",
        ),
        download_gb=1.9,
    ),
    EngineInfo("asr-external-tcp", "外部ASR（TCP）", None, None),
    # speech.cpp（ggml。GPU は Vulkan）。実行ファイルと既定のモデル（Haas-ja）を取る。
    # ほかのモデル（ReazonSpeech NeMo v2 など）は選んだときに speech.cpp が初回に取ってくる。
    EngineInfo(
        "asr-speechcpp", "speech.cpp（Haas-ja ほか）", None, None,
        setup_steps=("SpeechCppRuntime", "SpeechCppAsrModel"),
        probe_paths=("vendor/speechcpp/speech.exe", "speechcpp_models/sakasegawa--Haas-ja-v1.0-GGUF"),
        download_gb=0.7,
    ),
    # sherpa-onnx OfflineRecognizer 系。sherpa-onnx 本体は asr-sherpa の extra で入るので、
    # ここで要るのは各モデルの取得だけ（probe_paths は展開済みかどうかの判定）。
    EngineInfo(
        "asr-sensevoice", "SenseVoiceSmall（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("SenseVoiceModel",),
        probe_paths=("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx",),
        download_gb=0.2,
    ),
    EngineInfo(
        "asr-parakeet", "Parakeet TDT-CTC ja（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("ParakeetJaModel",),
        probe_paths=("sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/model.int8.onnx",),
        download_gb=0.5,
    ),
    EngineInfo(
        "asr-cohere", "Cohere Transcribe（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("CohereAsrModel",),
        probe_paths=("sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt",),
        download_gb=1.7,
    ),
    EngineInfo(
        "asr-funasr-nano", "Fun-ASR-Nano（sherpa-onnx）", "asr-sherpa", "sherpa_onnx",
        setup_steps=("FunAsrNanoModel",),
        probe_paths=("sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx",),
        download_gb=0.8,
    ),
    EngineInfo(
        "tts-voicevox", "VOICEVOX Core", "tts-voicevox", "voicevox_core",
        setup_steps=("VoicevoxRuntime",),
        download_gb=0.3,
    ),
    # アプリ版 VOICEVOX。GUI の TTS 一覧の "VOICEVOX"（HTTP localhost:50021）がこれ。
    # VOICEVOX-Core とは別物で、こちらは Python パッケージを使わず外部プロセスを起動する
    # だけなので extra は無い。インストーラ版と同じ位置を見るので、既に手で入れている人は
    # そのまま「インストール済み」と出る。
    EngineInfo(
        "tts-voicevox-app", "VOICEVOX（アプリ）", None, None,
        setup_steps=("VoicevoxApp",),
        probe_paths=("%LOCALAPPDATA%/Programs/VOICEVOX/VOICEVOX.exe",),
        download_gb=2.1,
    ),
    EngineInfo(
        "tts-kokoro", "Kokoro（RealtimeTTS）", "tts-kokoro", "RealtimeTTS",
        setup_steps=("UnidicDict",),
        download_gb=0.8,
    ),
    EngineInfo("tts-cartesia", "Cartesia（クラウド）", "tts-cartesia", "cartesia"),
    # Irodori の既定の動かし方は speech.cpp（専用の環境が要らず速い）。PyTorch 版は別に選べる。
    EngineInfo(
        "tts-irodori", "Irodori（speech.cpp）", None, None,
        setup_steps=("SpeechCppRuntime", "SpeechCppIrodoriModel"),
        probe_paths=("vendor/speechcpp/speech.exe", "speechcpp_models/sakasegawa--Irodori-TTS-v4.1-Small-MF-GGUF"),
        download_gb=1.2,
    ),
    # Irodori-TTS の audio.cpp 版（500M v3 / 600M v3 VoiceDesign。irodori_backend = audiocpp）。
    EngineInfo(
        "tts-irodori-audiocpp", "Irodori（audio.cpp）", None, None,
        setup_steps=("AudioCppServer", "IrodoriAudioCppGguf"),
        probe_paths=(
            "audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe",
            "audio.cpp/models/Irodori-TTS-500M-v3-GGUF/irodori-tts-500m-v3-q8_0.gguf",
            "audio.cpp/models/Irodori-TTS-600M-v3-VoiceDesign-GGUF/irodori-tts-600m-v3-voicedesign-q8_0.gguf",
        ),
        download_gb=2.4,
        shared_runtime="audiocpp",
    ),
    EngineInfo(
        "tts-irodori-pytorch", "Irodori（PyTorch専用環境）", None, None,
        setup_venv_script="scripts/setup_irodori_venv.ps1",
        probe_paths=(".venv_irodori/Scripts/python.exe",),
    ),
    EngineInfo(
        "tts-tsukasa", "Tsukasa", "tts-tsukasa", "munch",
        setup_steps=("TsukasaVendor",),
        download_gb=5.0,
    ),
    EngineInfo(
        "tts-aivis", "AivisSpeech（アプリ）", None, None,
        setup_steps=("AivisSpeech",),
        probe_paths=("%LOCALAPPDATA%/Programs/AivisSpeech/AivisSpeech-Engine/run.exe",),
        download_gb=0.4,
    ),
    EngineInfo(
        "tts-supertonic", "Supertonic 3（audio.cpp）", None, None,
        setup_steps=("AudioCppServer", "SupertonicGguf"),
        probe_paths=(
            "audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe",
            "audio.cpp/models/Supertonic-3-GGUF/supertonic-3-orig.gguf",
        ),
        download_gb=0.5,
        shared_runtime="audiocpp",
    ),
    EngineInfo(
        "tts-qwen3-cpp", "Qwen3-TTS（audio.cpp）", None, None,
        setup_steps=("AudioCppServer", "Qwen3TTSCppGguf"),
        probe_paths=(
            "audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe",
            "audio.cpp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf",
        ),
        download_gb=2.8,
        shared_runtime="audiocpp",
    ),
    EngineInfo(
        "tts-lfm2-audio", "LFM2.5-Audio（audio.cpp）", None, None,
        setup_steps=("AudioCppServer", "Lfm2AudioGguf"),
        probe_paths=(
            "audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe",
            "audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF/LFM2.5-Audio-1.5B-JP-Q8_0.gguf",
        ),
        download_gb=1.8,
        shared_runtime="audiocpp",
    ),
    EngineInfo("tts-template", "TEMPLATE（HTTP）", None, None),
)

def _state_path() -> Path:
    return _paths.repo_path("enabled_engines.json")


def _module_is_present(module_name: str) -> bool:
    if not module_name:
        return False
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    return spec is not None and spec.origin is not None


def require_module_installed(module_name: str, label: str | None = None) -> None:
    """Raise a clear, actionable RuntimeError if `module_name` isn't actually importable.

    Call this at the START of any ASR/TTS engine's init/load, before the real import, so a
    disabled-in-Engine-Manager engine fails with a message pointing at the fix instead of a
    raw ImportError/ModuleNotFoundError from deep inside engine code.
    """
    if _module_is_present(module_name):
        return
    name = label or module_name
    raise RuntimeError(
        f"{name} はインストールされていません。GUIの「エンジン管理」ウィンドウでチェックを入れて"
        f"「適用」を押してからもう一度お試しください。"
    )


def _resolve_probe_path(raw: str) -> Path:
    """Expand %ENV% vars (e.g. AivisSpeech's %LOCALAPPDATA% install path) and resolve
    repo-relative paths against REPO_ROOT / data_root, mirroring how the setup scripts
    place things. 専用 venv の python だけは常に REPO_ROOT（コンテナの volume）側。"""
    expanded = os.path.expandvars(raw)
    norm = expanded.replace("\\", "/")
    if "/bin/python" in norm or norm.endswith("python.exe"):
        match = None
        for part in Path(norm).parts:
            if part.startswith(".venv_"):
                match = part
                break
        if match:
            return _paths.resolve_venv_python(expanded, match)
    return _paths.resolve_under_repo(expanded)


def _posixify_probe_path(raw: str) -> str | None:
    """probe_paths の Windows 向け表記を POSIX に読み替える。

    AivisSpeech の run.exe は Linux に無い（ホストの HTTP エンジンへ繋ぐ）ので None。
    """
    text = raw.replace("\\", "/")
    if os.name == "nt":
        return text
    if "Scripts/python.exe" in text or text.endswith("Scripts/python"):
        return text.replace("Scripts/python.exe", "bin/python").replace("Scripts/python", "bin/python")
    # AivisSpeech・アプリ版 VOICEVOX の本体は、macOS では setup_assets.sh が置くエンジン（paths.macos_tts_engine_run）。
    # Linux ではホスト側の HTTP エンジンへ繋ぐので None。
    if "AivisSpeech" in text or text.endswith("run.exe"):
        return _paths.macos_tts_engine_run("aivisspeech")
    if text.endswith("VOICEVOX.exe"):
        return _paths.macos_tts_engine_run("voicevox")
    if text.endswith("audiocpp_server.exe"):
        # setup_assets.sh が置く場所（paths.resolve_audiocpp_server が探す場所と同じ）
        if sys.platform == "darwin":
            return "audio.cpp/build/macos-metal-release/bin/audiocpp_server"
        return "audio.cpp/build/linux-cuda-release/bin/audiocpp_server"
    if text.endswith("speechcpp/speech.exe"):
        return text[: -len(".exe")]
    return text


def _probe_paths_satisfied(item: EngineInfo) -> bool:
    """インストール判定。POSIX では Windows EXE を必須にしない。"""
    if not item.probe_paths:
        return False
    required: list[str] = []
    optional_bins: list[str] = []
    for raw in item.probe_paths:
        adapted = _posixify_probe_path(raw)
        if adapted is None:
            continue
        original = raw.replace("\\", "/")
        if original.endswith("audiocpp_server.exe") or adapted.endswith("audiocpp_server"):
            optional_bins.append(adapted)
            if sys.platform != "darwin":  # macOS では Windows・Linux のサーバーは動かない
                optional_bins.append("audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe")
                optional_bins.append("audio.cpp/build/linux-cuda-release/bin/audiocpp_server")
            continue
        required.append(adapted)
    if not required and not optional_bins:
        # AivisSpeech のように本体バイナリがこの OS に無いエンジンは、リモート接続前提で
        # インストール済みとみなす。
        return True
    if any(not _resolve_probe_path(path).exists() for path in required):
        return False
    if optional_bins:
        if any(_resolve_probe_path(path).is_file() for path in optional_bins):
            return True
        # GGUF だけあればホスト／別コンテナのサーバーへ繋げる。
        return bool(required)
    return True


def list_engines() -> list[EngineInfo]:
    """Return known engines with current optional dependency availability."""
    def _is_installed(item: EngineInfo) -> bool:
        if item.probe_paths:
            if not _probe_paths_satisfied(item):
                return False
            if item.probe_module:
                return _module_is_present(item.probe_module)
            return True
        if item.extra is None:
            return True
        if not item.probe_module:
            return False
        return _module_is_present(item.probe_module)

    # `replace` rather than rebuilding EngineInfo field by field: `installed` is the only
    # thing computed here, and a hand-written constructor call silently drops any field
    # added to EngineInfo later (probe_paths was added after this function was written).
    return [replace(item, installed=_is_installed(item)) for item in _ENGINE_DEFINITIONS]


def engine_kind(item: EngineInfo) -> str:
    """一覧の見出し（"ASR" / "TTS"）。"""
    return "ASR" if item.id.startswith("asr-") else "TTS"


def has_apply_action(item: EngineInfo) -> bool:
    """「適用」で何かを入れる・外すエンジンか（外部サービスだけのものは常に使える）。"""
    return bool(item.extra or item.setup_steps or item.setup_venv_script)


def plan_engine_changes(engines: list[EngineInfo], selected_ids) -> dict:
    """チェックの状態から、「適用」で起きることを求める（確認の表示と各行の状態に使う）。

    - 入れる：チェックがあり、まだ入っていないもの。
    - 外す：チェックが無く、入っているもの のうち、uv の extra を他のチェック済みのエンジンと
      共有していないもの。`uv sync` は渡さなかった extra の Python パッケージを消す。
      モデルなどのファイルは消さない。
    取得量は各エンジンの download_gb の合計に、まだ入っていない共有の実行系を 1 回分足したもの。
    """
    selected = set(selected_ids)
    selected_extras = {e.extra for e in engines if e.id in selected and e.extra}
    install = [e for e in engines if e.id in selected and not e.installed and has_apply_action(e)]
    remove = [
        e for e in engines
        if e.id not in selected and e.installed and e.extra and e.extra not in selected_extras
    ]
    total = sum(e.download_gb or 0.0 for e in install)
    present_runtimes = {e.shared_runtime for e in engines if e.installed and e.shared_runtime}
    for runtime in {e.shared_runtime for e in install if e.shared_runtime} - present_runtimes:
        total += SHARED_RUNTIME_GB.get(runtime, 0.0)
    return {
        "install": [e.id for e in install],
        "remove": [e.id for e in remove],
        "download_gb": round(total, 1),
    }


def get_enabled_engines() -> list[str]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return [str(value) for value in data if isinstance(value, str)] if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


# scripts/setup_env.ps1 builds a custom sherpa-onnx (NeMo mel + max_symbols=30 patches,
# needed for Japanese ASR quality — plain PyPI sherpa-onnx is a thin wheel that carries no
# runtime of its own and depends on a separate sherpa-onnx-core package for it) and stashes
# it here specifically because `uv sync` overwrites it with the plain PyPI package otherwise.
# A version string containing "+" (e.g. "1.13.5+cuda") identifies the custom build; PyPI
# releases never have that suffix.
_SHERPA_SNAPSHOT_DIR = _paths.repo_path("vendor", "sherpa_onnx_prebuilt")


def _site_packages_dir() -> Path:
    return Path(sysconfig.get_path("purelib"))


def _current_sherpa_version() -> str | None:
    """Return the installed sherpa-onnx version without importing its native extension.

    `uv sync` can replace sherpa-onnx's dist-info with the PyPI package's while leaving
    the custom build's files in site-packages (observed: code 1.13.5+cuda, dist-info
    1.13.2). Reading the metadata there makes _save_sherpa_snapshot_if_custom believe the
    custom build is gone and skip the snapshot -- the one moment it exists to cover. The
    package keeps a literal `__version__` in __init__.py, so parse that assignment without
    executing the file (which would load _sherpa_onnx.pyd and its DLL dependencies).
    """
    try:
        metadata_version = importlib.metadata.version("sherpa-onnx")
    except importlib.metadata.PackageNotFoundError:
        metadata_version = None
    if metadata_version and "+" in metadata_version:
        return metadata_version

    init_path = _site_packages_dir() / "sherpa_onnx" / "__init__.py"
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except (OSError, UnicodeError, SyntaxError):
        return metadata_version
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            source_version = node.value.value.strip()
            if source_version:
                return source_version
    return metadata_version


def _sherpa_dist_infos(site_pkgs: Path) -> list[Path]:
    """sherpa-onnx **本体だけ** の dist-info。

    `glob("sherpa_onnx*.dist-info")` にすると `sherpa_onnx_core-*.dist-info` にも
    当たる。復元側はこれを rmtree するので、`uv sync` が入れた sherpa-onnx-core の
    メタデータを毎回消してしまい、インストール済みなのに未インストールに見える
    （実際に主環境で core が消えていた）。バージョン部は必ず `-` 区切りなので、
    `sherpa_onnx-` で始まるものだけに絞る。
    """
    return [p for p in site_pkgs.glob("sherpa_onnx-*.dist-info") if p.is_dir()]


def _save_sherpa_snapshot_if_custom() -> None:
    """Stash the currently-installed sherpa-onnx into vendor/sherpa_onnx_prebuilt/ if it's the
    custom build — mirrors setup_env.ps1's Save-SherpaSnapshot, run before `uv sync` so a
    same-session custom build survives an unrelated engine toggle."""
    version = _current_sherpa_version()
    if not version or "+" not in version:
        return
    site_pkgs = _site_packages_dir()
    src = site_pkgs / "sherpa_onnx"
    if not src.is_dir():
        return
    try:
        if _SHERPA_SNAPSHOT_DIR.exists():
            shutil.rmtree(_SHERPA_SNAPSHOT_DIR)
        _SHERPA_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, _SHERPA_SNAPSHOT_DIR / "sherpa_onnx")
        for dist_info in _sherpa_dist_infos(site_pkgs):
            shutil.copytree(dist_info, _SHERPA_SNAPSHOT_DIR / dist_info.name)
        (_SHERPA_SNAPSHOT_DIR / "VERSION.txt").write_text(version, encoding="utf-8")
    except OSError:
        pass  # best-effort; a failed snapshot just means the next restore falls back to PyPI


def _restore_sherpa_snapshot() -> tuple[bool, str | None]:
    """Restore vendor/sherpa_onnx_prebuilt/ over whatever `uv sync` just installed — mirrors
    setup_env.ps1's Restore-SherpaSnapshot.

    Returns (True, None) on success, (False, None) when no snapshot exists, and
    (False, reason) when restoration failed after it started.
    """
    src = _SHERPA_SNAPSHOT_DIR / "sherpa_onnx"
    if not src.is_dir():
        return False, None
    site_pkgs = _site_packages_dir()
    try:
        target = site_pkgs / "sherpa_onnx"
        if target.exists():
            shutil.rmtree(target)
        for dist_info in _sherpa_dist_infos(site_pkgs):
            shutil.rmtree(dist_info)
        shutil.copytree(src, target)
        for entry in _SHERPA_SNAPSHOT_DIR.iterdir():
            if entry.is_dir() and entry.name.endswith(".dist-info"):
                shutil.copytree(entry, site_pkgs / entry.name)
    except OSError as ex:
        return False, f"{type(ex).__name__}: {ex}"
    return True, None


def setup_is_supported() -> bool:
    """このプラットフォームでエンジンのセットアップを実行できるか。

    Windows は PowerShell の setup_assets.ps1、Linux/macOS は setup_assets.sh を使う。
    どちらも無い環境（スクリプトを配布していないクローン）では False。
    """
    return _setup_assets_script() is not None


def _setup_assets_script() -> Path | None:
    name = "setup_assets.ps1" if os.name == "nt" else "setup_assets.sh"
    script = _paths.repo_path("scripts", name)
    return script if script.is_file() else None


def _setup_command(script: Path, *args: str) -> list[str]:
    """OS ごとの起動コマンド。.ps1 は PowerShell、.sh は bash 経由で叩く。"""
    if script.suffix.lower() == ".ps1":
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script), *args]
    return ["bash", str(script), *args]


def _platform_script(script: str) -> Path | None:
    """設定に書かれた .ps1 パスを、POSIX では同名の .sh に読み替える。"""
    path = _paths.resolve_under_repo(script)
    if os.name == "nt":
        return path if path.is_file() else None
    posix = path.with_suffix(".sh")
    return posix if posix.is_file() else None


def _emit(on_output: Callable[[str], None] | None, message: str) -> None:
    """進捗を 1 行流す。**コールバック側の例外は握り潰す**。

    `set_enabled_engines` は `uv sync` で環境を書き換えたあと、最後に
    enabled_engines.json を書く。その間はセットアップ出力をこのコールバックへ
    流し続けるので、コールバックが例外を投げると「環境は変わったのに状態ファイルは
    古いまま」で終わる。実際に踏んだ: cp932 のコンソールへ出す呼び出し元で
    `UnicodeEncodeError` が上がり、5 エンジンが「有効」と記録されたまま
    パッケージ未導入という食い違いが残った。

    ログが 1 行落ちることより、環境と状態ファイルがズレることの方が高くつく。
    """
    if on_output is None:
        return
    try:
        on_output(message)
    except Exception:  # noqa: BLE001
        pass



# onnxruntime-gpu の wheel は 280MB 近くあり、uv の既定 30 秒では読み切れずに
#   Failed to download distribution due to network timeout
# で落ちる（実機で発生）。uv は版によって UV_HTTP_TIMEOUT / UV_REQUEST_TIMEOUT の
# どちらを見るかが違うので両方入れる。利用者が明示していればそちらを尊重する。
_UV_DOWNLOAD_TIMEOUT_SEC = "300"


def _uv_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """uv を起動するための環境変数。巨大な wheel でタイムアウトしないようにする。"""
    env = os.environ.copy()
    # PowerShell 7 から起動されたプロセスは、PowerShell 7 用の PSModulePath を持っている。それを
    # そのまま Windows PowerShell 5.1（setup_assets.ps1 などの powershell）に渡すと、5.1 が 7 用の
    # Microsoft.PowerShell.Utility を読みに行き、Get-FileHash などが「見つからない」で落ちる
    # （実機で発生）。外して渡せば、5.1 は自分の既定の場所から組み立てる。
    # Windows の os.environ は変数名を大文字で持つ（PSMODULEPATH）ので、大文字・小文字を区別せずに外す。
    if os.name == "nt":
        for key in [k for k in env if k.upper() == "PSMODULEPATH"]:
            del env[key]
    for key in ("UV_HTTP_TIMEOUT", "UV_REQUEST_TIMEOUT"):
        env.setdefault(key, _UV_DOWNLOAD_TIMEOUT_SEC)
    if extra:
        env.update(extra)
    return env


def _run_setup_command(
    cmd: list[str],
    *,
    engine_id: str,
    setup_name: str,
    on_output: Callable[[str], None] | None,
    extra_env: dict[str, str] | None = None,
) -> bool:
    """Run one optional setup action, stream tagged output, and report success."""
    env = _uv_env(extra_env)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=_paths.REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except OSError as ex:
        _emit(on_output, f"[setup:{engine_id}] {setup_name} could not start: {ex}")
        return False
    if proc.stdout is not None:
        for line in proc.stdout:
            _emit(on_output, f"[setup:{engine_id}] {line.rstrip(chr(13) + chr(10))}")
    rc = proc.wait()
    if rc != 0:
        _emit(on_output, f"[setup:{engine_id}] {setup_name} failed (exit code {rc})")
    return rc == 0


# GPU 構成 / CPU 構成を選ぶ排他 extra（pyproject の [tool.uv] conflicts）。
# どちらも渡さないと torch も onnxruntime-gpu も入らないので、必ず片方を付ける。
_DEVICE_EXTRAS = ("cu128", "cpu")
_DEVICE_MARKER = _paths.repo_path(".device_extra")


def device_extra() -> str:
    """選ばれているデバイス extra。**選んでいなければ空文字**。

    空文字は「デバイスを指定しない」を意味し、適用は cpu / cu128 のどちらも渡さない。
    そうしないと `uv sync --extra gui` だけで作った環境（PyPI 版 torch）と、
    常に cu128 を渡す適用との間で torch が毎回入れ替わる。PyPI 版 torch は
    このプラットフォームでは CPU 専用なので、未指定のままでも CPU 環境として動く。

    設定されるのは scripts/setup_env.ps1 の -Device か、GUI のデバイス選択を
    操作したとき。実行時に CUDA の有無で決めないのは、`uv sync` が torch を
    入れ替えることになり数 GB の再ダウンロードが起きるため。
    """
    # 環境変数を先に見るのは、コンテナで上書きしたいケースがあるため
    # （イメージを建て直さずに compose 側で切り替えられる）。
    env_value = os.environ.get("MAAI_SDS_DEVICE_EXTRA", "").strip()
    if env_value in _DEVICE_EXTRAS:
        return env_value
    try:
        value = _DEVICE_MARKER.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return value if value in _DEVICE_EXTRAS else ""


def device_sync_extra() -> str:
    """`uv sync` に渡すデバイス extra。GPU 以外は空。

    CPU モードで `--extra cpu` を渡さないのは、既定（extra なし）で入る PyPI 版
    torch がこのプラットフォームでは CPU 専用で、+cpu ビルドとの実質差が無いため。
    渡すと数 GB の入れ替えが起きるだけで何も得られない（無印が引く nvidia-* 依存は
    4 件すべて `platform_system == "Linux"` 条件付き）。

    Linux の Docker は compose が MAAI_SDS_DEVICE_EXTRA / DEVICE_EXTRA=cpu を
    明示するので、そちらは env 経由で "cpu" が返り +cpu に固定される。
    """
    choice = device_extra()
    if choice == "cu128":
        return "cu128"
    # env で明示された cpu は尊重する（Docker の CPU イメージ）。
    if os.environ.get("MAAI_SDS_DEVICE_EXTRA", "").strip() == "cpu":
        return "cpu"
    return ""


def set_device_extra(value: str) -> None:
    """GPU 構成 / CPU 構成の選択を記録する。

    環境変数 MAAI_SDS_DEVICE_EXTRA が設定されている環境（コンテナ）では、そちらが
    優先されるので書いても効かない。UI からはその状況を隠さず、
    `device_extra_is_locked()` で入力を無効化する。
    """
    if value not in _DEVICE_EXTRAS:
        raise ValueError(f"unknown device extra: {value!r}")
    _DEVICE_MARKER.write_text(value, encoding="utf-8")


def device_extra_is_locked() -> bool:
    """環境変数で固定されているか（この場合 UI から変えても無意味）。"""
    return os.environ.get("MAAI_SDS_DEVICE_EXTRA", "").strip() in _DEVICE_EXTRAS


def _installed_host_ui_extras() -> tuple[str, ...]:
    """いま入っている UI 用 extra（`gui` / `web`）。`uv sync` の exact sync から守る。

    呼び出し側は自分の分だけを `always_extras` に渡していた（PySide は `("gui",)`、
    Web サーバは `("web",)`）。`uv sync` は渡されなかった extra を消すので、ホストで
    両方使っていると、エンジン適用のたびに「もう一方」が黙ってアンインストールされる。
    Docker は `_env_always_extras()` が両方カバーしているので発生しない。

    「入っているものだけ」返すのがポイント。無条件に両方渡すと、Web だけ使いたい環境に
    PySide6 を、GUI だけの環境に FastAPI を新規インストールしてしまう。
    """
    extras: list[str] = []
    if _module_is_present("PySide6"):
        extras.append("gui")
    if _module_is_present("fastapi"):
        extras.append("web")
    return tuple(extras)


def _env_always_extras() -> tuple[str, ...]:
    """`uv sync` が消してはいけない extra。

    Docker イメージは web + ASR/TTS extra を同梱する。`uv sync` は exact sync なので、
    エンジン適用時にそれらを渡さないとサーバ自身と既存エンジンが消える。
    """
    raw = os.environ.get("MAAI_SDS_ALWAYS_EXTRAS", "").strip()
    if raw:
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if _paths.running_in_docker():
        return (
            "web",
            "asr-faster-whisper",
            "asr-sherpa",
            "asr-chirp3",
            "tts-voicevox",
            "tts-kokoro",
            "tts-cartesia",
            "tts-tsukasa",
        )
    return ()


def installed_torch_device_extra() -> str | None:
    """今 site-packages に入っている torch がどちらのビルドか。判別できなければ None。

    ローカル版（+cpu / +cu128）で見る。`uv sync` が torch を入れ替える必要があるか、
    つまり実行中プロセスの DLL とぶつかるかを事前に知るために使う。
    """
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    local = version.partition("+")[2]
    if not local:
        return None
    return "cpu" if local == "cpu" else "cu128" if local.startswith("cu") else None


def device_swap_required(device: str | None) -> bool:
    """この適用で torch の入れ替えが起きるか。"""
    # 比較するのは「uv に渡す extra」。CPU モードは渡さないので、無印が入っていれば
    # 入れ替えは起きない（+cpu が入っている場合も CPU 専用同士なので触らない）。
    target = "cu128" if (device or device_extra()) == "cu128" else device_sync_extra()
    installed = installed_torch_device_extra()
    if not target:
        # extra を渡さない = 無印へ寄せる。+cu128 が入っているときだけ入れ替わる
        # （+cpu も無印も CPU 専用なので触らない）。
        return installed == "cu128"
    # extra を渡す = そのビルドに固定する。無印（None）が入っている場合も入れ替わる。
    return installed != target


def _selected_device_sync_extra(
    device: str | None, on_output: Callable[[str], None] | None
) -> str:
    """今回の適用で `uv sync` に渡すデバイス extra。指定があれば選択を記録してから返す。

    環境変数で固定されている場合は書き込まない（書いても読み側が env を先に見るので
    食い違うだけ）。CPU モードで空を返すのは device_sync_extra の注記のとおり。
    """
    if device and device in _DEVICE_EXTRAS and not device_extra_is_locked():
        if device != device_extra():
            _emit(on_output, f"[engine-manager] デバイス構成を {device} に切り替えます。")
        try:
            set_device_extra(device)
        except OSError as ex:
            _emit(on_output, f"[engine-manager] デバイス構成を記録できませんでした: {ex}")
    extra = device_sync_extra()
    if not extra and device_extra() == "cpu":
        _emit(
            on_output,
            "[engine-manager] CPU 構成では torch を入れ替えません"
            "（既定で入る PyPI 版がこのプラットフォームでは CPU 専用のため）。",
        )
    return extra


# Windows ではソースからのビルドがある extra（misaki[ja] → pyopenjtalk、monotonic-align）。
# scripts/setup_env.ps1 の $compilerExtras と揃える。既定の構成には入らないので、
# これらを「適用」したときだけ MSVC Build Tools と CMake が要る。
COMPILER_EXTRAS = ("tts-kokoro", "tts-tsukasa")


def _msvc_available() -> bool:
    """MSVC の C++ ツールセットが入っているか（vswhere で見る。setup_env.ps1 と同じ判定）。"""
    root = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return False
    try:
        out = subprocess.run(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def build_tools_hint(extras, *, platform: str | None = None) -> str | None:
    """ソースからのビルドが要る extra を入れるのにツールが足りなければ、その案内を返す（足りていれば None）。"""
    if (platform or os.name) != "nt":
        return None
    needs = [e for e in extras if e in COMPILER_EXTRAS]
    if not needs:
        return None
    lack = []
    if not _msvc_available():
        lack.append("Visual Studio 2022 Build Tools（C++。winget install Microsoft.VisualStudio.2022.BuildTools "
                    '--override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"）')
    if shutil.which("cmake") is None:
        lack.append("CMake（winget install Kitware.CMake）")
    if not lack:
        return None
    return (
        f"{', '.join(needs)} は Windows ではソースからのビルドがあり、次が要ります：{'、'.join(lack)}。"
        "入れたあとはアプリを起動し直してから、もう一度「適用」してください"
        "（入れる前に失敗していた場合は、先に uv cache clean pyopenjtalk monotonic-align）。"
    )


def set_enabled_engines(
    ids: list[str],
    *,
    always_extras: tuple[str, ...] = (),
    device: str | None = None,
    on_output: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Persist enabled IDs and synchronize their optional dependency extras.

    `uv sync` is an *exact* sync: any optional-dependency extra not passed on this
    call gets uninstalled, even ones unrelated to ASR/TTS engines (e.g. `gui`).
    Callers running inside a process that itself depends on an extra not tracked
    here (the GUI process needs PySide6/`gui` to keep running) must pass it via
    `always_extras`, or `uv sync` will try to delete that extra's native DLLs out
    from under the running process and fail (Windows file lock) or worse.

    Also protects the custom sherpa-onnx build (see _SHERPA_SNAPSHOT_DIR above): `uv sync`
    always resolves `asr-sherpa` from plain PyPI, which would silently downgrade Japanese ASR
    quality (and, if PyPI's sherpa-onnx-core companion package isn't available for this
    platform, break sherpa-onnx entirely) every time any engine gets toggled here — snapshot
    before, restore after, so this function is safe to call regardless of what's selected.
    """
    known = {item.id: item for item in _ENGINE_DEFINITIONS}
    selected = list(dict.fromkeys(engine_id for engine_id in ids if engine_id in known))
    extras = list(dict.fromkeys(
        [e for e in (_selected_device_sync_extra(device, on_output),) if e]
        + list(_env_always_extras())
        + list(_installed_host_ui_extras())
        + list(always_extras)
        + [known[engine_id].extra for engine_id in selected if known[engine_id].extra]
    ))
    cmd = ["uv", "sync", "--frozen"]
    for extra in extras:
        cmd.extend(("--extra", extra))
    tools_hint = build_tools_hint(extras)
    if tools_hint:
        _emit(on_output, f"[engine-manager] {tools_hint}")
    _save_sherpa_snapshot_if_custom()
    try:
        proc = subprocess.Popen(cmd, cwd=_paths.REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=_uv_env())
    except OSError as ex:
        return False, f"uv sync could not start: {ex}"
    if proc.stdout is not None:
        for line in proc.stdout:
            _emit(on_output, line.rstrip("\r\n"))
    rc = proc.wait()
    if rc != 0:
        message = f"uv sync failed (exit code {rc})"
        if tools_hint:
            message += f" {tools_hint}"
        return False, message
    # 判定は「エンジン id」ではなく「解決後の extra」で行う。sherpa-onnx を使うエンジンは
    # asr-sherpa だけではなく、qwen3-onnx / sensevoice / parakeet / cohere / funasr-nano
    # も同じ extra を共有する。id で見ていると、それらだけを選んだときに `uv sync` が
    # 入れた PyPI 版がそのまま残り、カスタムビルドが黙って失われる。
    if "asr-sherpa" in extras:
        # スナップショットが無ければ、配布されているプリビルドを取りに行く。
        # ここでやるのは順序のため: setup_steps は下のループ（復元の後）なので、
        # そこで落としても適用は次回の apply までかからない。取得先は
        # vendor/sherpa_onnx_prebuilt/ なので、あとは既存の復元処理がそのまま入れる。
        # 取得できなくても続行する（PyPI 版のまま動く。品質は落ちる）。
        prebuilt_script = _setup_assets_script()
        if not _SHERPA_SNAPSHOT_DIR.is_dir() and prebuilt_script is not None:
            _emit(on_output, "[engine-manager] カスタムビルド sherpa-onnx を取得します…")
            only_flag = "-Only" if prebuilt_script.suffix.lower() == ".ps1" else "--only"
            _run_setup_command(
                _setup_command(prebuilt_script, only_flag, "SherpaPrebuilt"),
                engine_id="asr-sherpa",
                setup_name="SherpaPrebuilt",
                on_output=on_output,
            )
        restored, restore_error = _restore_sherpa_snapshot()
        if restored:
            _emit(on_output, "[engine-manager] カスタムビルドの sherpa-onnx を復元しました。")
        elif restore_error is not None:
            message = (
                "[engine-manager] カスタムビルド sherpa-onnx の復元に失敗しました: "
                f"{restore_error} site-packages が部分的な状態の可能性があります。"
                "GUIを終了して scripts/setup_env.ps1 で環境を修復してください。"
            )
            _emit(on_output, message)
            return False, message
        else:
            _emit(
                on_output,
                    "[engine-manager] カスタムビルド sherpa-onnx を用意できませんでした。"
                    "PyPI 版で動作しますが、日本語 ASR の品質は落ちます"
                    "（NeMo mel 再計算と max_symbols=30 のパッチが入りません）。"
                    "ネットワークが繋がらない、または配布物が無い環境では "
                    "scripts/upgrade_sherpa_onnx.ps1 で自前ビルドできます"
                    "（Git + CMake + MSVC Build Tools が必要）。"
                )
    failed_steps: list[str] = []
    setup_assets_script = _setup_assets_script()
    for engine_id in selected:
        info = known[engine_id]
        if info.setup_venv_script:
            script_path = _platform_script(info.setup_venv_script)
            if script_path is None:
                _emit(
                    on_output,
                    f"[setup:{engine_id}] {Path(info.setup_venv_script).name} は"
                    f" このプラットフォーム（{os.name}）向けが未提供のため飛ばします。",
                )
                failed_steps.append(f"{engine_id}/{Path(info.setup_venv_script).name}")
            elif not _run_setup_command(
                _setup_command(script_path),
                engine_id=engine_id,
                setup_name=script_path.name,
                on_output=on_output,
            ):
                failed_steps.append(f"{engine_id}/{script_path.name}")
        for step in info.setup_steps:
            if setup_assets_script is None:
                _emit(
                    on_output,
                    f"[setup:{engine_id}] {step}: 資産取得スクリプトが見つかりません。")
                failed_steps.append(f"{engine_id}/{step}")
                continue
            only_flag = "-Only" if setup_assets_script.suffix.lower() == ".ps1" else "--only"
            args = [only_flag, step]
            if not _run_setup_command(
                _setup_command(setup_assets_script, *args),
                engine_id=engine_id,
                setup_name=step,
                on_output=on_output,
            ):
                failed_steps.append(f"{engine_id}/{step}")
    try:
        _state_path().write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as ex:
        return False, f"Dependencies installed, but state file could not be saved: {ex}"
    message = "Engine installation settings applied."
    if failed_steps:
        message += f" ただし追加セットアップに失敗: {', '.join(failed_steps)}"
    return True, message
