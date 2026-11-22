"""sherpa-onnx とプロジェクト同梱 ORT の ABI 競合を避ける。

Windows では DLL 検索順。Linux（Docker）の PyPI sherpa-onnx は
``libonnxruntime.so`` の ``VERS_1.24.4`` を要求するが、venv の
onnxruntime-gpu は 1.25.1 だけなので、1.24.4 を別キャッシュして
``sherpa_onnx/lib``（RPATH の $ORIGIN）へ置く。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_done = False

_LINUX_ORT_VERSION = "1.24.4"
_LINUX_ORT_TGZ = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    f"v{_LINUX_ORT_VERSION}/onnxruntime-linux-x64-{_LINUX_ORT_VERSION}.tgz"
)

# sherpa-onnx は同梱 onnxruntime 1.17.x でリンクされる。先に venv の 1.25 を
# import すると API version 24 vs 1.17 で推論が失敗する。
_SHERPA_ASR_ENGINES = frozenset({
    "sherpa-onnx",
    "qwen3-asr-onnx",
    "nemotron-onnx",
    "parakeet-onnx",
    "sensevoice-onnx",
    "funasr-nano-onnx",
    "cohere-asr-onnx",
})


def _venv_ort_capi_dir() -> Path | None:
    exe = Path(sys.executable).resolve()
    for root in (exe.parent, exe.parent.parent):
        capi = root / "Lib" / "site-packages" / "onnxruntime" / "capi"
        if (capi / "onnxruntime.dll").is_file():
            return capi
    return None


def _sherpa_package_dir() -> Path | None:
    try:
        import importlib.util

        spec = importlib.util.find_spec("sherpa_onnx")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(spec.submodule_search_locations[0]).resolve()


def _sherpa_ort_dll_dirs() -> list[Path]:
    base = _sherpa_package_dir()
    if base is None:
        return []
    dirs: list[Path] = []
    seen: set[str] = set()
    lib_dir = base / "lib"
    if lib_dir.is_dir():
        dirs.append(lib_dir)
        seen.add(str(lib_dir).lower())
    for dll in sorted(base.glob("**/onnxruntime.dll")):
        key = str(dll.parent).lower()
        if key in seen:
            continue
        seen.add(key)
        dirs.append(dll.parent)
    return dirs


def _cuda_dll_dirs() -> list[Path]:
    """venv 内 NVIDIA CUDA/cuDNN wheel の bin（+ torch/lib）。

    onnxruntime の CUDA Execution Provider は実行時に cudnn64_9.dll などを要求するが、
    pip の onnxruntime-gpu はそれらを同梱しない。pip 版 nvidia-cudnn/cublas wheel か
    torch 同梱 DLL を DLL 検索パスへ足して CUDAExecutionProvider のロード失敗を防ぐ。
    """
    exe = Path(sys.executable).resolve()
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in (exe.parent, exe.parent.parent):
        sp = root / "Lib" / "site-packages"
        if not sp.is_dir():
            continue
        nvidia = sp / "nvidia"
        if nvidia.is_dir():
            for sub in sorted(nvidia.glob("*/bin")):
                key = str(sub).lower()
                if sub.is_dir() and key not in seen:
                    seen.add(key)
                    dirs.append(sub)
        torch_lib = sp / "torch" / "lib"
        key = str(torch_lib).lower()
        if torch_lib.is_dir() and key not in seen:
            seen.add(key)
            dirs.append(torch_lib)
        if dirs:
            break
    return dirs


def _add_cuda_dll_dirs() -> None:
    """CUDA/cuDNN DLL ディレクトリを DLL 検索パスと PATH の両方へ追加する。

    add_dll_directory だけでは cuDNN の推移的依存（cudnn_graph 等）が解決されない
    ことがあるため、PATH 前置も行う（Windows の既定 DLL 探索が使う）。
    """
    if sys.platform != "win32":
        return
    extra = _cuda_dll_dirs()
    if not extra:
        return
    path_parts = [str(d) for d in extra]
    os.environ["PATH"] = os.pathsep.join(path_parts) + os.pathsep + os.environ.get("PATH", "")
    for d in extra:
        try:
            os.add_dll_directory(str(d))
        except (OSError, AttributeError, ValueError):
            pass


def _linux_ort_cache_dir() -> Path:
    from .. import paths as _paths

    return _paths.assets_root() / f"onnxruntime-linux-x64-{_LINUX_ORT_VERSION}"


def _download_linux_ort_1244(dest_root: Path) -> Path:
    """公式 CPU ORT 1.24.4 を dest_root に展開し、lib ディレクトリを返す。"""
    import tarfile
    import tempfile
    import urllib.request

    dest_root.mkdir(parents=True, exist_ok=True)
    lib_dir = dest_root / "lib"
    if (lib_dir / "libonnxruntime.so").is_file() or list(lib_dir.glob("libonnxruntime.so.*")):
        return lib_dir
    print(
        f"[ASR] sherpa-onnx 用に ONNX Runtime {_LINUX_ORT_VERSION} を取得します: {_LINUX_ORT_TGZ}",
        file=sys.stderr,
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="ort1244-") as tmp:
        tgz = Path(tmp) / "ort.tgz"
        urllib.request.urlretrieve(_LINUX_ORT_TGZ, tgz)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(path=tmp)
        extracted_lib = None
        for candidate in Path(tmp).rglob("libonnxruntime.so"):
            extracted_lib = candidate.parent
            break
        if extracted_lib is None:
            for candidate in Path(tmp).rglob("libonnxruntime.so.*"):
                if "providers" in candidate.name:
                    continue
                extracted_lib = candidate.parent
                break
        if extracted_lib is None:
            raise RuntimeError(f"ONNX Runtime {_LINUX_ORT_VERSION} の tarball に libonnxruntime.so がありません")
        lib_dir.mkdir(parents=True, exist_ok=True)
        for src in extracted_lib.iterdir():
            if not src.name.startswith("libonnxruntime"):
                continue
            target = lib_dir / src.name
            if target.exists() or target.is_symlink():
                target.unlink()
            if src.is_symlink():
                target.symlink_to(os.readlink(src))
            else:
                target.write_bytes(src.read_bytes())
    print(f"[ASR] ONNX Runtime {_LINUX_ORT_VERSION} を配置: {lib_dir}", file=sys.stderr, flush=True)
    return lib_dir


def _ensure_linux_sherpa_ort() -> None:
    """PyPI sherpa-onnx の NEEDED libonnxruntime.so (VERS_1.24.4) を $ORIGIN に置く。"""
    base = _sherpa_package_dir()
    if base is None:
        return
    origin = base / "lib"
    if not origin.is_dir():
        return
    try:
        cache_lib = _download_linux_ort_1244(_linux_ort_cache_dir())
    except Exception as ex:  # noqa: BLE001
        print(f"[ASR] sherpa 用 ORT {_LINUX_ORT_VERSION} の準備に失敗: {ex}", file=sys.stderr, flush=True)
        return
    for src in cache_lib.iterdir():
        if not src.name.startswith("libonnxruntime"):
            continue
        dest = origin / src.name
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src.resolve())
        except OSError:
            try:
                dest.write_bytes(src.read_bytes())
            except OSError as copy_ex:
                print(f"[ASR] {dest} を置けません: {copy_ex}", file=sys.stderr, flush=True)


def ensure_sherpa_bundled() -> None:
    """sherpa-onnx 同梱 ORT を先に DLL パスへ追加する（venv ORT は import しない）。"""
    global _done
    if _done:
        return
    _done = True
    if sys.platform.startswith("linux"):
        _ensure_linux_sherpa_ort()
        return
    if sys.platform != "win32":
        return
    for dll_dir in _sherpa_ort_dll_dirs():
        try:
            os.add_dll_directory(str(dll_dir))
        except (OSError, AttributeError, ValueError):
            pass
    # GPU sherpa-onnx ビルド（onnxruntime-win-x64-gpu）の CUDA EP 向けに cuDNN を解決可能にする。
    _add_cuda_dll_dirs()


def ensure() -> None:
    """venv の onnxruntime-gpu を先にロード（smart-turn 等、プロジェクト ORT 利用向け）。"""
    global _done
    if _done:
        return
    _done = True
    if sys.platform != "win32":
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            pass
        return
    capi = _venv_ort_capi_dir()
    if capi is not None:
        try:
            os.add_dll_directory(str(capi))
        except (OSError, AttributeError, ValueError):
            pass
    _add_cuda_dll_dirs()
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        pass


def ensure_for_asr_engine(engine: str) -> None:
    eng = (engine or "").strip().lower()
    if eng in _SHERPA_ASR_ENGINES:
        ensure_sherpa_bundled()
    else:
        ensure()


def peek_asr_engine_from_argv(argv: list[str] | None = None) -> str:
    args = argv if argv is not None else sys.argv
    for i, arg in enumerate(args):
        if arg != "--config-json" or i + 1 >= len(args):
            continue
        raw = args[i + 1]
        try:
            obj = json.loads(raw)
        except Exception:
            return ""
        if isinstance(obj, dict):
            return str(obj.get("asr_engine", "") or "").strip().lower()
        return ""
    return ""


def ensure_for_asr_engines(engines: "list[str]") -> None:
    """複数エンジン（compare の top/bottom 等）向け。いずれかが sherpa 系なら
    同梱 ORT を優先（venv ORT 1.25 を import すると sherpa 1.17 が壊れるため）。"""
    norm = {(e or "").strip().lower() for e in engines}
    if norm & _SHERPA_ASR_ENGINES:
        ensure_sherpa_bundled()
    else:
        ensure()


def peek_compare_engines_from_argv(argv: "list[str] | None" = None) -> "list[str]":
    args = argv if argv is not None else sys.argv
    for i, arg in enumerate(args):
        if arg != "--config-json" or i + 1 >= len(args):
            continue
        try:
            obj = json.loads(args[i + 1])
        except Exception:
            return []
        if isinstance(obj, dict):
            return [
                str(obj.get("compare_top_engine", "") or ""),
                str(obj.get("compare_bottom_engine", "") or ""),
            ]
        return []
    return []
