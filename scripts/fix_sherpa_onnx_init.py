"""Trim sherpa_onnx/__init__.py imports to symbols present in the built _sherpa_onnx.pyd."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _load_pyd_exports(pyd_path: Path) -> set[str]:
    if sys.platform == "win32":
        import os

        lib_dir = pyd_path.parent
        try:
            os.add_dll_directory(str(lib_dir))
        except (OSError, AttributeError, ValueError):
            pass
    mod_name = "_sherpa_onnx"
    spec = importlib.util.spec_from_file_location(mod_name, pyd_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pyd: {pyd_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {name for name in dir(mod) if not name.startswith("_")}


def fix_sherpa_init(package_dir: Path) -> int:
    init_path = package_dir / "__init__.py"
    lib_dir = package_dir / "lib"
    if not init_path.is_file():
        print(f"missing {init_path}", file=sys.stderr)
        return 1

    # 拡張子は OS で変わる（Windows: .pyd / Linux: .so）。Linux ビルドでも
    # 同じ調整が要る（TTS=OFF だと GenerationConfig が .so から落ちるのは同じ）。
    pyd_candidates = sorted(lib_dir.glob("_sherpa_onnx*.pyd")) or sorted(
        lib_dir.glob("_sherpa_onnx*.so")
    )
    if not pyd_candidates:
        print(f"no _sherpa_onnx extension under {lib_dir}", file=sys.stderr)
        return 1

    available = _load_pyd_exports(pyd_candidates[0])
    text = init_path.read_text(encoding="utf-8")
    match = re.search(
        r"(from sherpa_onnx\.lib\._sherpa_onnx import \(\s*)([\s\S]*?)(\s*\))",
        text,
    )
    if not match:
        print("import block not found in __init__.py", file=sys.stderr)
        return 1

    names = [line.strip().rstrip(",") for line in match.group(2).splitlines()]
    names = [n for n in names if n]
    kept = [n for n in names if n in available]
    dropped = [n for n in names if n not in available]
    if not kept:
        print("no overlapping symbols between __init__.py and pyd", file=sys.stderr)
        return 1

    new_block = match.group(1) + ",\n    ".join(kept) + ",\n" + match.group(3)
    new_text = text[: match.start()] + new_block + text[match.end() :]
    if "__version__" not in new_text:
        new_text = new_text.rstrip() + "\n\n__version__ = version\n"
    init_path.write_text(new_text, encoding="utf-8")
    print(f"fixed {init_path}: kept {len(kept)}, dropped {len(dropped)}")
    if dropped:
        print("dropped:", ", ".join(dropped[:12]) + ("..." if len(dropped) > 12 else ""))
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <site-packages/sherpa_onnx>", file=sys.stderr)
        return 2
    return fix_sherpa_init(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
