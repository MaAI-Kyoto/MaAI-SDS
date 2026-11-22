"""Patch sherpa-onnx Windows build for MaAI-SDS.

1. SHERPA_ONNX_MSVC_JOBS (default 1) — OpenFST parallel build fix
2. os.makedirs(out_bin_dir) — wheel copy when SHERPA_ONNX_ENABLE_BINARY=OFF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "SHERPA_ONNX_MSVC_JOBS"
BIN_DIR_MARKER = "os.makedirs(out_bin_dir, exist_ok=True)"


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    if MARKER not in text:
        old = '        if is_windows():\n            if not need_split_package():'
        new = f'''        if is_windows():
            msvc_jobs = os.environ.get("{MARKER}", "1")
            msvc_flag = f"-m:{{msvc_jobs}}"
            if not need_split_package():'''
        if old not in text:
            raise RuntimeError(f"MSVC anchor not found in {path}")
        text = text.replace(old, new, 1)
        text = text.replace(
            "cmake --build {self.build_temp} --target install --config Release -- -m:2",
            "cmake --build {self.build_temp} --target install --config Release -- {msvc_flag}",
            1,
        )
        text = text.replace(
            "cmake --build {self.build_temp} --target _sherpa_onnx --config Release -- -m:2",
            "cmake --build {self.build_temp} --target _sherpa_onnx --config Release -- {msvc_flag}",
            1,
        )
        text = text.replace('"--",\n                    "-m:2",', '"--",\n                    msvc_flag,')
        changed = True

    if BIN_DIR_MARKER not in text:
        anchor = (
            '        out_bin_dir = Path(self.build_lib).resolve().parent / "sherpa_onnx" / "bin"\n'
            '        install_dir = Path(self.build_lib).resolve() / "sherpa_onnx"'
        )
        repl = (
            '        out_bin_dir = Path(self.build_lib).resolve().parent / "sherpa_onnx" / "bin"\n'
            "        os.makedirs(out_bin_dir, exist_ok=True)\n"
            '        install_dir = Path(self.build_lib).resolve() / "sherpa_onnx"'
        )
        if anchor not in text:
            if BIN_DIR_MARKER in text:
                pass
            else:
                raise RuntimeError(f"out_bin_dir anchor not found in {path}")
        else:
            text = text.replace(anchor, repl, 1)
            changed = True

    if changed:
        path.write_text(text, encoding="utf-8", newline="\n")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sherpa_src", type=Path)
    args = parser.parse_args()
    target = args.sherpa_src / "cmake" / "cmake_extension.py"
    if not target.is_file():
        print(f"ERROR: missing {target}", file=sys.stderr)
        return 1
    changed = patch(target)
    print(f"  cmake_extension.py: {'patched' if changed else 'already patched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
