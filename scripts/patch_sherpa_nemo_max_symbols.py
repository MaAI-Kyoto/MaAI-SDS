"""Patch sherpa-onnx NeMo online decoder max_symbols_per_frame before Git build.

sherpa-onnx hardcodes max_symbols_per_frame=10 in online-transducer-greedy-search-nemo-decoder.cc.
Nemotron NeMo uses 30 for Japanese (see nemotron_asr.py / RNNTDecodingConfig.greedy.max_symbols).
10 causes noticeable word dropouts compared to nemotron-asr.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TARGET_REL = Path("sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.cc")
PATTERN = re.compile(
    r"int32_t max_symbols_per_frame = \d+;",
    re.MULTILINE,
)
REPLACEMENT_TEMPLATE = "int32_t max_symbols_per_frame = {value};"


def patch_file(path: Path, max_symbols: int) -> bool:
    text = path.read_text(encoding="utf-8")
    replacement = REPLACEMENT_TEMPLATE.format(value=int(max_symbols))
    if PATTERN.search(text) is None:
        if replacement in text:
            return False
        raise RuntimeError(f"max_symbols_per_frame assignment not found in {path}")
    new_text, count = PATTERN.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"expected 1 replacement in {path}, got {count}")
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sherpa_src",
        type=Path,
        help="Path to sherpa-onnx git clone (vendor/sherpa-onnx)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=30,
        help="max_symbols_per_frame for NeMo greedy online decoder (default: 30)",
    )
    args = parser.parse_args()
    target = args.sherpa_src / TARGET_REL
    if not target.is_file():
        print(f"ERROR: file not found: {target}", file=sys.stderr)
        return 1
    if args.max_symbols < 1 or args.max_symbols > 128:
        print("ERROR: --max-symbols must be between 1 and 128", file=sys.stderr)
        return 1
    changed = patch_file(target, args.max_symbols)
    if changed:
        print(f"Patched {target} -> max_symbols_per_frame={args.max_symbols}")
    else:
        print(f"Already patched ({target}, max_symbols_per_frame={args.max_symbols})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
