"""NeMo greedy decoder cross-chunk double-feed.

Upstream sherpa-onnx v1.13.5 (PR #3785) caches decoder_out across chunks via
OnlineStream::{Get,Set}NeMoDecoderOut, which is the official fix. Do NOT copy
the old MaAI-SDS full-file template over that — it would revert #3785.

This script is a no-op on v1.13.5+. It only copies
`scripts/sherpa_patches/online-transducer-greedy-search-nemo-decoder.nemo.cc`
when the tree is older than #3785 (no GetNeMoDecoderOut).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET_REL = Path("sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.cc")
STREAM_H_REL = Path("sherpa-onnx/csrc/online-stream.h")
UPSTREAM_MARKER = "GetNeMoDecoderOut"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sherpa_src",
        type=Path,
        help="Path to sherpa-onnx git clone (vendor/sherpa-onnx)",
    )
    args = parser.parse_args()

    stream_h = args.sherpa_src / STREAM_H_REL
    target = args.sherpa_src / TARGET_REL
    if stream_h.is_file() and UPSTREAM_MARKER in stream_h.read_text(encoding="utf-8"):
        print(
            "  online-transducer-greedy-search-nemo-decoder.cc: "
            "skip (upstream #3785 GetNeMoDecoderOut present; use max_symbols patch only)"
        )
        return 0

    script_dir = Path(__file__).resolve().parent
    template = script_dir / "sherpa_patches" / "online-transducer-greedy-search-nemo-decoder.nemo.cc"
    if not template.is_file():
        print(f"missing template: {template}", file=sys.stderr)
        return 1

    if not target.is_file():
        print(f"target not found: {target}", file=sys.stderr)
        return 1

    new_text = template.read_text(encoding="utf-8")
    cur_text = target.read_text(encoding="utf-8")
    if cur_text == new_text:
        print("  online-transducer-greedy-search-nemo-decoder.cc: already patched")
        return 0

    target.write_text(new_text, encoding="utf-8", newline="\n")
    print(
        "  online-transducer-greedy-search-nemo-decoder.cc: patched "
        "(legacy double-feed fix; prefer upgrading to sherpa-onnx v1.13.5+)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
