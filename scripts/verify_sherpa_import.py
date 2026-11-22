"""Verify sherpa_onnx imports after Git build install."""
from __future__ import annotations

import sys


def main() -> int:
    import sherpa_onnx
    from sherpa_onnx import OnlineRecognizer

    ver = getattr(sherpa_onnx, "__version__", None) or getattr(sherpa_onnx, "version", "?")
    print("import OK", ver)
    _ = OnlineRecognizer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
