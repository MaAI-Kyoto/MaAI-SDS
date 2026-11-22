"""Align sherpa-onnx NeMo online fbank with Nemotron AudioToMelSpectrogramPreprocessor.

NeMo 3.5 (normalize=NA): 128 mel, hann, 25ms/10ms, librosa-style, preemph=0.97,
low/high 0–8000 Hz, dither=0 at inference, snip_edges=false, n_fft=512 (power-of-two).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("sherpa-onnx/csrc/online-recognizer-transducer-nemo-impl.h")
MARKER = "MaAI-SDS NeMo mel alignment"

OLD = """    config_.feat_config.dither = 0;
    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();"""

NEW = """    config_.feat_config.dither = 0;
    // MaAI-SDS NeMo mel alignment (nemotron-3.5-asr-streaming-0.6b preprocessor)
    config_.feat_config.preemph_coeff = 0.97f;
    config_.feat_config.frame_shift_ms = 10.0f;
    config_.feat_config.frame_length_ms = 25.0f;
    config_.feat_config.snip_edges = false;
    config_.feat_config.round_to_power_of_two = true;
    config_.feat_config.normalize_samples = true;
    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sherpa_src", type=Path, help="vendor/sherpa-onnx clone root")
    args = parser.parse_args()
    path = args.sherpa_src / TARGET
    if not path.is_file():
        print(f"ERROR: missing {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Already patched: {path}")
        return 0
    if OLD not in text:
        print(f"ERROR: anchor not found in {path}", file=sys.stderr)
        return 1
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8", newline="\n")
    print(f"Patched {path} ({MARKER})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
