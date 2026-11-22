"""Patch sherpa-onnx NeMo online path for MaAI-SDS nemotron-onnx.

1. online-stream.cc: 15s audio buffer + full-window STFT/mel recompute +
   per_feature normalize (nemotron-asr style), then encoder chunk extract.
2. online-recognizer-transducer-nemo-impl.h: call GetNeMoNormalizedFeatures in DecodeStreams.
3. online-stream.h: declare GetNeMoNormalizedFeatures.

Applies on sherpa-onnx v1.13.5 (tag; includes PR #3671 prompt_index and #3785
decoder_out cache). Copies scripts/sherpa_patches/online-stream.nemo.cc when the
full-STFT path is not yet present. That template also implements GetNeMoDecoderOut
so the official #3785 API is not dropped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FULL_STFT_MARKER = "ComputeNeMoMelFromWaveform"
PREPROCESSOR_MARKER = "ComputeNeMoMelFromPreprocessorOnnx"
MARKER = "GetNeMoNormalizedFeatures"
NEMO_IMPL_MARKER = 'use_nemo_window_mel'

ONLINE_STREAM_H_SNIPPET = """
  /** NeMo online: full-window STFT/mel recompute (15s audio buffer),
   *  per_feature-normalize, extract chunk (nemotron-asr style).
   *
   * @param frame_index  Same index as GetNumProcessedFrames() / GetFrames().
   * @param n            chunk_size mel frames.
   * @param raw_features Flattened (n, feature_dim) from GetFrames().
   * @param normalize_type  Model metadata (e.g. "per_feature"); empty skips normalization.
   */
  std::vector<float> GetNeMoNormalizedFeatures(
      int32_t frame_index, int32_t n,
      const std::vector<float> &raw_features,
      const std::string &normalize_type) const;

  // Generic per-stream option mechanism (key-value string pairs).
"""

NEMO_IMPL_DECODE = """    const std::string &normalize_type = config_.feat_config.nemo_normalize_type;
    const bool use_nemo_window_mel =
        !config_.feat_config.nemo_preprocessor_model.empty() ||
        normalize_type == "per_feature";

    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> raw_features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      std::vector<float> features =
          use_nemo_window_mel
              ? ss[i]->GetNeMoNormalizedFeatures(num_processed_frames,
                                                 chunk_size, raw_features,
                                                 normalize_type)
              : std::move(raw_features);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;
"""


def _replace_once(text: str, old: str, new: str, path: Path, label: str) -> tuple[str, bool]:
    if old not in text:
        if new.strip() in text:
            return text, False
        raise RuntimeError(f"{label}: expected anchor not found in {path}")
    return text.replace(old, new, 1), True


def patch_online_stream_h(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    anchor = "  // Generic per-stream option mechanism (key-value string pairs)."
    new_text, changed = _replace_once(text, anchor, ONLINE_STREAM_H_SNIPPET, path, "online-stream.h")
    if changed:
        path.write_text(new_text, encoding="utf-8", newline="\n")
    return changed


def patch_online_stream_cc_from_template(path: Path, template: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if PREPROCESSOR_MARKER in text:
        return False
    if not template.is_file():
        raise RuntimeError(f"Missing template: {template}")
    shutil.copyfile(template, path)
    return True


def patch_nemo_impl(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if NEMO_IMPL_MARKER in text:
        return False
    old = """    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;"""
    if old not in text:
        raise RuntimeError(f"DecodeStreams anchor not found in {path}")
    text = text.replace(old, NEMO_IMPL_DECODE, 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sherpa_src",
        type=Path,
        help="Path to sherpa-onnx git clone (vendor/sherpa-onnx)",
    )
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    template = script_dir / "sherpa_patches" / "online-stream.nemo.cc"
    root = args.sherpa_src / "sherpa-onnx" / "csrc"
    files = {
        "online-stream.h": patch_online_stream_h(root / "online-stream.h"),
        "online-stream.cc": patch_online_stream_cc_from_template(
            root / "online-stream.cc", template
        ),
        "online-recognizer-transducer-nemo-impl.h": patch_nemo_impl(
            root / "online-recognizer-transducer-nemo-impl.h"
        ),
    }
    cc_text = (root / "online-stream.cc").read_text(encoding="utf-8")
    if MARKER in cc_text and FULL_STFT_MARKER not in cc_text:
        print(
            "ERROR: online-stream.cc has old mel-ring patch; re-run after git checkout",
            file=sys.stderr,
        )
        return 1

    any_changed = any(files.values())
    for name, changed in files.items():
        status = "patched" if changed else "already patched"
        print(f"  {name}: {status}")
    if any_changed:
        print("NeMo online full-STFT + per_feature patch applied (15s audio window).")
    else:
        print("NeMo online full-STFT + per_feature patch already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
