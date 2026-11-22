"""Patch sherpa-onnx for MaAI-SDS NeMo preprocessor ONNX window-mel path.

1. features.h: nemo_preprocessor_model field on FeatureExtractorConfig
2. online-recognizer-transducer-nemo-impl.h: auto-detect preprocessor.onnx + DecodeStreams
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "nemo_preprocessor_model"
PREPROCESSOR_IMPL_MARKER = "use_nemo_window_mel"
POSTINIT_MARKER = "MaAI-SDS NeMo preprocessor auto-detect"

FEATURES_H_OLD = """  std::string nemo_normalize_type;

  // for MFCC
  int32_t num_ceps = 13;"""

FEATURES_H_NEW = """  std::string nemo_normalize_type;

  // MaAI-SDS: optional NeMo preprocessor ONNX for 15s window mel recompute.
  // Set automatically from encoder dir when preprocessor.onnx is present.
  std::string nemo_preprocessor_model;

  // for MFCC
  int32_t num_ceps = 13;"""

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

OLD_NEMO_IMPL_DECODE_PER_FEATURE = """    const std::string &normalize_type = config_.feat_config.nemo_normalize_type;

    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> raw_features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      std::vector<float> features =
          normalize_type == "per_feature"
              ? ss[i]->GetNeMoNormalizedFeatures(num_processed_frames,
                                                 chunk_size, raw_features,
                                                 normalize_type)
              : std::move(raw_features);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;
"""

OLD_NEMO_IMPL_DECODE_RAW = """    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;
"""

POSTINIT_OLD = """    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    int32_t vocab_size = model_->VocabSize();"""

POSTINIT_NEW = """    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    // MaAI-SDS NeMo preprocessor auto-detect (preprocessor.onnx beside encoder)
    {
      std::string encoder =
          ResolveAbsolutePath(config_.model_config.transducer.encoder);
      auto pos = encoder.find_last_of("/\\\\");
      if (pos != std::string::npos) {
        std::string preproc =
            ResolveAbsolutePath(encoder.substr(0, pos + 1) + "preprocessor.onnx");
        if (FileExists(preproc)) {
          config_.feat_config.nemo_preprocessor_model = preproc;
        }
      }
    }

    int32_t vocab_size = model_->VocabSize();"""


def patch_features_h(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    if FEATURES_H_OLD not in text:
        raise RuntimeError(f"features.h anchor not found: {path}")
    path.write_text(text.replace(FEATURES_H_OLD, FEATURES_H_NEW, 1), encoding="utf-8", newline="\n")
    return True


def patch_nemo_impl(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    if PREPROCESSOR_IMPL_MARKER not in text:
        if OLD_NEMO_IMPL_DECODE_PER_FEATURE in text:
            text = text.replace(OLD_NEMO_IMPL_DECODE_PER_FEATURE, NEMO_IMPL_DECODE, 1)
            changed = True
        elif OLD_NEMO_IMPL_DECODE_RAW in text:
            text = text.replace(OLD_NEMO_IMPL_DECODE_RAW, NEMO_IMPL_DECODE, 1)
            changed = True
        else:
            raise RuntimeError(f"DecodeStreams anchor not found: {path}")

    if POSTINIT_MARKER not in text:
        if POSTINIT_OLD not in text:
            raise RuntimeError(f"PostInit anchor not found: {path}")
        text = text.replace(POSTINIT_OLD, POSTINIT_NEW, 1)
        changed = True

    if "file-utils.h" not in text:
        anchor = '#include "sherpa-onnx/csrc/macros.h"'
        if anchor not in text:
            raise RuntimeError(f"include anchor not found: {path}")
        text = text.replace(
            anchor,
            anchor + '\n#include "sherpa-onnx/csrc/file-utils.h"',
            1,
        )
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8", newline="\n")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sherpa_src", type=Path, help="vendor/sherpa-onnx clone root")
    args = parser.parse_args()
    root = args.sherpa_src / "sherpa-onnx" / "csrc"
    files = {
        "features.h": patch_features_h(root / "features.h"),
        "online-recognizer-transducer-nemo-impl.h": patch_nemo_impl(
            root / "online-recognizer-transducer-nemo-impl.h"
        ),
    }
    for name, changed in files.items():
        print(f"  {name}: {'patched' if changed else 'already patched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
