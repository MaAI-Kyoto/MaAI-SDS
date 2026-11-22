"""ローカルモデルで完結する ASR エンジン（faster-whisper / sherpa-onnx）。

`asr/service.ASRService` に mixin する。faster-whisper はローカル Path →
HF 名 → サイズ名の順で読む。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import engine_manager, paths as _paths
from ..utils import to_float_or_none


class AsrLocalEnginesMixin:
    """ローカルモデルで完結する ASR エンジン（faster-whisper / sherpa-onnx）。 `ASRService` に mixin する。"""

    def _faster_whisper_model_ref(self) -> str:
        """ローカル Path → HF 名 → サイズ名の順。Path は実在するときだけ使う。"""
        path = str(self.whisper_model_path or "").strip()
        hf = str(getattr(self, "whisper_hf_id", "") or "").strip()
        name = str(self.whisper_model_name or "").strip()
        if path:
            candidate = Path(path)
            if not candidate.is_absolute():
                try:
                    resolved = _paths.resolve_under_repo(candidate)
                    if resolved.exists():
                        return str(resolved)
                except Exception:
                    pass
            elif candidate.exists():
                return path
            # 旧設定: Path 欄に HF repo ID が入っている
            if "/" in path.replace("\\", "/") and not candidate.is_absolute():
                hf = hf or path
        if hf:
            return hf
        if name:
            return name
        return "deepdml/faster-whisper-large-v3-turbo-ct2"

    def _init_engine_faster_whisper(self) -> None:
        engine_manager.require_module_installed("faster_whisper", "faster-whisper")
        try:
            from faster_whisper import WhisperModel
        except Exception as ex:
            raise RuntimeError("faster-whisper をインポートできません。依存関係を確認してください。") from ex
        model_ref = self._faster_whisper_model_ref()
        self._log(
            f"faster-whisper: loading {model_ref} "
            f"device={self.whisper_device} compute_type={self.whisper_compute_type}"
        )
        self._faster_whisper_model = WhisperModel(
            model_size_or_path=model_ref,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
        )
        self._sherpa_recognizer = None

    def _decode_with_faster_whisper(
        self,
        audio_segment: np.ndarray,
        start_ts: float | None,
        end_ts: float | None,
        *,
        is_final: bool,
        is_streaming: bool,
    ) -> bool:
        if self._faster_whisper_model is None:
            raise RuntimeError("faster-whisper モデルが初期化されていません。")
        normalized = np.asarray(audio_segment, dtype=np.float32).flatten()
        if not normalized.flags["C_CONTIGUOUS"]:
            normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        # base_start/base_end は無音付与前の元長で算出する（絶対時刻の対応を保つ）。
        base_start, base_end = self._resolve_segment_bounds(start_ts, end_ts, len(normalized))
        original_sample_count = int(len(normalized))
        # sherpa/qwen3-onnx と同様、先頭に 0.5 秒無音を付与して先頭を安定化させる。
        # whisper が返す相対タイムスタンプはこの無音分シフトするので、後段で差し引く。
        lead_sec = 0.5
        lead = np.zeros(int(self.sample_rate * lead_sec), dtype=np.float32)
        normalized = np.concatenate([lead, normalized])

        t0 = time.perf_counter()
        segments_iter, _ = self._faster_whisper_model.transcribe(
            normalized,
            language="ja",
            without_timestamps=False,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segments = list(segments_iter)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        audio_sec = original_sample_count / float(max(1, self.sample_rate))
        rtf = (decode_ms / 1000.0) / max(audio_sec, 1e-3)
        self._log(
            f"faster-whisper decode audio={audio_sec:.2f}s "
            f"in {decode_ms:.0f}ms rtf={rtf:.2f} "
            f"{'final' if is_final else 'partial'}"
        )
        emitted = 0
        for segment in segments:
            text = str(segment.text).strip()
            if not text:
                continue
            rel_start = to_float_or_none(getattr(segment, "start", None))
            rel_end = to_float_or_none(getattr(segment, "end", None))
            if rel_start is None:
                rel_start = 0.0
            if rel_end is None:
                rel_end = rel_start
            # 先頭に付与した lead_sec 分を差し引いて元音声基準の相対時刻へ戻す。
            rel_start = max(0.0, float(rel_start) - lead_sec)
            rel_end = max(float(rel_start), float(rel_end) - lead_sec)
            abs_start = float(base_start) + rel_start
            abs_end = float(base_start) + rel_end
            abs_end = min(abs_end, float(base_end))
            if abs_end < abs_start:
                abs_end = abs_start
            self._publish_result(
                text=text,
                end_ts=abs_end,
                engine="faster-whisper",
                is_final=is_final,
                is_streaming=is_streaming,
                segment_start_ts=abs_start,
                segment_end_ts=abs_end,
            )
            emitted += 1

        if emitted == 0:
            fallback_text = " ".join(str(getattr(s, "text", "")).strip() for s in segments).strip()
            if fallback_text:
                self._publish_result(
                    text=fallback_text,
                    end_ts=float(base_end),
                    engine="faster-whisper",
                    is_final=is_final,
                    is_streaming=is_streaming,
                    segment_start_ts=float(base_start),
                    segment_end_ts=float(base_end),
                )
                emitted = 1
        return bool(emitted > 0)

    def _init_engine_sherpa_onnx(self) -> None:
        engine_manager.require_module_installed("sherpa_onnx", "sherpa-onnx")
        try:
            import sherpa_onnx
        except Exception as ex:
            raise RuntimeError(
                f"sherpa-onnx をインポートできません: {ex}"
            ) from ex
        required_paths = [
            ("tokens", self.sherpa_tokens_path),
            ("encoder", self.sherpa_encoder_path),
            ("decoder", self.sherpa_decoder_path),
            ("joiner", self.sherpa_joiner_path),
        ]
        missing = [name for name, path in required_paths if not path or not Path(path).exists()]
        if missing:
            details = "; ".join(f"{name}={path}" for name, path in required_paths)
            raise RuntimeError(
                f"sherpa-onnx モデルファイルが不足しています: {', '.join(missing)} ({details})"
            )
        if self.sherpa_provider == "cuda" and "int8" in f"{self.sherpa_encoder_path}".lower():
            self._log(
                "sherpa-onnx: int8 encoder を GPU に載せても速くなりません"
                "（dynamic 量子化に CUDA カーネルが無いため。実測 CPU 122ms 対 GPU 203ms）。"
                " GPU を使うなら encoder/joiner を fp32（*-epoch-99-avg-1.onnx）に変えてください。"
            )
        self._sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self.sherpa_encoder_path,
            decoder=self.sherpa_decoder_path,
            joiner=self.sherpa_joiner_path,
            tokens=self.sherpa_tokens_path,
            num_threads=self.sherpa_num_threads,
            sample_rate=self.sherpa_sample_rate,
            feature_dim=self.sherpa_feature_dim,
            decoding_method=self.sherpa_decoding_method,
            provider=self.sherpa_provider,
        )
        self._faster_whisper_model = None
        self._sherpa_needs_lead_silence = True

    def _decode_with_sherpa(
        self,
        audio_segment: np.ndarray,
        start_ts: float | None,
        end_ts: float | None,
        *,
        is_final: bool,
        is_streaming: bool,
    ) -> bool:
        if self._sherpa_recognizer is None:
            raise RuntimeError("sherpa-onnx recognizer が初期化されていません。")

        normalized = np.asarray(audio_segment, dtype=np.float32).flatten()
        if not normalized.flags["C_CONTIGUOUS"]:
            normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        original_sample_count = int(len(normalized))
        # zipformer の先頭安定化用。オフラインで発話全体を再推論するエンジン
        # （qwen3 / parakeet / sensevoice / funasr-nano / cohere）は、疑似ストリーミングの
        # たびにこの無音ぶんも再推論することになるので付けない。判定は各 _init_engine_* が
        # 立てる `_sherpa_needs_lead_silence` で行う。
        if self._sherpa_needs_lead_silence:
            silence = np.zeros(int(self.sherpa_sample_rate * 0.5), dtype=np.float32)
            normalized = np.concatenate([silence, normalized])
        start_v, end_v = self._resolve_segment_bounds(start_ts, end_ts, original_sample_count)

        t0 = time.perf_counter()
        stream = self._sherpa_recognizer.create_stream()
        if self.asr_engine == "qwen3-asr-onnx" and self.qwen3_onnx_language:
            # 言語を強制（フル名 "Japanese" 等）。空だと自動判定で中国語等に流れることがある。
            try:
                stream.set_option("language", self.qwen3_onnx_language)
            except Exception:  # noqa: BLE001
                pass
        stream.accept_waveform(self.sherpa_sample_rate, normalized)
        self._sherpa_recognizer.decode_stream(stream)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        audio_sec = original_sample_count / float(max(1, self.sherpa_sample_rate))
        text = str(stream.result.text).strip()
        if self.asr_engine == "qwen3-asr-onnx":
            # Qwen3-ASR は "language Japanese<asr_text>..." のメタ接頭辞を返すため除去する。
            text = self._clean_qwen_text(text)
            rtf = (decode_ms / 1000.0) / max(audio_sec, 1e-3)
            self._log(
                f"qwen3-asr-onnx decode audio={audio_sec:.2f}s "
                f"in {decode_ms:.0f}ms rtf={rtf:.2f} "
                f"provider={self.qwen3_onnx_provider} "
                f"{'final' if is_final else 'partial'} chars={len(text)}"
            )
        if text:
            self._publish_result(
                text=text,
                end_ts=end_v,
                engine=self.asr_engine,
                is_final=is_final,
                is_streaming=is_streaming,
                segment_start_ts=start_v,
                segment_end_ts=end_v,
            )
            return True
        return False
