"""sherpa-onnx OnlineRecognizer 向け Nemotron ストリーミングセッション（feed/reset/finalize）。"""
from __future__ import annotations

from typing import Callable

import numpy as np


class NemotronOnnxStreamingSession:
    """nemotron-asr の StreamingSession と同じ feed/reset/finalize インターフェース。"""

    def __init__(
        self,
        recognizer,
        *,
        sample_rate: int = 16000,
        apply_language: Callable[[object], None] | None = None,
        finalize_tail_sec: float = 0.41,
    ) -> None:
        self.rec = recognizer
        self.sample_rate = int(sample_rate)
        self._apply_language = apply_language
        # Nemotron chunk 向け: NeMo finalize pad = window_size mel frames * 10ms（既定 ~0.41s @320ms）
        self._finalize_tail_sec = max(0.0, float(finalize_tail_sec))
        self.current_text = ""
        self._stream = None
        self.reset()

    def reset(self) -> None:
        self._stream = self.rec.create_stream()
        if self._apply_language is not None:
            self._apply_language(self._stream)
        self.current_text = ""

    def feed(self, audio_f32: np.ndarray) -> str:
        x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if x.size <= 0:
            return self.current_text
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x, dtype=np.float32)
        self._stream.accept_waveform(self.sample_rate, x)
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)
        self.current_text = str(self.rec.get_result(self._stream)).strip()
        return self.current_text

    def finalize(self) -> str:
        if self._finalize_tail_sec > 0.0:
            tail = np.zeros(int(self.sample_rate * self._finalize_tail_sec), dtype=np.float32)
            self._stream.accept_waveform(self.sample_rate, tail)
        self._stream.input_finished()
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)
        self.current_text = str(self.rec.get_result(self._stream)).strip()
        return self.current_text
