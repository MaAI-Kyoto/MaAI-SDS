"""`speechcpp-asr`：speech.cpp（ggml）で動く ASR。Haas-ja・ReazonSpeech NeMo v2 など。

speech.cpp の `speech worker` に発話全体を投げて全文を受け取る、sherpa-onnx の
OfflineRecognizer と同じ使い方になる。そこで sherpa の認識器と同じ形
（`create_stream` / `decode_stream` / `stream.result.text`）の薄い包みを
`_sherpa_recognizer` に入れ、区間検出・疑似ストリーミング・結果の扱いは
`_decode_with_sherpa` をそのまま共有する（エンジン名は `SHERPA_DECODE_ENGINES` に入れてある）。

実測（RTX 5070 Ti Laptop・Vulkan、1 発話の中央値）：Haas-ja 30 ms・ReazonSpeech NeMo v2 61 ms、
読み込み 0.6 秒、GPU メモリ 0.7〜0.8 GB。CPU でも Haas-ja は 180〜310 ms で動く
（0.25 秒ごとの疑似ストリーミングには遅いので、CPU では発話の終わりに 1 回だけ認識する使い方になる）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import speechcpp


@dataclass
class _Result:
    text: str = ""


@dataclass
class _Stream:
    sample_rate: int = 16000
    chunks: list[np.ndarray] = field(default_factory=list)
    result: _Result = field(default_factory=_Result)

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.sample_rate = int(sample_rate)
        self.chunks.append(np.asarray(samples, dtype=np.float32).reshape(-1))

    def set_option(self, _key: str, _value: str) -> None:  # sherpa の stream と同じ形にするだけ
        pass


class SpeechCppRecognizer:
    """sherpa-onnx の OfflineRecognizer と同じ呼び方で speech.cpp に認識させる。"""

    def __init__(self, worker: speechcpp.SpeechCppWorker, language: str = "") -> None:
        self.worker = worker
        self.language = language

    def create_stream(self) -> _Stream:
        return _Stream()

    def decode_stream(self, stream: _Stream) -> None:
        audio = np.concatenate(stream.chunks) if stream.chunks else np.zeros(0, dtype=np.float32)
        if audio.size == 0:
            stream.result.text = ""
            return
        options = {"language": self.language} if self.language else {}
        stream.result.text = self.worker.transcribe(audio, stream.sample_rate, **options)

    def close(self) -> None:
        self.worker.close()


def language_for(model: str, configured: str) -> str:
    """多言語のモデル（Qwen3-ASR）だけ言語を渡す。日本語専用のモデルは言語の指定を受け付けない。"""
    if configured.strip():
        return configured.strip()
    return "ja" if model.startswith("qwen3-asr") else ""


class AsrSpeechCppEngineMixin:
    """`speechcpp-asr` の読み込み。`ASRService` に mixin する。"""

    def _init_engine_speechcpp_asr(self) -> None:
        model = (self.speechcpp_asr_model or speechcpp.DEFAULT_ASR_MODEL).strip()
        self._close_speechcpp_asr()
        worker = speechcpp.SpeechCppWorker(
            model,
            device=self.speechcpp_asr_device,
            threads=self.speechcpp_asr_threads,
            exe=self.speechcpp_exe,
            on_log=self._log,
        ).start()
        self._sherpa_recognizer = SpeechCppRecognizer(worker, language_for(model, self.speechcpp_asr_language))
        self._speechcpp_asr_worker = worker
        self._faster_whisper_model = None
        self._nemotron = None
        self._nemotron_onnx_recognizer = None
        # 発話全体を疑似ストリーミングのたびに認識し直すので、先頭の無音は足さない
        # （sherpa のオフライン系と同じ理由）。
        self._sherpa_needs_lead_silence = False

    def _close_speechcpp_asr(self) -> None:
        worker = getattr(self, "_speechcpp_asr_worker", None)
        self._speechcpp_asr_worker = None
        if worker is not None:
            worker.close()
