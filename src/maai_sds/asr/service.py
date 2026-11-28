from __future__ import annotations

import math
import os
import queue
import socket
import threading
import time
import json
import re
import base64
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


from .. import engine_manager, paths as _paths
from .frame_types import AudioFrame, SegmentLoopHooks
from .engines_chirp3 import AsrChirp3EngineMixin
from .engines_external import AsrExternalEngineMixin
from .engines_local import AsrLocalEnginesMixin
from .engines_nemotron import AsrNemotronEnginesMixin
from .engines_qwen import AsrQwenEnginesMixin
from .engines_sherpa_offline import AsrSherpaOfflineEnginesMixin
from .engines_speechcpp import AsrSpeechCppEngineMixin
from .vad_turn import VadTurnMixin
from ..utils import to_float_or_none

# `_decode_with_sherpa` で復号するエンジン。どれも sherpa-onnx の OfflineRecognizer で、
# VAD が切り出した発話全体を 1 回投げて全文を受け取る（＝ VAD ループ＋推論ループの 2 本立て）。
# エンジン名の列挙が `_build_engine_threads` と `_decode_segment` に分かれていたせいで、
# 追加時に前者だけ漏れて「未対応のASRエンジンです」で落ちた。以後はここが唯一の定義。
SHERPA_DECODE_ENGINES = frozenset({
    "sherpa-onnx",
    "qwen3-asr-onnx",
    "parakeet-onnx",
    "sensevoice-onnx",
    "funasr-nano-onnx",
    "cohere-asr-onnx",
    # speech.cpp も発話全体を 1 回投げて全文を受け取るので、同じ復号を使う（engines_speechcpp.py）
    "speechcpp-asr",
})


@dataclass
class ASRResult:
    text: str
    engine: str
    is_final: bool
    is_streaming: bool
    latency_sec: float
    utterance_index: int = 0
    end_of_utterance_ts: float | None = None
    segment_start_ts: float | None = None
    segment_end_ts: float | None = None


class ASRService(
    AsrChirp3EngineMixin,
    AsrExternalEngineMixin,
    AsrLocalEnginesMixin,
    AsrNemotronEnginesMixin,
    AsrQwenEnginesMixin,
    AsrSherpaOfflineEnginesMixin,
    AsrSpeechCppEngineMixin,
    VadTurnMixin,
):
    QUEUE_POLICY_DROP_OLDEST = "drop_oldest"
    QUEUE_POLICY_DROP_NEWEST = "drop_newest"
    QUEUE_POLICY_BLOCK = "block"

    def __init__(
        self,
        asr_engine: str = "sherpa-onnx",
        vad_backend: str = "maai",
        turn_backend: str = "maai",
        sample_rate: int = 16000,
        frame_rate: float = 20.0,
        # Shared VAD parameters
        vad_start_threshold: float = 0.1,
        vad_start_hold_sec: float = 0.1,
        vad_end_threshold: float = 0.2,
        vad_end_hold_sec: float = 0.1,
        min_segment_sec: float = 0.5,
        pre_roll_buffer_sec: float = 1.0,
        nonstream_pseudo_enabled: bool = True,
        nonstream_pseudo_chunk_sec: float = 0.25,
        inference_queue_size: int = 4,
        # faster-whisper
        whisper_model_name: str = "large-v3",
        whisper_hf_id: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
        whisper_model_path: str = "",
        whisper_use_local_path: bool = False,
        whisper_device: str = "cuda",
        whisper_compute_type: str = "int8_float16",
        # sherpa-onnx
        sherpa_tokens_path: str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt",
        sherpa_encoder_path: str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.int8.onnx",
        sherpa_decoder_path: str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.onnx",
        sherpa_joiner_path: str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.int8.onnx",
        sherpa_num_threads: int = 2,
        # zipformer の実行デバイス。既定 cpu は現状維持で、GPU は明示選択したときだけ。
        # **int8 のまま GPU にしても遅くなる**（実測 122ms→203ms。dynamic 量子化に CUDA
        # カーネルが無いため）。GPU を使うなら encoder/joiner のパスも fp32
        # （`*-epoch-99-avg-1.onnx`）へ変えること。その組み合わせが最速かつ最良で、
        # 実測 102ms / CER 0.1367（int8+CPU 2 スレッドは 152ms / 0.1414）。
        sherpa_provider: str = "cpu",
        sherpa_decoding_method: str = "greedy_search",
        sherpa_sample_rate: int = 16000,
        sherpa_feature_dim: int = 80,
        # qwen3-asr-onnx (sherpa-onnx OfflineRecognizer.from_qwen3_asr)
        # GitHub 公開物は int8 のみ。fp32 は ModelScope の自前エクスポート。
        qwen3_onnx_conv_frontend_path: str = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx",
        qwen3_onnx_encoder_path: str = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx",
        qwen3_onnx_decoder_path: str = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx",
        qwen3_onnx_tokenizer_path: str = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer",
        qwen3_onnx_num_threads: int = 8,
        qwen3_onnx_feature_dim: int = 128,
        qwen3_onnx_provider: str = "cpu",
        qwen3_onnx_language: str = "",
        # --- sherpa-onnx OfflineRecognizer 系（engines_sherpa_offline.py）---
        # k2-fsa の release tarball を展開したディレクトリを指す。中のファイル名は
        # engines_sherpa_offline._pick が解決するので、int8/fp32 を入れ替えても設定は同じ。
        # provider が 4 つとも cpu なのは実測（int8 は CUDA カーネルが効かない）。
        parakeet_onnx_model_dir: str = "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
        parakeet_onnx_num_threads: int = 8,
        parakeet_onnx_provider: str = "cpu",
        # 2024-07-17 は本家 FunAudioLLM/SenseVoice からの変換。同名の 2025-09-09 版は
        # ASLP-lab/WSYue-ASR の**広東語ファインチューン**で、日本語を簡体字で書き起こす
        # （実測: 同梱 ja.wav すら「便当制」等になり CER 0.79）。取り違えないこと。
        sensevoice_onnx_model_dir: str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        sensevoice_onnx_num_threads: int = 8,
        sensevoice_onnx_provider: str = "cpu",
        sensevoice_onnx_language: str = "ja",
        sensevoice_onnx_use_itn: bool = True,
        funasr_nano_onnx_model_dir: str = "sherpa-onnx-funasr-nano-int8-2025-12-30",
        funasr_nano_onnx_num_threads: int = 8,
        funasr_nano_onnx_provider: str = "cpu",
        funasr_nano_onnx_language: str = "ja",
        funasr_nano_onnx_use_itn: bool = True,
        funasr_nano_onnx_max_new_tokens: int = 128,
        cohere_onnx_model_dir: str = "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
        cohere_onnx_num_threads: int = 8,
        cohere_onnx_provider: str = "cpu",
        cohere_onnx_language: str = "ja",
        cohere_onnx_use_punct: bool = True,
        cohere_onnx_use_itn: bool = True,
        # --- speech.cpp（engines_speechcpp.py）---
        # モデルは `speech models` の名前。device は gpu（外付けの GPU、無ければ CPU）/ cpu / デバイス名。
        speechcpp_asr_model: str = "haas-ja-v1.0",
        speechcpp_asr_device: str = "gpu",
        speechcpp_asr_threads: int = 0,
        speechcpp_asr_language: str = "",
        speechcpp_exe: str = "",
        # nemotron-onnx (sherpa-onnx OnlineRecognizer.from_transducer, multilingual streaming)
        nemotron_onnx_tokens_path: str = "",
        nemotron_onnx_encoder_path: str = "",
        nemotron_onnx_decoder_path: str = "",
        nemotron_onnx_joiner_path: str = "",
        nemotron_onnx_num_threads: int = 2,
        nemotron_onnx_feature_dim: int = 128,
        nemotron_onnx_language: str = "",
        nemotron_onnx_provider: str = "cpu",
        nemotron_onnx_quant: str = "fp32",
        # qwen3-asr (OpenAI-compatible endpoint)
        qwen_host: str = "127.0.0.1",
        qwen_port: int = 8000,
        qwen_model: str = "Qwen3-ASR-0.6B",
        qwen_language: str = "ja",
        qwen_chunk_sec: float = 2.0,
        qwen_timeout_sec: float = 30.0,
        qwen_transport: str = "ws",
        qwen_ws_path: str = "/ws",
        qwen_session_timeout_sec: float = 60.0,
        qwen_no_text_flush_sec: float = 0.0,
        qwen_min_return_audio_sec: float = 0.0,
        qwen_finish_on_vad_end: bool = True,
        # external-asr-tcp
        external_asr_host: str = "127.0.0.1",
        external_asr_port: int = 19090,
        external_asr_audio_host: str = "127.0.0.1",
        external_asr_audio_port: int = 14600,
        # nemotron streaming asr (NVIDIA Nemotron 3.5 ASR, cache-aware, in-process)
        nemotron_model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        nemotron_chunk_size_ms: int = 320,
        nemotron_window_sec: float = 15.0,
        nemotron_max_symbols: int = 30,
        nemotron_lang: str = "ja-JP",
        nemotron_device: str = "cuda",
        nemotron_finalize_mode: str = "vad",
        nemotron_python: str = "",
        nemotron_remote_host: str = "",
        nemotron_remote_port: int = 0,
        # chirp3 streaming asr (Google Cloud Speech-to-Text V2)
        chirp_project_id: str = "",
        chirp_location: str = "global",
        chirp_recognizer: str = "_",
        chirp_language_codes: str = "ja-JP",
        chirp_alternative_language_code: str = "",
        chirp_model: str = "chirp_3",
        # smart-turn
        smart_turn_model_path: str = "smart-turn-v3.2-gpu.onnx",
        smart_turn_threshold: float = 0.5,
        smart_turn_num_threads: int = 2,
        torch_cpu_intra_threads: int = 2,
        # MaAI VAP（vad_backend / turn_backend が maai のとき Maai(...) に渡す）
        maai_vap_mode: str = "vap_mc",
        maai_vap_lang: str = "jp_kyoto",
        maai_vap_device: str = "cuda",
        maai_vap_model_type: str = "normal",
        # maai output
        maai_output_gui_plot: bool = False,
        # callbacks
        on_log: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_result: Callable[[ASRResult], None] | None = None,
        on_turn_probability: Callable[[float, float], None] | None = None,
        on_maai_result: Callable[[dict[str, Any]], None] | None = None,
        on_inference_state: Callable[[bool], None] | None = None,
        on_force_finalize_outcome: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = False,
        # shared input
        external_raw_audio_queue: queue.Queue[AudioFrame] | None = None,
    ) -> None:
        self.asr_engine = (asr_engine or "sherpa-onnx").strip().lower()
        self.vad_backend = (vad_backend or "maai").strip().lower()
        self.turn_backend = (turn_backend or "maai").strip().lower()
        self.sample_rate = int(sample_rate)
        try:
            self.frame_rate = float(frame_rate)
        except Exception:
            self.frame_rate = 20.0
        if (not math.isfinite(self.frame_rate)) or self.frame_rate <= 0.0:
            self.frame_rate = 20.0
        self.vad_start_threshold = float(vad_start_threshold)
        self.vad_start_hold_sec = float(vad_start_hold_sec)
        self.vad_end_threshold = float(vad_end_threshold)
        self.vad_end_hold_sec = float(vad_end_hold_sec)
        self.min_segment_sec = float(min_segment_sec)
        self.pre_roll_buffer_sec = max(0.0, float(pre_roll_buffer_sec))
        self.nonstream_pseudo_enabled = bool(nonstream_pseudo_enabled)
        self.nonstream_pseudo_chunk_sec = max(0.0, float(nonstream_pseudo_chunk_sec))
        # 擬似ストリーミングの「発話開始直後の 1 発目」は chunk と同じ長さで出す
        # （かつて bootstrap_sec という独立ノブがあったが、UI 仕様の単純化で
        #  chunk に統合された。引数も属性も使われていなかったので削除した）。
        self.verbose = bool(verbose)

        self.whisper_model_name = whisper_model_name
        self.whisper_hf_id = whisper_hf_id
        self.whisper_model_path = whisper_model_path
        self.whisper_use_local_path = whisper_use_local_path
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type

        self.sherpa_tokens_path = self._resolve_model_path(sherpa_tokens_path) or sherpa_tokens_path
        self.sherpa_encoder_path = self._resolve_model_path(sherpa_encoder_path) or sherpa_encoder_path
        self.sherpa_decoder_path = self._resolve_model_path(sherpa_decoder_path) or sherpa_decoder_path
        self.sherpa_joiner_path = self._resolve_model_path(sherpa_joiner_path) or sherpa_joiner_path
        self.sherpa_num_threads = int(sherpa_num_threads)
        self.sherpa_provider = (sherpa_provider or "cpu").strip() or "cpu"
        self.sherpa_decoding_method = sherpa_decoding_method
        self.sherpa_sample_rate = int(sherpa_sample_rate)
        self.sherpa_feature_dim = int(sherpa_feature_dim)
        # qwen3-asr-onnx
        self.qwen3_onnx_conv_frontend_path = self._resolve_model_path(qwen3_onnx_conv_frontend_path)
        self.qwen3_onnx_encoder_path = self._resolve_model_path(qwen3_onnx_encoder_path)
        self.qwen3_onnx_decoder_path = self._resolve_model_path(qwen3_onnx_decoder_path)
        self.qwen3_onnx_tokenizer_path = self._resolve_model_path(qwen3_onnx_tokenizer_path)
        self.qwen3_onnx_num_threads = max(1, min(int(qwen3_onnx_num_threads), 32))
        self.qwen3_onnx_feature_dim = int(qwen3_onnx_feature_dim)
        self.qwen3_onnx_provider = (qwen3_onnx_provider or "cpu").strip() or "cpu"
        # Qwen3-ASR の言語強制（フル名: "Japanese"/"English"/"Chinese" 等。空=自動判定）
        self.qwen3_onnx_language = (qwen3_onnx_language or "").strip()
        # sherpa-onnx OfflineRecognizer 系（ディレクトリの実在確認は各 _init_engine_* 側。
        # モデル未取得のエンジンを選んでいなくても ASRService は作れる必要があるため）
        self.parakeet_onnx_model_dir = self._resolve_model_path(parakeet_onnx_model_dir)
        self.parakeet_onnx_num_threads = max(1, min(int(parakeet_onnx_num_threads), 32))
        self.parakeet_onnx_provider = (parakeet_onnx_provider or "cpu").strip() or "cpu"
        self.sensevoice_onnx_model_dir = self._resolve_model_path(sensevoice_onnx_model_dir)
        self.sensevoice_onnx_num_threads = max(1, min(int(sensevoice_onnx_num_threads), 32))
        self.sensevoice_onnx_provider = (sensevoice_onnx_provider or "cpu").strip() or "cpu"
        self.sensevoice_onnx_language = (sensevoice_onnx_language or "").strip()
        self.sensevoice_onnx_use_itn = bool(sensevoice_onnx_use_itn)
        self.funasr_nano_onnx_model_dir = self._resolve_model_path(funasr_nano_onnx_model_dir)
        self.funasr_nano_onnx_num_threads = max(1, min(int(funasr_nano_onnx_num_threads), 32))
        self.funasr_nano_onnx_provider = (funasr_nano_onnx_provider or "cpu").strip() or "cpu"
        self.funasr_nano_onnx_language = (funasr_nano_onnx_language or "").strip()
        self.funasr_nano_onnx_use_itn = bool(funasr_nano_onnx_use_itn)
        # 自己回帰デコーダの最悪遅延を縛る上限（1 発話ぶんに必要な長さで足りる）
        self.funasr_nano_onnx_max_new_tokens = max(16, min(int(funasr_nano_onnx_max_new_tokens), 2048))
        self.cohere_onnx_model_dir = self._resolve_model_path(cohere_onnx_model_dir)
        self.cohere_onnx_num_threads = max(1, min(int(cohere_onnx_num_threads), 32))
        self.cohere_onnx_provider = (cohere_onnx_provider or "cpu").strip() or "cpu"
        self.cohere_onnx_language = (cohere_onnx_language or "").strip()
        self.cohere_onnx_use_punct = bool(cohere_onnx_use_punct)
        self.cohere_onnx_use_itn = bool(cohere_onnx_use_itn)
        self.speechcpp_asr_model = (speechcpp_asr_model or "").strip()
        self.speechcpp_asr_device = (speechcpp_asr_device or "gpu").strip() or "gpu"
        self.speechcpp_asr_threads = max(0, min(int(speechcpp_asr_threads or 0), 64))
        self.speechcpp_asr_language = (speechcpp_asr_language or "").strip()
        self.speechcpp_exe = (speechcpp_exe or "").strip()
        self._speechcpp_asr_worker = None
        # nemotron-onnx（相対パスはリポジトリルート基準で解決）
        self.nemotron_onnx_tokens_path = self._resolve_model_path(nemotron_onnx_tokens_path)
        self.nemotron_onnx_encoder_path = self._resolve_model_path(nemotron_onnx_encoder_path)
        self.nemotron_onnx_decoder_path = self._resolve_model_path(nemotron_onnx_decoder_path)
        self.nemotron_onnx_joiner_path = self._resolve_model_path(nemotron_onnx_joiner_path)
        self.nemotron_onnx_num_threads = int(nemotron_onnx_num_threads)
        self.nemotron_onnx_feature_dim = int(nemotron_onnx_feature_dim)
        self.nemotron_onnx_language = (nemotron_onnx_language or "").strip()
        self.nemotron_onnx_provider = (nemotron_onnx_provider or "cpu").strip() or "cpu"
        self.nemotron_onnx_quant = (nemotron_onnx_quant or "fp32").strip().lower() or "fp32"
        self._nemotron_onnx_recognizer = None
        self._nemotron_onnx_session = None

        self.qwen_host = qwen_host.strip() or "127.0.0.1"
        self.qwen_port = int(qwen_port)
        self.qwen_model = qwen_model.strip() or "Qwen3-ASR-0.6B"
        self.qwen_language = self._normalize_qwen_language(qwen_language)
        self.qwen_chunk_sec = float(qwen_chunk_sec)
        self.qwen_timeout_sec = float(qwen_timeout_sec)
        self.qwen_transport = (qwen_transport or "ws").strip().lower()
        self.qwen_ws_path = self._normalize_qwen_ws_path(qwen_ws_path)
        self.qwen_session_timeout_sec = float(qwen_session_timeout_sec)
        self.qwen_no_text_flush_sec = float(qwen_no_text_flush_sec)
        self.qwen_min_return_audio_sec = max(0.0, float(qwen_min_return_audio_sec))
        self.qwen_finish_on_vad_end = bool(qwen_finish_on_vad_end)
        self.external_asr_host = (external_asr_host or "127.0.0.1").strip() or "127.0.0.1"
        self.external_asr_port = int(external_asr_port)
        self.external_asr_audio_host = (external_asr_audio_host or "127.0.0.1").strip() or "127.0.0.1"
        self.external_asr_audio_port = int(external_asr_audio_port)
        self.chirp_project_id = (chirp_project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
        self.chirp_location = (chirp_location or "global").strip() or "global"
        self.chirp_recognizer = (chirp_recognizer or "_").strip() or "_"
        self.chirp_language_codes = (chirp_language_codes or "ja-JP").strip()
        self.chirp_alternative_language_code = (chirp_alternative_language_code or "").strip()
        self.chirp_model = (chirp_model or "chirp_3").strip() or "chirp_3"

        self.nemotron_model = (nemotron_model or "nvidia/nemotron-3.5-asr-streaming-0.6b").strip() or "nvidia/nemotron-3.5-asr-streaming-0.6b"
        from .nemotron_chunk import normalize_chunk_size_ms

        self.nemotron_chunk_size_ms = normalize_chunk_size_ms(nemotron_chunk_size_ms)
        try:
            self.nemotron_window_sec = float(nemotron_window_sec)
        except Exception:
            self.nemotron_window_sec = 15.0
        try:
            self.nemotron_max_symbols = int(nemotron_max_symbols)
        except Exception:
            self.nemotron_max_symbols = 30
        self.nemotron_lang = (nemotron_lang or "ja-JP").strip() or "ja-JP"
        self.nemotron_device = (nemotron_device or "cuda").strip() or "cuda"
        self.nemotron_finalize_mode = (nemotron_finalize_mode or "vad").strip().lower() or "vad"
        if self.nemotron_finalize_mode not in {"vad", "continuous"}:
            self.nemotron_finalize_mode = "vad"
        self.nemotron_python = (nemotron_python or "").strip()
        self.nemotron_remote_host = (nemotron_remote_host or "").strip()
        self.nemotron_remote_port = int(nemotron_remote_port)

        self.smart_turn_model_path = self._resolve_model_path(smart_turn_model_path) or smart_turn_model_path
        self.smart_turn_threshold = float(smart_turn_threshold)
        self.smart_turn_num_threads = max(1, min(int(smart_turn_num_threads), 32))
        self.torch_cpu_intra_threads = max(1, min(int(torch_cpu_intra_threads), 32))
        self._bound_torch_threads()
        self.maai_vap_mode = str(maai_vap_mode or "vap_mc").strip() or "vap_mc"
        self.maai_vap_lang = str(maai_vap_lang or "jp_kyoto").strip() or "jp_kyoto"
        # CUDA を要求されても、実際に使えないなら cpu へ落とす。maai の
        # load_vap_model は `torch.load(sd, map_location=torch.device(device))` を
        # 呼ぶので、CUDA で保存された .pt に "cuda" を渡すと必ずこうなる:
        #   Attempting to deserialize object on a CUDA device but
        #   torch.cuda.is_available() is False
        # 判定をここでやるのは、このプロセスが torch を読む側だから。GUI 側で見ると
        # GUI プロセスに torch がロードされ、エンジン管理の「適用」が torch の DLL を
        # 置き換えられなくなる（Windows）。
        self.maai_vap_device = self._usable_torch_device(
            str(maai_vap_device or "cuda").strip() or "cuda"
        )
        self.maai_vap_model_type = str(maai_vap_model_type or "normal").strip() or "normal"
        self.maai_output_gui_plot = bool(maai_output_gui_plot)

        self.on_log = on_log
        self.on_error = on_error
        self.on_result = on_result
        self.on_turn_probability = on_turn_probability
        self.on_maai_result = on_maai_result
        self.on_inference_state = on_inference_state
        self.on_force_finalize_outcome = on_force_finalize_outcome

        # audio/input_worker 等は 320 サンプル/パケットで送る。セグメントループは block_size まで蓄積してから1ティック処理する。
        self.ingress_block_size = 320
        self._silero_model_chunk_size = 512
        self._vap_maai_chunk_size = max(
            1, int(round(float(self.sample_rate) / float(max(1e-9, self.frame_rate))))
        )
        # silero: モデル入力 512 に合わせる。maai: VAD のホップ（frame_rate 由来）に合わせる。
        if self.vad_backend == "silero":
            self.block_size = int(self._silero_model_chunk_size)
        else:
            self.block_size = int(self._vap_maai_chunk_size)

        self.raw_audio_queue: queue.Queue[AudioFrame] = (
            external_raw_audio_queue if external_raw_audio_queue is not None else queue.Queue()
        )
        if inference_queue_size > 0:
            self.inference_queue: queue.Queue[dict] = queue.Queue(maxsize=int(inference_queue_size))
        else:
            self.inference_queue = queue.Queue()
        self.transcript_queue: queue.Queue[ASRResult] = queue.Queue()
        self.smart_turn_queue: queue.Queue[dict] = queue.Queue(maxsize=4)

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._maai_result_event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        self._vad_init_lock = threading.Lock()
        self._faster_whisper_model = None
        self._sherpa_recognizer = None
        # zipformer は先頭 0.5 秒の無音で立ち上がりが安定する。発話全体を毎回
        # 再推論するオフライン系（qwen3 / parakeet / sensevoice / funasr-nano / cohere）は
        # その無音ぶんだけ毎回重くなるだけなので、各 _init_engine_* が False にする。
        self._sherpa_needs_lead_silence = True
        self._nemotron = None
        self._nemotron_proc = None
        self._nemotron_log_fp = None
        self._qwen_ws_endpoint = ""
        self._silero_model = None
        self._latest_silero_vad_score: float | None = None
        self._maai_instance = None
        self._maai_mic_chunk = None
        self._maai_zero_chunk = None
        self._smart_turn_session = None
        self._smart_turn_feature_extractor = None
        self._external_server_socket: socket.socket | None = None
        self._external_audio_socket: socket.socket | None = None
        self.turn_audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=32)
        self._silero_input_buffer = np.empty(0, dtype=np.float32)
        self._vap_maai_vad_buffer = np.empty(0, dtype=np.float32)
        self._external_segment_lock = threading.Lock()
        self._external_active_segment_start_ts: float | None = None
        self._external_last_segment_start_ts: float | None = None
        self._external_last_segment_end_ts: float | None = None
        self._utterance_index_counter = 0
        self._active_utterance_index: int | None = None
        self._system_audio_buffer_vad: deque[float] = deque()
        self._system_audio_lock = threading.Lock()
        self._force_finalize_event = threading.Event()
        self._force_finalize_state_lock = threading.Lock()
        self._force_finalize_speaking = False
        self._force_finalize_pending_request_id: int | None = None
        self._inference_busy = False
        self._inference_state_lock = threading.Lock()
        self._sample_clock_lock = threading.Lock()
        self._sample_clock_base_ts: float | None = None
        # フレームの受信記録。(音声時刻 ts_end, 送出側の wall 時刻)。
        # latency = now - 送出 wall … マイクのフレーム送出からテキストまで（端から端）
        # 音声時刻（ts_end）は「どのフレームか」を引き当てるキーにだけ使う。
        # 30 秒ぶん（16kHz / 20ms フレームでも 1500 件程度）あれば十分。
        self._frame_arrivals: deque[tuple[float, float | None]] = deque(maxlen=4096)
        self._frame_arrivals_lock = threading.Lock()
        # 音声クロックの進みを警告したか（初回だけ出す。_report_latency_sec 参照）
        self._audio_clock_skew_logged = False
        self._sample_clock_cursor = 0
        self._last_partial_for_flush: dict[str, Any] | None = None

    def start(self) -> None:
        if self._threads and any(t.is_alive() for t in self._threads):
            self._log("ASR service is already running")
            return

        self._stop_event.clear()
        self._force_finalize_event.clear()
        self._last_partial_for_flush = None
        while True:
            try:
                _ = self._maai_result_event_queue.get_nowait()
            except queue.Empty:
                break
        with self._force_finalize_state_lock:
            self._force_finalize_speaking = False

        self._initialize_turn_backend_if_needed()
        self._initialize_asr_engine()
        self._threads = self._build_engine_threads()
        if self.turn_backend == "smart-turn":
            # qwen/external/chirp は通常VADループが無いため、turn予測用に別セグメント化ループを回す。
            if self.asr_engine in {"external-asr-tcp", "chirp3-asr", "nemotron-asr", "nemotron-onnx"}:
                self._threads.append(
                    threading.Thread(target=self._turn_segment_loop, daemon=True, name="asr-turn-segment")
                )
            self._threads.append(
                threading.Thread(target=self._smart_turn_loop, daemon=True, name="asr-smart-turn")
            )
        # turn_backend=maai の場合: ターン推論は MaAI ワーカー側で行うため
        # ASR 内では _turn_maai_loop を起動しない。
        if self.on_maai_result is not None:
            self._threads.append(
                threading.Thread(target=self._maai_result_emit_loop, daemon=True, name="asr-maai-emit")
            )
        for thread in self._threads:
            thread.start()
        self._log(
            "ASR started: "
            f"engine={self.asr_engine}, vad={self.vad_backend}, turn={self.turn_backend}, "
            f"process_block={self.block_size}, ingress={self.ingress_block_size}"
            + (
                f", nemotron_finalize={self.nemotron_finalize_mode}"
                if self.asr_engine in {"nemotron-asr", "nemotron-onnx"}
                else ""
            )
        )

    def _build_engine_threads(self) -> list[threading.Thread]:
        threads: list[threading.Thread] = []
        # VAD で発話を区切って一括推論するエンジンは、どれも同じ 2 スレッド構成。
        # 個別に書き出すとエンジンを足したときにここだけ漏れる（実際に漏らした）。
        vad_and_infer = [(self._vad_loop, "asr-vad"), (self._asr_loop, "asr-infer")]
        engine_targets: dict[str, list[tuple[Callable[[], None], str]]] = {
            "qwen3-asr": [(self._qwen_stream_loop, "asr-qwen-stream")],
            "nemotron-asr": [(self._nemotron_stream_loop, "asr-nemotron-stream")],
            "chirp3-asr": [(self._chirp3_stream_loop, "asr-chirp3-stream")],
            "external-asr-tcp": [
                (self._external_asr_audio_send_loop, "asr-external-audio"),
                (self._external_asr_tcp_loop, "asr-external-tcp"),
            ],
            "faster-whisper": vad_and_infer,
            "nemotron-onnx": [(self._nemotron_onnx_stream_loop, "asr-nemotron-onnx-stream")],
            **{engine: vad_and_infer for engine in SHERPA_DECODE_ENGINES},
        }
        targets = engine_targets.get(self.asr_engine)
        if targets is None:
            raise RuntimeError(f"未対応のASRエンジンです: {self.asr_engine}")
        for target, name in targets:
            threads.append(threading.Thread(target=target, daemon=True, name=name))
        return threads

    def stop(self) -> None:
        self._stop_event.set()
        if self._external_server_socket is not None:
            try:
                self._external_server_socket.close()
            except Exception:
                pass
            self._external_server_socket = None
        if self._external_audio_socket is not None:
            try:
                self._external_audio_socket.close()
            except Exception:
                pass
            self._external_audio_socket = None
        for thread in self._threads:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        self._threads = []

        if self._nemotron is not None:
            try:
                self._nemotron.close()
            except Exception:
                pass
            self._nemotron = None
        self._terminate_nemotron_proc()
        self._nemotron_onnx_session = None
        self._close_speechcpp_asr()

        if self._maai_instance is not None:
            try:
                self._maai_instance.stop()
            except Exception:
                pass
            self._maai_instance = None
        self._log("ASR stopped")

    def inject_system_audio(self, audio_f32: np.ndarray) -> None:
        with self._system_audio_lock:
            samples = audio_f32.tolist()
            self._system_audio_buffer_vad.extend(samples)
            max_samples = self.sample_rate * 2
            while len(self._system_audio_buffer_vad) > max_samples:
                self._system_audio_buffer_vad.popleft()

    def _get_system_audio_chunk(self, n_samples: int) -> np.ndarray:
        buf = self._system_audio_buffer_vad
        with self._system_audio_lock:
            available = min(len(buf), n_samples)
            if available == 0:
                return np.zeros(n_samples, dtype=np.float32)
            chunk = np.zeros(n_samples, dtype=np.float32)
            for i in range(available):
                chunk[i] = buf.popleft()
            return chunk

    def request_force_finalize_current_utterance(self, request_id: int | None = None) -> bool:
        # non-streaming系で「現時点までをfinalとして確定」したいときに使用する。
        with self._force_finalize_state_lock:
            speaking = bool(self._force_finalize_speaking)
            if speaking:
                self._force_finalize_pending_request_id = int(request_id) if request_id is not None else None
                self._force_finalize_event.set()
            else:
                self._force_finalize_pending_request_id = None
        return speaking

    def _set_force_finalize_speaking(self, active: bool) -> None:
        with self._force_finalize_state_lock:
            self._force_finalize_speaking = bool(active)

    def _emit_force_finalize_outcome(self, payload: dict[str, Any]) -> None:
        if self.on_force_finalize_outcome is None:
            return
        try:
            self.on_force_finalize_outcome(dict(payload))
        except Exception:
            pass

    def _flush_with_last_partial(self, *, end_ts: float | None = None) -> bool:
        cached = self._last_partial_for_flush
        if not isinstance(cached, dict):
            return False
        text = str(cached.get("text", "")).strip()
        if not text:
            return False
        engine = str(cached.get("engine", self.asr_engine)).strip() or self.asr_engine
        seg_start = to_float_or_none(cached.get("segment_start_ts"))
        seg_end_cached = to_float_or_none(cached.get("segment_end_ts"))
        final_end = to_float_or_none(end_ts)
        if final_end is None:
            final_end = seg_end_cached if seg_end_cached is not None else time.time()
        if seg_start is not None and final_end < seg_start:
            final_end = seg_start
        self._publish_result(
            text=text,
            end_ts=final_end,
            engine=engine,
            is_final=True,
            is_streaming=True,
            segment_start_ts=seg_start,
            segment_end_ts=final_end,
        )
        self._last_partial_for_flush = None
        return True

    def _set_inference_busy(self, busy: bool) -> None:
        changed = False
        with self._inference_state_lock:
            if self._inference_busy != bool(busy):
                self._inference_busy = bool(busy)
                changed = True
        if changed and self.on_inference_state is not None:
            try:
                self.on_inference_state(bool(busy))
            except Exception:
                pass

    def _emit_maai_result(
        self,
        result: dict[str, Any],
        *,
        source_ts_start: float | None = None,
        source_ts_end: float | None = None,
        source_sample_start: int | None = None,
        source_sample_end: int | None = None,
    ) -> None:
        if self.on_maai_result is None:
            return
        try:
            payload = dict(result)
            if self._latest_silero_vad_score is not None:
                payload["silero_vad_score"] = float(self._latest_silero_vad_score)
            if source_ts_start is not None:
                payload["source_ts_start"] = float(source_ts_start)
            if source_ts_end is not None:
                payload["source_ts_end"] = float(source_ts_end)
            if source_sample_start is not None:
                payload["source_sample_start"] = int(source_sample_start)
            if source_sample_end is not None:
                payload["source_sample_end"] = int(source_sample_end)
            self._queue_put(
                self._maai_result_event_queue,
                payload,
                policy=self.QUEUE_POLICY_DROP_OLDEST,
            )
        except Exception as ex:
            self._error(f"MAAI結果イベント送信に失敗しました: {ex}")

    def _maai_result_emit_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._maai_result_event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.on_maai_result is None:
                continue
            try:
                self.on_maai_result(dict(payload))
            except Exception:
                pass

    def _initialize_asr_engine(self) -> None:
        initializers: dict[str, Callable[[], None]] = {
            "faster-whisper": self._init_engine_faster_whisper,
            "sherpa-onnx": self._init_engine_sherpa_onnx,
            "qwen3-asr-onnx": self._init_engine_qwen3_onnx,
            "parakeet-onnx": self._init_engine_parakeet_onnx,
            "sensevoice-onnx": self._init_engine_sensevoice_onnx,
            "funasr-nano-onnx": self._init_engine_funasr_nano_onnx,
            "cohere-asr-onnx": self._init_engine_cohere_asr_onnx,
            "speechcpp-asr": self._init_engine_speechcpp_asr,
            "nemotron-onnx": self._init_engine_nemotron_onnx,
            "qwen3-asr": self._init_engine_qwen,
            "nemotron-asr": self._init_engine_nemotron,
            "chirp3-asr": self._init_engine_chirp3,
            "external-asr-tcp": self._init_engine_external,
        }
        init = initializers.get(self.asr_engine)
        if init is None:
            raise RuntimeError(f"未対応のASRエンジンです: {self.asr_engine}")
        init()

    def _resolve_nemotron_onnx_chunk_paths(self, chunk_ms: int) -> dict[str, str]:
        from .nemotron_chunk import (
            resolve_nemotron_onnx_model_paths,
            swap_fp32_onnx_path_to_int8,
        )

        onnx_chunk_ms = int(chunk_ms)
        use_int8 = self.nemotron_onnx_quant == "int8"
        paths = resolve_nemotron_onnx_model_paths(
            tokens_path=self.nemotron_onnx_tokens_path,
            encoder_path=self.nemotron_onnx_encoder_path,
            decoder_path=self.nemotron_onnx_decoder_path,
            joiner_path=self.nemotron_onnx_joiner_path,
            chunk_ms=onnx_chunk_ms,
            resolve_path=self._resolve_model_path,
            prefer_fp32=not use_int8,
        )
        if use_int8:
            # CPU int8 mode: switch encoder/decoder/joiner to the *.int8.onnx variant
            # (fall back to the resolved fp32 path if int8 is not present beside it).
            for name in ("encoder", "decoder", "joiner"):
                int8_path = swap_fp32_onnx_path_to_int8(paths[name])
                if int8_path == paths[name]:
                    continue
                if Path(int8_path).is_file():
                    paths[name] = int8_path
                else:
                    self._log(
                        f"nemotron-onnx: int8 requested but {int8_path} missing; "
                        f"using {paths[name]}"
                    )
        return paths

    def _connect_nemotron_client(self, timeout_sec: float = 300.0) -> None:
        """サブプロセスがモデルロードを終えて listen するまで接続をリトライする。

        サーバはモデルロード完了後に bind/listen するため、接続成功 = ready の合図。
        """
        from .nemotron_asr import NemotronStreamingClient

        proc = self._nemotron_proc
        host = getattr(self, "_nemotron_host", "127.0.0.1")
        port = int(getattr(self, "_nemotron_port", 0) or 0)
        if port <= 0 or (proc is None and not self.nemotron_remote_host):
            raise RuntimeError("Nemotron サブプロセスが起動していません。")
        deadline = time.time() + float(timeout_sec)
        last_err: Exception | None = None
        while time.time() < deadline and not self._stop_event.is_set():
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"Nemotron サブプロセスが起動に失敗しました（exit={proc.returncode}）。"
                    "ログ logs/nemotron_server_*.log を確認してください。"
                )
            try:
                self._nemotron = NemotronStreamingClient(host, port, timeout=60.0)
                return
            except (ConnectionError, OSError) as ex:
                last_err = ex
                time.sleep(0.5)
        if self._stop_event.is_set():
            raise RuntimeError("Nemotron 接続待ち中に停止要求を受けました。")
        raise RuntimeError(f"Nemotron サブプロセスへ接続できませんでした: {last_err}")

    def _terminate_nemotron_proc(self) -> None:
        proc = self._nemotron_proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except Exception:  # noqa: BLE001
                    proc.kill()
        except Exception:  # noqa: BLE001
            pass
        self._nemotron_proc = None
        fp = self._nemotron_log_fp
        if fp is not None:
            try:
                fp.close()
            except Exception:  # noqa: BLE001
                pass
            self._nemotron_log_fp = None

    def _validate_external_asr_payload(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            self._error("external-asr-tcp: JSON object以外を受信しました。")
            return None
        text = str(payload.get("text", "")).strip()
        if not text:
            self._error("external-asr-tcp: text が空です。")
            return None
        if "is_final" not in payload:
            self._error("external-asr-tcp: is_final が必須です。")
            return None
        if "is_streaming" not in payload:
            self._error("external-asr-tcp: is_streaming が必須です。")
            return None
        is_final_raw = payload.get("is_final")
        is_streaming_raw = payload.get("is_streaming")
        if not isinstance(is_final_raw, bool):
            self._error("external-asr-tcp: is_final は bool で指定してください。")
            return None
        if not isinstance(is_streaming_raw, bool):
            self._error("external-asr-tcp: is_streaming は bool で指定してください。")
            return None
        return {
            "text": text,
            "engine": str(payload.get("engine", "external-asr-tcp")).strip() or "external-asr-tcp",
            "is_final": is_final_raw,
            "is_streaming": is_streaming_raw,
            "payload_start_ts": to_float_or_none(payload.get("segment_start_ts")),
            "payload_end_ts": to_float_or_none(payload.get("segment_end_ts")),
        }

    def _new_audio_frame(self, audio: np.ndarray) -> AudioFrame:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        n = int(x.shape[0])
        with self._sample_clock_lock:
            if self._sample_clock_base_ts is None:
                # callbackが呼ばれる時点では、バッファ先頭サンプルは既に過去の時刻に属する。
                # 先頭サンプル時刻に合わせて基準を置き、波形/ASR帯の右ズレを抑える。
                frame_sec = float(n) / float(max(1, self.sample_rate))
                self._sample_clock_base_ts = time.time() - frame_sec
            sample_start = int(self._sample_clock_cursor)
            sample_end = sample_start + n
            self._sample_clock_cursor = sample_end
            ts_start = float(self._sample_clock_base_ts) + (sample_start / float(self.sample_rate))
            ts_end = float(self._sample_clock_base_ts) + (sample_end / float(self.sample_rate))
        return AudioFrame(
            audio=x,
            sample_start=sample_start,
            sample_end=sample_end,
            ts_start=ts_start,
            ts_end=ts_end,
        )

    def _to_audio_frame(self, item: AudioFrame | np.ndarray) -> AudioFrame:
        if isinstance(item, AudioFrame):
            # External ingress frames already carry absolute sample/timestamp.
            # Align internal sample clock to avoid future-shifted timestamps
            # when rebuilding partial segment bounds from sample indices.
            with self._sample_clock_lock:
                try:
                    incoming_base_ts = float(item.ts_start) - (
                        float(item.sample_start) / float(max(1, self.sample_rate))
                    )
                except Exception:
                    incoming_base_ts = float(item.ts_start)
                current_base_ts = self._sample_clock_base_ts
                sample_rewound = int(item.sample_start) < int(self._sample_clock_cursor)
                clock_rebased = (
                    current_base_ts is not None
                    and abs(float(incoming_base_ts) - float(current_base_ts)) > 0.25
                )
                if current_base_ts is None or sample_rewound or clock_rebased:
                    # WAV の stop -> play では送信元の sample cursor が 0 に戻り、
                    # ts は再開時刻へ再アンカーされる。旧 base を保持すると休止時間ぶん
                    # ASR セグメント時刻が過去へずれ、疑似ストリーミング遅延が肥大する。
                    self._sample_clock_base_ts = float(incoming_base_ts)
                    self._sample_clock_cursor = int(item.sample_end)
                else:
                    self._sample_clock_cursor = max(int(self._sample_clock_cursor), int(item.sample_end))
            self._note_frame_arrival(item)
            return item
        frame = self._new_audio_frame(np.asarray(item, dtype=np.float32))
        self._note_frame_arrival(frame)
        return frame

    def _note_frame_arrival(self, frame: AudioFrame) -> None:
        """フレームを受け取った壁時計時刻を記録する（処理遅延の計測用）。

        `_to_audio_frame` が唯一の入口なので、ここだけで全エンジンをカバーできる。
        """
        try:
            ts_end = float(frame.ts_end)
        except (TypeError, ValueError):
            return
        origin = getattr(frame, "origin_ts", None)
        try:
            origin_ts = float(origin) if origin is not None else None
        except (TypeError, ValueError):
            origin_ts = None
        with self._frame_arrivals_lock:
            self._frame_arrivals.append((ts_end, origin_ts))

    def _origin_ts_for(self, end_ts: float | None) -> float | None:
        """`end_ts`（音声時刻）を含むフレームの**送出側 wall 時刻**を引き当てる。

        `ts_end >= end_ts` を満たす**最も古い**フレーム = その音声を運んだフレーム。
        音声時刻同士で引き当てるので、サンプルクロックが wall clock からずれていても
        引き当て自体はずれない（同じ時間軸で比べているだけ）。

        記録より新しい範囲を指された場合は直近のフレームで代替する。
        """
        if not end_ts:
            return None
        target = float(end_ts)
        found = False
        origin: float | None = None
        with self._frame_arrivals_lock:
            if not self._frame_arrivals:
                return None
            for ts_end, origin_at in reversed(self._frame_arrivals):
                if ts_end + 1e-9 >= target:
                    origin, found = origin_at, True
                else:
                    break
            if not found:
                origin = self._frame_arrivals[-1][1]
        return origin

    def _end_to_end_latency_sec(self, end_ts: float | None) -> float:
        """「マイクのフレームが送り出されてからテキストが出るまで」の壁時計時間。

        `audio/input_worker` が `emitted_at` として載せた壁時計時刻を基準にするので、
        音声のサンプルクロックが壁時計からずれていても影響を受けない
        （音声時刻を基準にしていた旧実装は、そのズレで負の値になっていた）。

        origin が運ばれてこない経路（asr/service 単体起動など）では、
        従来の音声時刻ベース（`_report_latency_sec`）へ落とす。
        デバイス自身のバッファ遅延は含まない（ソフトからは測れない）。
        """
        origin = self._origin_ts_for(end_ts)
        if origin is None:
            return self._report_latency_sec(end_ts)
        return max(0.0, time.time() - float(origin))

    def _sample_index_to_ts(self, sample_index: int) -> float:
        with self._sample_clock_lock:
            if self._sample_clock_base_ts is None:
                self._sample_clock_base_ts = time.time()
            return float(self._sample_clock_base_ts) + (float(sample_index) / float(self.sample_rate))

    def _build_segment_payload_from_frames(
        self,
        frames: list[AudioFrame],
        tail_sec: float | None,
        *,
        result_mode: str,
    ) -> dict[str, Any] | None:
        if not frames:
            return None
        try:
            merged = np.concatenate([np.asarray(f.audio, dtype=np.float32).reshape(-1) for f in frames], axis=0)
        except Exception:
            return None
        if merged.size <= 0:
            return None
        if tail_sec is not None:
            target_samples = int(self.sample_rate * max(0.0, float(tail_sec)))
            if target_samples <= 0:
                target_samples = int(merged.size)
            if merged.size > target_samples:
                merged = merged[-target_samples:]
        sample_end = int(frames[-1].sample_end)
        sample_start = max(0, sample_end - int(merged.size))
        start_ts, end_ts = self._resolve_segment_bounds_from_samples(sample_start, sample_end)
        return {
            "audio": merged,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "result_mode": result_mode,
        }

    @staticmethod
    def _split_fixed_chunks(buffer: np.ndarray, chunk_size: int) -> tuple[list[np.ndarray], np.ndarray]:
        x = np.asarray(buffer, dtype=np.float32).reshape(-1)
        n = max(1, int(chunk_size))
        chunks: list[np.ndarray] = []
        while x.size >= n:
            chunks.append(x[:n].copy())
            x = x[n:]
        return chunks, x

    def _take_aligned_frame(self, buffer: list[AudioFrame], n: int) -> AudioFrame | None:
        """buffer 先頭からちょうど n サンプル分を1つの AudioFrame にまとめて取り出す。足りなければ None（buffer は不変）。"""
        nn = int(n)
        if nn <= 0 or not buffer:
            return None
        total = sum(int(np.asarray(f.audio, dtype=np.float32).size) for f in buffer)
        if total < nn:
            return None
        parts: list[np.ndarray] = []
        first_sample_start: int | None = None
        need = nn
        while need > 0:
            f = buffer[0]
            a = np.asarray(f.audio, dtype=np.float32).reshape(-1)
            if first_sample_start is None:
                first_sample_start = int(f.sample_start)
            if a.size <= need:
                parts.append(a)
                need -= int(a.size)
                buffer.pop(0)
            else:
                parts.append(a[:need].copy())
                rest = a[need:].copy()
                advance = need
                buffer[0] = AudioFrame(
                    audio=rest,
                    sample_start=int(f.sample_start) + advance,
                    sample_end=int(f.sample_end),
                    ts_start=float(f.ts_start) + float(advance) / float(max(1, self.sample_rate)),
                    ts_end=float(f.ts_end),
                )
                need = 0
        assert first_sample_start is not None
        audio = np.concatenate(parts, axis=0)
        if int(audio.shape[0]) != nn:
            return None
        sample_end = int(first_sample_start) + nn
        ts_start = self._sample_index_to_ts(int(first_sample_start))
        ts_end = self._sample_index_to_ts(sample_end)
        return AudioFrame(
            audio=audio,
            sample_start=int(first_sample_start),
            sample_end=sample_end,
            ts_start=float(ts_start),
            ts_end=float(ts_end),
        )

    def _resolve_segment_bounds(
        self,
        start_ts: float | None,
        end_ts: float | None,
        sample_count: int,
    ) -> tuple[float, float]:
        duration = max(0.0, float(sample_count) / float(max(1, self.sample_rate)))
        start_v = to_float_or_none(start_ts)
        end_v = to_float_or_none(end_ts)
        if start_v is None and end_v is None:
            with self._sample_clock_lock:
                base_ts = self._sample_clock_base_ts
                cursor = int(self._sample_clock_cursor)
            if base_ts is not None:
                end_v = float(base_ts) + (float(cursor) / float(self.sample_rate))
                start_v = max(0.0, float(end_v) - duration)
            else:
                end_v = time.time()
                start_v = max(0.0, float(end_v) - duration)
        elif start_v is None:
            start_v = max(0.0, float(end_v) - duration) if end_v is not None else time.time()
        elif end_v is None:
            end_v = float(start_v) + duration
        if end_v is None:
            end_v = float(start_v) + duration
        if start_v is None:
            start_v = max(0.0, float(end_v) - duration)
        if float(end_v) < float(start_v):
            end_v = float(start_v)
        return float(start_v), float(end_v)

    def _ts_to_sample_index(self, ts: float) -> int:
        ts_v = float(ts)
        with self._sample_clock_lock:
            if self._sample_clock_base_ts is None:
                self._sample_clock_base_ts = ts_v
            base_ts = float(self._sample_clock_base_ts)
        return max(0, int(round((ts_v - base_ts) * float(self.sample_rate))))

    def _resolve_segment_bounds_from_samples(self, sample_start: int, sample_end: int) -> tuple[float, float]:
        s_idx = max(0, int(sample_start))
        e_idx = max(s_idx, int(sample_end))
        start_ts = self._sample_index_to_ts(s_idx)
        end_ts = self._sample_index_to_ts(e_idx)
        return self._resolve_segment_bounds(start_ts, end_ts, max(0, e_idx - s_idx))

    def _run_segment_loop(
        self,
        *,
        source_queue: queue.Queue[AudioFrame],
        hooks: SegmentLoopHooks,
    ) -> None:
        try:
            self._ensure_vad_backend_initialized()
        except Exception as ex:
            self._error(str(ex))
            return
        # 生フレーム投入周期（sample_rate / block_size）で hold 秒をフレーム数へ換算する。
        # Qwen 側と同じ換算に揃えることで、vad_start/end_hold_sec の時間意味を統一する。
        frame_hz = max(1.0, float(self.sample_rate) / float(max(1, self.block_size)))
        activation_frames = max(1, int(math.ceil(self.vad_start_hold_sec * frame_hz)))
        deactivation_frames = max(1, int(math.ceil(self.vad_end_hold_sec * frame_hz)))
        use_pre_roll = bool(hooks.emit_to_inference)
        pre_frames = (
            max(1, int(math.ceil(max(0.0, self.pre_roll_buffer_sec) * frame_hz)))
            if use_pre_roll
            else activation_frames
        )
        pre_buffer: deque[AudioFrame] = deque(maxlen=pre_frames)
        pseudo_chunk_samples = max(1, int(self.sample_rate * max(0.0, self.nonstream_pseudo_chunk_sec)))
        speech_active = False
        activation_count = 0
        deactivation_count = 0
        end_of_utterance_ts: float | None = None
        segment_start_ts: float | None = None
        current_segment_frames: list[AudioFrame] = []
        pseudo_partial_frames: list[AudioFrame] = []
        next_partial_sample_end: int | None = None
        ingress_buffer: list[AudioFrame] = []
        last_input_sample_end: int | None = None
        last_input_ts_end: float | None = None
        while not self._stop_event.is_set():
            try:
                raw = source_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # 起動時のモデルロード/ウォームアップ中に無制限キューへ数秒分の音声が
            # 溜まると、VAD（silero/MaAI）はほぼ実時間速度でしか処理できないため
            # バックログが排出されず、VAD/プロットが常時数秒遅れる。
            # 発話中でないときに限り古いフレームを捨てて実時間へ追いつかせる。
            # speech_active 中はドロップしないので ASR セグメントは欠落しない。
            # process_only_frames（ストリーミング）時は speech_active が常に False のため、
            # エンジン側の発話状態を確認し、発話中はバックログを破棄しない（ASR音声欠落の防止）。
            _engine_in_segment = (
                bool(hooks.is_in_active_segment()) if hooks.is_in_active_segment is not None else False
            )
            if not speech_active and not _engine_in_segment:
                max_backlog_frames = max(2, int(frame_hz * 0.5))
                while source_queue.qsize() > max_backlog_frames:
                    try:
                        stale = source_queue.get_nowait()
                    except queue.Empty:
                        break
                    # サンプルクロックのカーソルだけ前進させ、音声本体は破棄する。
                    self._to_audio_frame(stale)
            incoming = self._to_audio_frame(raw)
            timeline_discontinuous = False
            if last_input_sample_end is not None and int(incoming.sample_start) != int(last_input_sample_end):
                timeline_discontinuous = True
            if last_input_ts_end is not None:
                frame_sec = float(max(1, incoming.audio.size)) / float(max(1, self.sample_rate))
                ts_gap = float(incoming.ts_start) - float(last_input_ts_end)
                if ts_gap < -0.01 or ts_gap > max(0.25, frame_sec * 4.0):
                    timeline_discontinuous = True
            last_input_sample_end = int(incoming.sample_end)
            last_input_ts_end = float(incoming.ts_end)
            if timeline_discontinuous:
                # WAV 停止中も壁時計プロットは流し続けるが、停止前の発話状態は
                # 再開後へ接続しない。接続すると休止時間が ASR latency / span に入る。
                speech_active = False
                if hooks.emit_to_inference:
                    self._set_force_finalize_speaking(False)
                activation_count = 0
                deactivation_count = 0
                end_of_utterance_ts = None
                segment_start_ts = None
                current_segment_frames = []
                pseudo_partial_frames = []
                next_partial_sample_end = None
                ingress_buffer.clear()
                pre_buffer.clear()
                self._last_partial_for_flush = None
                self._silero_input_buffer = np.empty(0, dtype=np.float32)
                self._vap_maai_vad_buffer = np.empty(0, dtype=np.float32)
                if hooks.emit_to_inference:
                    while True:
                        try:
                            self.inference_queue.get_nowait()
                        except queue.Empty:
                            break
            ingress_buffer.append(incoming)
            while True:
                frame = self._take_aligned_frame(ingress_buffer, self.block_size)
                if frame is None:
                    break
                pre_buffer.append(frame)
                audio_chunk = frame.audio
                if (hooks.emit_to_inference or hooks.forward_to_turn_queue) and self.turn_backend != "maai":
                    self._queue_put(self.turn_audio_queue, frame, policy=self.QUEUE_POLICY_DROP_OLDEST)
                if hooks.on_frame is not None:
                    hooks.on_frame(frame)
                if hooks.process_only_frames:
                    continue
                vad_score = self._get_vad_score(audio_chunk, frame)
                if vad_score is None:
                    continue
                if not speech_active:
                    if vad_score > self.vad_start_threshold:
                        activation_count += 1
                        if activation_count >= activation_frames:
                            speech_active = True
                            self._last_partial_for_flush = None
                            if hooks.emit_to_inference:
                                self._set_force_finalize_speaking(True)
                            deactivation_count = 0
                            current_segment_frames = list(pre_buffer)
                            segment_start_ts = (
                                float(current_segment_frames[0].ts_start)
                                if current_segment_frames
                                else float(frame.ts_start)
                            )
                            if hooks.on_segment_start is not None:
                                try:
                                    hooks.on_segment_start(current_segment_frames[0] if current_segment_frames else frame)
                                except Exception:
                                    pass
                            if hooks.emit_to_inference and self.nonstream_pseudo_enabled and self.nonstream_pseudo_chunk_sec > 0.0:
                                pseudo_partial_frames = list(pre_buffer)
                                bootstrap_payload = self._build_segment_payload_from_frames(
                                    pseudo_partial_frames,
                                    self.nonstream_pseudo_chunk_sec,
                                    result_mode="partial",
                                )
                                if bootstrap_payload is not None:
                                    self._queue_put(
                                        self.inference_queue, bootstrap_payload, policy=self.QUEUE_POLICY_DROP_OLDEST
                                    )
                            next_partial_sample_end = int(frame.sample_end) + pseudo_chunk_samples
                    else:
                        activation_count = 0
                        if not use_pre_roll:
                            pre_buffer.clear()
                    continue
                current_segment_frames.append(frame)
                if (
                    hooks.emit_to_inference
                    and self.nonstream_pseudo_enabled
                    and self.nonstream_pseudo_chunk_sec > 0.0
                    and pseudo_partial_frames
                ):
                    pseudo_partial_frames.append(frame)
                if (
                    hooks.emit_to_inference
                    and next_partial_sample_end is not None
                    and int(frame.sample_end) >= int(next_partial_sample_end)
                    and self.nonstream_pseudo_enabled
                    and self.nonstream_pseudo_chunk_sec > 0.0
                ):
                    partial_payload = self._build_segment_payload_from_frames(
                        pseudo_partial_frames,
                        None,
                        result_mode="partial",
                    )
                    if partial_payload is not None:
                        self._queue_put(self.inference_queue, partial_payload, policy=self.QUEUE_POLICY_DROP_OLDEST)
                    while next_partial_sample_end <= int(frame.sample_end):
                        next_partial_sample_end += pseudo_chunk_samples
                if hooks.emit_to_inference and self._force_finalize_event.is_set():
                    self._force_finalize_event.clear()
                    with self._force_finalize_state_lock:
                        request_id = self._force_finalize_pending_request_id
                        self._force_finalize_pending_request_id = None
                    if current_segment_frames:
                        segment_audio = np.concatenate([f.audio for f in current_segment_frames], axis=0)
                        min_samples = int(self.sample_rate * self.min_segment_sec)
                        if len(segment_audio) >= min_samples:
                            payload = {
                                "audio": segment_audio,
                                "start_ts": segment_start_ts,
                                "end_ts": float(frame.ts_end),
                                "result_mode": "final",
                                "force_finalize_request_id": request_id,
                            }
                            self._queue_put(self.inference_queue, payload, policy=self.QUEUE_POLICY_DROP_OLDEST)
                            if self.turn_backend == "smart-turn":
                                self._queue_put(self.smart_turn_queue, payload, policy=self.QUEUE_POLICY_DROP_OLDEST)
                        else:
                            flushed = self._flush_with_last_partial(end_ts=float(frame.ts_end))
                            self._emit_force_finalize_outcome(
                                {
                                    "status": "flushed_from_partial" if flushed else "skipped_short",
                                    "request_id": request_id,
                                    "samples": int(len(segment_audio)),
                                    "min_samples": int(min_samples),
                                }
                            )
                    else:
                        flushed = self._flush_with_last_partial(end_ts=float(frame.ts_end))
                        self._emit_force_finalize_outcome(
                            {
                                "status": "flushed_from_partial" if flushed else "no_audio",
                                "request_id": request_id,
                            }
                        )
                    speech_active = False
                    self._set_force_finalize_speaking(False)
                    activation_count = 0
                    deactivation_count = 0
                    end_of_utterance_ts = None
                    segment_start_ts = None
                    current_segment_frames = []
                    pseudo_partial_frames = []
                    next_partial_sample_end = None
                    continue
                if vad_score < self.vad_end_threshold:
                    if deactivation_count == 0:
                        end_of_utterance_ts = float(frame.ts_end)
                    deactivation_count += 1
                    if deactivation_count >= deactivation_frames:
                        if current_segment_frames:
                            segment_audio = np.concatenate([f.audio for f in current_segment_frames], axis=0)
                            min_samples = int(self.sample_rate * self.min_segment_sec)
                            if len(segment_audio) >= min_samples:
                                payload = {
                                    "audio": segment_audio,
                                    "start_ts": segment_start_ts,
                                    "end_ts": end_of_utterance_ts or float(current_segment_frames[-1].ts_end),
                                }
                                if hooks.emit_to_inference:
                                    payload["result_mode"] = "final"
                                    self._queue_put(self.inference_queue, payload, policy=self.QUEUE_POLICY_DROP_OLDEST)
                                    if self.turn_backend == "smart-turn":
                                        self._queue_put(
                                            self.smart_turn_queue, payload, policy=self.QUEUE_POLICY_DROP_OLDEST
                                        )
                                elif hooks.emit_to_smart_turn_on_noninference:
                                    self._queue_put(self.smart_turn_queue, payload, policy=self.QUEUE_POLICY_DROP_OLDEST)
                                if hooks.on_segment_end is not None:
                                    try:
                                        hooks.on_segment_end(float(payload["end_ts"]))
                                    except Exception:
                                        pass
                            elif hooks.emit_to_inference:
                                # 擬似ストリーミングで短発話が min_segment_sec 未満だと
                                # final 推論が走らず partial のまま残るため、直近partialをfinal化する。
                                _ = self._flush_with_last_partial(
                                    end_ts=end_of_utterance_ts or float(current_segment_frames[-1].ts_end)
                                )
                        speech_active = False
                        if hooks.emit_to_inference:
                            self._set_force_finalize_speaking(False)
                        activation_count = 0
                        deactivation_count = 0
                        end_of_utterance_ts = None
                        segment_start_ts = None
                        current_segment_frames = []
                        pseudo_partial_frames = []
                        next_partial_sample_end = None
                        pre_buffer.clear()
                else:
                    deactivation_count = 0
                    end_of_utterance_ts = None

    def _decode_segment(
        self,
        audio: np.ndarray,
        start_ts: float | None,
        end_ts: float | None,
        *,
        is_final: bool,
        is_streaming: bool,
    ) -> bool:
        if self.asr_engine == "faster-whisper":
            return self._decode_with_faster_whisper(
                audio,
                start_ts,
                end_ts,
                is_final=is_final,
                is_streaming=is_streaming,
            )
        if self.asr_engine in SHERPA_DECODE_ENGINES:
            return self._decode_with_sherpa(
                audio,
                start_ts,
                end_ts,
                is_final=is_final,
                is_streaming=is_streaming,
            )
        if self.asr_engine == "nemotron-onnx":
            return self._decode_with_nemotron_onnx(
                audio,
                start_ts,
                end_ts,
                is_final=is_final,
                is_streaming=is_streaming,
            )
        raise RuntimeError(f"inference loop未対応のASRエンジンです: {self.asr_engine}")

    def _asr_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_segment = item["audio"]
            start_ts = item.get("start_ts")
            end_ts = item.get("end_ts")
            force_finalize_request_id_obj = item.get("force_finalize_request_id")
            try:
                force_finalize_request_id = (
                    int(force_finalize_request_id_obj)
                    if force_finalize_request_id_obj is not None
                    else None
                )
            except Exception:
                force_finalize_request_id = None
            result_mode = str(item.get("result_mode", "final")).strip().lower() or "final"
            is_partial = result_mode == "partial"
            start_ts, end_ts = self._resolve_segment_bounds(
                item.get("start_ts"),
                item.get("end_ts"),
                len(audio_segment),
            )
            if is_partial:
                if len(audio_segment) < max(1, int(self.sample_rate * 0.05)):
                    if force_finalize_request_id is not None:
                        self._emit_force_finalize_outcome(
                            {
                                "status": "skipped_short",
                                "request_id": force_finalize_request_id,
                                "samples": int(len(audio_segment)),
                                "min_samples": int(max(1, int(self.sample_rate * 0.05))),
                            }
                        )
                    continue
            elif len(audio_segment) < int(self.sample_rate * self.min_segment_sec):
                if force_finalize_request_id is not None:
                    self._emit_force_finalize_outcome(
                        {
                            "status": "skipped_short",
                            "request_id": force_finalize_request_id,
                            "samples": int(len(audio_segment)),
                            "min_samples": int(int(self.sample_rate * self.min_segment_sec)),
                        }
                    )
                continue

            emitted_result = False
            failed = False
            # 疑似ストリーミング時は真のストリーミング（Qwen 等）と同様、partial/final とも is_streaming=True。
            pseudo_streaming_active = bool(self.nonstream_pseudo_enabled) and float(
                self.nonstream_pseudo_chunk_sec
            ) > 0.0
            is_streaming = bool(is_partial) or pseudo_streaming_active
            try:
                self._set_inference_busy(True)
                emitted_result = self._decode_segment(
                    audio_segment,
                    start_ts,
                    end_ts,
                    is_final=(not is_partial),
                    is_streaming=is_streaming,
                )
            except Exception as ex:
                failed = True
                self._error(f"ASR推論に失敗しました ({self.asr_engine}): {ex}")
                if force_finalize_request_id is not None:
                    self._emit_force_finalize_outcome(
                        {
                            "status": "error",
                            "request_id": force_finalize_request_id,
                            "error": str(ex),
                        }
                    )
            finally:
                self._set_inference_busy(False)
            if force_finalize_request_id is not None and (not failed):
                self._emit_force_finalize_outcome(
                    {
                        "status": "result_emitted" if emitted_result else "empty_text",
                        "request_id": force_finalize_request_id,
                    }
                )

    @staticmethod
    def _clean_qwen_text(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        # Qwen3-ASR は "language Japanese<asr_text>本文</asr_text>" 形式を返す。
        # <asr_text> 以降が本文なので、タグ以降を取り出してメタ接頭辞を確実に除く。
        # （タグ除去後に "language X" を貪欲に消すと英語本文まで巻き込むため、この順で処理する）
        m = re.search(r"<\s*asr_text\s*>", cleaned, flags=re.IGNORECASE)
        if m:
            cleaned = cleaned[m.end():]
        cleaned = re.sub(r"</?\s*asr_text\s*>", "", cleaned, flags=re.IGNORECASE)
        if m is None:
            # タグが無い場合のみ、先頭の "language X" メタ接頭辞を保守的に除去する。
            cleaned = re.sub(
                r"^\s*language\s+[A-Za-z\-]+\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE
            )
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _project_root() -> Path:
        return _paths.REPO_ROOT


    def _bound_torch_threads(self) -> None:
        """このプロセスの torch のスレッド上限を決める。

        torch の既定はコア数そのまま（24 論理コアのマシンで intra 24 / interop 24）。
        turn_backend="maai" のときはターン判定用の MaAI がこのプロセスで torch を使うので、
        放置すると MaAI ワーカー（torch 8 + ONNX 8）や TTS とコアを奪い合う。
        このマシンは 8P+16E のハイブリッドで P コアは 8 本しかない。

        以前は silero_vad の import 副作用（モジュール直下の set_num_threads(1)）が
        たまたま上限になっていたが、それは silero を使わない設定でも効いてしまう
        事故だったので取り除いた。代わりにここで明示する。
        """
        try:
            import torch  # noqa: PLC0415

            torch.set_num_threads(self.torch_cpu_intra_threads)
            # interop は一度実行すると変更できないので、失敗しても致命的ではない。
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        except Exception as ex:  # noqa: BLE001
            # __init__ の途中から呼ばれるため self.on_log はまだ無い。stderr に出す。
            import sys  # noqa: PLC0415

            print(f"[ASRService] torch スレッド数の設定をスキップしました: {ex}",
                  file=sys.stderr, flush=True)

    @staticmethod
    def _usable_torch_device(requested: str) -> str:
        """CUDA を要求されたが使えない場合は cpu を返す。"""
        if not requested.lower().startswith(("cuda", "gpu")):
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[device] MaAI VAP: CUDA が使えないため cpu で実行します（要求 {requested!r}）。",
            flush=True,
        )
        return "cpu"

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        text = (path or "").strip()
        if not text:
            return ""
        candidate = Path(text)
        if candidate.is_absolute():
            return text
        return str(_paths.resolve_under_repo(candidate))

    @staticmethod
    def _sherpa_onnx_git_build_marker() -> Path:
        return ASRService._project_root() / ".sherpa_onnx_git_built"

    def _ensure_nemotron_onnx_git_sherpa(self) -> None:
        marker = self._sherpa_onnx_git_build_marker()
        if marker.is_file():
            return
        # macOS・Linux 用は scripts/upgrade_sherpa_onnx.sh（以前は Windows の .ps1 しか案内していなかった）
        how = (
            " PowerShell で scripts/upgrade_sherpa_onnx.ps1 を実行し、"
            if os.name == "nt"
            else " bash scripts/upgrade_sherpa_onnx.sh を実行し（C++ コンパイラーが要ります。数分かかります）、"
        )
        raise RuntimeError(
            "多言語 Nemotron ONNX には sherpa-onnx v1.13.5 Git ビルド（品質パッチ込み）が必要です。"
            f" マーカー {marker.name} がありません（素の PyPI は max_symbols=10 / NeMo mel 未対応。"
            " prompt_index だけなら PyPI 1.13.3+ で足ります）。"
            + how
            + " 直後に uv sync しないで GUI を再起動してください。"
        )

    @staticmethod
    def _sherpa_onnx_version_str() -> str:
        try:
            engine_manager.require_module_installed("sherpa_onnx", "sherpa-onnx")
            import sherpa_onnx

            # モジュールの __version__ を先に見る。`uv sync` は dist-info だけを PyPI 版で
            # 置き換えることがあり（実機で site-packages のコードは 1.13.5+cuda のまま
            # dist-info が 1.13.2 になっていた）、metadata を先に読むと動いている版と
            # 食い違うログが出る。setup_env.ps1 の Get-SherpaVersion も __version__ を見る。
            return str(getattr(sherpa_onnx, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            pass
        try:
            import importlib.metadata

            for name in ("sherpa-onnx", "sherpa_onnx"):
                try:
                    return str(importlib.metadata.version(name))
                except importlib.metadata.PackageNotFoundError:
                    continue
        except Exception:  # noqa: BLE001
            pass
        return "unknown"

    @staticmethod
    def _nemotron_onnx_encoder_has_prompt_index(encoder_path: str) -> bool:
        path = ASRService._resolve_model_path(encoder_path)
        if not path or not Path(path).is_file():
            return False
        try:
            import onnx

            model = onnx.load(path, load_external_data=False)
            return any(str(i.name) == "prompt_index" for i in model.graph.input)
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _normalize_nemotron_onnx_language(value: str) -> str:
        text = (value or "").strip()
        if not text or text.lower() in {"auto", "automatic", "detect", "default"}:
            return ""
        low = text.lower().replace("_", "-")
        aliases = {
            "ja": "ja",
            "jp": "ja",
            "japanese": "ja",
            "ja-jp": "ja",
            "en": "en",
            "english": "en",
            "zh": "zh",
            "chinese": "zh",
        }
        mapped = aliases.get(low, text)
        if mapped.lower() in {"ja", "ja-jp"}:
            return "ja"
        return mapped

    def _apply_nemotron_onnx_stream_language(self, stream) -> None:
        lang = self._normalize_nemotron_onnx_language(self.nemotron_onnx_language)
        try:
            stream.set_option("language", lang)
        except Exception as ex:  # noqa: BLE001
            self._log(f"nemotron-onnx: set_option(language={lang!r}) failed: {ex}")

    def _verify_nemotron_onnx_runtime(self) -> None:
        rec = self._nemotron_onnx_recognizer
        if rec is None:
            return
        needs_prompt = self._nemotron_onnx_encoder_has_prompt_index(self.nemotron_onnx_encoder_path)
        stream = rec.create_stream()
        self._apply_nemotron_onnx_stream_language(stream)
        sample_rate = int(self.sherpa_sample_rate)
        silence = np.zeros(max(1, sample_rate // 10), dtype=np.float32)
        stream.accept_waveform(sample_rate, silence)
        tail = np.zeros(max(1, sample_rate // 10), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail)
        stream.input_finished()
        try:
            while rec.is_ready(stream):
                rec.decode_stream(stream)
        except Exception as ex:
            msg = str(ex).lower()
            if not needs_prompt and "prompt_index" not in msg:
                raise
            ver = self._sherpa_onnx_version_str()
            marker = self._sherpa_onnx_git_build_marker()
            hint = (
                " scripts/upgrade_sherpa_onnx.ps1 を実行し、uv sync で PyPI 版に戻さないでください。"
                if not marker.is_file()
                else " Git ビルドが上書きされた可能性があります。upgrade スクリプトを再実行してください。"
            )
            raise RuntimeError(
                "多言語 Nemotron ONNX には sherpa-onnx v1.13.3+（GitHub PR #3671）が必要です。"
                f" 現在: {ver}。{hint} ({ex})"
            ) from ex

    @staticmethod
    def _normalize_qwen_language(language: str) -> str:
        raw = (language or "").strip()
        if not raw:
            return "ja"
        normalized = raw.lower().replace("_", "-")
        aliases = {
            "japanese": "ja",
            "japanese-ja": "ja",
            "ja-jp": "ja",
            "jp": "ja",
            "english": "en",
            "chinese": "zh",
        }
        return aliases.get(normalized, raw)

    @staticmethod
    def _normalize_qwen_ws_path(path: str) -> str:
        raw = (path or "").strip()
        if not raw:
            return "/ws"
        return raw if raw.startswith("/") else f"/{raw}"

    @staticmethod
    def _to_pcm16le_bytes(audio: np.ndarray) -> bytes:
        normalized = np.asarray(audio, dtype=np.float32).flatten()
        clipped = np.clip(normalized, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    # 音声クロックが wall clock より進んでいることを何回警告したか（初回だけ出す）。
    _AUDIO_CLOCK_SKEW_LOG_THRESHOLD_SEC = 0.001

    def _report_latency_sec(self, end_ts: float | None) -> float:
        """発話終端から結果までの遅延。**負の値は返さない。**

        `end_ts` は音声のサンプルクロック由来で、wall clock とは別のクロック:
        `audio/input_worker` が最初のマイクコールバックで
        `time.time() - duration` を基準に置き、以後はサンプル数で進める。そのため

          - 起動時のパイプライン遅延（デバイスバッファ + IPC）を基準に吸収してしまう
          - デバイスの実サンプルレートが公称値とずれていると累積する

ため、`end_ts` が `time.time()` を追い越すことがある。音声は未来には録れないので、
        負の遅延は「音声クロックが進んでいる」ことの表れ。表示は 0 で止めるが、
        黙って潰すと原因が追えないのでズレの大きさを**初回だけ**ログに出す。
        """
        if not end_ts:
            return 0.0
        raw = time.time() - float(end_ts)
        if raw >= 0.0:
            return raw
        if not self._audio_clock_skew_logged and -raw > self._AUDIO_CLOCK_SKEW_LOG_THRESHOLD_SEC:
            self._audio_clock_skew_logged = True
            self._log(
                f"音声クロックが wall clock より {-raw:.3f}s 進んでいます"
                "（デバイスのレート誤差や起動時のパイプライン遅延によるもの）。"
                "遅延表示は 0 で止めます。"
            )
        return 0.0

    def _publish_result(
        self,
        text: str,
        end_ts: float | None,
        engine: str,
        is_final: bool,
        is_streaming: bool,
        segment_start_ts: float | None = None,
        segment_end_ts: float | None = None,
    ) -> None:
        """ASRResult には常に utterance_index を付与する（同一発話の partial/final を対応付ける）。"""
        # マイクのフレーム送出からテキストまで（端から端まで・壁時計）
        latency = self._end_to_end_latency_sec(end_ts)
        if segment_end_ts is None:
            segment_end_ts = end_ts
        if self._active_utterance_index is None:
            self._active_utterance_index = int(self._utterance_index_counter)
            self._utterance_index_counter += 1
        utterance_index = int(self._active_utterance_index)
        if bool(is_final):
            self._active_utterance_index = None
            self._last_partial_for_flush = None
        result = ASRResult(
            text=text,
            engine=engine,
            is_final=bool(is_final),
            is_streaming=bool(is_streaming),
            latency_sec=latency,
            utterance_index=utterance_index,
            end_of_utterance_ts=end_ts,
            segment_start_ts=segment_start_ts,
            segment_end_ts=segment_end_ts,
        )
        self._queue_put(self.transcript_queue, result, policy=self.QUEUE_POLICY_DROP_OLDEST)
        if self.on_result is not None:
            self.on_result(result)
        if (not bool(is_final)) and bool(is_streaming):
            self._last_partial_for_flush = {
                "text": text,
                "engine": engine,
                "segment_start_ts": segment_start_ts,
                "segment_end_ts": segment_end_ts,
            }
        if self.verbose:
            kind = "final" if result.is_final else "partial"
            stream = "stream" if result.is_streaming else "non-stream"
            self._log(f"[{engine}][{stream}/{kind}] {text} latency={latency:.3f}s")

    def _queue_put(self, target_queue: queue.Queue, item, policy: str = QUEUE_POLICY_DROP_OLDEST) -> bool:
        if policy == self.QUEUE_POLICY_BLOCK:
            try:
                target_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                return False
        try:
            target_queue.put_nowait(item)
            return True
        except queue.Full:
            if policy == self.QUEUE_POLICY_DROP_NEWEST:
                return False
            if policy != self.QUEUE_POLICY_DROP_OLDEST:
                return False
            try:
                _ = target_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                target_queue.put_nowait(item)
                return True
            except queue.Full:
                return False

    def _log(self, message: str) -> None:
        if self.on_log is not None:
            self.on_log(message)

    def _error(self, message: str) -> None:
        if self.on_error is not None:
            self.on_error(message)
        if self.on_log is not None:
            self.on_log(f"ERROR: {message}")


def main() -> int:
    def on_log(message: str) -> None:
        print(f"[ASRService] {message}")

    def on_error(message: str) -> None:
        print(f"[ASRService][ERROR] {message}")

    def on_result(result: ASRResult) -> None:
        print(f"[ASRService][{result.engine}] {result.text} (latency={result.latency_sec:.3f}s)")

    service = ASRService(
        asr_engine="faster-whisper",
        whisper_hf_id="RoachLin/kotoba-whisper-v2.2-faster",
        whisper_device="cuda",
        whisper_compute_type="int8_float16",
        vad_backend="silero",
        turn_backend="maai",
        sample_rate=16000,
        frame_rate=20.0,
        vad_start_threshold=0.1,
        vad_start_hold_sec=0.1,
        vad_end_threshold=0.2,
        vad_end_hold_sec=0.1,
        min_segment_sec=0.5,
        inference_queue_size=4,
        verbose=True,
        on_log=on_log,
        on_error=on_error,
        on_result=on_result,
    )

    try:
        service.start()
        print("Listening... Ctrl+C で終了します。")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("停止します。")
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
