"""`controller_config.json` の読み書き（controller.py から移設）。

- `_load_app_config`: cfg → ウィジェット
- `_collect_app_config_payload`: ウィジェット → cfg（純粋に dict を組み立てるだけ）
- `_save_app_config`: 上を呼んでファイルへ書く

平坦なフィールドは `app_config_spec.APP_CONFIG_FIELDS` のループで処理し、
エンジン切替のマイグレーションや `tts_template` の JSON など、宣言化すると
かえって読みにくくなる特殊ケースだけを明示コードで持つ。

本文は controller.py から 1 文字も変えずに移してある。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .. import paths as _paths
from ..dialog.settings import normalize_tts_engine
from .app_config_spec import (
    APP_CONFIG_FIELDS,
    _LLM_CUSTOM_DEFAULT_MODEL,
    _LLM_OPENAI_DEFAULT_MODEL,
)
from ..maai.plot_window import MAAI_PLOT_VISIBILITY_SPECS
from ..utils import json_config_bool, to_bool


class AppConfigIOMixin:
    """`controller_config.json` の読み書き。`MMDAgentController` に mixin する。"""

    def _load_app_config(self) -> None:
        self._is_loading_app_config = True
        try:
            cfg = self.config_store.load()
            exec_path = str(cfg.get("exec_path", "")).strip()
            mdf_path = str(cfg.get("mdf_path", "")).strip()
            if exec_path and Path(exec_path).exists():
                self.exec_edit.setText(exec_path)
            mdf_host_path = self._host_path_for_backend_value(mdf_path) if mdf_path else None
            if mdf_path and mdf_host_path is not None and mdf_host_path.exists():
                self.mdf_edit.setText(mdf_path)
                self.load_mdf_settings()
            saved_tts_engine = str(cfg.get("tts_engine", "VOICEVOX")).strip() or "VOICEVOX"
            # Kokoro系は TTFT 実測で最速の Kokoro-RealtimeTTS のみ残したため、
            # 旧設定(Kokoro-onnx / Kokoro-FastAPI)は RealtimeTTS に移行する。
            if saved_tts_engine in ("Kokoro-onnx", "Kokoro-FastAPI"):
                saved_tts_engine = "Kokoro-RealtimeTTS"
            # 削除した PyTorch 版 Qwen3-TTS（旧 "Qwen3-TTS"）は audio.cpp 版へ移行する。
            saved_tts_engine = normalize_tts_engine(saved_tts_engine)
            self.tts_engine_combo.setCurrentText(saved_tts_engine)
            self._apply_app_config_field(cfg, "voicevox_exe_path")
            self._apply_app_config_field(cfg, "tts_host")
            self._apply_app_config_field(cfg, "tts_port")
            self._apply_app_config_field(cfg, "tts_base_path")
            migrated_voice = str(cfg.get("tts_voice", "")).strip()
            voice_voicevox = str(cfg.get("tts_voice_voicevox", "")).strip()
            voice_kokoro_realtime = str(cfg.get("tts_voice_kokoro_realtimetts", "")).strip()
            voice_voicevox_core = str(cfg.get("tts_voice_voicevox_core", "")).strip()
            voice_cartesia = str(cfg.get("tts_voice_cartesia", "")).strip()
            if not voice_cartesia:
                voice_cartesia = str(cfg.get("cartesia_tts_voice_id", "")).strip()
            voice_template = str(cfg.get("tts_voice_template", "")).strip()
            voice_aivis = str(cfg.get("tts_voice_aivisspeech", "")).strip()
            voice_supertonic = str(cfg.get("tts_voice_supertonic", "")).strip()
            voice_irodori = str(cfg.get("tts_voice_irodori", "")).strip()
            if voice_voicevox:
                self._tts_voice_by_engine["VOICEVOX"] = voice_voicevox
            if voice_kokoro_realtime:
                self._tts_voice_by_engine["Kokoro-RealtimeTTS"] = voice_kokoro_realtime
            if voice_voicevox_core:
                self._tts_voice_by_engine["VOICEVOX-Core"] = voice_voicevox_core
            if voice_cartesia:
                self._tts_voice_by_engine["Cartesia-Realtime"] = voice_cartesia
            if voice_template:
                self._tts_voice_by_engine["TEMPLATE"] = voice_template
            if voice_aivis:
                self._tts_voice_by_engine["AivisSpeech"] = voice_aivis
            if voice_supertonic:
                self._tts_voice_by_engine["Supertonic-3"] = voice_supertonic
            if voice_irodori:
                self._tts_voice_by_engine["Irodori-v4"] = voice_irodori
            current_engine = self.tts_engine_combo.currentText().strip() or "VOICEVOX"
            if not migrated_voice:
                if current_engine == "VOICEVOX":
                    migrated_voice = str(cfg.get("voicevox_speaker", "1"))
                else:
                    migrated_voice = self._tts_voice_by_engine.get(current_engine, "")
            if migrated_voice and not self._tts_voice_by_engine.get(current_engine):
                self._tts_voice_by_engine[current_engine] = migrated_voice
            self._restore_engine_voice(current_engine)
            self._apply_app_config_field(cfg, "tts_model")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_server_exe")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_model_path")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_host")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_port")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_model_id")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_speaker")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_instruct")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_language")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_do_sample")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_temperature")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_top_k")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_top_p")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_repetition_penalty")
            self._apply_app_config_field(cfg, "qwen3_tts_cpp_seed")
            self._apply_app_config_field(cfg, "voicevox_core_onnxruntime_file")
            self._apply_app_config_field(cfg, "voicevox_core_openjtalk_dict")
            self._apply_app_config_field(cfg, "voicevox_core_vvm_path")
            self._apply_app_config_field(cfg, "voicevox_core_cpu_num_threads")
            self._apply_app_config_field(cfg, "voicevox_core_speed_scale")
            self.voicevox_core_device_combo.setCurrentText(str(cfg.get("voicevox_core_device", "GPU")))
            self._apply_app_config_field(cfg, "cartesia_tts_model_id")
            self._apply_app_config_field(cfg, "cartesia_tts_sample_rate")
            self._apply_app_config_field(cfg, "cartesia_tts_api_key")
            self._apply_app_config_field(cfg, "cartesia_tts_api_key_env")
            self._apply_app_config_field(cfg, "aivisspeech_host")
            self._apply_app_config_field(cfg, "aivisspeech_port")
            self._apply_app_config_field(cfg, "aivisspeech_exe_path")
            self._apply_app_config_field(cfg, "supertonic_host")
            self._apply_app_config_field(cfg, "supertonic_port")
            self._apply_app_config_field(cfg, "supertonic_model_id")
            self._apply_app_config_field(cfg, "supertonic_server_exe")
            self._apply_app_config_field(cfg, "supertonic_model_path")
            self._apply_app_config_field(cfg, "supertonic_steps")
            self._apply_app_config_field(cfg, "irodori_python")
            self._apply_app_config_field(cfg, "irodori_checkpoint")
            self._apply_app_config_field(cfg, "irodori_reference_audio")
            self._apply_app_config_field(cfg, "irodori_steps")
            self._apply_app_config_field(cfg, "irodori_backend")
            self._apply_app_config_field(cfg, "irodori_speechcpp_device")
            self._apply_app_config_field(cfg, "irodori_speechcpp_steps")
            self._apply_app_config_field(cfg, "irodori_seed")
            self._apply_app_config_field(cfg, "irodori_audiocpp_model")
            self._apply_app_config_field(cfg, "irodori_audiocpp_host")
            self._apply_app_config_field(cfg, "irodori_audiocpp_port")
            self._apply_app_config_field(cfg, "lfm2_audio_host")
            self._apply_app_config_field(cfg, "lfm2_audio_port")
            self._apply_app_config_field(cfg, "lfm2_audio_model_id")
            self._apply_app_config_field(cfg, "lfm2_audio_model_path")
            self._apply_app_config_field(cfg, "lfm2_audio_server_exe")
            self._apply_app_config_field(cfg, "tsukasa_root")
            self._apply_app_config_field(cfg, "tsukasa_reference_audio")
            self._apply_app_config_field(cfg, "tsukasa_steps")
            self._apply_app_config_field(cfg, "tsukasa_alpha")
            self._apply_app_config_field(cfg, "tsukasa_beta")
            self.tts_auth_mode_combo.setCurrentText(str(cfg.get("tts_auth_mode", "none")))
            self._apply_app_config_field(cfg, "tts_auth_env")
            self._apply_app_config_field(cfg, "tts_auth_header")
            self._apply_app_config_field(cfg, "tts_auth_prefix")
            self.tts_template_edit.setPlainText(str(cfg.get("tts_template", self._load_default_tts_template_text())))
            self._apply_app_config_field(cfg, "tts_test_text")
            input_streaming_enabled = json_config_bool(cfg, "tts_input_streaming_enabled", True)
            self.tts_input_streaming_check.setChecked(input_streaming_enabled)
            self._apply_app_config_field(cfg, "tts_segment_gap_silence_sec")
            if hasattr(self, "mmd_burst_snd_frames_edit"):
                try:
                    mbf = max(0, min(500, int(str(cfg.get("mmd_burst_snd_frames", "20")).strip() or "20")))
                except ValueError:
                    mbf = 20
                self.mmd_burst_snd_frames_edit.setText(str(mbf))
            if hasattr(self, "mmd_blink_enable_check"):
                from ..mmd.blink import (
                    DEFAULT_BLINK_INTERVAL_MEAN_SEC,
                    normalize_blink_interval_mean_sec,
                    sanitize_blink_morph,
                )

                self._apply_app_config_field(cfg, "mmd_blink_enable")
                self.mmd_blink_interval_mean_sec_edit.setText(
                    str(
                        normalize_blink_interval_mean_sec(
                            cfg.get("mmd_blink_interval_mean_sec", DEFAULT_BLINK_INTERVAL_MEAN_SEC)
                        )
                    )
                )
                self.mmd_blink_morph_edit.setText(
                    sanitize_blink_morph(cfg.get("mmd_blink_morph", "まばたき"))
                )
            if hasattr(self, "mmd_blink_mode_combo"):
                from ..mmd.blink import normalize_blink_mode

                idx = self.mmd_blink_mode_combo.findData(
                    normalize_blink_mode(cfg.get("mmd_blink_mode"))
                )
                self.mmd_blink_mode_combo.setCurrentIndex(max(0, idx))
            if hasattr(self, "audio_output_mode_combo"):
                om = str(cfg.get("audio_output_mode", "mmd")).strip().lower() or "mmd"
                oi = self.audio_output_mode_combo.findData(om)
                if oi >= 0:
                    self.audio_output_mode_combo.setCurrentIndex(oi)
                else:
                    self.audio_output_mode_combo.setCurrentIndex(0)
            self._apply_app_config_field(cfg, "audio_output_tcp_host")
            self._apply_app_config_field(cfg, "audio_output_tcp_port")
            saved_backend = str(cfg.get("llm_backend", "LM Studio")).strip() or "LM Studio"
            legacy_llm_model = str(cfg.get("llm_model", "")).strip()
            lmstudio_model = str(cfg.get("llm_model_lmstudio", "")).strip()
            openai_model = str(cfg.get("llm_model_openai", "")).strip()
            custom_model = str(cfg.get("llm_model_custom", "")).strip()
            # 旧形式（llm_model のみ）や分割キー追加前の JSON から、バックエンドごとに復元する。
            if legacy_llm_model:
                if "llm_model_lmstudio" not in cfg and saved_backend == "LM Studio":
                    lmstudio_model = lmstudio_model or legacy_llm_model
                if "llm_model_openai" not in cfg and saved_backend == "OpenAI":
                    openai_model = openai_model or legacy_llm_model
                if "llm_model_custom" not in cfg and saved_backend == "Custom":
                    custom_model = custom_model or legacy_llm_model
            self._llm_model_by_backend["LM Studio"] = lmstudio_model
            self._llm_model_by_backend["OpenAI"] = openai_model or _LLM_OPENAI_DEFAULT_MODEL
            self._llm_model_by_backend["Custom"] = custom_model or _LLM_CUSTOM_DEFAULT_MODEL
            self._last_llm_backend = saved_backend
            self.llm_backend_combo.setCurrentText(saved_backend)
            self._apply_app_config_field(cfg, "llm_host")
            self._write_llm_model_for_backend(
                "LM Studio",
                self._llm_model_by_backend.get("LM Studio", ""),
            )
            self._write_llm_model_for_backend(
                "OpenAI",
                self._llm_model_by_backend.get("OpenAI", _LLM_OPENAI_DEFAULT_MODEL) or _LLM_OPENAI_DEFAULT_MODEL,
            )
            self._write_llm_model_for_backend(
                "Custom",
                self._llm_model_by_backend.get("Custom", _LLM_CUSTOM_DEFAULT_MODEL) or _LLM_CUSTOM_DEFAULT_MODEL,
            )
            self._apply_app_config_field(cfg, "llm_api_key")
            self._apply_app_config_field(cfg, "llm_api_key_env")
            if hasattr(self, "llm_custom_base_url_edit"):
                self._apply_app_config_field(cfg, "llm_custom_base_url")
                self._apply_app_config_field(cfg, "llm_custom_api_key")
                self._apply_app_config_field(cfg, "llm_custom_api_key_env")
                self.llm_custom_send_temperature_check.setChecked(
                    json_config_bool(cfg, "llm_custom_send_temperature", True)
                )
                self.llm_custom_send_max_tokens_check.setChecked(
                    json_config_bool(cfg, "llm_custom_send_max_tokens", True)
                )
                self._apply_app_config_field(cfg, "llm_custom_temperature")
                if hasattr(self, "llm_custom_extra_body_thinking_check"):
                    self.llm_custom_extra_body_thinking_check.setChecked(
                        json_config_bool(cfg, "llm_custom_extra_body_thinking", False)
                    )
            self._apply_app_config_field(cfg, "llm_system_prompt")
            self._apply_app_config_field(cfg, "llm_context_len")
            self._apply_app_config_field(cfg, "llm_max_tokens")
            self.llm_tts_stream_check.setChecked(json_config_bool(cfg, "llm_tts_stream_enabled", False))
            self.llm_test_tts_input_streaming_check.setChecked(
                json_config_bool(cfg, "llm_test_tts_input_streaming_enabled", True)
            )
            llm_split_custom_delimiters = str(
                cfg.get("llm_split_custom_delimiters", "。．！？!?\\n")
            )
            llm_split_custom_strip_emoji = json_config_bool(cfg, "llm_split_custom_strip_emoji", True)
            llm_split_custom_strip_parenthetical = json_config_bool(cfg, "llm_split_custom_strip_parenthetical", False)
            llm_split_custom_strip_standalone_warai = json_config_bool(
                cfg, "llm_split_custom_strip_standalone_warai", False
            )
            self.llm_split_custom_delimiters_edit.setText(llm_split_custom_delimiters or "。．！？!?\\n")
            self.llm_split_custom_strip_emoji_check.setChecked(llm_split_custom_strip_emoji)
            self.llm_split_custom_strip_parenthetical_check.setChecked(llm_split_custom_strip_parenthetical)
            self.llm_split_custom_strip_standalone_warai_check.setChecked(llm_split_custom_strip_standalone_warai)
            self._sync_llm_test_tts_input_streaming_widgets()
            self._voice_dialog_interrupt_on_user_speech = to_bool(
                str(cfg.get("voice_dialog_interrupt_on_user_speech", True))
            )
            interrupt_check = getattr(self, "voice_dialog_interrupt_check", None)
            if interrupt_check is not None:
                interrupt_check.setChecked(self._voice_dialog_interrupt_on_user_speech)
            self.voice_dialog_interrupt_vad_threshold_edit.setText(
                str(
                    cfg.get(
                        "voice_dialog_interrupt_vad_threshold",
                        cfg.get("asr_vad_on_threshold", "0.1"),
                    )
                )
            )
            self.voice_dialog_interrupt_pnow_threshold_edit.setText(
                str(
                    cfg.get(
                        "voice_dialog_interrupt_p_now_threshold",
                        cfg.get("asr_maai_turn_threshold", "0.5"),
                    )
                )
            )
            self._apply_app_config_field(cfg, "voice_dialog_opening_utterance")
            if hasattr(self, "voice_dialog_scheduled_exit_sec_edit"):
                self.voice_dialog_scheduled_exit_sec_edit.setText(
                    str(cfg.get("voice_dialog_scheduled_exit_sec", "0"))
                )
            if hasattr(self, "voice_dialog_scheduled_exit_utterance_edit"):
                self.voice_dialog_scheduled_exit_utterance_edit.setText(
                    str(cfg.get("voice_dialog_scheduled_exit_utterance", ""))
                )
            if hasattr(self, "voice_dialog_idle_take_turn_sec_edit"):
                self.voice_dialog_idle_take_turn_sec_edit.setText(
                    str(cfg.get("voice_dialog_idle_take_turn_sec", "0"))
                )
            if hasattr(self, "obs_remote_host_edit"):
                self._apply_app_config_field(cfg, "obs_remote_host")
                self._apply_app_config_field(cfg, "obs_remote_port")
                self.obs_remote_password_edit.setText(str(cfg.get("obs_remote_password", "")))
                self.obs_remote_enabled_check.setChecked(
                    json_config_bool(cfg, "obs_remote_enabled", False)
                )
                self.obs_remote_link_maai_check.setChecked(
                    json_config_bool(cfg, "obs_remote_link_maai", False)
                )
                self.obs_remote_link_voice_check.setChecked(
                    json_config_bool(cfg, "obs_remote_link_voice", False)
                )
            self._on_llm_backend_changed(self.llm_backend_combo.currentText().strip())
            self._last_lmstudio_model = self._read_llm_model_for_backend("LM Studio")
            self.asr_engine_combo.setCurrentText(str(cfg.get("asr_engine", "sherpa-onnx")))
            self.asr_compare_mode_check.setChecked(to_bool(str(cfg.get("asr_compare_mode_enabled", False))))
            self.asr_compare_top_engine_combo.setCurrentText(str(cfg.get("asr_compare_top_engine", "sherpa-onnx")))
            self.asr_compare_bottom_engine_combo.setCurrentText(
                str(cfg.get("asr_compare_bottom_engine", "faster-whisper"))
            )
            self.asr_vad_backend_combo.setCurrentText(str(cfg.get("asr_vad_backend", "maai")))
            self.asr_turn_backend_combo.setCurrentText(str(cfg.get("asr_turn_backend", "maai")))
            self.asr_device_combo.setCurrentText(str(cfg.get("asr_device", "cuda")))
            self.asr_compute_type_combo.setCurrentText(str(cfg.get("asr_compute_type", "int8_float16")))
            self._apply_whisper_fields_from_cfg(cfg)
            self._apply_app_config_field(cfg, "asr_sherpa_tokens")
            self._apply_app_config_field(cfg, "asr_sherpa_encoder")
            self._apply_app_config_field(cfg, "asr_sherpa_decoder")
            self._apply_app_config_field(cfg, "asr_sherpa_joiner")
            self._apply_app_config_field(cfg, "asr_sherpa_num_threads")
            if hasattr(self, "asr_sherpa_provider_combo"):
                _sherpa_prov = str(cfg.get("asr_sherpa_provider", "cpu")).strip().lower()
                self.asr_sherpa_provider_combo.setCurrentText(
                    "GPU (cuda)" if _sherpa_prov == "cuda" else "CPU"
                )
            self._apply_app_config_field(cfg, "asr_qwen3_onnx_conv_frontend")
            self._apply_app_config_field(cfg, "asr_qwen3_onnx_encoder")
            self._apply_app_config_field(cfg, "asr_qwen3_onnx_decoder")
            self._apply_app_config_field(cfg, "asr_qwen3_onnx_tokenizer")
            self._apply_app_config_field(cfg, "asr_qwen3_onnx_language")
            if hasattr(self, "asr_qwen3_onnx_provider_combo"):
                _qwen_prov = str(cfg.get("asr_qwen3_onnx_provider", "cpu")).strip().lower()
                self.asr_qwen3_onnx_provider_combo.setCurrentText(
                    "GPU (cuda)" if _qwen_prov == "cuda" else "CPU"
                )
            if hasattr(self, "asr_qwen3_onnx_num_threads_edit"):
                self.asr_qwen3_onnx_num_threads_edit.setText(
                    str(cfg.get("asr_qwen3_onnx_num_threads", "8"))
                )
            # sherpa-onnx OfflineRecognizer 系のテキスト欄。以前は「APP_CONFIG_FIELDS が見る」として
            # 読み込みを書いていなかったが、宣言テーブルは保存にしか使われておらず、起動のたびに
            # 既定値へ戻っていた。
            for _key in (
                "asr_parakeet_onnx_dir", "asr_parakeet_onnx_num_threads",
                "asr_sensevoice_onnx_dir", "asr_sensevoice_onnx_num_threads", "asr_sensevoice_onnx_language",
                "asr_funasr_nano_onnx_dir", "asr_funasr_nano_onnx_num_threads",
                "asr_funasr_nano_onnx_language", "asr_funasr_nano_onnx_max_new_tokens",
                "asr_cohere_onnx_dir", "asr_cohere_onnx_num_threads", "asr_cohere_onnx_language",
                "asr_speechcpp_model", "asr_speechcpp_device", "asr_speechcpp_threads",
            ):
                self._apply_app_config_field(cfg, _key)
            # 宣言テーブルに載らない combo / checkbox（既定は全て CPU / 有効）。
            for _key, _attr in (
                ("asr_parakeet_onnx_provider", "asr_parakeet_onnx_provider_combo"),
                ("asr_sensevoice_onnx_provider", "asr_sensevoice_onnx_provider_combo"),
                ("asr_funasr_nano_onnx_provider", "asr_funasr_nano_onnx_provider_combo"),
                ("asr_cohere_onnx_provider", "asr_cohere_onnx_provider_combo"),
            ):
                _combo = getattr(self, _attr, None)
                if _combo is not None:
                    _prov = str(cfg.get(_key, "cpu")).strip().lower()
                    _combo.setCurrentText("GPU (cuda)" if _prov == "cuda" else "CPU")
            for _key, _attr in (
                ("asr_sensevoice_onnx_itn", "asr_sensevoice_onnx_itn_check"),
                ("asr_funasr_nano_onnx_itn", "asr_funasr_nano_onnx_itn_check"),
                ("asr_cohere_onnx_punct", "asr_cohere_onnx_punct_check"),
                ("asr_cohere_onnx_itn", "asr_cohere_onnx_itn_check"),
            ):
                _check = getattr(self, _attr, None)
                if _check is not None:
                    _check.setChecked(to_bool(str(cfg.get(_key, True))))
            self._apply_app_config_field(cfg, "asr_nemotron_onnx_tokens")
            self._apply_app_config_field(cfg, "asr_nemotron_onnx_encoder")
            self._apply_app_config_field(cfg, "asr_nemotron_onnx_decoder")
            self._apply_app_config_field(cfg, "asr_nemotron_onnx_joiner")
            self._apply_app_config_field(cfg, "asr_nemotron_onnx_language")
            if hasattr(self, "asr_nemotron_onnx_compute_combo"):
                _compute = str(cfg.get("asr_nemotron_onnx_compute", "gpu-fp32")).strip()
                self.asr_nemotron_onnx_compute_combo.setCurrentText(
                    "CPU (int8)" if _compute == "cpu-int8" else "GPU (fp32)"
                )
            self._apply_app_config_field(cfg, "asr_qwen_host")
            self._apply_app_config_field(cfg, "asr_qwen_port")
            self._apply_app_config_field(cfg, "asr_external_host")
            self._apply_app_config_field(cfg, "asr_external_port")
            self._apply_app_config_field(cfg, "asr_external_audio_host")
            self._apply_app_config_field(cfg, "asr_external_audio_port")
            self._apply_app_config_field(cfg, "asr_external_start_script_path")
            self._write_input_text(
                self.asr_qwen_model_edit,
                str(cfg.get("asr_qwen_model", "Qwen/Qwen3-ASR-1.7B")),
            )
            self._apply_app_config_field(cfg, "asr_qwen_custom_model_path")
            self._apply_app_config_field(cfg, "asr_qwen_gpu_memory_utilization")
            self._apply_app_config_field(cfg, "asr_qwen_max_model_len")
            self._apply_app_config_field(cfg, "asr_qwen_chunk_sec")
            self._apply_app_config_field(cfg, "asr_qwen_timeout_sec")
            self._apply_app_config_field(cfg, "asr_qwen_ws_path")
            self._apply_app_config_field(cfg, "asr_qwen_session_timeout_sec")
            self._apply_app_config_field(cfg, "asr_qwen_no_text_flush_sec")
            self._apply_app_config_field(cfg, "asr_qwen_min_return_audio_sec")
            self._apply_app_config_field(cfg, "asr_nemotron_model")
            self.asr_nemotron_chunk_combo.setCurrentText(str(cfg.get("asr_nemotron_chunk_size_ms", "320")))
            if hasattr(self, "asr_nemotron_onnx_chunk_combo"):
                self.asr_nemotron_onnx_chunk_combo.setCurrentText(
                    str(cfg.get("asr_nemotron_chunk_size_ms", "320"))
                )
            finalize_mode = str(cfg.get("asr_nemotron_finalize_mode", "vad"))
            self.asr_nemotron_finalize_combo.setCurrentText(finalize_mode)
            if hasattr(self, "asr_nemotron_onnx_finalize_combo"):
                self.asr_nemotron_onnx_finalize_combo.setCurrentText(finalize_mode)
            self._apply_app_config_field(cfg, "asr_nemotron_window_sec")
            self._apply_app_config_field(cfg, "asr_nemotron_max_symbols")
            self._apply_app_config_field(cfg, "asr_nemotron_lang")
            self.asr_nemotron_device_combo.setCurrentText(str(cfg.get("asr_nemotron_device", "cuda")))
            self._apply_app_config_field(cfg, "asr_nemotron_python")
            self.asr_qwen_finish_on_vad_end_check.setChecked(
                to_bool(str(cfg.get("asr_qwen_finish_on_vad_end", True)))
            )
            self.asr_qwen_stop_container_check.setChecked(
                to_bool(str(cfg.get("asr_qwen_stop_with_compose_down", False)))
            )
            if "asr_qwen_vllm_enforce_eager" in cfg:
                self.asr_qwen_enforce_eager_check.setChecked(
                    to_bool(str(cfg.get("asr_qwen_vllm_enforce_eager")))
                )
            else:
                self.asr_qwen_enforce_eager_check.setChecked(os.name == "nt")
            self._apply_app_config_field(cfg, "asr_chirp_project_id")
            self._apply_app_config_field(cfg, "asr_chirp_location")
            self._apply_app_config_field(cfg, "asr_chirp_recognizer")
            self._apply_app_config_field(cfg, "asr_chirp_model")
            self._apply_app_config_field(cfg, "asr_smart_turn_model")
            if hasattr(self, "maai_device_combo"):
                dev = str(cfg.get("maai_device") or "cuda").strip().lower()
                if dev not in {"cpu", "cuda"}:
                    dev = "cuda"
                self.maai_device_combo.setCurrentText("cuda" if dev == "cuda" else "cpu")
            self._apply_app_config_field(cfg, "maai_enable_bc")
            self._apply_app_config_field(cfg, "maai_enable_nod")
            self._apply_app_config_field(cfg, "maai_enable_turn_vap")
            if hasattr(self, "maai_test_turn_vap_frame_rate_edit"):
                self.maai_test_turn_vap_frame_rate_edit.setText(
                    str(
                        cfg.get(
                            "maai_test_turn_vap_frame_rate",
                            cfg.get("asr_vap_frame_rate", "20"),
                        )
                    )
                )
            if hasattr(self, "maai_test_turn_vap_mode_edit"):
                self.maai_test_turn_vap_mode_edit.setText(
                    str(cfg.get("maai_test_turn_vap_mode", cfg.get("asr_vap_mode", "vap_mc")))
                )
            if hasattr(self, "maai_test_turn_vap_lang_edit"):
                self.maai_test_turn_vap_lang_edit.setText(
                    str(cfg.get("maai_test_turn_vap_lang", cfg.get("asr_vap_lang", "jp_kyoto")))
                )
            if hasattr(self, "maai_test_turn_vap_model_type_edit"):
                self.maai_test_turn_vap_model_type_edit.setCurrentText(
                    str(
                        cfg.get(
                            "maai_test_turn_vap_model_type",
                            cfg.get("asr_vap_model_type", "normal"),
                        )
                    )
                )
            self._apply_app_config_field(cfg, "maai_test_turn_vap_local_model")
            self._apply_app_config_field(cfg, "maai_test_bc_frame_rate")
            self._apply_app_config_field(cfg, "maai_test_bc_lang")
            if hasattr(self, "maai_test_bc_model_type_edit"):
                self.maai_test_bc_model_type_edit.setCurrentText(str(cfg.get("maai_test_bc_model_type", "normal")))
            self._apply_app_config_field(cfg, "maai_test_bc_local_model")
            self._apply_app_config_field(cfg, "maai_test_nod_frame_rate")
            self._apply_app_config_field(cfg, "maai_test_nod_lang")
            if hasattr(self, "maai_test_nod_model_type_edit"):
                self.maai_test_nod_model_type_edit.setCurrentText(str(cfg.get("maai_test_nod_model_type", "normal-ver2")))
            if hasattr(self, "maai_multi_enabled_check"):
                self.maai_multi_enabled_check.setChecked(json_config_bool(cfg, "maai_multi_enabled", False))
            self._apply_app_config_field(cfg, "maai_multi_frame_rate")
            if hasattr(self, "maai_multi_model_type_edit"):
                self.maai_multi_model_type_edit.setCurrentText(str(cfg.get("maai_multi_model_type", "normal-ver2")))
            if hasattr(self, "maai_use_mimi_onnx_check"):
                self.maai_use_mimi_onnx_check.setChecked(json_config_bool(cfg, "maai_use_mimi_onnx", True))
            if hasattr(self, "maai_use_local_mimi_onnx_check"):
                self.maai_use_local_mimi_onnx_check.setChecked(
                    json_config_bool(cfg, "maai_use_local_mimi_onnx", False)
                )
            self._apply_app_config_field(cfg, "maai_mimi_onnx_cpu_intra_threads")
            self._apply_app_config_field(cfg, "maai_mimi_onnx_cpu_inter_threads")
            self._apply_app_config_field(cfg, "maai_torch_cpu_intra_threads")
            self._apply_app_config_field(cfg, "maai_torch_cpu_interop_threads")
            self._apply_app_config_field(cfg, "maai_inference_chunk_frames")
            if hasattr(self, "maai_root_edit"):
                self.maai_root_edit.setText(
                    str(cfg.get("maai_root", "") or "")
                    or str(_paths.assets_root())
                )
            for _edit_attr, _cfg_key in (
                ("maai_mimi_onnx_path_edit", "maai_mimi_onnx_path"),
                ("maai_mimi_onnx_meta_path_edit", "maai_mimi_onnx_meta_path"),
            ):
                if hasattr(self, _edit_attr):
                    getattr(self, _edit_attr).setText(str(cfg.get(_cfg_key, "") or ""))
            self._sync_maai_local_mimi_onnx_controls()
            self._apply_app_config_field(cfg, "maai_test_nod_count_t0")
            self._apply_app_config_field(cfg, "maai_test_nod_count_t1")
            self._apply_app_config_field(cfg, "maai_test_nod_count_t2")
            self._apply_app_config_field(cfg, "maai_test_nod_swing_threshold")
            if hasattr(self, "maai_backchannel_enabled_check"):
                self.maai_backchannel_enabled_check.setChecked(
                    json_config_bool(cfg, "maai_backchannel_enabled", False)
                )
            if hasattr(self, "maai_backchannel_playback_combo"):
                pb = str(cfg.get("maai_backchannel_playback", "play")).strip() or "play"
                ri = self.maai_backchannel_playback_combo.findData(pb)
                if ri >= 0:
                    self.maai_backchannel_playback_combo.setCurrentIndex(ri)
                else:
                    self.maai_backchannel_playback_combo.setCurrentIndex(0)
            if hasattr(self, "maai_backchannel_prob_threshold_edit"):
                self.maai_backchannel_prob_threshold_edit.setText(
                    str(cfg.get("maai_backchannel_prob_threshold", "0.5"))
                )
            if hasattr(self, "maai_backchannel_min_interval_edit"):
                self.maai_backchannel_min_interval_edit.setText(
                    str(cfg.get("maai_backchannel_min_interval_sec", "1.0"))
                )
            if hasattr(self, "maai_backchannel_consecutive_frames_edit"):
                self.maai_backchannel_consecutive_frames_edit.setText(
                    str(cfg.get("maai_backchannel_consecutive_frames", "1"))
                )
            if hasattr(self, "maai_backchannel_dictionary_path_edit"):
                default_dict = str(_paths.repo_path("backchannel_forms.json"))
                self.maai_backchannel_dictionary_path_edit.setText(
                    str(cfg.get("maai_backchannel_dictionary_path", default_dict))
                )
            if hasattr(self, "audio_input_mode_combo"):
                mode = str(cfg.get("audio_input_mode", "mic")).strip().lower() or "mic"
                i = self.audio_input_mode_combo.findText(mode)
                if i >= 0:
                    self.audio_input_mode_combo.setCurrentIndex(i)
            if hasattr(self, "_sync_audio_input_aec_ref_combo"):
                try:
                    self._sync_audio_input_aec_ref_combo()
                except Exception:
                    pass
            if hasattr(self, "audio_input_mic_device_combo"):
                try:
                    dev_idx = int(str(cfg.get("audio_input_mic_device_index", "-1")).strip() or "-1")
                except Exception:
                    dev_idx = -1
                i = self.audio_input_mic_device_combo.findData(dev_idx)
                if i >= 0:
                    self.audio_input_mic_device_combo.setCurrentIndex(i)
            self._apply_app_config_field(cfg, "audio_input_wav_path")
            self._apply_app_config_field(cfg, "audio_input_wav_start_sec")
            self._apply_app_config_field(cfg, "audio_input_wav_end_sec")
            if hasattr(self, "audio_input_wav_local_playback_check"):
                self.audio_input_wav_local_playback_check.setChecked(
                    json_config_bool(cfg, "audio_input_wav_local_playback", False)
                )
            if hasattr(self, "audio_input_wav_other_ch_to_mmd_check"):
                self.audio_input_wav_other_ch_to_mmd_check.setChecked(
                    json_config_bool(cfg, "audio_input_wav_other_ch_to_mmd", False)
                )
            if hasattr(self, "audio_input_wav_other_ch_to_maai_ch2_check"):
                self.audio_input_wav_other_ch_to_maai_ch2_check.setChecked(
                    json_config_bool(cfg, "audio_input_wav_other_ch_to_maai_ch2", False)
                )
            if hasattr(self, "audio_input_wav_channel_combo"):
                try:
                    wch = int(str(cfg.get("audio_input_wav_channel_index", "0")).strip() or "0")
                except Exception:
                    wch = 0
                wch = max(0, min(1, wch))
                wi = self.audio_input_wav_channel_combo.findData(wch)
                if wi >= 0:
                    self.audio_input_wav_channel_combo.setCurrentIndex(wi)
                else:
                    self.audio_input_wav_channel_combo.setCurrentIndex(0)
            if hasattr(self, "audio_input_wav_aec_check"):
                self.audio_input_wav_aec_check.setChecked(
                    json_config_bool(
                        cfg,
                        "audio_input_enable_aec",
                        json_config_bool(cfg, "audio_input_wav_enable_aec", False),
                    )
                )
            if hasattr(self, "audio_input_wav_aec_ref_combo"):
                ref_val = str(
                    cfg.get("audio_input_aec_ref", cfg.get("audio_input_wav_aec_ref", "other_ch"))
                ).strip() or "other_ch"
                ri = self.audio_input_wav_aec_ref_combo.findData(ref_val)
                if ri >= 0:
                    self.audio_input_wav_aec_ref_combo.setCurrentIndex(ri)
                else:
                    self.audio_input_wav_aec_ref_combo.setCurrentIndex(0)
            self._apply_app_config_field(cfg, "audio_input_tcp_host")
            self._apply_app_config_field(cfg, "audio_input_tcp_port")
            self._apply_app_config_field(cfg, "audio_input_gain")
            # nod_para motion
            if hasattr(self, "nod_para_enabled_check"):
                self.nod_para_enabled_check.setChecked(json_config_bool(cfg, "nod_para_enabled", False))
            _nod_para_load_map = {
                "nod_para_prob_threshold_edit": ("nod_para_prob_threshold", "0.6"),
                "nod_para_min_consecutive_frames_edit": ("nod_para_min_consecutive_frames", "3"),
                "nod_para_min_interval_ms_edit": ("nod_para_min_interval_ms", "1500"),
                "nod_para_decay_rate_edit": ("nod_para_decay_rate", "0.6"),
                "nod_para_range_clip_min_edit": ("nod_para_range_clip_min", "0.0"),
                "nod_para_range_clip_max_edit": ("nod_para_range_clip_max", "1.0"),
                "nod_para_speed_clip_min_edit": ("nod_para_speed_clip_min", "0.0"),
                "nod_para_speed_clip_max_edit": ("nod_para_speed_clip_max", "1.0"),
                "nod_para_fps_edit": ("nod_para_fps", "30"),
                "nod_para_pre_rise_ratio_edit": ("nod_para_pre_rise_ratio", "0.8"),
                "nod_para_scale_edit": ("nod_para_scale", "1.0"),
                "nod_para_speed_scale_edit": ("nod_para_speed_scale", "1.0"),
                "nod_para_eye_scale_edit": ("nod_para_eye_scale", "1.0"),
                "nod_para_bone_edit": ("nod_para_bone", "頭"),
            }
            for _attr, (_key, _def) in _nod_para_load_map.items():
                if hasattr(self, _attr):
                    getattr(self, _attr).setText(str(cfg.get(_key, _def)))
            if hasattr(self, "nod_para_interpolation_combo"):
                _interp = str(cfg.get("nod_para_interpolation", "cosine"))
                idx = self.nod_para_interpolation_combo.findText(_interp)
                if idx >= 0:
                    self.nod_para_interpolation_combo.setCurrentIndex(idx)
            self._apply_app_config_field(cfg, "asr_vad_on_threshold")
            self._apply_app_config_field(cfg, "asr_vad_on_sec")
            self._apply_app_config_field(cfg, "asr_vad_off_threshold")
            self._apply_app_config_field(cfg, "asr_vad_off_sec")
            self._apply_app_config_field(cfg, "asr_min_segment_sec")
            pre_roll_sec = str(
                cfg.get("asr_pre_roll_buffer_sec", cfg.get("asr_nonstream_pseudo_chunk_sec", "0.25"))
            )
            self.asr_pre_roll_buffer_sec_edit.setText(pre_roll_sec)
            self.asr_nonstream_pseudo_enabled_check.setChecked(
                to_bool(str(cfg.get("asr_nonstream_pseudo_enabled", True)))
            )
            # `asr_nonstream_pseudo_bootstrap_sec` は旧版が書いていたミラー。
            # 保存はもうしないが、既存の設定ファイルを読めるようフォールバックは残す。
            pseudo_chunk = str(cfg.get("asr_nonstream_pseudo_chunk_sec", cfg.get("asr_nonstream_pseudo_bootstrap_sec", "0.25")))
            self.asr_nonstream_pseudo_chunk_sec_edit.setText(pseudo_chunk)
            self.asr_maai_gui_plot_check.setChecked(to_bool(str(cfg.get("asr_maai_gui_plot", False))))
            self.asr_segment_arrival_align_check.setChecked(
                to_bool(str(cfg.get("asr_segment_arrival_align", False)))
            )
            self._apply_app_config_field(cfg, "asr_maai_turn_threshold")
            self._apply_app_config_field(cfg, "asr_smart_turn_threshold")
            self._apply_app_config_field(cfg, "asr_smart_turn_num_threads")
            self._apply_app_config_field(cfg, "asr_torch_cpu_intra_threads")
            self._apply_app_config_field(cfg, "maai_plot_flush_interval_ms")
            self._apply_app_config_field(cfg, "maai_plot_asr_scroll_interval_ms")
            # 新キーは ms。旧キー(_sec)しか無い設定は秒→ms へ換算して後方互換を保つ。
            if "maai_plot_raw_draw_interval_ms" in cfg:
                _raw_ms_text = str(cfg.get("maai_plot_raw_draw_interval_ms"))
            elif "maai_plot_raw_draw_interval_sec" in cfg:
                try:
                    _raw_ms_text = f"{float(cfg.get('maai_plot_raw_draw_interval_sec')) * 1000.0:.6g}"
                except (TypeError, ValueError):
                    _raw_ms_text = f"{(1000.0 / 15.0):.6g}"
            else:
                _raw_ms_text = f"{(1000.0 / 15.0):.6g}"
            self.maai_plot_raw_draw_interval_edit.setText(_raw_ms_text)
            self._sync_maai_plot_thresholds()
            self._apply_maai_plot_timing_from_ui()
            for panel_id, _label in MAAI_PLOT_VISIBILITY_SPECS:
                cb = getattr(self, f"maai_plot_visibility_{panel_id}_check", None)
                if cb is not None:
                    cb.setChecked(json_config_bool(cfg, f"maai_plot_show_{panel_id}", True))
            self._sync_maai_plot_panel_visibility_from_ui()
            self._on_asr_engine_changed(self.asr_engine_combo.currentText().strip() or "sherpa-onnx")
            self._on_asr_compare_mode_changed()
            # 設定読込後にエンジン既定値を再適用して、他エンジン値の取り違えを防ぐ。
            self._apply_engine_defaults(self.tts_engine_combo.currentText().strip() or "VOICEVOX")
            self._save_current_engine_voice(self.tts_engine_combo.currentText().strip() or "VOICEVOX")
            self._sync_tts_exe_paths_from_ui()
        finally:
            self._is_loading_app_config = False
        self._refresh_module_status_summaries()

    def _collect_app_config_payload(self, *, sync_engine_voice: bool = True) -> dict[str, object]:
        """ウィジェットの現在値から `controller_config.json` へ保存する dict を組み立てる。

        `_save_app_config` から機械的に切り出しただけで、ロジックは変えていない。
        エンジン別の声設定と LLM モデル記憶の更新も、元と同じ順序でここで行う。
        headless 側の `dialog/settings.build_*(cfg)` と同じ形の dict を
        GUI からも得られるようにするための足場。
        """
        current_engine = self.tts_engine_combo.currentText().strip() or "VOICEVOX"
        if sync_engine_voice:
            self._save_current_engine_voice(current_engine)
        self._llm_model_by_backend["LM Studio"] = self._read_llm_model_for_backend("LM Studio")
        self._llm_model_by_backend["OpenAI"] = self._read_llm_model_for_backend("OpenAI")
        self._llm_model_by_backend["Custom"] = self._read_llm_model_for_backend("Custom")
        payload = {
            "exec_path": self.exec_edit.text().strip(),
            "mdf_path": self.mdf_edit.text().strip(),
            "tts_engine": self.tts_engine_combo.currentText().strip(),
            "tts_voice": self.tts_voice_edit.text().strip(),
            "tts_voice_voicevox": self._tts_voice_by_engine.get("VOICEVOX", "1"),
            "tts_voice_voicevox_core": self._tts_voice_by_engine.get("VOICEVOX-Core", "8"),
            "tts_voice_kokoro_realtimetts": self._tts_voice_by_engine.get("Kokoro-RealtimeTTS", "jf_alpha"),
            "tts_voice_cartesia": self._tts_voice_by_engine.get(
                "Cartesia-Realtime", "f786b574-daa5-4673-aa0c-cbe3e8534c02"
            ),
            "tts_voice_template": self._tts_voice_by_engine.get("TEMPLATE", "1"),
            "tts_voice_aivisspeech": self._tts_voice_by_engine.get("AivisSpeech", "888753761"),
            "tts_voice_supertonic": self._tts_voice_by_engine.get("Supertonic-3", "F1"),
            "tts_voice_irodori": self._tts_voice_by_engine.get("Irodori-v4", "明るく自然な日本語の声"),
            "voicevox_core_device": self.voicevox_core_device_combo.currentText().strip(),
            # 互換用に保存しつつ、実体は Voice ID (Cartesia) の単一入力欄を使う。
            "cartesia_tts_voice_id": self.tts_voice_edit.text().strip(),
            "tts_auth_mode": self.tts_auth_mode_combo.currentText().strip(),
            "tts_template": self.tts_template_edit.toPlainText(),
            "tts_input_streaming_enabled": self.tts_input_streaming_check.isChecked(),
            "mmd_burst_snd_frames": self._get_mmd_burst_snd_frames(),
            "audio_output_mode": self._get_audio_output_mode(),
            "asr_engine": self.asr_engine_combo.currentText().strip(),
            "asr_compare_mode_enabled": self.asr_compare_mode_check.isChecked(),
            "asr_compare_top_engine": self.asr_compare_top_engine_combo.currentText().strip(),
            "asr_compare_bottom_engine": self.asr_compare_bottom_engine_combo.currentText().strip(),
            "asr_vad_backend": self.asr_vad_backend_combo.currentText().strip(),
            "asr_turn_backend": self.asr_turn_backend_combo.currentText().strip(),
            "maai_test_turn_vap_frame_rate": self.maai_test_turn_vap_frame_rate_edit.text().strip()
            if hasattr(self, "maai_test_turn_vap_frame_rate_edit")
            else "20",
            "maai_test_turn_vap_mode": self.maai_test_turn_vap_mode_edit.text().strip()
            if hasattr(self, "maai_test_turn_vap_mode_edit")
            else "vap_mc",
            "maai_test_turn_vap_lang": self.maai_test_turn_vap_lang_edit.text().strip()
            if hasattr(self, "maai_test_turn_vap_lang_edit")
            else "jp_kyoto",
            "maai_test_turn_vap_model_type": self.maai_test_turn_vap_model_type_edit.currentText().strip()
            if hasattr(self, "maai_test_turn_vap_model_type_edit")
            else "normal",
            # 旧キー互換: 外部スクリプト・既存 JSON 向けに MaAI テスト欄と同値を保存
            "asr_vap_frame_rate": self.maai_test_turn_vap_frame_rate_edit.text().strip()
            if hasattr(self, "maai_test_turn_vap_frame_rate_edit")
            else "",
            "asr_vap_mode": self.maai_test_turn_vap_mode_edit.text().strip()
            if hasattr(self, "maai_test_turn_vap_mode_edit")
            else "",
            "asr_vap_lang": self.maai_test_turn_vap_lang_edit.text().strip()
            if hasattr(self, "maai_test_turn_vap_lang_edit")
            else "",
            "asr_vap_device": self._maai_unified_device_str(),
            "asr_vap_model_type": self.maai_test_turn_vap_model_type_edit.currentText().strip()
            if hasattr(self, "maai_test_turn_vap_model_type_edit")
            else "",
            "asr_device": self.asr_device_combo.currentText().strip(),
            "asr_compute_type": self.asr_compute_type_combo.currentText().strip(),
            "asr_qwen3_onnx_provider": self._qwen3_onnx_provider(),
            "asr_qwen3_onnx_num_threads": str(self._qwen3_onnx_num_threads()),
            "asr_sherpa_provider": self._sherpa_offline_provider("asr_sherpa_provider_combo"),
            "asr_parakeet_onnx_provider": self._sherpa_offline_provider("asr_parakeet_onnx_provider_combo"),
            "asr_sensevoice_onnx_provider": self._sherpa_offline_provider("asr_sensevoice_onnx_provider_combo"),
            "asr_funasr_nano_onnx_provider": self._sherpa_offline_provider("asr_funasr_nano_onnx_provider_combo"),
            "asr_cohere_onnx_provider": self._sherpa_offline_provider("asr_cohere_onnx_provider_combo"),
            "asr_sensevoice_onnx_itn": self._checkbox_state("asr_sensevoice_onnx_itn_check"),
            "asr_funasr_nano_onnx_itn": self._checkbox_state("asr_funasr_nano_onnx_itn_check"),
            "asr_cohere_onnx_punct": self._checkbox_state("asr_cohere_onnx_punct_check"),
            "asr_cohere_onnx_itn": self._checkbox_state("asr_cohere_onnx_itn_check"),
            "asr_nemotron_onnx_compute": self._nemotron_onnx_compute_key(),
            "asr_qwen_model": self._read_input_text(self.asr_qwen_model_edit),
            "asr_nemotron_chunk_size_ms": self.asr_nemotron_chunk_combo.currentText().strip(),
            "asr_nemotron_finalize_mode": self._get_nemotron_finalize_mode(),
            "asr_nemotron_device": self.asr_nemotron_device_combo.currentText().strip(),
            "asr_qwen_finish_on_vad_end": self.asr_qwen_finish_on_vad_end_check.isChecked(),
            "asr_qwen_stop_with_compose_down": self.asr_qwen_stop_container_check.isChecked(),
            "asr_qwen_vllm_enforce_eager": self.asr_qwen_enforce_eager_check.isChecked(),
            "maai_device": self._maai_unified_device_str(),
            "maai_enable_bc": bool(hasattr(self, "maai_enable_bc_box") and self.maai_enable_bc_box.isChecked()),
            "maai_enable_nod": bool(hasattr(self, "maai_enable_nod_box") and self.maai_enable_nod_box.isChecked()),
            "maai_enable_turn_vap": bool(hasattr(self, "maai_enable_turn_vap_box") and self.maai_enable_turn_vap_box.isChecked()),
            "maai_multi_enabled": bool(hasattr(self, "maai_multi_enabled_check") and self.maai_multi_enabled_check.isChecked()),
            "maai_multi_model_type": self.maai_multi_model_type_edit.currentText().strip() if hasattr(self, "maai_multi_model_type_edit") else "normal-ver2",
            "maai_use_mimi_onnx": bool(
                getattr(self, "maai_use_mimi_onnx_check", None) is None
                or self.maai_use_mimi_onnx_check.isChecked()
            ),
            "maai_use_local_mimi_onnx": bool(
                getattr(self, "maai_use_local_mimi_onnx_check", None) is not None
                and self.maai_use_local_mimi_onnx_check.isChecked()
            ),
            "maai_root": (
                self.maai_root_edit.text().strip()
                if hasattr(self, "maai_root_edit")
                else ""
            ) or str(_paths.assets_root()),
            "maai_mimi_onnx_path": self.maai_mimi_onnx_path_edit.text().strip()
            if hasattr(self, "maai_mimi_onnx_path_edit")
            else "",
            "maai_mimi_onnx_meta_path": self.maai_mimi_onnx_meta_path_edit.text().strip()
            if hasattr(self, "maai_mimi_onnx_meta_path_edit")
            else "",
            "maai_test_bc_model_type": self.maai_test_bc_model_type_edit.currentText().strip()
            if hasattr(self, "maai_test_bc_model_type_edit")
            else "normal",
            "maai_test_nod_model_type": self.maai_test_nod_model_type_edit.currentText().strip()
            if hasattr(self, "maai_test_nod_model_type_edit")
            else "normal-ver2",
            "maai_backchannel_enabled": self._is_maai_backchannel_enabled(),
            "maai_backchannel_playback": str(
                self.maai_backchannel_playback_combo.currentData()
                if hasattr(self, "maai_backchannel_playback_combo")
                and self.maai_backchannel_playback_combo.currentData() is not None
                else "play"
            ),
            "maai_backchannel_prob_threshold": str(self._get_maai_backchannel_prob_threshold()),
            "maai_backchannel_min_interval_sec": str(self._get_maai_backchannel_min_interval_sec()),
            "maai_backchannel_consecutive_frames": str(self._get_maai_backchannel_consecutive_frames()),
            "maai_backchannel_dictionary_path": self._get_maai_backchannel_dictionary_path(),
            "audio_input_mode": self.audio_input_mode_combo.currentText().strip().lower()
            if hasattr(self, "audio_input_mode_combo")
            else "mic",
            "audio_input_mic_device_index": str(
                int(self.audio_input_mic_device_combo.currentData())
                if hasattr(self, "audio_input_mic_device_combo")
                and self.audio_input_mic_device_combo.currentData() is not None
                else -1
            ),
            "audio_input_wav_local_playback": self._is_audio_input_wav_local_playback_enabled(),
            "audio_input_wav_other_ch_to_mmd": bool(
                hasattr(self, "audio_input_wav_other_ch_to_mmd_check")
                and self.audio_input_wav_other_ch_to_mmd_check.isChecked()
            ),
            "audio_input_wav_other_ch_to_maai_ch2": bool(
                hasattr(self, "audio_input_wav_other_ch_to_maai_ch2_check")
                and self.audio_input_wav_other_ch_to_maai_ch2_check.isChecked()
            ),
            "audio_input_wav_channel_index": str(
                int(self.audio_input_wav_channel_combo.currentData())
                if hasattr(self, "audio_input_wav_channel_combo")
                and self.audio_input_wav_channel_combo.currentData() is not None
                else 0
            ),
            "audio_input_enable_aec": bool(
                hasattr(self, "audio_input_wav_aec_check")
                and self.audio_input_wav_aec_check.isChecked()
            ),
            "audio_input_wav_enable_aec": bool(
                hasattr(self, "audio_input_wav_aec_check")
                and self.audio_input_wav_aec_check.isChecked()
            ),
            "audio_input_aec_ref": str(
                self.audio_input_wav_aec_ref_combo.currentData()
                if hasattr(self, "audio_input_wav_aec_ref_combo")
                and self.audio_input_wav_aec_ref_combo.currentData() is not None
                else "other_ch"
            ),
            "audio_input_wav_aec_ref": str(
                self.audio_input_wav_aec_ref_combo.currentData()
                if hasattr(self, "audio_input_wav_aec_ref_combo")
                and self.audio_input_wav_aec_ref_combo.currentData() is not None
                else "other_ch"
            ),
            "nod_para_enabled": self._is_nod_para_enabled(),
            "nod_para_prob_threshold": str(self._get_nod_para_float("nod_para_prob_threshold_edit", 0.6)),
            "nod_para_min_consecutive_frames": str(self._get_nod_para_int("nod_para_min_consecutive_frames_edit", 3)),
            "nod_para_min_interval_ms": str(self._get_nod_para_float("nod_para_min_interval_ms_edit", 1500)),
            "nod_para_decay_rate": str(self._get_nod_para_float("nod_para_decay_rate_edit", 0.6)),
            "nod_para_range_clip_min": str(self._get_nod_para_float("nod_para_range_clip_min_edit", 0.0)),
            "nod_para_range_clip_max": str(self._get_nod_para_float("nod_para_range_clip_max_edit", 1.0)),
            "nod_para_speed_clip_min": str(self._get_nod_para_float("nod_para_speed_clip_min_edit", 0.0)),
            "nod_para_speed_clip_max": str(self._get_nod_para_float("nod_para_speed_clip_max_edit", 1.0)),
            "nod_para_fps": str(self._get_nod_para_int("nod_para_fps_edit", 30)),
            "nod_para_interpolation": self._get_nod_para_combo("nod_para_interpolation_combo", "cosine"),
            "nod_para_pre_rise_ratio": str(self._get_nod_para_float("nod_para_pre_rise_ratio_edit", 0.8)),
            "nod_para_scale": str(self._get_nod_para_float("nod_para_scale_edit", 1.0)),
            "nod_para_speed_scale": str(self._get_nod_para_float("nod_para_speed_scale_edit", 1.0)),
            "nod_para_eye_scale": str(self._get_nod_para_float("nod_para_eye_scale_edit", 1.0)),
            "nod_para_bone": self._get_nod_para_text("nod_para_bone_edit", "頭"),
            "asr_pre_roll_buffer_sec": self.asr_pre_roll_buffer_sec_edit.text().strip(),
            "asr_nonstream_pseudo_enabled": self.asr_nonstream_pseudo_enabled_check.isChecked(),
            "asr_nonstream_pseudo_chunk_sec": self.asr_nonstream_pseudo_chunk_sec_edit.text().strip(),
            "asr_maai_gui_plot": self.asr_maai_gui_plot_check.isChecked(),
            "maai_plot_raw_draw_interval_ms": self.maai_plot_raw_draw_interval_edit.text().strip(),
            "asr_segment_arrival_align": self.asr_segment_arrival_align_check.isChecked(),
            "voicevox_speaker": self.tts_voice_edit.text().strip(),
            "llm_backend": self.llm_backend_combo.currentText().strip() or "LM Studio",
            "llm_model_lmstudio": self._llm_model_by_backend.get("LM Studio", ""),
            "llm_model_openai": self._llm_model_by_backend.get("OpenAI", _LLM_OPENAI_DEFAULT_MODEL),
            "llm_model_custom": self._llm_model_by_backend.get("Custom", _LLM_CUSTOM_DEFAULT_MODEL),
            "llm_custom_send_temperature": self.llm_custom_send_temperature_check.isChecked()
            if hasattr(self, "llm_custom_send_temperature_check")
            else True,
            "llm_custom_send_max_tokens": self.llm_custom_send_max_tokens_check.isChecked()
            if hasattr(self, "llm_custom_send_max_tokens_check")
            else True,
            "llm_custom_extra_body_thinking": self.llm_custom_extra_body_thinking_check.isChecked()
            if hasattr(self, "llm_custom_extra_body_thinking_check")
            else False,
            "llm_tts_stream_enabled": self.llm_tts_stream_check.isChecked(),
            "llm_test_tts_input_streaming_enabled": self.llm_test_tts_input_streaming_check.isChecked(),
            "llm_split_custom_delimiters": self.llm_split_custom_delimiters_edit.text() or "。．！？!?\\n",
            "llm_split_custom_strip_emoji": self.llm_split_custom_strip_emoji_check.isChecked(),
            "llm_split_custom_strip_parenthetical": self.llm_split_custom_strip_parenthetical_check.isChecked(),
            "llm_split_custom_strip_standalone_warai": self.llm_split_custom_strip_standalone_warai_check.isChecked(),
            "voice_dialog_interrupt_on_user_speech": bool(
                getattr(self, "voice_dialog_interrupt_check", None).isChecked()
            )
            if getattr(self, "voice_dialog_interrupt_check", None) is not None
            else bool(self._voice_dialog_interrupt_on_user_speech),
            "voice_dialog_interrupt_vad_threshold": self.voice_dialog_interrupt_vad_threshold_edit.text().strip(),
            "voice_dialog_interrupt_p_now_threshold": self.voice_dialog_interrupt_pnow_threshold_edit.text().strip(),
            "voice_dialog_opening_utterance": (
                self.voice_dialog_opening_utterance_edit.text().strip()
                if hasattr(self, "voice_dialog_opening_utterance_edit")
                else ""
            ),
            "voice_dialog_scheduled_exit_sec": max(
                0.0, self._get_nod_para_float("voice_dialog_scheduled_exit_sec_edit", 0.0)
            ),
            "voice_dialog_scheduled_exit_utterance": self._get_nod_para_text(
                "voice_dialog_scheduled_exit_utterance_edit", ""
            ),
            "voice_dialog_idle_take_turn_sec": max(
                0.0, self._get_nod_para_float("voice_dialog_idle_take_turn_sec_edit", 0.0)
            ),
            "obs_remote_password": self.obs_remote_password_edit.text()
            if hasattr(self, "obs_remote_password_edit")
            else "",
            "obs_remote_enabled": bool(
                hasattr(self, "obs_remote_enabled_check") and self.obs_remote_enabled_check.isChecked()
            ),
            "obs_remote_link_maai": bool(
                hasattr(self, "obs_remote_link_maai_check") and self.obs_remote_link_maai_check.isChecked()
            ),
            "obs_remote_link_voice": bool(
                hasattr(self, "obs_remote_link_voice_check") and self.obs_remote_link_voice_check.isChecked()
            ),
            "llm_stream": True,
        }
        # 平坦なフィールドは宣言テーブル（app_config_spec）を正とする。
        # 上の dict リテラルにも同じキーがあるが、ここで spec 由来の値に統一する。
        # （キー名と既定値の二重管理をやめるため。dict 側の重複行はこの後で削除する）
        for _spec_row in APP_CONFIG_FIELDS:
            _spec_key = _spec_row[0]
            payload[_spec_key] = self._read_app_config_field(_spec_key)
        payload["mmd_blink_enable"] = self._get_mmd_blink_enable()
        payload["mmd_blink_interval_mean_sec"] = self._get_mmd_blink_interval_mean_sec()
        payload["mmd_blink_morph"] = self._get_mmd_blink_morph()
        # 表示名ではなく itemData（always / dialog）で保存する（Web UI も同じ値）。
        # 読み込み側（normalize_blink_mode）は表示名も受け付ける。
        payload["mmd_blink_mode"] = str(
            (
                self.mmd_blink_mode_combo.currentData()
                or self.mmd_blink_mode_combo.currentText().strip()
            )
            if hasattr(self, "mmd_blink_mode_combo")
            else "always"
        )
        for _pid, _ in MAAI_PLOT_VISIBILITY_SPECS:
            _cb = getattr(self, f"maai_plot_visibility_{_pid}_check", None)
            payload[f"maai_plot_show_{_pid}"] = bool(_cb.isChecked()) if _cb is not None else True
        return payload

    def _apply_whisper_fields_from_cfg(self, cfg: dict) -> None:
        """faster-whisper の HF / サイズ / Path を読む。旧設定の HF ID 混入もここで分ける。"""
        merged = dict(cfg)
        if not str(merged.get("asr_whisper_hf", "") or "").strip():
            path = str(merged.get("asr_whisper_model_path", "") or "").strip()
            name = str(merged.get("asr_whisper_model", "") or "").strip()
            path_host = self._host_path_for_backend_value(path) if path else None
            name_host = self._host_path_for_backend_value(name) if name else None
            path_is_absolute = Path(path).is_absolute() or bool(
                self.remote_backend_enabled and path.replace("\\", "/").startswith("/")
            )
            name_is_absolute = Path(name).is_absolute() or bool(
                self.remote_backend_enabled and name.replace("\\", "/").startswith("/")
            )
            path_as_hf = (
                "/" in path.replace("\\", "/")
                and not bool(path_host is not None and path_host.exists())
                and not path_is_absolute
            )
            name_as_hf = (
                "/" in name.replace("\\", "/")
                and not bool(name_host is not None and name_host.exists())
                and not name_is_absolute
            )
            if path_as_hf:
                merged["asr_whisper_hf"] = path
                merged["asr_whisper_model_path"] = ""
            elif name_as_hf:
                merged["asr_whisper_hf"] = name
                merged["asr_whisper_model"] = "large-v3"
        self._apply_app_config_field(merged, "asr_whisper_hf")
        self._apply_app_config_field(merged, "asr_whisper_model")
        self._apply_app_config_field(merged, "asr_whisper_model_path")

    def _save_app_config(self, *, sync_orchestrator: bool = True) -> None:
        # ロード中は何もしない（副作用も起こさない）。元の実装と同じ位置のガード。
        if self._is_loading_app_config:
            return
        payload = self._collect_app_config_payload()
        self.config_store.save(payload)
        self._refresh_module_status_summaries()
        self._push_dialog_orchestrator_settings()
        if sync_orchestrator:
            self._sync_orchestrator_audio_output_from_ui()
            self._sync_orchestrator_audio_input_from_ui()
        if self.process_service.is_running():
            self._send_remote_plugin_settings()
