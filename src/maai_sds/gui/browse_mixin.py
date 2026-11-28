"""controller.py から切り出した mixin（Phase 3-2）。

**mixin にしているのが要点**。`self` は `MMDAgentController` のままなので、
メソッド本文を 1 文字も書き換えずに移せる（協調オブジェクトにすると self が
別物になり、ウィジェット参照を全部書き換える必要が出る）。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from .. import paths as _paths


class FileBrowseMixin:
    """FileBrowseMixin: controller.py から移設したメソッド群。"""

    def _store_backend_picker_path(
        self, edit, host_path: str, label: str, *, host_only: bool = False,
    ) -> bool:
        value = self._docker_backend_path_from_picker(
            host_path, label=label, host_only=host_only,
        )
        if value is None:
            return False
        edit.setText(value)
        return True

    def _picker_parent_dir(self, configured_path: str) -> str:
        host_path = self._docker_picker_initial_path(configured_path)
        return str(Path(host_path).parent) if host_path else str(Path.cwd())

    def _browse_audio_input_wav_file(self) -> None:
        current = ""
        if hasattr(self, "audio_input_wav_path_edit"):
            try:
                current = self.audio_input_wav_path_edit.text().strip()
            except Exception:
                current = ""
        start_dir = self._picker_parent_dir(current)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "WAVファイルを選択",
            start_dir,
            "WAV Files (*.wav);;All Files (*)",
        )
        if not path:
            return
        if hasattr(self, "audio_input_wav_path_edit") and not self._store_backend_picker_path(
            self.audio_input_wav_path_edit, path, "入力 WAV",
        ):
            return
        self._save_app_config()

    def _browse_tts_reference_audio(self, edit_attr: str, title: str) -> None:
        """Choose the single reference recording required by cloning engines."""
        edit = getattr(self, edit_attr, None)
        if edit is None:
            return
        current = edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            initial_dir,
            "Audio files (*.wav *.flac *.mp3 *.m4a *.ogg);;All Files (*.*)",
        )
        if file_path and self._store_backend_picker_path(edit, file_path, title):
            self._save_app_config()

    def _browse_exec(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "実行ファイルを選択", "", "Executable (*.exe)")
        if file_path:
            self._store_backend_picker_path(
                self.exec_edit, file_path, "MMDAgent-EX 実行ファイル", host_only=True,
            )
            self._save_app_config()

    def _browse_mdf(self) -> None:
        current = self.mdf_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, "MDFファイルを選択", initial_dir, "MDF (*.mdf)")
        if file_path and self._store_backend_picker_path(self.mdf_edit, file_path, "MDF"):
            self.load_mdf_settings()
            self._save_app_config()

    def _browse_avatar_browser_web_dir(self) -> None:
        current = self.avatar_browser_web_dir_edit.text().strip()
        folder = QFileDialog.getExistingDirectory(self, "MMDAgent-EX Web のフォルダーを選択", current or "")
        if folder:
            self.avatar_browser_web_dir_edit.setText(folder)
            self._save_app_config()

    def _browse_maai_local_model(self, edit_attr: str, title: str) -> None:
        edit = getattr(self, edit_attr, None)
        if edit is None:
            return
        current = edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, initial_dir,
            "PyTorch checkpoint (*.pt *.pth *.ckpt);;All Files (*.*)",
        )
        if file_path and self._store_backend_picker_path(edit, file_path, title):
            self._save_app_config()

    def _browse_maai_onnx_file(self, path_edit_attr: str, meta_edit_attr: str, title: str) -> None:
        edit = getattr(self, path_edit_attr, None)
        if edit is None:
            return
        current = edit.text().strip()
        if current:
            initial_dir = self._picker_parent_dir(current)
        else:
            initial_dir = self._maai_onnx_browse_initial_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"{title} を選択", initial_dir, "ONNX (*.onnx);;All Files (*.*)",
        )
        if not file_path:
            return
        if not self._store_backend_picker_path(edit, file_path, title):
            return
        meta_edit = getattr(self, meta_edit_attr, None)
        if meta_edit is not None and not meta_edit.text().strip():
            guessed = self._guess_maai_onnx_meta_path(file_path)
            self._store_backend_picker_path(meta_edit, guessed, f"{title} metadata")
        self._save_app_config()

    def _browse_maai_root_dir(self) -> None:
        edit = getattr(self, "maai_root_edit", None)
        if edit is None:
            return
        current = edit.text().strip()
        current_host = self._docker_picker_initial_path(current)
        initial = current_host if current_host and Path(current_host).is_dir() else str(
            _paths.assets_root()
        )
        chosen = QFileDialog.getExistingDirectory(self, "maai_root（ONNX アセット基準）を選択", initial)
        if chosen and self._store_backend_picker_path(edit, chosen, "maai_root"):
            self._save_app_config()

    def _browse_maai_onnx_meta_file(self, meta_edit_attr: str, title: str) -> None:
        edit = getattr(self, meta_edit_attr, None)
        if edit is None:
            return
        current = edit.text().strip()
        if current:
            initial_dir = self._picker_parent_dir(current)
        else:
            initial_dir = self._maai_onnx_browse_initial_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, initial_dir, "JSON (*.json);;All Files (*.*)",
        )
        if file_path and self._store_backend_picker_path(edit, file_path, title):
            self._save_app_config()

    def _browse_maai_turn_vap_local_model(self) -> None:
        self._browse_maai_local_model("maai_test_turn_vap_local_model_edit", "Turn VAP チェックポイント(.pt)を選択")

    def _browse_maai_bc_local_model(self) -> None:
        self._browse_maai_local_model("maai_test_bc_local_model_edit", "BC チェックポイント(.pt)を選択")

    def _browse_voicevox_exe(self) -> None:
        current = self.voicevox_exe_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, "VOICEVOX 実行ファイルを選択", initial_dir, "Executable (*.exe)")
        if file_path and self._store_backend_picker_path(
            self.voicevox_exe_edit, file_path, "VOICEVOX 実行ファイル",
        ):
            self._sync_tts_exe_paths_from_ui()
            self._save_app_config()

    def _browse_voicevox_core_onnxruntime(self) -> None:
        current = self.voicevox_core_onnxruntime_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "VOICEVOX Core ONNXRuntime DLL を選択", initial_dir, "DLL (*.dll);;All Files (*.*)"
        )
        if file_path and self._store_backend_picker_path(
            self.voicevox_core_onnxruntime_edit, file_path, "VOICEVOX Core ONNXRuntime",
        ):
            self._save_app_config()

    def _browse_voicevox_core_openjtalk_dict(self) -> None:
        current = self.voicevox_core_openjtalk_dict_edit.text().strip()
        current_host = self._docker_picker_initial_path(current)
        initial = current_host if current_host and Path(current_host).is_dir() else str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "VOICEVOX Core OpenJTalk 辞書フォルダを選択", initial)
        if chosen and self._store_backend_picker_path(
            self.voicevox_core_openjtalk_dict_edit, chosen, "VOICEVOX Core OpenJTalk 辞書",
        ):
            self._save_app_config()

    def _browse_voicevox_core_vvm(self) -> None:
        current = self.voicevox_core_vvm_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "VOICEVOX Core VVM モデルを選択", initial_dir, "VVM (*.vvm);;All Files (*.*)"
        )
        if file_path and self._store_backend_picker_path(
            self.voicevox_core_vvm_edit, file_path, "VOICEVOX Core VVM",
        ):
            self._save_app_config()

    def _browse_aivisspeech_exe(self) -> None:
        current = self.aivisspeech_exe_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "AivisSpeech Engine run.exe を選択", initial_dir, "Executable (*.exe)"
        )
        if file_path and self._store_backend_picker_path(
            self.aivisspeech_exe_edit, file_path, "AivisSpeech Engine",
        ):
            self._sync_tts_exe_paths_from_ui()
            self._save_app_config()

    def _browse_supertonic_server_exe(self) -> None:
        current = self.supertonic_server_exe_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Supertonic (audio.cpp) サーバー EXE を選択", initial_dir, "Executable (*.exe)"
        )
        if file_path and self._store_backend_picker_path(
            self.supertonic_server_exe_edit, file_path, "Supertonic server",
        ):
            self._save_app_config()

    def _browse_supertonic_model_path(self) -> None:
        current = self.supertonic_model_path_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Supertonic GGUF モデルを選択", initial_dir, "GGUF (*.gguf);;All Files (*.*)"
        )
        if file_path and self._store_backend_picker_path(
            self.supertonic_model_path_edit, file_path, "Supertonic model",
        ):
            self._save_app_config()

    def _browse_asr_whisper_model_path(self) -> None:
        current = self.asr_whisper_model_path_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Whisper モデル(ローカル)を選択",
            initial_dir,
            "All Files (*.*)",
        )
        if file_path and self._store_backend_picker_path(
            self.asr_whisper_model_path_edit, file_path, "Whisper model",
        ):
            self._save_app_config()

    def _browse_asr_sherpa_tokens(self) -> None:
        current = self.asr_sherpa_tokens_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, "Sherpa tokens.txt を選択", initial_dir, "Text (*.txt);;All Files (*.*)")
        if file_path and self._store_backend_picker_path(
            self.asr_sherpa_tokens_edit, file_path, "Sherpa tokens",
        ):
            self._save_app_config()

    def _browse_asr_sherpa_encoder(self) -> None:
        current = self.asr_sherpa_encoder_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, "Sherpa encoder.onnx を選択", initial_dir, "ONNX (*.onnx);;All Files (*.*)")
        if file_path and self._store_backend_picker_path(
            self.asr_sherpa_encoder_edit, file_path, "Sherpa encoder",
        ):
            self._save_app_config()

    def _browse_asr_sherpa_decoder(self) -> None:
        current = self.asr_sherpa_decoder_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, "Sherpa decoder.onnx を選択", initial_dir, "ONNX (*.onnx);;All Files (*.*)")
        if file_path and self._store_backend_picker_path(
            self.asr_sherpa_decoder_edit, file_path, "Sherpa decoder",
        ):
            self._save_app_config()

    def _browse_asr_sherpa_joiner(self) -> None:
        current = self.asr_sherpa_joiner_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, "Sherpa joiner.onnx を選択", initial_dir, "ONNX (*.onnx);;All Files (*.*)")
        if file_path and self._store_backend_picker_path(
            self.asr_sherpa_joiner_edit, file_path, "Sherpa joiner",
        ):
            self._save_app_config()

    def _browse_asr_model_file(self, edit, title: str, file_filter: str = "ONNX (*.onnx);;All Files (*.*)") -> None:
        """ASR モデルファイル選択の汎用ハンドラ（qwen3-onnx / nemotron-onnx 用）。"""
        current = edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, title, initial_dir, file_filter)
        if file_path and self._store_backend_picker_path(edit, file_path, title):
            self._save_app_config()

    def _browse_asr_smart_turn_model(self) -> None:
        current = self.asr_smart_turn_model_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(self, "smart-turn ONNX を選択", initial_dir, "ONNX (*.onnx);;All Files (*.*)")
        if file_path and self._store_backend_picker_path(
            self.asr_smart_turn_model_edit, file_path, "smart-turn model",
        ):
            self._save_app_config()

    def _browse_maai_backchannel_dictionary_path(self) -> None:
        current = (
            self.maai_backchannel_dictionary_path_edit.text().strip()
            if hasattr(self, "maai_backchannel_dictionary_path_edit")
            else ""
        )
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "MaAI 相槌辞書を選択",
            initial_dir,
            "JSON (*.json);;All Files (*.*)",
        )
        if file_path:
            if hasattr(self, "maai_backchannel_dictionary_path_edit") and not self._store_backend_picker_path(
                self.maai_backchannel_dictionary_path_edit, file_path, "MaAI backchannel dictionary",
            ):
                return
            self._save_app_config()

    def _browse_asr_qwen_custom_model_path(self) -> None:
        current = self._docker_picker_initial_path(
            self.asr_qwen_custom_model_path_edit.text().strip()
        )
        initial_dir = str(Path.cwd())
        if current:
            p = Path(current).expanduser()
            try:
                p = p.resolve()
            except OSError:
                pass
            else:
                initial_dir = str(p if p.is_dir() else p.parent)
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Qwen ASR モデル（チェックポイント）フォルダを選択",
            initial_dir,
        )
        if dir_path and self._store_backend_picker_path(
            self.asr_qwen_custom_model_path_edit, dir_path, "Qwen ASR model",
        ):
            self._save_app_config()

    def _browse_asr_nemotron_python(self) -> None:
        current = self.asr_nemotron_python_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Nemotron 専用 venv の Python 実行ファイルを選択",
            initial_dir,
            "Python (python.exe);;All Files (*.*)",
        )
        if file_path and self._store_backend_picker_path(
            self.asr_nemotron_python_edit, file_path, "Nemotron Python",
        ):
            self._save_app_config()

    def _browse_asr_external_start_script(self) -> None:
        current = self.asr_external_start_script_edit.text().strip()
        initial_dir = self._picker_parent_dir(current)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "External ASR 起動ファイルを選択",
            initial_dir,
            "PowerShell Script (*.ps1);;Batch (*.bat *.cmd);;All Files (*.*)",
        )
        if file_path and self._store_backend_picker_path(
            self.asr_external_start_script_edit, file_path, "External ASR start script",
        ):
            self._save_app_config()
