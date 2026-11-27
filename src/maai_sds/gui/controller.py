from __future__ import annotations

import base64
import hashlib
import json
import locale
import numbers
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any
from pathlib import Path

from PySide6.QtCore import QEvent, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QWidget,
)

from ..ipc.jsonline import JsonLineDuplexClient
from ..docker_paths import DockerPathMapper
from . import settings as cs
from ..config_store import ConfigStore
from ..mmd.mdf_service import MdfService
from ..ipc.process_service import (
    DialogOrchestratorProcessService,
    MMDProcessService,
)
from ..mmd.console_filter import is_per_frame_line
from .engine_choices import link_engine_choice_combos, resync_engine_choice_combos
from .numeric_fields import apply_numeric_fields
from .ui import apply_device_choice_limits, build_ui
from ..utils import apply_maai_nod_para_prediction_aliases, json_config_bool, to_bool
from ..maai.plot_window import MAAI_PLOT_VISIBILITY_SPECS, MaaiPlotWindow, maai_plot_default_visible
from ..llm import s2s_preview as llm_s2s_preview
from ..llm import admin_http as _llm_admin_http
from ..obs import remote as obs_remote
from ..tts import playback_service as tps
from ..dialog import settings as vds
from .obs_mixin import ObsRemoteMixin
from .module_status_mixin import ModuleStatusPanelMixin
from .qwen_asr_mixin import QwenAsrServerMixin
from .browse_mixin import FileBrowseMixin
from .orchestrator_events import (
    ORCH_EVENT_HANDLERS_EARLY,
    ORCH_EVENT_HANDLERS_LATE,
    OrchestratorEventMixin,
)
from .app_config_spec import (
    APP_CONFIG_FIELD_BY_KEY,
    APP_CONFIG_FIELDS,
    _LLM_CUSTOM_DEFAULT_MODEL,
    _LLM_OPENAI_DEFAULT_MODEL,
)
from .config_io import AppConfigIOMixin
from .theme import STATE_COLORS, TOKENS, lamp_style
from .remote_mic import RemoteMicSender

try:
    import sounddevice as sd
except Exception:
    sd = None

from .. import paths as _paths
from .. import engine_manager


# ワーカー子プロセスの行に既に付いた [name][STDOUT|STDERR] 前置を検出する（二重前置の抑止）。
_WORKER_PREFIX_RE = re.compile(r"^\[.+\]\[(STDOUT|STDERR)\] ")
# maai ライブラリが stdout に出す平均処理時間（約 100 フレームごと）。
# c94e134 以降は RTF も同ログに載る（分母は inference_chunk_frames / frame_rate）。
_MAAI_AVG_PROCESS_TIME_RE = re.compile(
    r"\[(?P<mode>[^\]]+)\] Average processing time: (?P<sec>[0-9.eE+-]+) \[sec\], "
    r"#process/sec: (?P<pps>[0-9.eE+-]+)"
    r"(?:, RTF: (?P<rtf>[0-9.eE+-]+))?"
)


def parse_remote_orchestrator_endpoint(value: str) -> tuple[str, int]:
    """Parse the Qt remote-backend ``HOST:PORT`` setting."""
    text = str(value or "").strip()
    host, sep, port_text = text.rpartition(":")
    if not sep or not host.strip() or not port_text.strip():
        raise ValueError(f"remote orchestrator must be HOST:PORT, got {value!r}")
    try:
        port = int(port_text)
    except ValueError as ex:
        raise ValueError(f"remote orchestrator port must be an integer: {port_text!r}") from ex
    if not 1 <= port <= 65535:
        raise ValueError(f"remote orchestrator port is outside 1..65535: {port}")
    return host.strip(), port


def _engine_label(combo) -> str:
    """一覧の表示名（モデル（実行系））。エンジンの値しか持たない欄はその値。"""
    label = getattr(combo, "current_label", None)
    return (label() if callable(label) else combo.currentText()).strip()


class MMDAgentController(
    ModuleStatusPanelMixin,
    ObsRemoteMixin,
    QwenAsrServerMixin,
    FileBrowseMixin,
    OrchestratorEventMixin,
    AppConfigIOMixin,
    QWidget,
):
    process_output_signal = Signal(str)
    latency_signal = Signal(float, str)
    # engine-manager install worker (plain threading.Thread, not QThread — see
    # _apply_engine_installation). Signal.emit() from any thread is queued safely to the
    # GUI thread regardless of the emitting thread's type, same as process_output_signal above.
    _engine_install_output_signal = Signal(str)
    _engine_install_completed_signal = Signal(bool, str)
    asr_result_signal = Signal(str)
    asr_result_event_signal = Signal(object)
    maai_result_event_signal = Signal(object)
    maai_test_event_signal = Signal(object)
    raw_audio_event_signal = Signal(object)
    tts_test_synthesis_finished_signal = Signal(object)
    asr_start_finished_signal = Signal(bool, str)
    qwen_asr_start_finished_signal = Signal(bool, str)
    llm_test_chat_chunk_signal = Signal(object)
    llm_test_chat_finished_signal = Signal(object)
    llm_tts_stream_push_signal = Signal(object)
    voice_dialog_llm_chunk_signal = Signal(object)
    llm_models_fetched_signal = Signal(object)
    llm_openai_models_fetched_signal = Signal(object)
    llm_model_unload_finished_signal = Signal(object)
    voice_dialog_latency_signal = Signal(object)
    voice_dialog_history_refresh_signal = Signal()
    dialog_orchestrator_event_signal = Signal(object)
    # IPC リーダースレッドからは直接コールバックさせず emit する（QTimer は GUI スレッド専用）。
    dialog_orchestrator_ipc_disconnected_signal = Signal()
    # モジュール状態パネル更新（module, state, detail）。非GUIスレッドからも安全に発火できる。
    module_panel_state_signal = Signal(str, str, str)

    APP_CONFIG_PATH = _paths.config_path()
    TTS_TEMPLATE_PATH = _paths.tts_template_path()
    DEFAULT_FST_CONTENT = cs.DEFAULT_FST_CONTENT
    REMOVED_LEGACY_KEYS = cs.REMOVED_LEGACY_KEYS
    FIELD_SPECS = cs.FIELD_SPECS
    BOOL_KEYS = cs.BOOL_KEYS
    FIELD_DESCRIPTIONS = cs.FIELD_DESCRIPTIONS
    CONFLICT_CHECKBOX_PAIRS = cs.CONFLICT_CHECKBOX_PAIRS
    TTS_STREAM_MODE_BY_ENGINE: dict[str, str] = {
        "Cartesia-Realtime": "full_streaming",
        "Kokoro-RealtimeTTS": "full_streaming",
        "Qwen3-TTS-CPP": "batch_only",
        "VOICEVOX": "batch_only",
        "VOICEVOX-Core": "batch_only",
        "TEMPLATE": "batch_only",
    }

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        remote_orchestrator: str | None = None,
    ) -> None:
        super().__init__()
        remote_text = (
            str(remote_orchestrator).strip()
            if remote_orchestrator is not None
            else os.environ.get("MAAI_SDS_REMOTE_ORCHESTRATOR", "").strip()
        )
        self._remote_orchestrator_endpoint = (
            parse_remote_orchestrator_endpoint(remote_text) if remote_text else None
        )
        self._remote_backend_enabled = self._remote_orchestrator_endpoint is not None
        self._docker_path_mapper = (
            DockerPathMapper.from_compose_project(_paths.REPO_ROOT)
            if self._remote_backend_enabled
            else None
        )
        self._remote_orchestrator_retry_scheduled = False
        self._remote_mic_sender: RemoteMicSender | None = None
        self._remote_unchecked_model_paths: set[str] = set()
        configured_path = config_path
        if configured_path is None:
            configured_path = os.environ.get("MAAI_SDS_CONTROLLER_CONFIG", "").strip() or None
        if configured_path is None:
            app_config_path = self.APP_CONFIG_PATH
        else:
            app_config_path = Path(configured_path).expanduser()
            if not app_config_path.is_absolute():
                app_config_path = _paths.REPO_ROOT / app_config_path
            app_config_path = app_config_path.resolve()
        # Preserve the class default while allowing all existing mixin code to
        # keep referring to the instance attribute.
        self.APP_CONFIG_PATH = app_config_path
        # 既定の設定ファイルが無ければ、例（controller_config.example.json）から始める。main() でも
        # 同じことをしているが、main() を通らずに GUI を作る経路（setup_env.ps1 の検証など）では写されず、
        # 画面の部品の初期値がそのまま保存され、そのあと初めて起動しても例が使われなかった。
        if configured_path is None and _paths.ensure_config_file(app_config_path):
            print("[maai-sds] controller_config.json が無いので controller_config.example.json から作りました", flush=True)
        self._is_loading_app_config = True
        self.field_widgets: dict[str, QWidget] = {}
        self.default_values = {
            key: default
            for specs in self.FIELD_SPECS.values()
            for key, _label, default in specs
        }
        self.touched_keys: set[str] = set()
        self.loaded_keys: set[str] = set()
        self._suppress_checkbox_handler = False
        # 添付済みコンソールは Popen オブジェクト識別で管理する。生 PID で管理すると Windows の
        # PID 再利用で、別プロセスが同じ PID を得たとき添付がスキップされパイプ詰まりで固まる(#15)。
        self._attached_console_procs: set = set()
        self._last_selected_tts_engine: str | None = None
        self.dialog_orchestrator_process_service = DialogOrchestratorProcessService()

        # Orchestrator-based module lifecycle events
        self._module_ready_events: dict[str, threading.Event] = {
            "asr": threading.Event(),
            "asr_compare": threading.Event(),
            "llm": threading.Event(),
            "tts": threading.Event(),
            "maai": threading.Event(),
            "audio_input": threading.Event(),
            "audio_output": threading.Event(),
        }
        self._module_start_errors: dict[str, str] = {}
        self._audio_output_worker_ready_event = threading.Event()
        self._audio_output_worker_started_output_mode: str | None = None
        self._asr_worker_script = _paths.worker_script("asr", "worker.py")
        self._asr_compare_worker_script = _paths.worker_script("asr", "compare_worker.py")
        self._tts_worker_engine: str | None = None
        # 実際に動いている TTS 設定の指紋。Orchestrator が module_state で送ってくる
        # ものを正とする（GUI が自分の組み立てた値を覚えるだけだと、TTS テスト送信が
        # 別設定で ensure_tts を呼んだときに古いまま残る）。
        self._tts_worker_config_digest: str | None = None
        self._dialog_orchestrator_ipc_client: JsonLineDuplexClient | None = None
        self._dialog_orchestrator_worker_script = _paths.worker_script("dialog", "worker.py")
        self._dialog_orchestrator_ready_event = threading.Event()
        # 対話サーバーの ▶ / ■（module_status_mixin）。■ で止めたら自動で起動し直さない。
        self._dialog_server_user_stopped = False
        self._dialog_server_busy = ""
        self._dialog_server_error = ""
        self._dialog_orchestrator_settings_digest: str | None = None
        self._remote_plugin_settings_digest_sent: str | None = None
        self._controller_stop_event = threading.Event()
        self._dialog_orchestrator_sender_thread: threading.Thread | None = None
        self._maai_forward_thread: threading.Thread | None = None
        self._llm_worker_ready_event = threading.Event()
        self._llm_ipc_lifecycle_lock = threading.RLock()
        self._tts_bridge_rpc_lock = threading.Lock()
        self._tts_bridge_rpc_pending: dict[str, tuple[threading.Event, dict]] = {}
        self._llm_admin_rpc_lock = threading.Lock()
        self._llm_admin_rpc_pending: dict[str, tuple[threading.Event, dict]] = {}
        self._tts_worker_ready_event = threading.Event()
        self._asr_ready = False
        self._audio_input_worker_ready_event = threading.Event()
        self._audio_input_processing_event = threading.Event()
        self._audio_input_wav_paused = True
        self._audio_input_wav_command_inflight = False
        self._start_audio_input_after_asr_ready = True
        self._maai_ready = False
        self._maai_starting = False
        self._voice_dialog_maai_preexisting = False
        self._voice_dialog_asr_preexisting = False
        self._maai_mmd_av_preamble_sent = False
        self._maai_mmd_av_preamble_fail_count = 0
        self._maai_mmd_av_preamble_wait_for_mmd = False
        self._maai_test_log_last_emit_bc = 0.0
        self._maai_test_log_last_emit_nod = 0.0
        self._maai_test_surface_jsonl_fp: object | None = None
        self._maai_test_surface_jsonl_path: Path | None = None
        self._maai_test_bc_trigger_log_buffer: dict[str, object] | None = None
        self._maai_nod_threshold_defaults: dict[str, object] | None = None
        self._asr_starting = False
        self._asr_stop_requested_during_start = False
        self._pending_asr_config: dict | None = None
        self._last_asr_config: dict | None = None
        self._external_asr_start_proc = None
        self._qwen_asr_logs_proc = None
        self._engine_install_thread: threading.Thread | None = None
        self._qwen_asr_logs_lock = threading.Lock()
        self._qwen_asr_logs_follow_enabled = False
        self._tts_test_in_progress = False
        self._tts_engine_launching: str | None = None
        self._llm_test_in_progress = False
        self._llm_messages: list[dict[str, str]] = []
        self._last_lmstudio_model = ""
        self._lmstudio_bulk_unload_in_progress = False
        self._controller_closing = False
        self._fast_shutdown = False
        self._voice_dialog_active = False
        self._voice_dialog_starting = False
        self._voice_dialog_messages: list[dict[str, str]] = []
        self._voice_dialog_pending_texts: list[str] = []
        self._voice_dialog_current_partial = ""
        self._voice_dialog_asr_latency_sec: float | None = None
        self._voice_dialog_llm_latency_sec: float | None = None
        self._voice_dialog_tts_latency_sec: float | None = None
        self._voice_dialog_interrupt_on_user_speech = True
        # ASR Silero 推論の最新スコア（MaAI ワーカー経由の turn_vap 等へマージして Orchestrator の割り込み判定に使う）
        self._voice_dialog_last_silero_vad: float | None = None
        self._voice_dialog_tts_confirmed_segments: list[str] = []
        self._voice_dialog_tts_segments_snapshot: list[str] = []
        self._voice_dialog_tts_snap_prefix: str = ""
        self._voice_dialog_streaming_assistant_full: str = ""
        # 開始時発話ありのとき _ensure_voice_dialog_modules_ready で LLM ウォームアップを送らず、開始発話 TTS 完了後に送る
        self._llm_test_tts_confirmed_segments: list[str] = []
        self._llm_test_streaming_assistant_full: str = ""
        self._llm_test_assistant_block_start: int = -1
        self._obs_remote_refcount: int = 0
        self._obs_acquired_maai: bool = False
        self._obs_acquired_voice: bool = False
        self._obs_acquired_manual: bool = False
        self._llm_model_by_backend: dict[str, str] = {
            "LM Studio": "",
            "OpenAI": _LLM_OPENAI_DEFAULT_MODEL,
            "Custom": _LLM_CUSTOM_DEFAULT_MODEL,
        }
        self._last_llm_backend = "LM Studio"
        self._tts_voice_by_engine: dict[str, str] = {
            "VOICEVOX": "1",
            "VOICEVOX-Core": "8",
            "Kokoro-RealtimeTTS": "jf_alpha",
            "Qwen3-TTS-CPP": "Vivian",
            "LFM2.5-Audio": "",
            "Cartesia-Realtime": "f786b574-daa5-4673-aa0c-cbe3e8534c02",
            "AivisSpeech": "888753761",
            "Supertonic-3": "F1",
            "Irodori-v4": "明るく自然な日本語の声",
            "Tsukasa": "",
            "TEMPLATE": "1",
        }
        self.checkbox_conflicts: dict[str, set[str]] = {}
        for left, right in self.CONFLICT_CHECKBOX_PAIRS:
            self.checkbox_conflicts.setdefault(left, set()).add(right)
            self.checkbox_conflicts.setdefault(right, set()).add(left)

        self.config_store = ConfigStore(self.APP_CONFIG_PATH)
        self.process_service = MMDProcessService()
        self._orch_mmd_bridge_connected = False
        self._orch_mmd_bridge_mode = ""
        self._audio_mmd_bridge_connected = False
        self._audio_mmd_bridge_mode = ""
        self.mdf_service = MdfService(
            bool_keys=self.BOOL_KEYS,
            default_values=self.default_values,
            removed_legacy_keys=self.REMOVED_LEGACY_KEYS,
            default_fst_content=self.DEFAULT_FST_CONTENT,
        )

        build_ui(self)
        apply_numeric_fields(self)
        link_engine_choice_combos(self)
        self._refresh_audio_input_device_list()
        self.process_output_signal.connect(
            self._append_console_output, Qt.ConnectionType.QueuedConnection
        )
        self.latency_signal.connect(self._set_tts_latency)
        self.asr_result_signal.connect(self._append_asr_result)
        self.asr_result_event_signal.connect(self._append_asr_result_event)
        self.maai_result_event_signal.connect(self._append_maai_result_event)
        self.maai_test_event_signal.connect(self._append_maai_test_event)
        self.raw_audio_event_signal.connect(self._append_raw_audio_event)
        self.tts_test_synthesis_finished_signal.connect(self._on_tts_test_synthesis_finished)
        self.asr_start_finished_signal.connect(self._on_asr_start_finished)
        self.qwen_asr_start_finished_signal.connect(self._on_qwen_asr_start_finished)
        self.llm_test_chat_chunk_signal.connect(self._on_llm_test_chat_chunk)
        self.llm_test_chat_finished_signal.connect(self._on_llm_test_chat_finished)
        self.llm_tts_stream_push_signal.connect(self._on_llm_tts_stream_push_ui)
        self.voice_dialog_llm_chunk_signal.connect(self._on_voice_dialog_llm_chunk_ui)
        self.llm_models_fetched_signal.connect(self._on_lmstudio_models_fetched)
        self.llm_openai_models_fetched_signal.connect(self._on_openai_models_fetched)
        self.llm_model_unload_finished_signal.connect(self._on_lmstudio_model_unload_finished)
        self.voice_dialog_latency_signal.connect(self._on_voice_dialog_latency_event)
        self.voice_dialog_history_refresh_signal.connect(self._refresh_voice_dialog_history_view)
        self.dialog_orchestrator_event_signal.connect(
            self._on_dialog_orchestrator_event, Qt.ConnectionType.QueuedConnection
        )
        self.dialog_orchestrator_ipc_disconnected_signal.connect(self._on_dialog_orchestrator_ipc_disconnected)
        self._module_panel_states: dict[str, dict[str, str]] = {
            m: {"state": "stopped", "detail": ""} for m in ("asr", "llm", "tts", "maai")
        }
        self._maai_process_metrics: dict[str, dict[str, float]] = {}
        self.module_panel_state_signal.connect(self._on_module_panel_state, Qt.ConnectionType.QueuedConnection)
        self._engine_install_output_signal.connect(self._append_console_output, Qt.ConnectionType.QueuedConnection)
        self._engine_install_completed_signal.connect(
            self._on_engine_installation_complete, Qt.ConnectionType.QueuedConnection
        )
        self._dialog_orchestrator_command_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=256)
        # 高頻度の音声中継(send_to_audio_output)専用キュー。制御コマンドと分離し、音声
        # バーストで制御コマンド(start/stop_module, tts_done 等)が押し出されないようにする。
        self._dialog_orchestrator_audio_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=512)
        self._dialog_orchestrator_sender_thread = threading.Thread(
            target=self._dialog_orchestrator_command_sender_loop,
            daemon=True,
            name="controller-dialog-orch-cmd-sender",
        )
        self._dialog_orchestrator_sender_thread.start()
        self._maai_result_forward_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=64)
        self._maai_forward_thread = threading.Thread(
            target=self._maai_result_forward_loop,
            daemon=True,
            name="controller-maai-forward",
        )
        self._maai_forward_thread.start()
        self._refresh_voice_dialog_latency_label()
        if hasattr(self, "llm_test_window") and self.llm_test_window is not None:
            self.llm_test_window.installEventFilter(self)
        self.maai_plot_window = MaaiPlotWindow(frame_rate=20, sample_rate=16000)
        self.maai_plot_window.init_full_layout()
        self._maai_plot_pending: list[dict[str, Any]] = []
        self._maai_plot_wall_clock_timer = QTimer(self)
        self._maai_plot_wall_clock_timer.timeout.connect(self._on_maai_plot_wall_clock_tick)
        self._apply_maai_plot_timing_from_ui()
        self._on_asr_segment_arrival_align_changed()
        self._set_default_paths()
        try:
            self._load_app_config()
        finally:
            self._is_loading_app_config = False
        # 隠した欄（speech.cpp のモデル・Irodori の実行系）を読んだ後で、一覧の行をそれに合わせる。
        resync_engine_choice_combos(self)
        # 設定を読み込んだ **後** に無効化する。先に当てると、GPU 機で書かれた設定を
        # 読んだ時点で無効項目が選択された状態に戻ってしまう。
        self._init_device_extra_controls()
        apply_device_choice_limits(self)
        self._set_asr_controls_state()
        self._update_maai_plot_wall_clock_timer()
        self._refresh_module_status_panel()
        if self._remote_backend_enabled and os.environ.get("MAAI_SDS_QT_COMPOSE_UP_ON_START") == "1":
            # GUI の「対話サーバー」で Docker を選んで開いたとき: コンテナの起動からつなぐ（▶ と同じ）。
            QTimer.singleShot(0, lambda: self._dialog_server_start())
        else:
            QTimer.singleShot(0, self._ensure_dialog_orchestrator_worker_startup)

    @property
    def remote_backend_enabled(self) -> bool:
        return bool(self._remote_backend_enabled)

    def _docker_backend_path_from_picker(
        self,
        host_path: str,
        *,
        label: str,
        host_only: bool = False,
    ) -> str | None:
        """Convert a host file-picker result before storing it in Docker config."""
        if not self._remote_backend_enabled or host_only:
            return str(host_path)
        mapper = self._docker_path_mapper
        mapped = mapper.host_to_container(host_path) if mapper is not None else None
        if mapped is not None:
            return mapped
        QMessageBox.warning(
            self,
            "Dockerパス変換エラー",
            f"{label}はDockerのバインドマウント外にあるためコンテナから読み込めません。\n\n"
            f"{host_path}\n\n"
            "リポジトリ配下、INPUT_WAV_HOST、または MMDAGENT_ASSETS_HOST 配下のファイルを選択してください。",
        )
        return None

    def _docker_picker_initial_path(self, configured_path: str) -> str:
        if not self._remote_backend_enabled or not configured_path:
            return str(configured_path)
        mapper = self._docker_path_mapper
        mapped = mapper.container_to_host(configured_path) if mapper is not None else None
        return str(mapped) if mapped is not None else str(configured_path)

    def _host_path_for_backend_value(self, configured_path: str) -> Path | None:
        text = str(configured_path or "").strip()
        if not text:
            return None
        if not self._remote_backend_enabled:
            return Path(text)
        mapper = self._docker_path_mapper
        return mapper.host_equivalent(text) if mapper is not None else None

    def _container_path_for_backend_value(self, configured_path: str) -> str | None:
        text = str(configured_path or "").strip()
        if not text or not self._remote_backend_enabled:
            return text
        mapper = self._docker_path_mapper
        if mapper is None:
            return None
        if mapper.container_to_host(text) is not None:
            return text.replace("\\", "/")
        return mapper.host_to_container(text)

    def _remote_whisper_use_local_path(self, configured_path: str) -> bool:
        """Check a bind-mounted model on the host or defer to the container."""
        text = str(configured_path or "").strip()
        if not text:
            return False
        host_path = self._host_path_for_backend_value(text)
        if host_path is not None:
            return host_path.exists()
        normalized = text.replace("\\", "/")
        if normalized.startswith("/"):
            if normalized not in self._remote_unchecked_model_paths:
                self._remote_unchecked_model_paths.add(normalized)
                self.process_output_signal.emit(
                    "[DockerBackend] モデルパスは bind mount 外のコンテナパスのため、"
                    f"ホスト側の存在確認を省略します: {normalized}"
                )
            return True
        return False

    def _remote_mic_log(self, message: str) -> None:
        self.process_output_signal.emit(f"[DockerMic] {message}")

    def _stop_remote_mic_capture(self) -> None:
        sender = self._remote_mic_sender
        if sender is not None:
            sender.stop()

    def _clear_remote_orchestrator_state_cache(self) -> None:
        """Discard edge-triggered worker state before a remote reconnect."""
        if not self._remote_backend_enabled:
            return
        for event in self._module_ready_events.values():
            event.clear()
        self._audio_output_worker_ready_event.clear()
        self._audio_output_worker_started_output_mode = None
        self._audio_input_worker_ready_event.clear()
        self._audio_input_processing_event.clear()
        self._llm_worker_ready_event.clear()
        self._tts_worker_ready_event.clear()
        self._tts_worker_engine = None
        self._tts_worker_config_digest = None
        self._asr_ready = False
        self._asr_starting = False
        self._maai_ready = False
        self._maai_starting = False
        self._remote_plugin_settings_digest_sent = None

    def _sync_remote_mic_capture(self) -> None:
        if not self._remote_backend_enabled:
            return
        mode = ""
        combo = getattr(self, "audio_input_mode_combo", None)
        if combo is not None:
            mode = str(combo.currentText()).strip().lower()
        should_run = bool(
            not self._controller_closing
            and mode == "mic"
            and self._dialog_orchestrator_ready_event.is_set()
            and self._audio_input_worker_ready_event.is_set()
            and self._audio_input_processing_event.is_set()
        )
        if not should_run:
            self._stop_remote_mic_capture()
            return
        if sd is None:
            self._remote_mic_log("sounddevice が利用できないためホストマイクを開始できません。")
            return
        device = None
        device_combo = getattr(self, "audio_input_mic_device_combo", None)
        if device_combo is not None:
            try:
                raw_device = device_combo.currentData()
                if raw_device is not None and int(raw_device) >= 0:
                    device = int(raw_device)
            except (TypeError, ValueError):
                device = None
        if self._remote_mic_sender is None:
            self._remote_mic_sender = RemoteMicSender(
                self._send_dialog_orchestrator_command_sync,
                sounddevice_module=sd,
                log=self._remote_mic_log,
            )
        try:
            self._remote_mic_sender.start(device)
        except Exception as ex:
            self._remote_mic_log(f"ホストマイクの開始に失敗しました: {ex}")

    def _validate_remote_backend_constraints(self, *, tts_engine: str | None = None) -> None:
        if not self._remote_backend_enabled:
            return
        mode = self._get_audio_output_mode()
        if mode != "mmd":
            raise RuntimeError(
                "Dockerバックエンドでは音声出力先は「MMD（Remote）」のみ利用できます。"
                "ローカルスピーカー／TCP送出は選択できません。"
            )
        engine = str(tts_engine or "").strip()
        if engine in {"VOICEVOX", "AivisSpeech"}:
            raise RuntimeError(
                f"DockerバックエンドからWindows版{engine}を起動できません。"
                "VOICEVOX-Coreなどコンテナ内で動くTTSを選択してください。"
            )

    def _ensure_dialog_orchestrator_worker_startup(self) -> None:
        if self._controller_closing or self._dialog_server_user_stopped or self._dialog_server_busy:
            return
        try:
            self._ensure_dialog_orchestrator_worker()
            if self._remote_backend_enabled:
                host, port = self._remote_orchestrator_endpoint or ("", 0)
                self.process_output_signal.emit(
                    f"[DockerBackend] remote orchestrator 接続完了: {host}:{port}"
                )
            else:
                self.process_output_signal.emit("[DialogOrchestrator] 常駐起動完了")
        except Exception as ex:
            if self._remote_backend_enabled:
                host, port = self._remote_orchestrator_endpoint or ("", 0)
                message = (
                    f"remote orchestrator {host}:{port} に接続できません: {ex}。"
                    "1.5 秒後に再試行します。"
                )
                self.process_output_signal.emit(f"[DockerBackend][WARN] {message}")
                if hasattr(self, "status_label"):
                    self.status_label.setText("状態：Dockerバックエンド接続待ち")
                self._schedule_remote_orchestrator_retry()
            else:
                self._dialog_server_error = str(ex)
                self.process_output_signal.emit(f"[DialogOrchestrator][ERROR] 起動失敗: {ex}")
                QMessageBox.warning(self, "対話サーバーの起動エラー", str(ex))

    def _schedule_remote_orchestrator_retry(self) -> None:
        if not self._remote_backend_enabled or self._remote_orchestrator_retry_scheduled:
            return
        self._remote_orchestrator_retry_scheduled = True

        def _retry() -> None:
            self._remote_orchestrator_retry_scheduled = False
            if (
                self._controller_closing
                or self._dialog_orchestrator_ready_event.is_set()
                or self._dialog_server_user_stopped
                or self._dialog_server_busy
            ):
                return
            self._ensure_dialog_orchestrator_worker_startup()

        QTimer.singleShot(1500, _retry)

    def _open_settings_window(self) -> None:
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def _open_audio_input_settings_window(self) -> None:
        self._sync_audio_input_wav_transport_button()
        self.audio_input_settings_window.show()
        self.audio_input_settings_window.raise_()
        self.audio_input_settings_window.activateWindow()

    def _open_audio_output_settings_window(self) -> None:
        self.audio_output_settings_window.show()
        self.audio_output_settings_window.raise_()
        self.audio_output_settings_window.activateWindow()

    def _open_tts_template_auth_window(self) -> None:
        if self.tts_engine_combo.currentText().strip() != "TEMPLATE":
            return
        win = getattr(self, "tts_template_auth_window", None)
        if win is None:
            return
        win.show()
        win.raise_()
        win.activateWindow()

    def _open_tts_test_window(self) -> None:
        self.tts_test_window.show()
        self.tts_test_window.raise_()
        self.tts_test_window.activateWindow()

    def _open_asr_test_window(self) -> None:
        self.asr_test_window.show()
        self.asr_test_window.raise_()
        self.asr_test_window.activateWindow()

    def _open_llm_test_window(self) -> None:
        backend = self.llm_backend_combo.currentText().strip()
        if backend == "LM Studio":
            self.refresh_lmstudio_models(silent=True)
        elif backend == "OpenAI":
            self.refresh_openai_models(silent=True)
        self.llm_test_window.show()
        self.llm_test_window.raise_()
        self.llm_test_window.activateWindow()

    def _open_llm_s2s_test_window(self) -> None:
        self.llm_s2s_test_window.show()
        self.llm_s2s_test_window.raise_()
        self.llm_s2s_test_window.activateWindow()

    def _open_maai_test_window(self) -> None:
        self.maai_test_window.show()
        self.maai_test_window.raise_()
        self.maai_test_window.activateWindow()

    def _open_obs_settings_window(self) -> None:
        win = getattr(self, "obs_settings_window", None)
        if win is None:
            return
        win.show()
        win.raise_()
        win.activateWindow()

    def _should_obs_link_maai(self) -> bool:
        """OBS: MaAIテストの起動／停止でのみ連動（launch_maai_test / stop_maai_test 経由）。"""
        if not self._is_obs_remote_enabled():
            return False
        cb = getattr(self, "obs_remote_link_maai_check", None)
        return bool(cb is not None and cb.isChecked())

    def _should_obs_link_voice(self) -> bool:
        """OBS: 音声対話の開始／終了でのみ連動。音声対話用チェックが必須（MaAIテストと連動のみでは録画しない）。"""
        if not self._is_obs_remote_enabled():
            return False
        voice_cb = getattr(self, "obs_remote_link_voice_check", None)
        # 「音声対話と連動」がオフなら False。MaAIテストと連動だけONのときもここが False になる。
        return bool(voice_cb is not None and voice_cb.isChecked())

    def eventFilter(self, watched, event):  # noqa: N802
        if (
            hasattr(self, "llm_test_window")
            and watched is self.llm_test_window
            and event is not None
            and event.type() == QEvent.Type.Close
        ):
            if bool(getattr(self, "_controller_closing", False)):
                return super().eventFilter(watched, event)
            # LLMテスト画面を閉じたら、LM Studio上の残留モデルを解放する。
            if not (self._voice_dialog_active or self._voice_dialog_starting):
                self.unload_lmstudio_non_current_models(silent=True, reason="window_close", unload_all=True)
        return super().eventFilter(watched, event)

    def _open_maai_plot_window(self) -> None:
        self._sync_maai_plot_thresholds()
        self._apply_maai_plot_timing_from_ui()
        # 未初期化のときだけパネル可視化を同期する。
        # 初期化済みで再度呼ばれた場合は _relayout_panels() を起動しない（グラフ崩れ防止）。
        # パネル可視性はUIチェックボックスのシグナルで既にリアルタイム反映されている。
        if not self.maai_plot_window.initialized:
            self._sync_maai_plot_panel_visibility_from_ui()
        self.maai_plot_window.show()
        self.maai_plot_window.raise_()
        self.maai_plot_window.activateWindow()

    def _open_nod_motion_chart_window(self) -> None:
        win = getattr(self, "nod_motion_chart_window", None)
        if win is None:
            return
        win.show()
        win.raise_()
        win.activateWindow()

    def _sync_maai_plot_thresholds(self) -> None:
        if not hasattr(self, "maai_plot_window") or self.maai_plot_window is None:
            return

        def _parse_or_default(text: str, default: float) -> float:
            try:
                v = float(str(text).strip())
            except Exception:
                v = float(default)
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return float(v)

        vad_edit = getattr(self, "voice_dialog_interrupt_vad_threshold_edit", None)
        pnow_edit = getattr(self, "voice_dialog_interrupt_pnow_threshold_edit", None)
        turn_edit = getattr(self, "asr_maai_turn_threshold_edit", None)
        vad_on_edit = getattr(self, "asr_vad_on_threshold_edit", None)
        vad_off_edit = getattr(self, "asr_vad_off_threshold_edit", None)
        interrupt_vad = _parse_or_default(vad_edit.text() if vad_edit is not None else "0.1", 0.1)
        interrupt_p_now = _parse_or_default(pnow_edit.text() if pnow_edit is not None else "0.5", 0.5)
        turn_p_now = _parse_or_default(turn_edit.text() if turn_edit is not None else "0.5", 0.5)
        vad_on = _parse_or_default(vad_on_edit.text() if vad_on_edit is not None else "0.1", 0.1)
        vad_off = _parse_or_default(vad_off_edit.text() if vad_off_edit is not None else "0.2", 0.2)
        self.maai_plot_window.set_thresholds(
            interrupt_vad_threshold=interrupt_vad,
            interrupt_p_now_threshold=interrupt_p_now,
            turn_p_now_threshold=turn_p_now,
            vad_on_threshold=vad_on,
            vad_off_threshold=vad_off,
        )

    def _get_maai_plot_flush_interval_ms(self) -> int:
        edit = getattr(self, "maai_plot_flush_ms_edit", None)
        try:
            ms = int(float(str(edit.text()).strip())) if edit is not None else 200
        except Exception:
            ms = 200
        return max(20, min(5000, ms))

    def _get_maai_plot_asr_scroll_wall_clock_interval_ms(self) -> int:
        """GuiPlot の ASR セグメント壁時計スクロール専用（MaAI flush とは独立）。"""
        edit = getattr(self, "maai_plot_asr_scroll_ms_edit", None)
        try:
            ms = int(float(str(edit.text()).strip())) if edit is not None else 200
        except Exception:
            ms = 200
        return max(20, min(10000, ms))

    @staticmethod
    def _parse_maai_plot_float_interval(
        edit: object | None, default: float, lo: float, hi: float
    ) -> float:
        if edit is None:
            v = default
        else:
            try:
                v = float(str(edit.text()).strip())  # type: ignore[attr-defined]
            except Exception:
                v = default
        return max(lo, min(hi, float(v)))

    def _apply_maai_plot_timing_from_ui(self) -> None:
        w = getattr(self, "maai_plot_window", None)
        if w is None:
            return
        # UI は ms 表示。プロット側は秒で消費するため ms→sec に変換する。
        raw_ms = self._parse_maai_plot_float_interval(
            getattr(self, "maai_plot_raw_draw_interval_edit", None),
            1000.0 / 15.0,
            1000.0 / 240.0,
            2000.0,
        )
        w.fixed_draw_interval_raw_sec = raw_ms / 1000.0

    def _on_maai_plot_timing_ui_changed(self) -> None:
        self._apply_maai_plot_timing_from_ui()
        self._update_maai_plot_wall_clock_timer()

    def _on_maai_plot_wall_clock_tick(self) -> None:
        if not hasattr(self, "asr_maai_gui_plot_check") or not self.asr_maai_gui_plot_check.isChecked():
            return
        w = getattr(self, "maai_plot_window", None)
        if w is None:
            return
        # ingest は結果到着時に即実行済み。ここでは残存 pending の drain（互換）と描画のみ。
        self._drain_maai_plot_pending_ingest()
        w.tick_wall_clock_plot_scroll()

    def _update_maai_plot_wall_clock_timer(self) -> None:
        t = getattr(self, "_maai_plot_wall_clock_timer", None)
        if t is None:
            return
        if hasattr(self, "asr_maai_gui_plot_check") and self.asr_maai_gui_plot_check.isChecked():
            # 描画＝ingest＋スクロールを一本化したので、flush(ms) と ASR壁時計(ms) の
            # 小さい方を周期にする（どちらを下げても再描画が速くなる）。
            t.setInterval(
                min(
                    self._get_maai_plot_flush_interval_ms(),
                    self._get_maai_plot_asr_scroll_wall_clock_interval_ms(),
                )
            )
            if not t.isActive():
                t.start()
        else:
            t.stop()

    def _on_asr_maai_gui_plot_toggled(self) -> None:
        if not self._is_loading_app_config:
            self._save_app_config(sync_orchestrator=False)
        self._push_dialog_orchestrator_settings()
        self._update_maai_plot_wall_clock_timer()

    def _on_maai_plot_visibility_check_changed(self) -> None:
        if self._is_loading_app_config:
            return
        self._save_app_config()
        self._sync_maai_plot_panel_visibility_from_ui()

    def _sync_maai_plot_panel_visibility_from_ui(self) -> None:
        w = getattr(self, "maai_plot_window", None)
        if w is None or not hasattr(w, "configure_maai_plot_panel_visibility"):
            return
        vis: dict[str, bool] = {}
        for panel_id, _label in MAAI_PLOT_VISIBILITY_SPECS:
            cb = getattr(self, f"maai_plot_visibility_{panel_id}_check", None)
            vis[panel_id] = bool(cb.isChecked()) if cb is not None else maai_plot_default_visible(panel_id)
        w.configure_maai_plot_panel_visibility(vis)
        self._save_app_config()

    def _open_console_window(self) -> None:
        self.console_window.show()
        self.console_window.raise_()
        self.console_window.activateWindow()

    def _open_engine_manager_window(self) -> None:
        self.engine_manager_window.show()
        self.engine_manager_window.raise_()
        self.engine_manager_window.activateWindow()

    def _append_console_output(self, line: str) -> None:
        text = line.rstrip("\r\n")
        if not text:
            return
        try:
            # ANSI制御文字を落として表示を安定させる。
            text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
            self.console_text_edit.appendPlainText(text)
        except Exception:
            pass
        self._write_console_log(text)
        self._note_maai_process_metrics_from_console(text)

    _CONSOLE_LOG_KEEP = 20

    def _write_console_log(self, text: str) -> None:
        """コンソールに出た行を、デバッグ用に logs/console/ にも残す（アプリ起動ごとに 1 ファイル）。

        コンソール窓は閉じると消えるので、不具合が起きた後から追えるようにしている。
        行ごとに書き出す（落ちてもそこまでは残る）。古いファイルは新しい順に _CONSOLE_LOG_KEEP 個だけ残す。
        """
        fp = getattr(self, "_console_log_fp", None)
        if fp is None:
            if getattr(self, "_console_log_failed", False):
                return
            try:
                folder = _paths.logs_dir() / "console"
                folder.mkdir(parents=True, exist_ok=True)
                old_logs = sorted(folder.glob("console_*.log"), key=lambda f: f.stat().st_mtime)
                for stale in old_logs[: max(0, len(old_logs) - (self._CONSOLE_LOG_KEEP - 1))]:
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                path = folder / f"console_{datetime.now():%Y%m%d_%H%M%S}.log"
                fp = open(path, "a", encoding="utf-8", buffering=1)
                self._console_log_fp = fp
            except Exception:
                self._console_log_failed = True
                return
        try:
            fp.write(f"{datetime.now():%H:%M:%S.%f}"[:-3] + " " + text + "\n")
        except Exception:
            pass

    def _close_console_log(self) -> None:
        fp = getattr(self, "_console_log_fp", None)
        self._console_log_fp = None
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass

    def _on_device_extra_radio_toggled(self, checked: bool) -> None:
        """ラジオを操作したら、それを明示的な選択として記録する。

        マーカーが無い間は適用がデバイス extra を渡さない（= torch を触らない）。
        ここで書いて初めて +cu128 / +cpu に固定される。表示の同期中は無視する。
        """
        if not checked or getattr(self, "_device_radio_loading", False):
            return
        if engine_manager.device_extra_is_locked():
            return
        choice = "cpu" if self.device_extra_cpu_radio.isChecked() else "cu128"
        try:
            engine_manager.set_device_extra(choice)
        except (OSError, ValueError) as ex:
            self._append_console_output(f"[engine-manager] デバイス構成を記録できません: {ex}")
            return
        self._set_engine_install_notice(
            f"デバイス構成を {choice} にしました。「適用」で torch を入れ替えます。"
        )
        apply_device_choice_limits(self)

    def _init_device_extra_controls(self) -> None:
        """デバイス構成のラジオを現在の選択に合わせる。

        環境変数 MAAI_SDS_DEVICE_EXTRA で固定されている場合（コンテナ）は、
        ここで変えても読み側が env を先に見るので触れないようにする。
        """
        current = engine_manager.device_extra()
        # 未選択なら、今入っている torch のビルドを表示に使う（PyPI 版なら CPU 扱い）。
        shown = current or engine_manager.installed_torch_device_extra() or "cpu"
        self._device_radio_loading = True
        try:
            self.device_extra_cpu_radio.setChecked(shown == "cpu")
            self.device_extra_gpu_radio.setChecked(shown != "cpu")
        finally:
            self._device_radio_loading = False
        if engine_manager.device_extra_is_locked():
            return
        choice = "cpu" if self.device_extra_cpu_radio.isChecked() else "cu128"
        try:
            engine_manager.set_device_extra(choice)
        except (OSError, ValueError) as ex:
            self._append_console_output(f"[engine-manager] デバイス構成を記録できません: {ex}")
            return
        self._set_engine_install_notice(
            f"デバイス構成を {choice} にしました。「適用」で torch を入れ替えます。"
        )
        apply_device_choice_limits(self)

    def _init_device_extra_controls(self) -> None:
        """デバイス構成のラジオを現在の選択に合わせる。

        環境変数 MAAI_SDS_DEVICE_EXTRA で固定されている場合（コンテナ）は、
        ここで変えても読み側が env を先に見るので触れないようにする。
        """
        current = engine_manager.device_extra()
        # 未選択なら、今入っている torch のビルドを表示に使う（PyPI 版なら CPU 扱い）。
        shown = current or engine_manager.installed_torch_device_extra() or "cpu"
        self._device_radio_loading = True
        try:
            self.device_extra_cpu_radio.setChecked(shown == "cpu")
            self.device_extra_gpu_radio.setChecked(shown != "cpu")
        finally:
            self._device_radio_loading = False
        if engine_manager.device_extra_is_locked():
            for radio in (self.device_extra_cpu_radio, self.device_extra_gpu_radio):
                radio.setEnabled(False)
            self.device_extra_note.setText(
                "起動環境でデバイス構成が固定されているため変更できません。"
            )
        else:
            self.device_extra_note.setText(
                "「適用」で切り替わります。GPU→CPU は数GBの入れ替えになります。"
            )

    def _set_engine_install_notice(self, text: str, *, error: bool = False) -> None:
        """エンジン管理画面のメッセージ欄。空文字で隠す。"""
        label = getattr(self, "engine_install_notice", None)
        if label is None:
            return
        label.setText(text)
        label.setStyleSheet(
            f"color:{STATE_COLORS['error']};font-weight:bold;" if error else f"color:{STATE_COLORS['running']};"
        )
        label.setVisible(bool(text))

    def _running_modules(self) -> list[str]:
        """今動いている（または起動中の）モジュール名。

        OS のプロセス一覧は見ない。アプリは自分が起動したワーカーの状態を
        `_module_panel_states` に持っているので、それが唯一の判断材料でよい。
        コマンドライン照合は誤判定しやすく、実際に無関係なプロセスまで拾った。
        """
        busy: list[str] = []
        for module in self._MODULE_PANEL_KEYS:
            state = str(self._module_panel_states.get(module, {}).get("state", "stopped"))
            if state in ("running", "starting"):
                busy.append(module)
        return busy

    def _engine_manager_is_available(self) -> tuple[bool, str]:
        """エンジン管理を使える状態か。使えないなら理由を返す。

        `uv sync` は exact sync なので、どのパッケージも入れ替えうる。ワーカーが
        動いている間はそれらがロードしている DLL を置き換えられず（Windows）、
        torch の入れ替えは実際に

            failed to rename ... torch\\lib\\c10.dll: アクセスが拒否されました

        で落ちた。途中で失敗すると venv が壊れた状態で残るので、走らせる前に断る。
        """
        busy = self._running_modules()
        if busy:
            return False, (
                "エンジン管理は全モジュールが停止しているときだけ使えます。"
                f"稼働中: {', '.join(busy)}。"
                "対話を停止（またはモジュールを個別に停止）してからもう一度お試しください。"
            )
        return True, ""

    def _apply_engine_installation(self) -> None:
        if self._engine_install_thread is not None and self._engine_install_thread.is_alive():
            return
        available, reason = self._engine_manager_is_available()
        if not available:
            self._append_console_output(f"[engine-manager] {reason}")
            self._set_engine_install_notice(reason, error=True)
            return
        ids = [engine_id for engine_id, check in self.engine_install_checks.items() if check.isChecked()]
        # 何が入り、何が外れるかを先に見せる。数 GB の取得や、使っているエンジンの
        # Python パッケージの削除が、押しただけで黙って始まらないようにする。
        if not self._confirm_engine_changes(ids):
            return
        # デバイス構成を変える適用は torch そのものを入れ替える。Windows では
        # ロード済み DLL を置き換えられないので、torch を読むプロセス（対話ワーカー）が
        # 動いていると uv sync が c10.dll の rename で必ず失敗する。生の uv エラーの
        # 代わりに、先に手順を伝えて止める。
        self.engine_install_apply_btn.setEnabled(False)
        self._set_engine_install_notice("適用中… 完了までこのウィンドウを閉じないでください。")
        self._append_console_output("[engine-manager] Applying optional engine dependencies...")
        thread = threading.Thread(
            target=self._run_engine_install_worker,
            args=(ids, engine_manager.device_extra()),
            daemon=True,
            name="controller-engine-install",
        )
        self._engine_install_thread = thread
        thread.start()

    def _run_engine_install_worker(self, ids: list[str], device: str) -> None:
        """Runs off the GUI thread. Only touches Qt via signal `.emit()` (thread-safe queuing) —
        never widgets directly. Plain threading.Thread, not QThread: this app already bridges
        background threads to the GUI via Signal elsewhere (see
        _dialog_orchestrator_sender_thread), and QThread's own dual Python/C++ object lifecycle
        was the direct cause of two prior bugs here (a stale-reference crash on re-apply, and a
        "QThreadStorage destroyed before end of thread" warning on app exit)."""
        # This thread runs inside the GUI's own PySide6 process — `gui` must always stay in the
        # synced extra set, or `uv sync` tries (and fails on Windows, since the running process
        # holds the DLLs open) to uninstall PySide6 out from under itself. See
        # engine_manager.set_enabled_engines docstring.
        ok, message = engine_manager.set_enabled_engines(
            ids,
            always_extras=("gui",),
            device=device,
            on_output=self._engine_install_output_signal.emit,
        )
        self._engine_install_completed_signal.emit(ok, message)

    def _on_engine_installation_complete(self, ok: bool, message: str) -> None:
        self._engine_install_thread = None
        self._append_console_output(f"[engine-manager] {message}")
        self._set_engine_install_notice(message, error=not ok)
        # デバイス構成が変わっていれば、GPU 専用の選択肢の可否も変わる。掛け直さないと
        # 再起動するまで古い状態のままになる（CPU→GPU にしたのに GPU が選べない）。
        self._init_device_extra_controls()
        apply_device_choice_limits(self)
        self._refresh_engine_install_status()
        self.engine_install_apply_btn.setEnabled(True)

    def _engine_install_plan(self) -> tuple[list, dict]:
        engines = engine_manager.list_engines()
        selected = [eid for eid, check in getattr(self, "engine_install_checks", {}).items() if check.isChecked()]
        return engines, engine_manager.plan_engine_changes(engines, selected)

    def _refresh_engine_install_status(self) -> None:
        """各行の状態（インストール済み・適用で入れる・適用で外す・未インストール）と合計を出す。"""
        labels = getattr(self, "engine_install_status_labels", None)
        if not labels:
            return
        engines, plan = self._engine_install_plan()
        install, remove = set(plan["install"]), set(plan["remove"])
        for info in engines:
            label = labels.get(info.id)
            if label is None:
                continue
            if info.id in install:
                text, color = "適用で入れる", TOKENS["accent"]
            elif info.id in remove:
                text, color = "適用で外す", TOKENS["warn"]
            elif info.installed:
                text, color = "インストール済み", TOKENS["ok"]
            else:
                text, color = "未インストール", TOKENS["muted"]
            label.setText(text)
            label.setStyleSheet(f"color: {color};")
        summary = getattr(self, "engine_install_summary", None)
        if summary is not None:
            parts = []
            if install:
                parts.append(f"入れる：{len(install)} 件（取得 約{plan['download_gb']:g}GB）")
            if remove:
                parts.append(f"外す：{len(remove)} 件")
            summary.setText("「適用」で " + "、".join(parts) if parts else "変更はありません。")

    def _confirm_engine_changes(self, ids: list[str]) -> bool:
        engines, plan = self._engine_install_plan()
        if not plan["install"] and not plan["remove"]:
            return True
        names = {e.id: e.label for e in engines}
        sizes = {e.id: e.download_gb for e in engines}
        lines = []
        if plan["install"]:
            lines.append(f"入れるもの（取得 約{plan['download_gb']:g}GB）：")
            for eid in plan["install"]:
                gb = sizes.get(eid)
                lines.append(f"  ・{names.get(eid, eid)}" + (f"（約{gb:g}GB）" if gb else ""))
        if plan["remove"]:
            lines.append("外すもの（Pythonパッケージを削除。モデルなどのファイルは残る）：")
            lines.extend(f"  ・{names.get(eid, eid)}" for eid in plan["remove"])
        lines.append("")
        lines.append("容量は目安で、Pythonパッケージの分は含みません。適用しますか？")
        answer = QMessageBox.question(
            self, "エンジン管理", "\n".join(lines),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Ok

    def _attach_process_console_output(self, name: str, proc) -> None:
        if proc is None or proc.pid is None:
            return
        if proc in self._attached_console_procs:
            return
        self._attached_console_procs.add(proc)
        # MMDAgent-EX は、まばたき・頷きの値を毎フレーム受けるたびに 2 行出す。コンソールが埋まるので出さない。
        drop = is_per_frame_line if name == "MMDAgent" else None

        def _pump(stream, stream_name: str) -> None:
            if stream is None:
                return
            try:
                for raw in iter(stream.readline, ""):
                    line = raw.rstrip("\r\n")
                    if line and not (drop and drop(line)):
                        if _WORKER_PREFIX_RE.match(line):
                            self.process_output_signal.emit(line)
                        else:
                            self.process_output_signal.emit(f"[{name}][{stream_name}] {line}")
            except Exception as ex:
                self.process_output_signal.emit(f"[{name}][{stream_name}] <read error: {ex}>")

        threading.Thread(target=_pump, args=(getattr(proc, "stdout", None), "STDOUT"), daemon=True).start()
        threading.Thread(target=_pump, args=(getattr(proc, "stderr", None), "STDERR"), daemon=True).start()

        def _reap_pid() -> None:
            # プロセス終了後に添付エントリを解放する。Popen 識別なので PID 再利用と無関係。
            try:
                proc.wait()
            except Exception:
                pass
            self._attached_console_procs.discard(proc)

        threading.Thread(target=_reap_pid, daemon=True, name=f"pid-reap-{proc.pid}").start()

    def _on_tts_engine_changed(self, engine: str) -> None:
        previous_engine = self._last_selected_tts_engine
        if previous_engine and previous_engine != engine:
            self._save_current_engine_voice(previous_engine)
            self._auto_stop_engine_process(previous_engine)

        if self.tts_engine_combo.currentText() != engine:
            self.tts_engine_combo.blockSignals(True)
            self.tts_engine_combo.setCurrentText(engine)
            self.tts_engine_combo.blockSignals(False)

        is_voicevox = (engine == "VOICEVOX")
        is_voicevox_core = (engine == "VOICEVOX-Core")
        is_kokoro_realtime = (engine == "Kokoro-RealtimeTTS")
        is_qwen_cpp = (engine == "Qwen3-TTS-CPP")
        is_lfm2_audio = (engine == "LFM2.5-Audio")
        is_cartesia = (engine == "Cartesia-Realtime")
        is_aivis = (engine == "AivisSpeech")
        is_supertonic = (engine == "Supertonic-3")
        is_irodori = (engine == "Irodori-v4")
        is_tsukasa = (engine == "Tsukasa")
        is_template = (engine == "TEMPLATE")

        self.voicevox_exe_label.setVisible(is_voicevox)
        self.voicevox_exe_edit.setVisible(is_voicevox)
        self.voicevox_exe_browse_btn.setVisible(is_voicevox)
        self.voicevox_launch_btn.setVisible(is_voicevox)
        self.voicevox_stop_btn.setVisible(is_voicevox)
        self.voicevox_core_launch_btn.setVisible(is_voicevox_core)
        self.voicevox_core_stop_btn.setVisible(is_voicevox_core)
        self.kokoro_realtimetts_launch_btn.setVisible(is_kokoro_realtime)
        self.kokoro_realtimetts_stop_btn.setVisible(is_kokoro_realtime)
        self.qwen_cpp_launch_btn.setVisible(is_qwen_cpp)
        self.qwen_cpp_stop_btn.setVisible(is_qwen_cpp)
        self.lfm2_audio_launch_btn.setVisible(is_lfm2_audio)
        self.lfm2_audio_stop_btn.setVisible(is_lfm2_audio)
        self.aivis_launch_btn.setVisible(is_aivis)
        self.aivis_stop_btn.setVisible(is_aivis)
        self.supertonic_launch_btn.setVisible(is_supertonic)
        self.supertonic_stop_btn.setVisible(is_supertonic)
        self.irodori_launch_btn.setVisible(is_irodori)
        self.irodori_stop_btn.setVisible(is_irodori)
        self.tsukasa_launch_btn.setVisible(is_tsukasa)
        self.tsukasa_stop_btn.setVisible(is_tsukasa)
        # EXE設定は手動起動が必要なエンジンのみ表示する。
        self.tts_exe_box.setVisible(is_voicevox)
        self.voicevox_exe_box.setVisible(is_voicevox)
        self.tts_settings_box.setVisible(
            is_template
            or is_qwen_cpp
            or is_lfm2_audio
            or is_kokoro_realtime
            or is_cartesia
            or is_voicevox_core
            or is_aivis
            or is_supertonic
            or is_irodori
            or is_tsukasa
        )
        self.tts_connection_box.setVisible(is_template)
        # 中の項目（tts_model）は Kokoro でしか出さないので、枠も Kokoro のときだけ出す
        # （以前は Qwen3 / Cartesia / VOICEVOX-Core で空の枠が残っていた）。
        self.tts_model_box.setVisible(is_kokoro_realtime)
        self.voicevox_core_settings_box.setVisible(is_voicevox_core)
        self.qwen3_tts_cpp_settings_box.setVisible(is_qwen_cpp)
        self.lfm2_audio_settings_box.setVisible(is_lfm2_audio)
        self.cartesia_tts_settings_box.setVisible(is_cartesia)
        self.aivisspeech_settings_box.setVisible(is_aivis)
        self.supertonic_settings_box.setVisible(is_supertonic)
        self.irodori_settings_box.setVisible(is_irodori)
        self.tsukasa_settings_box.setVisible(is_tsukasa)
        self.template_tts_settings_box.setVisible(is_template)
        self.tts_voice_label.setVisible(True)
        self.tts_voice_edit.setVisible(True)
        # LFM2.5-Audio はモデルの声 1 つだけで選べない。
        self.tts_voice_edit.setEnabled(not is_lfm2_audio)
        if is_voicevox:
            self.tts_voice_label.setText("話者ID（VOICEVOX）")
        elif is_voicevox_core:
            self.tts_voice_label.setText("話者ID（VOICEVOX Core）")
        elif is_kokoro_realtime:
            self.tts_voice_label.setText("音声ID（Kokoro-RealtimeTTS）")
        elif is_qwen_cpp:
            self.tts_voice_label.setText("話者（Qwen3-TTS C++）")
        elif is_lfm2_audio:
            self.tts_voice_label.setText("声（LFM2.5-Audio は選べません）")
        elif is_cartesia:
            self.tts_voice_label.setText("音声ID（Cartesia）")
        elif is_aivis:
            self.tts_voice_label.setText("話者スタイルID（AivisSpeech）")
        elif is_supertonic:
            self.tts_voice_label.setText("プリセット音声（F1～F5／M1～M5）")
        elif is_irodori:
            self.tts_voice_label.setText("声質キャプション（参照音声なし時）")
        elif is_tsukasa:
            self.tts_voice_label.setText("音声（参照音声ファイルで指定）")
        else:
            self.tts_voice_label.setText("話者／音声")
        self.tts_host_label.setVisible(is_template)
        self.tts_host_edit.setVisible(is_template)
        self.tts_port_label.setVisible(is_template)
        self.tts_port_edit.setVisible(is_template)
        self.tts_base_path_label.setVisible(is_template)
        self.tts_base_path_edit.setVisible(is_template)
        self.tts_model_label.setVisible(is_kokoro_realtime)
        self.tts_model_edit.setVisible(is_kokoro_realtime)
        auth_win = getattr(self, "tts_template_auth_window", None)
        if auth_win is not None and not is_template:
            auth_win.hide()
        for widget in (
            self.qwen3_tts_cpp_server_exe_label,
            self.qwen3_tts_cpp_server_exe_edit,
            self.qwen3_tts_cpp_model_path_label,
            self.qwen3_tts_cpp_model_path_edit,
            self.qwen3_tts_cpp_host_label,
            self.qwen3_tts_cpp_host_edit,
            self.qwen3_tts_cpp_port_label,
            self.qwen3_tts_cpp_port_edit,
            self.qwen3_tts_cpp_model_id_label,
            self.qwen3_tts_cpp_model_id_edit,
            self.qwen3_tts_cpp_speaker_label,
            self.qwen3_tts_cpp_speaker_edit,
            self.qwen3_tts_cpp_instruct_label,
            self.qwen3_tts_cpp_instruct_edit,
            self.qwen3_tts_cpp_language_label,
            self.qwen3_tts_cpp_language_edit,
            self.qwen3_tts_cpp_do_sample_label,
            self.qwen3_tts_cpp_do_sample_check,
            self.qwen3_tts_cpp_temperature_label,
            self.qwen3_tts_cpp_temperature_edit,
            self.qwen3_tts_cpp_top_k_label,
            self.qwen3_tts_cpp_top_k_edit,
            self.qwen3_tts_cpp_top_p_label,
            self.qwen3_tts_cpp_top_p_edit,
            self.qwen3_tts_cpp_repetition_penalty_label,
            self.qwen3_tts_cpp_repetition_penalty_edit,
            self.qwen3_tts_cpp_seed_label,
            self.qwen3_tts_cpp_seed_edit,
        ):
            widget.setVisible(is_qwen_cpp)
        self.voicevox_core_onnxruntime_label.setVisible(is_voicevox_core)
        self.voicevox_core_onnxruntime_edit.setVisible(is_voicevox_core)
        self.voicevox_core_openjtalk_dict_label.setVisible(is_voicevox_core)
        self.voicevox_core_openjtalk_dict_edit.setVisible(is_voicevox_core)
        self.voicevox_core_vvm_label.setVisible(is_voicevox_core)
        self.voicevox_core_vvm_edit.setVisible(is_voicevox_core)
        self.voicevox_core_device_label.setVisible(is_voicevox_core)
        self.voicevox_core_device_combo.setVisible(is_voicevox_core)
        self.cartesia_tts_model_id_label.setVisible(is_cartesia)
        self.cartesia_tts_model_id_edit.setVisible(is_cartesia)
        self.cartesia_tts_sample_rate_label.setVisible(is_cartesia)
        self.cartesia_tts_sample_rate_edit.setVisible(is_cartesia)
        self.cartesia_tts_api_key_label.setVisible(is_cartesia)
        self.cartesia_tts_api_key_edit.setVisible(is_cartesia)
        self.cartesia_tts_api_key_env_label.setVisible(is_cartesia)
        self.cartesia_tts_api_key_env_edit.setVisible(is_cartesia)
        self.template_load_btn.setVisible(is_template)
        self.template_save_btn.setVisible(is_template)
        self.tts_template_label.setVisible(is_template)
        self.tts_template_edit.setVisible(is_template)
        self._restore_engine_voice(engine)
        self._apply_engine_defaults(engine)
        self._save_current_engine_voice(engine)
        self._last_selected_tts_engine = engine
        self._sync_tts_launch_test_controls()

    def _begin_tts_engine_launch(self, engine: str) -> None:
        self._tts_engine_launching = engine
        self._set_module_panel_state("tts", "starting", engine)
        self._sync_tts_launch_test_controls()

    def _end_tts_engine_launch(self) -> None:
        self._tts_engine_launching = None
        self._sync_tts_launch_test_controls()

    def _sync_tts_launch_test_controls(self) -> None:
        """TTS 各エンジン起動中は起動/停止/テスト送信を無効にする。"""
        launching = self._tts_engine_launching
        busy = launching is not None
        test_ok = (not self._tts_test_in_progress) and not busy
        if hasattr(self, "tts_test_send_btn"):
            self.tts_test_send_btn.setEnabled(test_ok)
        pairs = [
            ("voicevox_launch_btn", "voicevox_stop_btn"),
            ("voicevox_core_launch_btn", "voicevox_core_stop_btn"),
            ("kokoro_realtimetts_launch_btn", "kokoro_realtimetts_stop_btn"),
            ("qwen_cpp_launch_btn", "qwen_cpp_stop_btn"),
            ("lfm2_audio_launch_btn", "lfm2_audio_stop_btn"),
            ("aivis_launch_btn", "aivis_stop_btn"),
            ("supertonic_launch_btn", "supertonic_stop_btn"),
            ("irodori_launch_btn", "irodori_stop_btn"),
            ("tsukasa_launch_btn", "tsukasa_stop_btn"),
        ]
        for launch_attr, stop_attr in pairs:
            launch_btn = getattr(self, launch_attr, None)
            stop_btn = getattr(self, stop_attr, None)
            if launch_btn is None or stop_btn is None:
                continue
            if not launch_btn.isVisible():
                continue
            launch_btn.setEnabled(not busy)
            stop_btn.setEnabled(not busy)

    def _save_current_engine_voice(self, engine: str) -> None:
        if not engine:
            return
        value = self.tts_voice_edit.text().strip()
        if value:
            self._tts_voice_by_engine[engine] = value

    def _restore_engine_voice(self, engine: str) -> None:
        if not engine:
            return
        value = self._tts_voice_by_engine.get(engine, "")
        if value and self.tts_voice_edit.text().strip() != value:
            self.tts_voice_edit.setText(value)

    def _apply_engine_defaults(self, engine: str) -> None:
        host = self.tts_host_edit.text().strip()
        port = self.tts_port_edit.text().strip()
        base_path = self.tts_base_path_edit.text().strip()
        voice = self.tts_voice_edit.text().strip()
        model = self.tts_model_edit.text().strip()

        known_default_ports = {"14500", "50021", "8880", "8000", "8080"}
        default_host = "127.0.0.1"

        if engine == "VOICEVOX":
            if not host:
                self.tts_host_edit.setText(default_host)
            if (not port) or (port in known_default_ports and port != "50021"):
                self.tts_port_edit.setText("50021")
            if not voice or voice == "kyoto_shiori":
                self.tts_voice_edit.setText("1")
            if base_path == "/v1":
                self.tts_base_path_edit.setText("")
            return

        if engine == "VOICEVOX-Core":
            if not voice or voice == "kyoto_shiori":
                self.tts_voice_edit.setText("8")
            if not self.voicevox_core_onnxruntime_edit.text().strip():
                self.voicevox_core_onnxruntime_edit.setText(
                    r"voicevox_runtime/onnxruntime/lib/voicevox_onnxruntime.dll"
                )
            if not self.voicevox_core_openjtalk_dict_edit.text().strip():
                self.voicevox_core_openjtalk_dict_edit.setText(
                    r"voicevox_runtime/dict/open_jtalk_dic_utf_8-1.11"
                )
            if not self.voicevox_core_vvm_edit.text().strip():
                self.voicevox_core_vvm_edit.setText(r"voicevox_runtime/models/vvms/0.vvm")
            if not self.voicevox_core_device_combo.currentText().strip():
                self.voicevox_core_device_combo.setCurrentText("GPU")
            return

        if engine == "Kokoro-RealtimeTTS":
            if not model:
                self.tts_model_edit.setText("kokoro")
            if (not voice) or voice.isdigit() or voice in {"kyoto_shiori", "1"}:
                self.tts_voice_edit.setText("jf_alpha")
            return

        if engine == "Qwen3-TTS-CPP":
            cpp_defaults = (
                (self.qwen3_tts_cpp_server_exe_edit, r"audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe"),
                (
                    self.qwen3_tts_cpp_model_path_edit,
                    r"audio.cpp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf",
                ),
                (self.qwen3_tts_cpp_host_edit, "127.0.0.1"),
                (self.qwen3_tts_cpp_port_edit, "18081"),
                (self.qwen3_tts_cpp_model_id_edit, "qwen3-tts-customvoice"),
                (self.qwen3_tts_cpp_speaker_edit, "Vivian"),
                (self.qwen3_tts_cpp_language_edit, "japanese"),
                (self.qwen3_tts_cpp_temperature_edit, "0.9"),
                (self.qwen3_tts_cpp_top_k_edit, "50"),
                (self.qwen3_tts_cpp_top_p_edit, "1.0"),
                (self.qwen3_tts_cpp_repetition_penalty_edit, "1.05"),
            )
            for edit, default in cpp_defaults:
                if not edit.text().strip():
                    edit.setText(default)
            return

        if engine == "LFM2.5-Audio":
            lfm2_defaults = (
                (self.lfm2_audio_host_edit, "127.0.0.1"),
                (self.lfm2_audio_port_edit, "18082"),
                (self.lfm2_audio_model_id_edit, "lfm2.5-audio-jp"),
                (self.lfm2_audio_model_path_edit, r"audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF"),
                (self.lfm2_audio_server_exe_edit, r"audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe"),
            )
            for edit, default in lfm2_defaults:
                if not edit.text().strip():
                    edit.setText(default)
            return

        if engine == "Cartesia-Realtime":
            if not self.cartesia_tts_model_id_edit.text().strip():
                self.cartesia_tts_model_id_edit.setText("sonic-3")
            if not self.tts_voice_edit.text().strip():
                self.tts_voice_edit.setText("f786b574-daa5-4673-aa0c-cbe3e8534c02")
            if not self.cartesia_tts_sample_rate_edit.text().strip():
                self.cartesia_tts_sample_rate_edit.setText("16000")
            if not self.cartesia_tts_api_key_env_edit.text().strip():
                self.cartesia_tts_api_key_env_edit.setText("CARTESIA_API_KEY")
            return

        if engine == "AivisSpeech":
            if not self.tts_voice_edit.text().strip().isdigit():
                self.tts_voice_edit.setText("888753761")
            return

        if engine == "Supertonic-3":
            if self.tts_voice_edit.text().strip().upper() not in {"F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"}:
                self.tts_voice_edit.setText("F1")
            return

        if engine == "Irodori-v4":
            if not self.tts_voice_edit.text().strip() or self.tts_voice_edit.text().strip().upper() in {"F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"} or self.tts_voice_edit.text().strip().isdigit():
                self.tts_voice_edit.setText("明るく自然な日本語の声")
            return

    def _resolve_llm_split_settings(self) -> dict[str, object]:
        # `vds.resolve_llm_split_settings` に一本化（Phase 1-2）。
        return vds.resolve_llm_split_settings(self._widget_config_snapshot())

    def run_llm_s2s_test(self) -> None:
        text = self.llm_s2s_test_input_edit.toPlainText()
        if not text.strip():
            self.llm_s2s_test_output_edit.setPlainText("入力テキストを指定してください。")
            return
        try:
            chunk_size = int(float(self.llm_s2s_test_chunk_size_edit.text().strip() or "2"))
        except Exception:
            self.llm_s2s_test_output_edit.setPlainText("chunk size は整数で指定してください。")
            return
        if chunk_size < 1:
            chunk_size = 1
        settings = self._resolve_llm_split_settings()
        enabled = bool(self.llm_test_tts_input_streaming_check.isChecked())
        out = llm_s2s_preview.build_preview_lines(
            text,
            enabled=enabled,
            chunk_size=chunk_size,
            settings=settings,
        )
        self.llm_s2s_test_output_edit.setPlainText(out)

    def _auto_stop_engine_process(self, engine: str) -> None:
        try:
            if engine in {"AivisSpeech", "Supertonic-3", "Qwen3-TTS-CPP", "LFM2.5-Audio", "Irodori-v4", "Tsukasa"}:
                if self._tts_worker_ready_event.is_set() and self._tts_worker_engine == engine:
                    self._teardown_tts_ipc()
                    self._log_stream(f"エンジン切替により {engine} TTS worker を停止")
                if engine == "AivisSpeech":
                    self._tts_bridge_rpc("stop_aivisspeech", timeout_sec=20.0)
                return
            if engine == "VOICEVOX":
                if self._tts_bridge_engine_running("VOICEVOX"):
                    self._sync_tts_exe_paths_to_orchestrator()
                    self._tts_bridge_rpc("stop_voicevox", timeout_sec=15.0)
                    self._log_stream("エンジン切替により VOICEVOX を自動停止")
            elif engine == "Kokoro-RealtimeTTS":
                if self._tts_worker_ready_event.is_set() and self._tts_worker_engine == "Kokoro-RealtimeTTS":
                    self._teardown_tts_ipc()
                    self._log_stream("エンジン切替により Kokoro-RealtimeTTS IPCワーカーを自動停止")
            elif engine == "VOICEVOX-Core":
                if self._tts_worker_ready_event.is_set() and self._tts_worker_engine == "VOICEVOX-Core":
                    self._teardown_tts_ipc()
                    self._log_stream("エンジン切替により VOICEVOX-Core IPCワーカーを自動停止")
            elif engine == "Cartesia-Realtime":
                if self._tts_worker_ready_event.is_set() and self._tts_worker_engine == "Cartesia-Realtime":
                    self._teardown_tts_ipc()
                    self._log_stream("エンジン切替により Cartesia-Realtime IPCワーカーを自動停止")
        except Exception as ex:
            self._log_stream(f"エンジン切替時の自動停止に失敗: {engine} / {ex}")

    def _is_tts_engine_running(self, engine: str) -> bool:
        if engine == "VOICEVOX":
            return self._tts_bridge_engine_running("VOICEVOX")
        if engine == "Kokoro-RealtimeTTS":
            return self._tts_worker_ready_event.is_set() and self._tts_worker_engine == "Kokoro-RealtimeTTS"
        if engine == "VOICEVOX-Core":
            return self._tts_worker_ready_event.is_set() and self._tts_worker_engine == "VOICEVOX-Core"
        if engine == "Cartesia-Realtime":
            return True
        return True

    def _ensure_tts_engine_for_test(self, engine: str) -> None:
        """テスト再生に必要な、GUI 管理下の外部 TTS を起動する。

        worker 型エンジンは Dialog Orchestrator 側の ``ensure_tts`` が起動するため、
        ここで明示的な起動が必要なのは外部プロセスを持つ VOICEVOX と
        AivisSpeech Engine だけである。TEMPLATE は任意の外部 HTTP API を指す
        方式であり、起動コマンドを持たないため対象外。
        """
        self._validate_remote_backend_constraints(tts_engine=engine)
        if engine == "VOICEVOX":
            if self._tts_bridge_engine_running("VOICEVOX"):
                return
            result = self._tts_bridge_rpc("launch_voicevox", timeout_sec=90.0)
            if not result or not result.get("ok"):
                message = str(result.get("message", "")) if result else "Orchestrator から応答がありません。"
                raise RuntimeError(message or "VOICEVOX の起動に失敗しました。")
            self._set_module_panel_state("tts", "running", "VOICEVOX")
            self._log_stream("TTSテストのため VOICEVOX を自動起動")
            return

        if engine == "AivisSpeech":
            host = self.aivisspeech_host_edit.text().strip() or "127.0.0.1"
            try:
                port = int(self.aivisspeech_port_edit.text().strip() or "10104")
            except ValueError as ex:
                raise RuntimeError("AivisSpeech のポートは整数で指定してください。") from ex
            result = self._tts_bridge_rpc(
                "launch_aivisspeech", timeout_sec=75.0, host=host, port=port, use_gpu=True
            )
            if not result or not result.get("ok"):
                message = str(result.get("message", "")) if result else "Orchestrator から応答がありません。"
                raise RuntimeError(message or "AivisSpeech Engine の起動に失敗しました。")
            self._log_stream("TTSテストのため AivisSpeech Engine を自動起動")

    def _tts_stream_mode(self, engine: str) -> str:
        return self.TTS_STREAM_MODE_BY_ENGINE.get(engine, "batch_only")

    def _load_default_tts_template_text(self) -> str:
        if self.TTS_TEMPLATE_PATH.exists():
            return self.TTS_TEMPLATE_PATH.read_text(encoding="utf-8")
        return "{}"

    def _load_tts_template_from_json(self) -> None:
        current = self.TTS_TEMPLATE_PATH if self.TTS_TEMPLATE_PATH.exists() else Path.cwd() / "tts_template.json"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "TTSテンプレートJSONを読込",
            str(current),
            "JSON (*.json);;All Files (*.*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as ex:
            QMessageBox.warning(self, "エラー", f"JSON読込に失敗しました。\n{path}\n\n{ex}")
            return
        if not isinstance(data, dict):
            QMessageBox.warning(self, "エラー", "テンプレートJSONの最上位はオブジェクトである必要があります。")
            return
        self.tts_template_edit.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        self.status_label.setText(f"状態：テンプレートJSON読込済み（{path.name}）")
        self._save_app_config()

    def _save_tts_template_to_json(self) -> None:
        raw = self.tts_template_edit.toPlainText().strip()
        if not raw:
            QMessageBox.warning(self, "エラー", "保存するテンプレートJSONが空です。")
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as ex:
            QMessageBox.warning(self, "エラー", f"テンプレートJSONが不正です。\n{ex}")
            return
        if not isinstance(data, dict):
            QMessageBox.warning(self, "エラー", "テンプレートJSONの最上位はオブジェクトである必要があります。")
            return
        default_target = self.TTS_TEMPLATE_PATH if self.TTS_TEMPLATE_PATH.name else Path.cwd() / "tts_template.json"
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "TTSテンプレートJSONを保存",
            str(default_target),
            "JSON (*.json);;All Files (*.*)",
        )
        if not target_path:
            return
        path = Path(target_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized = json.dumps(data, ensure_ascii=False, indent=2)
            path.write_text(normalized, encoding="utf-8")
        except Exception as ex:
            QMessageBox.warning(self, "エラー", f"JSON保存に失敗しました。\n{path}\n\n{ex}")
            return
        self.tts_template_edit.setPlainText(normalized)
        self.status_label.setText(f"状態：テンプレートJSON保存済み（{path.name}）")
        self._save_app_config()

    def _load_tts_template_dict(self) -> dict:
        # 空のときに既定テンプレートをウィジェットへ書き戻す**副作用は GUI 固有**なので残す。
        # 解析だけ vds と共有する（strict=True で従来と同じ RuntimeError を出す）。
        raw = self.tts_template_edit.toPlainText().strip()
        if not raw:
            raw = self._load_default_tts_template_text()
            self.tts_template_edit.setPlainText(raw)
        return vds.parse_tts_template_text(raw, strict=True)

    def _send_audio_output_command(self, payload: dict[str, object]) -> bool:
        return self._send_to_worker_via_orchestrator("audio_output", dict(payload))

    def _ensure_tts_playback_workers(self) -> None:
        """再生経路は Orchestrator 内 supervisor → audio/output_worker に集約（MMD も同じ）。

        audio_output は Bootstrap / 設定保存時の同期で既に起動している前提とし、
        ここでは **未 ready のときだけ** ensure する（テスト開始のたびにプロセスを取り直さない）。
        """
        self._ensure_dialog_orchestrator_worker()
        if self._effective_audio_output_mode() == "mmd":
            return
        if self._audio_output_worker_ready_event.is_set():
            return
        self._ensure_audio_output_worker()

    def _build_tts_inputs(
        self,
        engine: str,
        utterance: str,
        *,
        input_streaming_enabled: bool = False,
        split_settings: dict[str, object] | None = None,
    ) -> dict:
        """`vds.build_tts_inputs` に一本化（Phase 1-2）。

        ウィジェット補正は **GUI 固有の副作用**なので残す（読込直後などに他エンジンの
        既定値が残っていても、ここで直してから送る）。
        """
        self._restore_engine_voice(engine)
        self._apply_engine_defaults(engine)
        self._save_current_engine_voice(engine)
        if engine not in ("Kokoro-RealtimeTTS", "Cartesia-Realtime", "VOICEVOX-Core", "Qwen3-TTS-CPP", "LFM2.5-Audio"):
            if engine == "VOICEVOX" and self.tts_port_edit.text().strip() == "":
                self.tts_port_edit.setText("50021")
            if self._get_tts_input_port() is None:
                raise RuntimeError("TTSサーバーポートは1～65535の整数で指定してください。")
        # 上の補正を反映した状態でスナップショットを撮る。
        # `sync_engine_voice=False` が重要: この時点で声ウィジェットは「引数の engine」の
        # 値になっているため、コンボの engine で記憶を更新すると別エンジンの声を
        # 上書きして壊してしまう。
        return vds.build_tts_inputs(
            self._widget_config_snapshot(sync_engine_voice=False),
            engine=engine,
            utterance=utterance,
            input_streaming_enabled=input_streaming_enabled,
            split_settings=split_settings,
        )

    # ターミナル(stdout)へ出すのは主要コンポーネント(TTS/ASR/MMD/MaAI)の起動・停止のみに絞る。
    # （LLMモデルのアンロード完了などは除外）。詳細ログはアプリ内コンソール窓へ集約する。
    _TERMINAL_LIFECYCLE_KEYWORDS = ("起動", "停止")

    def _log_stream(self, message: str) -> None:
        msg = str(message)
        # 起動/停止/アンロードを含むメッセージだけターミナルに出す（ノイズ削減）。
        if any(kw in msg for kw in self._TERMINAL_LIFECYCLE_KEYWORDS):
            print(f"[MMDAgentController] {msg}", flush=True)
        try:
            # アプリ内コンソール窓には従来通り全ログを出す。
            self.process_output_signal.emit(f"[Controller][INFO] {msg}")
        except Exception:
            pass

    def _append_asr_result(self, text: str) -> None:
        line = text.strip()
        if not line:
            return
        self.asr_result_edit.appendPlainText(line)

    @staticmethod
    def _append_asr_line_to_editor(
        editor,
        *,
        engine: str,
        text: str,
        latency: float,
        is_final: bool,
        is_streaming: bool,
    ) -> None:
        partial_tag = f"[{engine}-partial]"
        final_tag = f"[{engine}]"
        tag = final_tag if is_final else partial_tag
        # latency = マイクのフレーム送出からテキストまで（端から端まで・壁時計）
        line = f"{tag} {text} (latency={latency:.3f}s)"
        current = editor.toPlainText()
        lines = current.splitlines() if current else []
        if is_streaming and (not is_final):
            last_match = re.match(r"^\[([^\]]+)\]", lines[-1]) if lines else None
            last_engine = last_match.group(1).strip() if last_match else ""
            if last_engine == f"{engine}-partial":
                lines[-1] = line
            else:
                lines.append(line)
            editor.setPlainText("\n".join(lines))
            editor.moveCursor(QTextCursor.MoveOperation.End)
            return
        if is_final and lines:
            last_match = re.match(r"^\[([^\]]+)\]", lines[-1])
            last_engine = last_match.group(1).strip() if last_match else ""
            if last_engine == f"{engine}-partial":
                lines[-1] = line
                editor.setPlainText("\n".join(lines))
                editor.moveCursor(QTextCursor.MoveOperation.End)
                return
        editor.appendPlainText(line)

    def _append_asr_result_event(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        from_orchestrator = bool(payload.get("from_orchestrator", False))
        engine = str(payload.get("engine", "asr")).strip() or "asr"
        text = str(payload.get("text", "")).strip()
        if not text:
            return
        latency = float(payload.get("latency_sec", 0.0) or 0.0)
        is_final = bool(payload.get("is_final", True))
        is_streaming = bool(payload.get("is_streaming", False))
        compare_slot = str(payload.get("compare_slot", "")).strip().lower()
        segment_start_ts = payload.get("segment_start_ts")
        segment_end_ts = payload.get("segment_end_ts")
        final_arrival_ts = payload.get("final_arrival_ts")
        utterance_index = int(payload.get("utterance_index", 0) or 0)
        if compare_slot == "top":
            target_edit = self.asr_compare_top_result_edit
        elif compare_slot == "bottom":
            target_edit = self.asr_compare_bottom_result_edit
        else:
            target_edit = self.asr_result_edit
        compare_enabled = bool(self.asr_compare_mode_check.isChecked())
        is_compare_top = (not compare_enabled) or (not compare_slot) or (compare_slot == "top")
        self._append_asr_line_to_editor(
            target_edit,
            engine=engine,
            text=text,
            latency=latency,
            is_final=is_final,
            is_streaming=is_streaming,
        )
        if self.asr_maai_gui_plot_check.isChecked():
            self.maai_plot_window.update_asr_segment(
                engine=engine,
                text=text,
                segment_start_ts=segment_start_ts,
                segment_end_ts=segment_end_ts,
                is_final=is_final,
                is_streaming=is_streaming,
                compare_slot=compare_slot,
                final_arrival_ts=final_arrival_ts,
            )
        if self._is_dialog_orchestrator_runtime_active() and (not from_orchestrator) and is_compare_top:
            self._send_dialog_orchestrator_command(
                {
                    "type": "asr_result",
                    "engine": engine,
                    "text": text,
                    "latency_sec": latency,
                    "is_final": is_final,
                    "is_streaming": is_streaming,
                    "segment_start_ts": segment_start_ts,
                    "segment_end_ts": segment_end_ts,
                    "final_arrival_ts": final_arrival_ts,
                    "utterance_index": utterance_index,
                }
            )

    @staticmethod
    def _maai_p_bc_scalar(value: object) -> float | None:
        """MaAI の p_bc（スカラー・配列・ネストリスト）を 1 つの確率に正規化。MaaiPlotWindow._first_float と同様に末尾を採用。"""
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, numbers.Real):
            v = float(value)
            return None if v != v else v
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            return MMDAgentController._maai_p_bc_scalar(value[-1])
        if isinstance(value, str):
            try:
                v = float(value.strip())
                return None if v != v else v
            except ValueError:
                return None
        try:
            import numpy as np

            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 0:
                return None
            v = float(arr[-1])
            return None if v != v else v
        except Exception:
            return None

    def _append_maai_result_event(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        from_orchestrator = bool(payload.get("from_orchestrator", False))
        result = payload.get("result")
        if not isinstance(result, dict):
            return
        try:
            _sv = result.get("silero_vad_score")
            if _sv is not None:
                self._voice_dialog_last_silero_vad = float(
                    _sv[0] if isinstance(_sv, (list, tuple)) else _sv
                )
        except Exception:
            pass
        # nod_para の閾値→離散化はここだけで行う（ASR 内・MaAI IPC 受信側での二重適用を避ける）。
        self._apply_maai_nod_para_prediction_aliases_with_ui(result)
        compare_slot = str(payload.get("compare_slot", "")).strip().lower()
        compare_enabled = bool(self.asr_compare_mode_check.isChecked())
        if compare_enabled and compare_slot and compare_slot != "top":
            return
        if (
            self._is_dialog_orchestrator_runtime_active()
            and (not from_orchestrator)
            and (not bool(result.get("silero_vad_plot_only")))
        ):
            self._enqueue_maai_result_for_orchestrator(result)
        if not self.asr_maai_gui_plot_check.isChecked():
            return
        plot_result = dict(result)
        # p_bc は bc（相槌）にのみ存在する。無い結果に float(None)→0 を入れていたため nod/turn 等で 0 が点々と付いていた。
        # 配列型 p_bc は float() 不能で同様に 0 になっていたため、末尾スカラーで解釈する。
        if "p_bc" in result:
            prob = self._maai_p_bc_scalar(result.get("p_bc"))
            if prob is None:
                plot_result.pop("p_bc", None)
                plot_result.pop("p_bc_threshold", None)
            else:
                prob = max(0.0, min(1.0, prob))
                plot_result["p_bc"] = prob
                threshold = self._get_maai_backchannel_prob_threshold()
                plot_result["p_bc_threshold"] = max(0.0, min(1.0, threshold))
        else:
            plot_result.pop("p_bc", None)
            plot_result.pop("p_bc_threshold", None)
        raw_su = plot_result.pop("nod_swing_up_prob", None)
        if raw_su is None and "nod_swing_up" in plot_result:
            raw_su = plot_result.get("nod_swing_up")
        if "nod_swing_up" in result or raw_su is not None:
            prob = self._maai_p_bc_scalar(raw_su)
            if prob is None:
                plot_result.pop("nod_swing_up", None)
                plot_result.pop("nod_swing_up_threshold", None)
            else:
                prob = max(0.0, min(1.0, float(prob)))
                plot_result["nod_swing_up"] = prob
                plot_result["nod_swing_up_threshold"] = max(
                    0.0, min(1.0, self._get_maai_nod_swing_threshold())
                )
        else:
            plot_result.pop("nod_swing_up", None)
            plot_result.pop("nod_swing_up_threshold", None)
        # 到着時に即 ingest して pending キュー待ちを無くす。描画は壁時計タイマーで行う。
        win = getattr(self, "maai_plot_window", None)
        if win is not None:
            win.ingest_result(plot_result)

    def _drain_maai_plot_pending_ingest(self) -> None:
        """溜まった MaAI 結果をプロットのバッファへ反映する（描画はしない）。

        描画は壁時計タイマー(_on_maai_plot_wall_clock_tick → tick_wall_clock_plot_scroll)で
        一括して行う。各点は自身の source_ts_end で系列に入るため、ここで描画原点は決めない。
        """
        pending = self._maai_plot_pending
        if not pending:
            return
        self._maai_plot_pending = []
        win = getattr(self, "maai_plot_window", None)
        if win is None:
            return
        for r in pending:
            win.ingest_result(r)

    @staticmethod
    def _maai_test_log_format_scalar(value: object) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, numbers.Integral):
            return str(int(value))
        if isinstance(value, numbers.Real):
            try:
                return f"{float(value):.6g}"
            except Exception:
                return str(value)
        if isinstance(value, (list, tuple)):
            inner = ", ".join(MMDAgentController._maai_test_log_format_scalar(x) for x in value)
            return f"[{inner}]"
        return str(value)

    def _append_line_maai_test_log(self, line: str) -> None:
        edit = getattr(self, "maai_test_log_edit", None)
        if edit is None:
            return
        current = edit.toPlainText().strip()
        merged = f"{current}\n{line}" if current else line
        edit.setPlainText(merged)
        edit.moveCursor(QTextCursor.MoveOperation.End)

    def _maybe_append_maai_throttled_test_log(self, event_type: str, result: dict[str, object]) -> None:
        """MaAI テストログ: bc / nod_para をそれぞれ最大 1 秒に 1 行。"""
        now = time.time()
        mode = str(result.get("maai_mode", "")).strip().lower()
        if event_type == "maai_bc_result" or mode == "bc":
            if (now - self._maai_test_log_last_emit_bc) < 1.0:
                return
            self._maai_test_log_last_emit_bc = now
            line = f"[bc] p_bc={self._maai_test_log_format_scalar(result.get('p_bc'))}"
            self._append_line_maai_test_log(line)
            return
        if event_type == "maai_nod_result" or mode == "nod_para":
            if (now - self._maai_test_log_last_emit_nod) < 1.0:
                return
            self._maai_test_log_last_emit_nod = now
            line = (
                f"[nod_para] p_nod={self._maai_test_log_format_scalar(result.get('p_nod'))} "
                f"count={self._maai_test_log_format_scalar(result.get('nod_count'))} "
                f"range={self._maai_test_log_format_scalar(result.get('nod_range'))} "
                f"speed={self._maai_test_log_format_scalar(result.get('nod_speed'))} "
                f"swing_up={self._maai_test_log_format_scalar(result.get('nod_swing_up'))}"
            )
            self._append_line_maai_test_log(line)

    def _append_maai_test_event(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        if not hasattr(self, "maai_test_log_edit"):
            return
        event_type = str(payload.get("type", "")).strip().lower()
        if not event_type:
            return
        if event_type in {"maai_nod_result", "maai_bc_result", "maai_result"}:
            result_obj = payload.get("result")
            if isinstance(result_obj, dict):
                self._maybe_append_maai_throttled_test_log(event_type, dict(result_obj))
            return
        if event_type in {"log", "error", "fatal"}:
            line = f"[{event_type}] {str(payload.get('message', ''))}"
        else:
            line = str(payload)
        self._append_line_maai_test_log(line)

    def _stop_maai_test_surface_jsonl(self) -> None:
        self._maai_test_bc_trigger_log_buffer = None
        fp = self._maai_test_surface_jsonl_fp
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass
            self._maai_test_surface_jsonl_fp = None
        self._maai_test_surface_jsonl_path = None

    def _write_maai_test_surface_jsonl_line(self, record: dict[str, object]) -> None:
        fp = self._maai_test_surface_jsonl_fp
        if fp is None:
            return
        try:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()
        except Exception:
            pass

    def _start_maai_test_surface_jsonl(self) -> None:
        self._stop_maai_test_surface_jsonl()
        try:
            log_dir = _paths.logs_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            fname = f"maai_test_surface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            path = log_dir / fname
            self._maai_test_surface_jsonl_fp = open(path, "a", encoding="utf-8")
            self._maai_test_surface_jsonl_path = path
        except Exception as ex:
            self._log_stream(f"[MaAIテスト] セッション表出ログを開けませんでした: {ex}")
            self._maai_test_surface_jsonl_fp = None
            self._maai_test_surface_jsonl_path = None
            return
        ain = self._build_audio_input_worker_config()
        in_mode = str(ain.get("input_mode", "") or "")
        wav_path = str(ain.get("wav_path", "") or "") if in_mode == "wav" else ""
        wav_start_sec = str(ain.get("wav_start_sec", "0") or "0") if in_mode == "wav" else ""
        wav_end_sec = str(ain.get("wav_end_sec", "") or "") if in_mode == "wav" else ""
        meta: dict[str, object] = {
            "kind": "session_meta",
            "started_at_wall_iso": datetime.now().isoformat(timespec="seconds"),
            "audio_input_mode": in_mode,
            "wav_path": wav_path,
            "wav_start_sec": wav_start_sec,
            "wav_end_sec": wav_end_sec,
            "maai_worker_config": dict(self._build_maai_worker_config()),
            "nod_para_scale_ref": float(self._get_nod_para_float("nod_para_scale_edit", 3.0)),
            "nod_para_speed_scale_ref": float(self._get_nod_para_float("nod_para_speed_scale_edit", 1.2)),
            "nod_para_eye_scale_ref": float(self._get_nod_para_float("nod_para_eye_scale_edit", 1.0)),
            "note": (
                "表出行は Dialog Orchestrator 経由のイベントのみ。"
                " オーケストレータ未接続時はメタのみのことがあります。"
                " audio_timeline_sec は 16k サンプル基準の概算です。"
            ),
        }
        self._write_maai_test_surface_jsonl_line(meta)
        p = self._maai_test_surface_jsonl_path
        if p is not None:
            self.process_output_signal.emit(f"[MaAIテスト] surface_jsonl={p}")

    def _maai_test_log_surface_backchannel(self, payload: dict, assistant_text: str) -> None:
        if self._maai_test_surface_jsonl_fp is None:
            return
        buf = self._maai_test_bc_trigger_log_buffer
        line: dict[str, object] = {
            "kind": "surface",
            "event": "backchannel",
            "assistant_text": assistant_text,
            "wall_time": time.time(),
        }
        if isinstance(buf, dict):
            for k in (
                "used_start_ts",
                "used_end_ts",
                "trigger_ts",
                "buffer_frames",
                "audio_timeline_sec",
                "source_sample_end",
            ):
                if k in buf:
                    line[k] = buf[k]
        for k in ("audio_timeline_sec", "source_sample_end"):
            if k in payload and k not in line:
                line[k] = payload[k]
        self._write_maai_test_surface_jsonl_line(line)
        self._maai_test_bc_trigger_log_buffer = None

    def _maai_test_log_surface_nod(self, payload: dict) -> None:
        if self._maai_test_surface_jsonl_fp is None:
            return
        skip = frozenset({"type", "scale", "speed_scale", "eye_scale"})
        line: dict[str, object] = {
            "kind": "surface",
            "event": "nod_motion",
            "wall_time": time.time(),
        }
        for k, v in payload.items():
            if k in skip:
                continue
            line[k] = v
        self._write_maai_test_surface_jsonl_line(line)

    def _update_nod_motion_chart(self, payload: dict) -> None:
        w = getattr(self, "maai_nod_motion_chart", None)
        if w is None:
            return
        motion = payload.get("motion")
        time_axis = payload.get("time_axis")
        if not isinstance(motion, list) or not isinstance(time_axis, list):
            return
        w.set_chart_data(
            motion=motion,
            time_axis=time_axis,
            interpolation=str(payload.get("interpolation", "")),
            velocity=float(payload.get("velocity", 0.0) or 0.0),
            nod_count=int(payload.get("nod_count", 1) or 1),
            p_nod=float(payload.get("p_nod", 0.0) or 0.0),
            range_rad=float(payload.get("range_rad", 0.0) or 0.0),
            swing_up_prob=float(payload.get("swing_up_prob", 0.0) or 0.0),
            use_pre_rise=bool(payload.get("use_pre_rise", False)),
            buffer_frames=int(payload.get("buffer_frames", 0) or 0),
            raw_range=float(payload.get("raw_range", 0.0) or 0.0),
            raw_speed=float(payload.get("raw_speed", 0.0) or 0.0),
        )

    def _update_maai_plot_nod_trigger(self, payload: dict) -> None:
        win = getattr(self, "maai_plot_window", None)
        if win is None:
            return
        try:
            used_start_ts = float(payload.get("used_start_ts"))
            used_end_ts = float(payload.get("used_end_ts"))
            trigger_ts = float(payload.get("trigger_ts"))
        except (TypeError, ValueError):
            return
        win.add_nod_trigger_region(
            used_start_ts=used_start_ts,
            used_end_ts=used_end_ts,
            trigger_ts=trigger_ts,
            buffer_frames=int(payload.get("buffer_frames", 0) or 0),
            count=int(payload.get("count", 0) or 0),
        )
        if self.asr_maai_gui_plot_check.isChecked():
            win.flush_maai_draw(now_ts=time.time())

    def _update_maai_plot_bc_trigger(self, payload: dict) -> None:
        win = getattr(self, "maai_plot_window", None)
        if win is None:
            return
        try:
            used_start_ts = float(payload.get("used_start_ts"))
            used_end_ts = float(payload.get("used_end_ts"))
            trigger_ts = float(payload.get("trigger_ts"))
        except (TypeError, ValueError):
            return
        win.add_bc_trigger_region(
            used_start_ts=used_start_ts,
            used_end_ts=used_end_ts,
            trigger_ts=trigger_ts,
            buffer_frames=int(payload.get("buffer_frames", 0) or 0),
        )
        if self.asr_maai_gui_plot_check.isChecked():
            win.flush_maai_draw(now_ts=time.time())

    def _update_maai_test_button_states(self) -> None:
        if not hasattr(self, "maai_test_launch_btn") or not hasattr(self, "maai_test_stop_btn"):
            return
        running = self._maai_ready
        self.maai_test_launch_btn.setEnabled((not self._maai_starting) and (not running))
        self.maai_test_stop_btn.setEnabled((not self._maai_starting) and running)

    def launch_maai_test(self) -> None:
        self._open_console_window()
        self._maai_starting = True
        self._set_module_panel_state("maai", "starting")
        self._update_maai_test_button_states()
        maai_was_ready = self._maai_ready
        try:
            self._ensure_maai_worker(check_backend_conflict=False)
            # 音声出力: Orchestrator 先行起動の audio_output をそのまま使う。
            # MMD 起動時だけ _ensure_tts_playback_workers を呼ぶと、出力モードが mmd 以外のとき
            # に毎回 ensure が走りプロセスを取り直し得るため、ここでは呼ばない。
            # MaAI ワーカー起動の start_module が既に Dialog Orchestrator を確保する。
            self._ensure_audio_input_worker()
        except Exception as ex:
            if self._maai_ready and not maai_was_ready:
                self._teardown_maai_ipc()
            self._maai_starting = False
            self._set_module_panel_state("maai", "error", str(ex))
            QMessageBox.warning(self, "エラー", f"MaAIの起動に失敗しました：\n{ex}")
            self.status_label.setText("状態：MaAI起動失敗")
            self._log_stream(f"MaAI 起動失敗: {ex}")
            self._update_maai_test_button_states()
            return
        self._maai_starting = False
        self._set_module_panel_state("maai", "running")
        self.status_label.setText("状態：MaAI起動")
        self._log_stream("MaAI IPCワーカー起動")
        self._push_dialog_orchestrator_settings()
        self._open_maai_plot_window()
        if hasattr(self, "maai_test_status_label"):
            self.maai_test_status_label.setText("状態：MaAI起動")
        self._update_maai_test_button_states()
        self._start_maai_test_surface_jsonl()
        self._obs_acquired_maai = False
        if self._should_obs_link_maai():
            if self._obs_remote_try_start():
                self._obs_acquired_maai = True
        self._maai_mmd_av_preamble_wait_for_mmd = not self._avatar_output_available()
        self._try_send_maai_mmd_av_preamble_once()
        self._try_reset_dialog_orchestrator_nod_para_timing()

    def _try_send_maai_mmd_av_preamble_once(self) -> None:
        if self._maai_mmd_av_preamble_sent:
            return
        if not self._maai_ready:
            return
        if not (
            self._is_maai_backchannel_enabled() or self._is_nod_para_enabled()
        ):
            return
        if self._effective_audio_output_mode() != "mmd":
            return
        if (
            self._maai_mmd_av_preamble_wait_for_mmd
            and not self._avatar_output_available()
        ):
            return
        if not self._is_dialog_orchestrator_runtime_active():
            return
        self._send_dialog_orchestrator_command_sync({"type": "mmd_av_preamble_once"})

    def stop_maai_test(self) -> None:
        if not self._maai_ready:
            self._stop_maai_test_surface_jsonl()
            self.status_label.setText("状態：MaAIは未起動")
            self._log_stream("MaAI は未起動")
            if hasattr(self, "maai_test_status_label"):
                self.maai_test_status_label.setText("状態：MaAIは未起動")
            self._update_maai_test_button_states()
            return
        self._teardown_maai_ipc()
        self._try_reset_dialog_orchestrator_nod_para_timing()
        self._maai_starting = False
        self._maai_mmd_av_preamble_sent = False
        self._maai_mmd_av_preamble_fail_count = 0
        self._maai_mmd_av_preamble_wait_for_mmd = False
        self._maai_test_log_last_emit_bc = 0.0
        self._maai_test_log_last_emit_nod = 0.0
        self.status_label.setText("状態：MaAI停止")
        self._log_stream("MaAI IPCワーカー停止")
        if hasattr(self, "maai_test_status_label"):
            self.maai_test_status_label.setText("状態：MaAI停止")
        self._update_maai_test_button_states()
        self._stop_maai_test_surface_jsonl()

    def _audio_input_needed_by_active_modules(self) -> bool:
        """未起動の Audio Input を、設定保存などをきっかけに起こしてよいか。"""
        if self._asr_ready or self._asr_starting:
            return True
        if self._maai_ready or self._maai_starting:
            return True
        if self._voice_dialog_active or self._voice_dialog_starting:
            return True
        return False

    def _enqueue_maai_result_for_orchestrator(self, result: dict[str, object]) -> None:
        mode = str(result.get("maai_mode", "")).strip().lower()
        if mode == "nod_para":
            event_type = "maai_nod_result"
        elif mode == "bc":
            event_type = "maai_bc_result"
        else:
            event_type = "maai_result"
        merged = dict(result)
        if "silero_vad_score" not in merged and self._voice_dialog_last_silero_vad is not None:
            merged["silero_vad_score"] = float(self._voice_dialog_last_silero_vad)
        payload = {"type": event_type, "result": merged}
        try:
            self._maai_result_forward_queue.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._maai_result_forward_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._maai_result_forward_queue.put_nowait(payload)
            except queue.Full:
                self.process_output_signal.emit(
                    "[MaAI][WARN] Dialog Orchestrator 向け MaAI 結果キューが満杯のため結果を破棄しました。"
                )

    def _maai_result_forward_loop(self) -> None:
        while not self._controller_stop_event.is_set():
            try:
                payload = self._maai_result_forward_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                break
            if not isinstance(payload, dict):
                continue
            if not self._is_dialog_orchestrator_runtime_active():
                continue
            self._send_dialog_orchestrator_command(payload)

    def _append_raw_audio_event(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        from_orchestrator = bool(payload.get("from_orchestrator", False))
        if not self.asr_maai_gui_plot_check.isChecked():
            return
        audio = payload.get("audio")
        if not isinstance(audio, list):
            return
        channel = str(payload.get("channel", "")).strip().lower()
        if channel == "system":
            # system音声はローカル送信元ですでに描画済み。
            # orchestrator経由の遅延エコーを重ねると古い時刻データが混ざり、
            # x2 が途中で消える/巻き戻る見え方になるため無視する。
            if from_orchestrator:
                return
            self.maai_plot_window.update_raw_audio_frame(
                audio=audio,
                ts_start=payload.get("ts_start"),
                ts_end=payload.get("ts_end"),
                compare_slot="",
                system_audio=True,
            )
            return
        compare_slot = str(payload.get("compare_slot", "")).strip().lower()
        compare_enabled = bool(self.asr_compare_mode_check.isChecked())
        if compare_enabled and compare_slot and compare_slot != "top":
            return
        self.maai_plot_window.update_raw_audio_frame(
            audio=audio,
            ts_start=payload.get("ts_start"),
            ts_end=payload.get("ts_end"),
            compare_slot=compare_slot,
        )

    def _on_asr_segment_arrival_align_changed(self, _state: object = None) -> None:
        del _state
        enabled = bool(
            hasattr(self, "asr_segment_arrival_align_check")
            and self.asr_segment_arrival_align_check.isChecked()
        )
        if hasattr(self, "maai_plot_window") and self.maai_plot_window is not None:
            self.maai_plot_window.align_segment_right_edge_to_arrival = enabled
            self.maai_plot_window._redraw_asr_segments_axis(draw=True)

    def clear_asr_results(self) -> None:
        self.asr_result_edit.clear()
        if hasattr(self, "asr_compare_top_result_edit"):
            self.asr_compare_top_result_edit.clear()
        if hasattr(self, "asr_compare_bottom_result_edit"):
            self.asr_compare_bottom_result_edit.clear()

    def _set_asr_controls_state(self) -> None:
        if not hasattr(self, "asr_launch_btn"):
            return
        running = self._asr_ready
        self.asr_launch_btn.setEnabled((not self._asr_starting) and (not running))
        self.asr_stop_btn.setEnabled((not self._asr_starting) and running)
        self.asr_clear_btn.setEnabled(not self._asr_starting)

    @staticmethod
    def _pump_ui_events() -> None:
        app = QApplication.instance()
        if app is None:
            return
        try:
            app.processEvents()
        except Exception:
            pass

    def _wait_flag_with_ui_pump(self, done_event: threading.Event, timeout_sec: float) -> bool:
        deadline = time.time() + max(0.1, float(timeout_sec))
        while time.time() < deadline:
            if done_event.is_set():
                return True
            self._pump_ui_events()
            time.sleep(0.01)
        return done_event.is_set()

    def _threading_event_wait_responsive(self, ev: threading.Event, timeout_sec: float) -> bool:
        """メインスレッドでは UI をポンプしながら待つ。バックグラウンドでは通常の Event.wait。"""
        app = QApplication.instance()
        if app is not None and QThread.currentThread() == app.thread():
            return self._wait_flag_with_ui_pump(ev, max(0.1, float(timeout_sec)))
        return ev.wait(timeout=max(0.0, float(timeout_sec)))

    def _asr_log(self, message: str) -> None:
        self.process_output_signal.emit(f"[ASR] {message}")

    def _teardown_asr_ipc(self) -> None:
        self._stop_module_via_orchestrator("asr")
        self._stop_module_via_orchestrator("asr_compare")
        self._asr_ready = False

    def _teardown_audio_input_ipc(self) -> None:
        self._stop_module_via_orchestrator("audio_input")
        self._audio_input_worker_ready_event.clear()
        self._audio_input_processing_event.clear()
        self._stop_remote_mic_capture()
        self._audio_input_wav_paused = True
        self._sync_audio_input_wav_transport_button()

    def _teardown_maai_ipc(self) -> None:
        self._stop_module_via_orchestrator("maai")
        self._maai_ready = False
        self._maai_nod_threshold_defaults = None
        self._update_maai_nod_threshold_default_labels()
        self._update_maai_test_button_states()
        self._obs_remote_release_maai()

    def _teardown_tts_ipc(self) -> None:
        self._tts_worker_ready_event.clear()
        self._stop_module_via_orchestrator("tts")
        self._tts_worker_engine = None
        self._tts_worker_config_digest = None

    def _teardown_llm_ipc_unlocked(self) -> None:
        self._llm_worker_ready_event.clear()
        self._stop_module_via_orchestrator("llm")

    def _teardown_llm_ipc(self) -> None:
        with self._llm_ipc_lifecycle_lock:
            self._teardown_llm_ipc_unlocked()

    def _teardown_dialog_orchestrator_ipc(self) -> None:
        self._dialog_orchestrator_ready_event.clear()
        self._dialog_orchestrator_settings_digest = None
        self._stop_remote_mic_capture()
        if self._dialog_orchestrator_ipc_client is not None:
            self._dialog_orchestrator_ipc_client.close()
            self._dialog_orchestrator_ipc_client = None
        if self._remote_backend_enabled:
            self._clear_remote_orchestrator_state_cache()
            return
        # IPC を閉じた時点で orchestrator は自己終了に入る（並列 stop_all で素早く終わる）。
        # graceful 待ちは短くして Ctrl+C 終了の体感を速くする。
        self.dialog_orchestrator_process_service.stop(timeout_sec=3, on_wait=self._pump_ui_events)

    def _on_dialog_orchestrator_ipc_event(self, payload: dict) -> None:
        p = dict(payload or {})
        # audio_output ワーカーが fan-in する remote_status は既に _remote_bridge_origin が付く。
        # ここで上書きすると AudioOutput 接続が Orchestrator 扱いになり、UI 状態・preamble 経路が壊れる。
        if str(p.get("type", "") or "").strip().lower() == "remote_status":
            if not str(p.get("_remote_bridge_origin", "") or "").strip():
                p["_remote_bridge_origin"] = "dialog_orchestrator"
        self.dialog_orchestrator_event_signal.emit(p)

    def _on_dialog_orchestrator_ipc_disconnected(self) -> None:
        self._dialog_orchestrator_ready_event.clear()
        self._stop_remote_mic_capture()
        self._clear_remote_orchestrator_state_cache()
        if self._controller_closing or self._dialog_server_user_stopped or self._dialog_server_busy:
            return  # ■ で止めたとき・▶ の処理中は、ここから起動し直さない
        self.process_output_signal.emit("[DialogOrchestrator][WARN] IPC切断を検知しました。再接続を試行します。")
        if self._remote_backend_enabled:
            self._schedule_remote_orchestrator_retry()
        else:
            QTimer.singleShot(300, self._ensure_dialog_orchestrator_worker_startup)

    def _send_dialog_orchestrator_command(self, payload: dict[str, object]) -> bool:
        # 終了処理中は sender スレッドが既に停止している。キューに積んでも消費者がおらず、
        # closeEvent 末尾の stop/teardown コマンドが黙って捨てられるため直接同期送信する(#6)。
        if self._controller_closing:
            return self._send_dialog_orchestrator_command_sync(payload)
        # 高頻度の音声中継は専用キューへ。同一キューだと音声バーストで溢れたとき
        # 最古の制御コマンドが捨てられてしまう。
        is_audio = str(payload.get("type", "")) == "send_to_audio_output"
        q = self._dialog_orchestrator_audio_queue if is_audio else self._dialog_orchestrator_command_queue
        try:
            q.put_nowait(dict(payload))
            return True
        except queue.Full:
            # 溢れたら最古を1つ捨てて入れ直す（音声は欠落許容。制御は別キューのため
            # 音声バーストでは溢れず、ここに到達するのは稀）。
            try:
                _ = q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(dict(payload))
                return True
            except queue.Full:
                return False

    def _send_dialog_orchestrator_command_sync(self, payload: dict[str, object]) -> bool:
        if self._dialog_orchestrator_ipc_client is None:
            return False
        try:
            self._dialog_orchestrator_ipc_client.send(dict(payload))
            return True
        except Exception:
            return False

    def _send_dialog_orchestrator_tts_done(self, *, playback_aborted: bool = False) -> bool:
        self._voice_dialog_tts_segments_snapshot = []
        self._voice_dialog_tts_snap_prefix = ""
        return self._send_dialog_orchestrator_command(
            {"type": "tts_done", "playback_aborted": bool(playback_aborted)}
        )

    def _dialog_orchestrator_command_sender_loop(self) -> None:
        _control_streak = 0
        while not self._controller_stop_event.is_set():
            # 制御コマンドを優先的に捌き、無ければ音声を捌く。ただし MaAI 結果転送等で制御が
            # 連続到着すると tts_play_chunk(音声)が枯渇するため、連続8件ごとに音声を1件割り込ませる(#14)。
            is_audio = False
            got = False
            payload = None
            if _control_streak >= 8:
                _control_streak = 0
                try:
                    payload = self._dialog_orchestrator_audio_queue.get_nowait()
                    is_audio = True
                    got = True
                except queue.Empty:
                    pass
            if not got:
                try:
                    payload = self._dialog_orchestrator_command_queue.get_nowait()
                    got = True
                    _control_streak += 1
                except queue.Empty:
                    _control_streak = 0
                    try:
                        payload = self._dialog_orchestrator_audio_queue.get(timeout=0.1)
                        is_audio = True
                        got = True
                    except queue.Empty:
                        got = False
            if not got:
                continue
            if payload is None:
                break
            if not isinstance(payload, dict):
                continue
            if not self._send_dialog_orchestrator_command_sync(payload):
                # 未接続中は再試行。制御は末尾へ戻し、音声は欠落許容で捨てる。
                if not is_audio:
                    try:
                        self._dialog_orchestrator_command_queue.put_nowait(dict(payload))
                    except queue.Full:
                        pass
                time.sleep(0.01)

    def _shutdown_controller_threads(self) -> None:
        self._controller_stop_event.set()
        try:
            self._dialog_orchestrator_command_queue.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            self._maai_result_forward_queue.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            pass
        for t in (self._dialog_orchestrator_sender_thread, self._maai_forward_thread):
            if t is None:
                continue
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

    def _is_dialog_orchestrator_runtime_active(self) -> bool:
        return bool(
            self._dialog_orchestrator_ipc_client is not None
            and self._dialog_orchestrator_ready_event.is_set()
        )

    # -- Orchestrator-mediated module lifecycle --------------------------------

    def _start_module_via_orchestrator(
        self,
        module: str,
        config: dict,
        *,
        source: str = "",
        engine: str = "",
        timeout_sec: float = 120.0,
    ) -> None:
        """Orchestrator にモジュール起動を依頼し、module_started を待つ。"""
        self._ensure_dialog_orchestrator_worker()
        ev = self._module_ready_events.get(module)
        if ev is None:
            raise RuntimeError(f"Unknown module: {module}")
        ev.clear()
        self._module_start_errors.pop(module, None)
        payload: dict[str, object] = {
            "type": "start_module",
            "module": module,
            "config": config,
            "source": source,
        }
        if engine:
            payload["engine"] = engine
        if not self._send_dialog_orchestrator_command(payload):
            raise RuntimeError(f"Orchestrator への {module} 起動コマンド送信に失敗しました。")
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if ev.is_set():
                err = self._module_start_errors.get(module)
                if err:
                    raise RuntimeError(err)
                return
            self._pump_ui_events()
            time.sleep(0.02)
        raise RuntimeError(f"{module} 起動がタイムアウトしました。")

    def _stop_module_via_orchestrator(self, module: str) -> bool:
        return self._send_dialog_orchestrator_command({"type": "stop_module", "module": module})

    def _send_to_worker_via_orchestrator(self, worker: str, payload: dict) -> bool:
        return self._send_dialog_orchestrator_command({
            "type": f"send_to_{worker}",
            "payload": payload,
        })

    def _send_tts_bridge_command(self, inner: dict) -> bool:
        self._ensure_dialog_orchestrator_worker()
        return self._send_dialog_orchestrator_command(
            {"type": "send_to_tts_bridge", "payload": inner}
        )

    def _send_llm_admin_command(self, inner: dict) -> bool:
        self._ensure_dialog_orchestrator_worker()
        return self._send_dialog_orchestrator_command(
            {"type": "send_to_llm_admin", "payload": inner}
        )

    def _orchestrator_rpc(
        self,
        *,
        op: str,
        rid_prefix: str,
        lock: threading.Lock,
        pending: dict,
        send_fn: Any,
        wait_timeout_sec: float,
        kwargs: dict,
    ) -> dict | None:
        """オーケストレータ経由の request/response RPC 共通処理。

        rid を発番して pending に登録 → コマンド送信 → 応答待ち → finally で必ず pending を掃除する。
        LLM admin / TTS bridge の差分（rid 接頭辞・ロック・pending・送信関数・待ち時間）のみ引数化する。
        """
        rid = f"{rid_prefix}{int(time.time() * 1000)}-{threading.get_ident()}"
        ev = threading.Event()
        out: dict = {}
        with lock:
            pending[rid] = (ev, out)
        try:
            body = {"op": op, "request_id": rid, **kwargs}
            if not send_fn(body):
                return None
            if not self._threading_event_wait_responsive(ev, wait_timeout_sec):
                return None
            return dict(out)
        finally:
            with lock:
                pending.pop(rid, None)

    def _llm_admin_rpc(self, op: str, timeout_sec: float = 120.0, **kwargs: Any) -> dict | None:
        self._ensure_dialog_orchestrator_worker()
        return self._orchestrator_rpc(
            op=op,
            rid_prefix="llm-adm-",
            lock=self._llm_admin_rpc_lock,
            pending=self._llm_admin_rpc_pending,
            send_fn=self._send_llm_admin_command,
            wait_timeout_sec=max(5.0, float(timeout_sec)),
            kwargs=kwargs,
        )

    def _on_llm_admin_ipc_event(self, payload: dict) -> None:
        if str(payload.get("type", "")).strip().lower() != "llm_admin_result":
            return
        rid = str(payload.get("request_id", "")).strip()
        if not rid:
            return
        with self._llm_admin_rpc_lock:
            tup = self._llm_admin_rpc_pending.get(rid)
        if tup is None:
            return
        ev, out = tup
        out.clear()
        for k, v in payload.items():
            if k != "type":
                out[k] = v
        ev.set()

    def _sync_tts_exe_paths_to_orchestrator(self) -> None:
        voicevox_path = self.voicevox_exe_edit.text().strip()
        aivisspeech_path = self.aivisspeech_exe_edit.text().strip()
        cfg: dict[str, object] = {
            "op": "configure",
            "voicevox_exe_path": voicevox_path,
            "aivisspeech_exe_path": aivisspeech_path,
        }
        self._send_tts_bridge_command(cfg)

    def _tts_bridge_rpc(self, op: str, timeout_sec: float = 8.0, **kwargs) -> dict | None:
        self._ensure_dialog_orchestrator_worker()
        self._sync_tts_exe_paths_to_orchestrator()
        return self._orchestrator_rpc(
            op=op,
            rid_prefix="rpc-",
            lock=self._tts_bridge_rpc_lock,
            pending=self._tts_bridge_rpc_pending,
            send_fn=self._send_tts_bridge_command,
            wait_timeout_sec=float(timeout_sec),
            kwargs=kwargs,
        )

    def _tts_bridge_engine_running(self, engine: str) -> bool:
        st = self._tts_bridge_rpc("status", timeout_sec=3.0)
        if not st:
            return False
        if engine == "VOICEVOX":
            return bool(st.get("voicevox_running"))
        return False

    def _connect_remote_dialog_orchestrator(self) -> None:
        endpoint = self._remote_orchestrator_endpoint
        if endpoint is None:
            raise RuntimeError("remote orchestrator endpoint is not configured")
        host, port = endpoint
        client = JsonLineDuplexClient(host, port)
        try:
            client.connect(attempts=1, sleep_sec=0.05)
        except Exception as ex:
            client.close()
            raise RuntimeError(f"{host}:{port} への接続に失敗しました: {ex}") from ex
        self._dialog_orchestrator_ipc_client = client
        client.start_reader(
            on_event=self._on_dialog_orchestrator_ipc_event,
            on_disconnected=lambda: self.dialog_orchestrator_ipc_disconnected_signal.emit(),
        )
        ready_deadline = time.time() + 5.0
        while time.time() < ready_deadline:
            if self._dialog_orchestrator_ready_event.is_set():
                return
            # Docker Desktop のポート転送は、コンテナの中がまだ待ち受けていなくても接続を受け付け、
            # すぐに切る。切られたら 5 秒待たずにやり直す（起動直後の接続が速くなる）。
            if not client.reader_alive():
                self._teardown_dialog_orchestrator_ipc()
                raise RuntimeError("接続がすぐに切れました（対話サーバーの準備中）")
            self._pump_ui_events()
            time.sleep(0.02)
        self._teardown_dialog_orchestrator_ipc()
        raise RuntimeError("接続後の ready 待機がタイムアウトしました")

    def _ensure_dialog_orchestrator_worker(self) -> None:
        if self._dialog_server_user_stopped:
            from .module_status_mixin import DIALOG_SERVER_STOPPED_MESSAGE

            raise RuntimeError(DIALOG_SERVER_STOPPED_MESSAGE)
        if (
            self._remote_backend_enabled
            and self._dialog_orchestrator_ipc_client is not None
            and self._dialog_orchestrator_ready_event.is_set()
        ):
            return
        if (
            not self._remote_backend_enabled
            and self.dialog_orchestrator_process_service.is_running()
            and self._dialog_orchestrator_ipc_client is not None
            and self._dialog_orchestrator_ready_event.is_set()
        ):
            return
        self._teardown_dialog_orchestrator_ipc()

        if self._remote_backend_enabled:
            self._connect_remote_dialog_orchestrator()
            return

        proc = self.dialog_orchestrator_process_service.start(
            python_exe=Path(sys.executable),
            worker_script=self._dialog_orchestrator_worker_script,
            on_wait=self._pump_ui_events,
        )
        # stdout は後で _attach_process_console_output が専有読取する。
        # ui_port 行を取るまで付けると別スレッドと readline が競合し固まる。
        # ui_port は別スレッドで読み取り、GUI スレッドはイベントポンプしながら待つ。
        # proc.stdout.readline を GUI スレッドで直接呼ぶと、ワーカーが無言のとき 10秒
        # デッドラインを越えて readline 内で UI が固まる。検出後は同じスレッドが stdout を
        # コンソール出力として読み続ける（_attach と二重に readline しない）。
        ui_port_holder = {"port": 0}
        ui_port_event = threading.Event()

        def _stdout_reader() -> None:
            try:
                stream = getattr(proc, "stdout", None)
                if stream is None:
                    return
                for raw in iter(stream.readline, ""):
                    if not ui_port_event.is_set():
                        line = raw.strip()
                        if line:
                            try:
                                info = json.loads(line)
                                if isinstance(info, dict) and "ui_port" in info:
                                    ui_port_holder["port"] = int(info["ui_port"])
                                    ui_port_event.set()
                                    continue
                            except (json.JSONDecodeError, ValueError, TypeError):
                                pass
                            self.process_output_signal.emit(f"[DialogOrchestrator][STDOUT] {line}")
                        continue
                    ln = raw.rstrip("\r\n")
                    if ln:
                        if _WORKER_PREFIX_RE.match(ln):
                            self.process_output_signal.emit(ln)
                        else:
                            self.process_output_signal.emit(f"[DialogOrchestrator][STDOUT] {ln}")
            except Exception:
                pass
            finally:
                ui_port_event.set()

        def _stderr_pump() -> None:
            stream = getattr(proc, "stderr", None)
            if stream is None:
                return
            try:
                for raw in iter(stream.readline, ""):
                    ln = raw.rstrip("\r\n")
                    if ln:
                        if _WORKER_PREFIX_RE.match(ln):
                            self.process_output_signal.emit(ln)
                        else:
                            self.process_output_signal.emit(f"[DialogOrchestrator][STDERR] {ln}")
            except Exception:
                pass

        threading.Thread(target=_stdout_reader, daemon=True, name="orch-stdout-reader").start()
        threading.Thread(target=_stderr_pump, daemon=True, name="orch-stderr-pump").start()

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if ui_port_event.wait(timeout=0.05):
                break
            self._pump_ui_events()
        ui_port = int(ui_port_holder["port"])
        if ui_port <= 0:
            self._teardown_dialog_orchestrator_ipc()
            raise RuntimeError("対話オーケストレーターのUIポート取得に失敗しました。")
        client = JsonLineDuplexClient("127.0.0.1", ui_port)
        try:
            client.connect(attempts=200, sleep_sec=0.05)
        except Exception as ex:
            self._teardown_dialog_orchestrator_ipc()
            raise RuntimeError(f"対話オーケストレーターへの接続に失敗しました：{ex}") from ex
        self._dialog_orchestrator_ipc_client = client
        client.start_reader(
            on_event=self._on_dialog_orchestrator_ipc_event,
            on_disconnected=lambda: self.dialog_orchestrator_ipc_disconnected_signal.emit(),
        )
        # stdout/stderr は上の専用スレッドが読むため _attach_process_console_output は使わない
        # （同一 stdout を二重に readline すると競合して固まるため）。

        ready_deadline = time.time() + 10.0
        while time.time() < ready_deadline:
            if self._dialog_orchestrator_ready_event.is_set():
                return
            self._pump_ui_events()
            time.sleep(0.02)
        self._teardown_dialog_orchestrator_ipc()
        raise RuntimeError("対話オーケストレーターの準備完了待機がタイムアウトしました。")

    @staticmethod
    def _log_snip(text: str, limit: int = 30) -> str:
        """ログ用に本文を1行・短縮する。"""
        s = " ".join(str(text or "").split())
        return s if len(s) <= limit else s[:limit] + "…"

    # -- モジュール状態パネル（メイン画面のランプ + 設定要約） -----------------

    _MODULE_PANEL_KEYS = ("asr", "llm", "tts", "maai")
    _MODULE_PANEL_STATE_STYLES: dict[str, tuple[str, str]] = {
        "stopped": ("停止", STATE_COLORS["stopped"]),
        "starting": ("起動中", STATE_COLORS["starting"]),
        "running": ("稼働", STATE_COLORS["running"]),
        "error": ("異常", STATE_COLORS["error"]),
    }

    def _module_settings_summary(self, module: str) -> str:
        """メイン画面向けの1行設定要約。UI ウィジェットの現在値から組み立てる。"""
        try:
            if module == "asr":
                cmp_check = getattr(self, "asr_compare_mode_check", None)
                if cmp_check is not None and cmp_check.isChecked():
                    top = self.asr_compare_top_engine_combo.currentText().strip()
                    bottom = self.asr_compare_bottom_engine_combo.currentText().strip()
                    return f"比較 {top} + {bottom}"
                engine = _engine_label(self.asr_engine_combo) or "sherpa-onnx"
                vad = self.asr_vad_backend_combo.currentText().strip() or "maai"
                turn = self.asr_turn_backend_combo.currentText().strip() or "maai"
                return f"{engine} / VAD:{vad} / Turn:{turn}"
            if module == "llm":
                backend = self.llm_backend_combo.currentText().strip() or "LM Studio"
                model = self._read_llm_model_for_backend(backend) or "-"
                return f"{backend} / {model}"
            if module == "tts":
                engine = _engine_label(self.tts_engine_combo) or "VOICEVOX"
                voice = self.tts_voice_edit.text().strip() or "-"
                return f"{engine} / voice:{voice}"
            if module == "maai":
                modes: list[str] = []
                bc_box = getattr(self, "maai_enable_bc_box", None)
                nod_box = getattr(self, "maai_enable_nod_box", None)
                tv_box = getattr(self, "maai_enable_turn_vap_box", None)
                if bc_box is not None and bc_box.isChecked():
                    modes.append("bc")
                if nod_box is not None and nod_box.isChecked():
                    modes.append("nod")
                if tv_box is not None and tv_box.isChecked():
                    modes.append("turn_vap")
                device = self._maai_unified_device_str()
                return f"{'+'.join(modes) if modes else 'none'} / {device}"
        except Exception:
            return "-"
        return "-"


    def _note_maai_process_metrics_from_console(self, text: str) -> None:
        if "Average processing time:" not in text:
            return
        if re.search(r"\[ASR(?:Worker)?\]\[", text):
            return
        m = _MAAI_AVG_PROCESS_TIME_RE.search(text)
        if m is None:
            return
        try:
            process_sec = float(m.group("sec"))
            pps = float(m.group("pps"))
        except (TypeError, ValueError):
            return
        if process_sec != process_sec or process_sec < 0:
            return
        mode = str(m.group("mode") or "maai").strip() or "maai"
        entry: dict[str, float] = {
            "process_ms": process_sec * 1000.0,
        }
        rtf_raw = m.group("rtf")
        if rtf_raw is not None:
            try:
                rtf = float(rtf_raw)
            except (TypeError, ValueError):
                rtf = None
            else:
                if rtf == rtf and rtf >= 0:
                    entry["rtf"] = rtf
        if pps == pps and pps > 0:
            entry["process_per_sec"] = pps
        self._maai_process_metrics[mode] = entry
        self._refresh_maai_status_metrics()

    def _refresh_maai_status_metrics(self) -> None:
        labels = getattr(self, "module_status_metrics_labels", None)
        if not labels:
            return
        label = labels.get("maai")
        if label is None:
            return
        stats = self._maai_process_metrics
        if not stats:
            label.setText("")
            label.setToolTip("MaAIコンソールの平均処理時間とRTFから表示します")
            return
        bottleneck = max(stats.values(), key=lambda d: float(d.get("process_ms", 0.0)))
        ms = float(bottleneck.get("process_ms", 0.0))
        rtf = bottleneck.get("rtf")
        text = f"{ms:.1f}ms" if rtf is None else f"{ms:.1f}ms RTF={float(rtf):.2f}"
        parts: list[str] = []
        for mode, d in stats.items():
            pms = float(d.get("process_ms", 0.0))
            prtf = d.get("rtf")
            pps = d.get("process_per_sec")
            bit = f"{mode} {pms:.1f}ms"
            if prtf is not None:
                bit += f" RTF={float(prtf):.2f}"
            if pps is not None:
                bit += f" {float(pps):.2f}/s"
            parts.append(bit)
        label.setText(text)
        label.setToolTip(" / ".join(parts) if parts else text)

    def _on_dialog_orchestrator_event(self, payload_obj: object) -> None:
        """Orchestrator から届いたイベントを対応するハンドラへ渡す。

        分岐本体は `gui/orchestrator_events.py` にある。
        テーブルが 2 枚あるのは順序を保つため（`_src` 判定を挟む位置が意味を持つ）。
        """
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        event_type = str(payload.get("type", "")).strip().lower()
        src_early = str(payload.get("_src", "")).strip().lower()

        handler = ORCH_EVENT_HANDLERS_EARLY.get(event_type)
        if handler is not None:
            getattr(self, handler)(payload, event_type)
            return

        if src_early == "llm_admin":
            self._on_llm_admin_ipc_event(payload)
            return
        src = str(payload.get("_src", "")).strip().lower()
        if src == "tts":
            self._on_tts_ipc_event(payload)
            return
        if src == "audio_output":
            self._on_audio_output_ipc_event(payload)
            return

        handler = ORCH_EVENT_HANDLERS_LATE.get(event_type)
        if handler is not None:
            getattr(self, handler)(payload, event_type)
            return

    # ---- voice dialog ----

    def _set_voice_dialog_controls_state(self) -> None:
        if not hasattr(self, "voice_dialog_start_btn"):
            return
        active = self._voice_dialog_active or self._voice_dialog_starting
        self.voice_dialog_start_btn.setEnabled(not active)
        self.voice_dialog_stop_btn.setEnabled(active)

    @staticmethod
    def _fmt_voice_dialog_latency(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value) * 1000.0:.0f}ms"

    def _refresh_voice_dialog_latency_label(self) -> None:
        if not hasattr(self, "voice_dialog_latency_label"):
            return
        asr_text = self._fmt_voice_dialog_latency(self._voice_dialog_asr_latency_sec)
        llm_text = self._fmt_voice_dialog_latency(self._voice_dialog_llm_latency_sec)
        tts_text = self._fmt_voice_dialog_latency(self._voice_dialog_tts_latency_sec)
        self.voice_dialog_latency_label.setText(f"遅延　ASR：{asr_text} / LLM：{llm_text} / TTS：{tts_text}")

    def _voice_dialog_chat_items(self) -> list[tuple[str, str]]:
        """会話の欄（吹き出し）に出す (role, text)。確定した発話と応答だけ。

        system（プロンプト）は出さない。認識中・生成中は下の「進行中」の欄に出す。
        """
        items: list[tuple[str, str]] = []
        for msg in self._voice_dialog_messages:
            role = str(msg.get("role", "")).strip().lower()
            text = str(msg.get("content", "")).strip()
            # user は本文が空でも残す（空スキップだと「発話が消えた」ように見える）。
            if role == "user":
                items.append(("user", text or "（空）"))
            elif role == "assistant" and text:
                items.append(("assistant", text))
        for text in self._voice_dialog_pending_texts:
            t = str(text).strip()
            if t:
                items.append(("user", t))  # 認識は確定済み（LLM へ送る前）
        return items

    def _voice_dialog_live_texts(self) -> tuple[str, str]:
        """(認識中の発話, 生成中の応答)。応答が履歴に入ったら生成中は空にする。"""
        last_assistant = ""
        for msg in self._voice_dialog_messages:
            if str(msg.get("role", "")).strip().lower() == "assistant":
                text = str(msg.get("content", "")).strip()
                if text:
                    last_assistant = text
        streaming = self._voice_dialog_streaming_text().strip()
        if streaming == last_assistant:
            streaming = ""
        return str(self._voice_dialog_current_partial or "").strip(), streaming

    def _render_voice_dialog_live(self) -> None:
        user_label = getattr(self, "voice_dialog_live_user_label", None)
        assistant_label = getattr(self, "voice_dialog_live_assistant_label", None)
        if user_label is None or assistant_label is None:
            return
        partial, streaming = self._voice_dialog_live_texts()
        for label, text in ((user_label, partial), (assistant_label, streaming)):
            shown = text or "—"
            if label.text() != shown:
                label.setText(shown)
            label.setProperty("empty", not text)
            label.style().unpolish(label)
            label.style().polish(label)

    def _voice_dialog_streaming_text(self) -> str:
        full = self._voice_dialog_streaming_assistant_full
        if full:
            return full
        return "".join(str(s) for s in self._voice_dialog_tts_confirmed_segments)

    def _refresh_voice_dialog_history_view(self) -> None:
        view = getattr(self, "voice_dialog_history_view", None)
        if view is None:
            return
        if hasattr(view, "set_items"):
            view.set_items(self._voice_dialog_chat_items())
        self._render_voice_dialog_live()
        self._refresh_voice_dialog_llm_stream_view(render_chat=False)

    def _request_voice_dialog_history_refresh(self) -> None:
        # UIウィジェット更新は必ずQtメインスレッドで実行する。
        self.voice_dialog_history_refresh_signal.emit()

    @staticmethod
    def _truncate_log_text(text: str, max_len: int = 200) -> str:
        t = str(text or "")
        if len(t) <= max_len:
            return t
        return t[: max_len - 3] + "..."

    def _refresh_voice_dialog_llm_stream_view(self, *, render_chat: bool = True) -> None:
        if render_chat:
            # 生成中の応答は会話の欄ではなく「進行中」の欄だけを書き換える。
            self._render_voice_dialog_live()
        if not hasattr(self, "voice_dialog_llm_stream_view") or self.voice_dialog_llm_stream_view is None:
            return
        vw = self.voice_dialog_llm_stream_view
        full = self._voice_dialog_streaming_assistant_full
        segments = self._voice_dialog_tts_confirmed_segments
        if (not full) and (not segments):
            # 応答完了後に状態だけ空になることがある。次の応答まで直前の表示を残す。
            return
        if not segments:
            body = full
        else:
            confirmed_text = "".join(segments)
            tail = full[len(confirmed_text) :] if full.startswith(confirmed_text) else ""
            parts: list[str] = [s for s in segments if str(s).strip() != ""]
            if tail:
                parts.append(tail)
            body = "/".join(parts) if parts else full
        vw.setPlainText(f"アシスタント（生成中）：{body}")
        vw.moveCursor(QTextCursor.MoveOperation.End)

    def _on_voice_dialog_llm_chunk_ui(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        self._voice_dialog_streaming_assistant_full = str(payload.get("assistant_text", ""))
        self._refresh_voice_dialog_llm_stream_view()

    def _on_llm_tts_stream_push_ui(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        cumulative = str(payload.get("cumulative", ""))
        pushed_segment = str(payload.get("pushed_segment", ""))
        if not cumulative:
            return
        self.process_output_signal.emit(
            f"[LLM→TTS push][llm_test] ここまで: {self._truncate_log_text(cumulative)}"
        )
        if pushed_segment:
            self._llm_test_tts_confirmed_segments.append(pushed_segment)
        self._redraw_llm_test_assistant_block()

    def _format_llm_test_assistant_display(self) -> str:
        full = self._llm_test_streaming_assistant_full
        segments = self._llm_test_tts_confirmed_segments
        if not segments:
            return full
        confirmed_text = "".join(segments)
        tail = full[len(confirmed_text) :] if full.startswith(confirmed_text) else ""
        lines: list[str] = list(segments)
        if tail:
            lines.append(tail)
        return "\n".join(lines)

    def _redraw_llm_test_assistant_block(self) -> None:
        if self._llm_test_assistant_block_start < 0:
            return
        if not hasattr(self, "llm_chat_view"):
            return
        view = self.llm_chat_view
        doc = view.document()
        cursor = QTextCursor(doc)
        cursor.setPosition(self._llm_test_assistant_block_start)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(self._format_llm_test_assistant_display())
        view.moveCursor(QTextCursor.MoveOperation.End)

    def _on_voice_dialog_latency_event(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        kind = str(payload.get("kind", "")).strip().lower()
        value_raw = payload.get("value")
        value = float(value_raw) if isinstance(value_raw, (float, int)) else None
        if kind == "asr":
            self._voice_dialog_asr_latency_sec = value
        elif kind == "llm":
            self._voice_dialog_llm_latency_sec = value
        elif kind == "tts":
            self._voice_dialog_tts_latency_sec = value
        elif kind == "interrupt":
            pass
        self._refresh_voice_dialog_latency_label()
        if (
            hasattr(self, "maai_plot_window")
            and self.maai_plot_window is not None
            and self.asr_maai_gui_plot_check.isChecked()
        ):
            if kind == "interrupt":
                self.maai_plot_window.update_voice_dialog_latency(
                    kind="interrupt",
                    latency_sec=None,
                    event_ts=payload.get("end_ts", time.time()),
                    start_ts=None,
                    end_ts=payload.get("end_ts", time.time()),
                )
            else:
                self.maai_plot_window.update_voice_dialog_latency(
                    kind=kind,
                    latency_sec=value,
                    event_ts=payload.get("end_ts", payload.get("start_ts", time.time())),
                    start_ts=payload.get("start_ts"),
                    end_ts=payload.get("end_ts"),
                )

    def _build_orchestrator_tts_stream_inputs(self) -> dict[str, object]:
        """`vds.build_orchestrator_tts_stream_inputs` に一本化（Phase 1-2）。"""
        return vds.build_orchestrator_tts_stream_inputs(
            self._widget_config_snapshot(sync_engine_voice=False)
        )

    def _build_dialog_orchestrator_settings(self) -> dict[str, object]:
        """`vds.build_dialog_orchestrator_settings` に一本化（Phase 1-2）。

        ライブ設定更新（シグナルハンドラ）からも呼ばれるため、フィールドが一時的に
        非数値でも例外で落とさず既定値へフォールバックする性質は `cfg_*` が担う。
        """
        return vds.build_dialog_orchestrator_settings(
            self._widget_config_snapshot(sync_engine_voice=False)
        )

    def _get_mmd_burst_snd_frames(self) -> int:
        edit = getattr(self, "mmd_burst_snd_frames_edit", None)
        if edit is not None:
            try:
                return max(0, min(500, int(edit.text().strip() or "20")))
            except ValueError:
                pass
        return 20

    def _get_mmd_blink_enable(self) -> bool:
        check = getattr(self, "mmd_blink_enable_check", None)
        return bool(check.isChecked()) if check is not None else True

    def _get_mmd_blink_interval_mean_sec(self) -> float:
        from ..mmd.blink import DEFAULT_BLINK_INTERVAL_MEAN_SEC, normalize_blink_interval_mean_sec

        edit = getattr(self, "mmd_blink_interval_mean_sec_edit", None)
        value = edit.text().strip() if edit is not None else DEFAULT_BLINK_INTERVAL_MEAN_SEC
        return normalize_blink_interval_mean_sec(value)

    def _get_mmd_blink_morph(self) -> str:
        from ..mmd.blink import sanitize_blink_morph

        edit = getattr(self, "mmd_blink_morph_edit", None)
        value = edit.text() if edit is not None else "まばたき"
        return sanitize_blink_morph(value)

    def _get_mmd_blink_mode(self) -> str:
        from ..mmd.blink import normalize_blink_mode

        combo = getattr(self, "mmd_blink_mode_combo", None)
        return normalize_blink_mode(combo.currentData() if combo is not None else None)

    def _get_audio_output_mode(self) -> str:
        combo = getattr(self, "audio_output_mode_combo", None)
        if combo is not None:
            try:
                data = combo.currentData()
                if data is not None:
                    return str(data).strip().lower() or "mmd"
            except Exception:
                pass
            try:
                return str(combo.currentText()).strip().lower() or "mmd"
            except Exception:
                pass
        return "mmd"

    def _effective_audio_output_mode(self) -> str:
        """UI が MMD でも MMDAgent 未起動のときはローカルスピーカーへフォールバック（保存値は変えない）。"""
        mode = self._get_audio_output_mode()
        if self._remote_backend_enabled:
            # The container cannot inspect the host-side MMD process.  In this
            # mode the published Remote bridge is the only supported output.
            return mode
        if mode == "mmd" and not self._avatar_output_available():
            return "local"
        return mode

    def _avatar_in_browser(self) -> bool:
        """アバターをブラウザ版（MMDAgent-EX Web）に表示する設定か。"""
        combo = getattr(self, "avatar_display_combo", None)
        return combo is not None and combo.currentData() == vds.AVATAR_DISPLAY_BROWSER

    def _avatar_output_available(self) -> bool:
        """音声と頷きをアバターへ送れるか。MMDAgent-EX は起動中のとき、ブラウザ版はいつでも
        （ページを開けば WebSocket でつながり、つながる前に送った分は捨てられるだけ）。"""
        return self.process_service.is_running() or self._avatar_in_browser()

    def _on_avatar_display_changed(self, _index: int = 0) -> None:
        """表示先を切り替えたら、保存して音声の出口とブリッジ（TCP / WebSocket）を取り直す。"""
        browser = self._avatar_in_browser()
        edit = getattr(self, "avatar_browser_ws_port_edit", None)
        if edit is not None:
            edit.setEnabled(browser)
        if getattr(self, "_is_loading_app_config", False):
            return
        # 保存すると音声の出口を取り直し、出せるならブリッジの設定も送る（_save_app_config）
        self._save_app_config()
        if browser and self.process_service.is_running():
            self.status_label.setText("状態：ブラウザ表示に切り替えました（MMDAgent-EX は停止してください）")
        self._maai_mmd_av_preamble_sent = False
        self._maai_mmd_av_preamble_wait_for_mmd = not self._avatar_output_available()
        self._try_send_maai_mmd_av_preamble_once()

    def _teardown_fallback_audio_output_if_unused(self) -> None:
        """MMD 起動後は SND を Orchestrator へ送るため、未起動時フォールバック用のローカル Audio Output を止める。"""
        if self._get_audio_output_mode() != "mmd" or not self._avatar_output_available():
            return
        if not self._audio_output_worker_ready_event.is_set():
            return
        if getattr(self, "_audio_output_worker_started_output_mode", None) != "local":
            return
        self._teardown_audio_output_ipc()
        # フォールバック(local)を止めただけだと再生経路が無くなる。MMD 向けに取り直す。
        self._ensure_audio_output_worker()

    # `dialog/settings` が MDF 由来として読む Plugin_Remote_* キー。
    # これらは controller_config.json ではなく .mdf に保存されるため、
    # ビルダーへ渡す cfg スナップショットにはウィジェットの現在値を重ねる。
    _MDF_BUILDER_KEYS = (
        "Plugin_Remote_EnableClient",
        "Plugin_Remote_EnableServer",
        "Plugin_Remote_ListenPort",
        "Plugin_Remote_Port",
        "Plugin_Remote_Websocket_Directory",
        "Plugin_Remote_Websocket_Host",
        "Plugin_Remote_Websocket_Port",
    )

    def _read_app_config_field(self, key: str) -> object:
        """spec に載っている平坦フィールドを **ウィジェット → 保存値** へ変換する。

        `app_config_spec.APP_CONFIG_FIELDS` は既存コードから機械抽出したので、
        既定値・ウィジェット名・ガード有無は従来と同一。
        """
        row = APP_CONFIG_FIELD_BY_KEY.get(key)
        if row is None:
            raise KeyError(f"app_config_spec に無いキー: {key}")
        _key, widget_attr, kind, guarded, save_default, _load_default, _setter, _empty = row
        widget = getattr(self, widget_attr, None)
        if widget is None:
            if guarded:
                return save_default
            raise AttributeError(f"ウィジェットが無い: {widget_attr} (key={key})")
        if kind == "bool":
            return bool(widget.isChecked())
        if kind == "combo":
            # 保存値は itemData（表示名と別）。data の無い項目は表示名。
            data = widget.currentData()
            return str(data if data is not None else widget.currentText()).strip()
        getter = getattr(widget, "text", None) or getattr(widget, "toPlainText")
        # `save_default` は **ウィジェットが存在しないとき**だけの代替値
        # （元コードの `... if hasattr(self, "X") else "既定"`）。
        # 空欄のときの代替ではないので、ここで空文字に既定を当ててはいけない
        # （当てると audio_input_tcp_host 等 7 キーで保存値が変わる。ゴールデンが検出した）。
        return str(getter()).strip()

    def _apply_app_config_field(self, cfg: dict, key: str) -> None:
        """spec に載っている平坦フィールドを **cfg → ウィジェット** へ反映する。

        `_load_app_config` から**元の位置のまま**呼ぶ。`setText` はハンドラを発火させる
        ため、まとめてループにすると発火順が変わって挙動が変わりうる。
        """
        row = APP_CONFIG_FIELD_BY_KEY.get(key)
        if row is None:
            raise KeyError(f"app_config_spec に無いキー: {key}")
        (_key, widget_attr, kind, _guarded, _save_default, load_default,
         setter, empty_default) = row
        widget = getattr(self, widget_attr, None)
        if widget is None:
            return
        if kind == "bool":
            widget.setChecked(json_config_bool(cfg, key, bool(load_default)))
            return
        if kind == "combo":
            value = str(cfg.get(key, load_default) or load_default).strip()
            index = widget.findData(value)
            if index < 0:
                index = widget.findText(value)
            if index < 0:
                index = widget.findData(str(load_default))
            if index >= 0:
                widget.setCurrentIndex(index)
            return
        value = str(cfg.get(key, load_default))
        if empty_default is not None:
            # 元コードが `str(cfg.get(...)).strip() or "既定"` だったキー
            # （空文字も既定へ落とす）。該当は 2 キーだけ。
            value = value.strip() or str(empty_default)
        getattr(widget, setter)(value)

    def _widget_config_snapshot(self, *, sync_engine_voice: bool = True) -> dict[str, object]:
        """ウィジェットの現在値を cfg dict として取り出す（ビルダーへ渡す入力）。

        `dialog/settings.build_*(cfg)` は cfg dict から読むので、GUI 側は
        「保存するときと同じ dict」を渡すことで同じ変換を共有する。
        `_collect_app_config_payload()` は冪等（ウィジェット→メモリ同期のみ）なので
        ビルダーから何度呼んでも安全。

        MDF 側の設定（`Plugin_Remote_*`）は `controller_config.json` に入らないため、
        ウィジェットの現在値を重ねる。**ファイルを読み直さない**のが重要で、
        GUI は編集中（.mdf 未保存）のウィジェット値で動くため、それに合わせる。
        """
        cfg = self._collect_app_config_payload(sync_engine_voice=sync_engine_voice)
        for key in self._MDF_BUILDER_KEYS:
            cfg[key] = self._widget_to_value(key)
        # 音声と口パクを渡すモデルの別名は MDF の隣の FST から読む。音声出力の設定を作るときは mdf_path を
        # 外すうえ、Docker の構成ではコンテナ側のパスなので、ここでホスト側のファイルから読んでおく。
        host_mdf = self._current_mdf_host_path()
        cfg["avatar_model_alias"] = vds.avatar_model_alias_from_mdf(str(host_mdf) if host_mdf else "")
        return cfg

    def _build_remote_plugin_tcp_settings_dict(self) -> dict[str, object]:
        # `vds.build_remote_plugin_tcp_settings_dict` に一本化（Phase 1-2）。
        # 旧実装の `or "9001"` フォールバックは、`load_mdf_settings` が
        # FIELD_SPECS の既定 "80" を必ず入れるためウィジェットが空にならず到達しない。
        return vds.build_remote_plugin_tcp_settings_dict(
            self._widget_config_snapshot(),
            # The config is assembled by native Qt, but the bridge itself runs
            # in the remote container.  Apply the same loopback substitutions
            # used by Docker Web rather than detecting this host process.
            running_in_docker=True if self._remote_backend_enabled else None,
            log_docker_substitution=not self._remote_backend_enabled,
        )

    def _build_audio_output_worker_config_dict(self) -> dict[str, object]:
        """`vds.build_audio_output_worker_config_dict` に一本化（Phase 1-2）。

        GUI 固有の状態を 2 つ明示的に渡す必要がある:
        - `output_mode` / `assume_mmd_running`: 旧実装は `_effective_audio_output_mode()`
          経由で **MMDAgent が起動中か** に依存していた（mmd かつ未起動なら local）。
          既定引数のままでは MMD 起動中に local へ落ちて挙動が変わる。
        - `mdf_path` を外す: vds 側は MDF ファイルを読み直して `Plugin_Remote_*` を
          上書きするが、GUI は**ウィジェットの現在値**（.mdf 未保存の編集も反映）で動く。
          スナップショットに載せた widget 値を勝ち残らせるため、ファイル読み込みを止める。
        """
        self._validate_remote_backend_constraints()
        cfg = self._widget_config_snapshot()
        cfg.pop("mdf_path", None)
        built = vds.build_audio_output_worker_config_dict(
            cfg,
            output_mode=self._get_audio_output_mode(),
            assume_mmd_running=(self._remote_backend_enabled or self._avatar_output_available()),
        )
        if self._remote_backend_enabled:
            built["remote_plugin_settings"] = self._build_remote_plugin_tcp_settings_dict()
        return built

    def _teardown_audio_output_ipc(self) -> None:
        self._stop_module_via_orchestrator("audio_output")
        self._audio_output_worker_ready_event.clear()
        self._audio_output_worker_started_output_mode = None
        self._remote_plugin_settings_digest_sent = None

    def _ensure_audio_output_worker(self) -> None:
        """Orchestrator 側 `ensure_audio_output` に委ねる（設定変更時のみプロセス再起動）。"""
        cfg = self._build_audio_output_worker_config_dict()
        self._start_module_via_orchestrator("audio_output", cfg, timeout_sec=60.0)

    def _sync_orchestrator_audio_output_from_ui(self, *, force: bool = False) -> None:
        """常駐 Orchestrator 上の AudioOutput を、いまの GUI 設定に追従させる。

        `ensure_audio_output` のダイジェストで再起動の要否を判定する。
        音声対話開始時は ensure しないため、設定変更は `_save_app_config` / MDF 保存などからここへ集約する。

        _save_app_config は ASR エンジン等の無関係な設定変更でも呼ばれるため、
        AudioOutput 設定が実際に変わったときだけ ensure する（不要な ensure RPC を抑止）。
        force=True は Orchestrator 接続直後など、状態を確実に再確立したいときに使う。
        """
        if not self._is_dialog_orchestrator_runtime_active():
            return
        try:
            cfg_json = json.dumps(
                self._build_audio_output_worker_config_dict(), sort_keys=True, ensure_ascii=False, default=str
            )
        except Exception:
            cfg_json = None
        if (not force) and cfg_json is not None and cfg_json == getattr(self, "_last_synced_audio_output_cfg_json", None):
            return
        try:
            self._ensure_audio_output_worker()
            self._last_synced_audio_output_cfg_json = cfg_json
        except Exception as ex:
            self.process_output_signal.emit(f"[AudioOutput][WARN] 設定同期に失敗: {ex}")

    def _sync_orchestrator_audio_input_from_ui(self, *, force: bool = False) -> None:
        """常駐 Orchestrator 上の AudioInput を、いまの GUI 設定に追従させる。

        `ensure_audio_input` が保持する設定 JSON で再起動の要否を判定する。
        音声対話開始のたびに ensure しないため、設定変更は `_save_app_config` / MDF 保存などからここへ集約する。

        _save_app_config は ASR エンジン等の無関係な設定変更でも呼ばれるため、
        AudioInput 設定が実際に変わったときだけ ensure する（不要な ensure RPC＝
        AudioInput 再起動とサンプルクロック再アンカーを抑止）。
        force=True は Orchestrator 接続直後など、状態を確実に再確立したいときに使う。
        """
        if not self._is_dialog_orchestrator_runtime_active():
            return
        # 未起動かつ ASR/MaAI/音声対話 も動いていないときは、設定保存や
        # Orchestrator ready をきっかけにマイクを新規起動しない。
        # 一度起動したワーカーはテスト停止では落とさず、GUI 終了時に teardown する。
        # 起動中なら設定変更を ensure で追従させる（同一 JSON ならプロセスは取り直さない）。
        worker_running = self._audio_input_worker_ready_event.is_set()
        if not worker_running and not self._audio_input_needed_by_active_modules():
            return
        try:
            cfg_json = json.dumps(
                self._build_audio_input_worker_config(), sort_keys=True, ensure_ascii=False, default=str
            )
        except Exception:
            cfg_json = None
        if (not force) and cfg_json is not None and cfg_json == getattr(self, "_last_synced_audio_input_cfg_json", None):
            # In Docker mic mode the selected host device is deliberately not
            # part of the container's browser-mode config.  It can therefore
            # change while the worker digest stays identical.
            self._sync_remote_mic_capture()
            return
        try:
            self._ensure_audio_input_worker()
            self._last_synced_audio_input_cfg_json = cfg_json
            self._sync_remote_mic_capture()
        except Exception as ex:
            self.process_output_signal.emit(f"[AudioInput][WARN] 設定同期に失敗: {ex}")

    def _sync_orchestrator_audio_io_after_ready(self) -> None:
        """Orchestrator ready 後に UI 設定を Audio I/O へ反映する（設定読込完了後に1回）。"""
        if self._controller_closing:
            return
        if not self._is_dialog_orchestrator_runtime_active():
            return
        if self._is_loading_app_config:
            QTimer.singleShot(0, self._sync_orchestrator_audio_io_after_ready)
            return
        # Orchestrator 接続直後は worker 状態を確実に再確立するため force で ensure する
        # （以降のダイジェスト判定の基準も同時に更新される）。
        self._sync_orchestrator_audio_output_from_ui(force=True)
        self._sync_orchestrator_audio_input_from_ui(force=True)

    def _on_audio_output_ipc_event(self, payload: dict) -> None:
        # remote_status 等は手前の分岐で処理済み。ここへ来るのはフォワードのみ。
        self.dialog_orchestrator_event_signal.emit(dict(payload or {}))

    def _try_reset_dialog_orchestrator_nod_para_timing(self) -> None:
        """MaAI テストの停止/再起動などで頷き判定状態をリセット（音声対話の start/stop 以外の経路）。"""
        if not self._is_dialog_orchestrator_runtime_active():
            return
        try:
            self._send_dialog_orchestrator_command({"type": "reset_nod_para_timing"})
        except Exception:
            pass

    def _push_dialog_orchestrator_settings(self, *, force: bool = False) -> None:
        if not self._is_dialog_orchestrator_runtime_active():
            return
        settings = self._build_dialog_orchestrator_settings()
        digest = hashlib.sha256(
            json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if not force and digest == self._dialog_orchestrator_settings_digest:
            return
        self._dialog_orchestrator_settings_digest = digest
        self._send_dialog_orchestrator_command(
            {
                "type": "update_settings",
                "settings": settings,
            }
        )
        if self._audio_output_worker_ready_event.is_set():
            self._send_audio_output_command(
                {
                    "type": "update_settings",
                    "settings": {
                        "mmd_burst_snd_frames": self._get_mmd_burst_snd_frames(),
                        "mmd_blink_enable": self._get_mmd_blink_enable(),
                        "mmd_blink_interval_mean_sec": self._get_mmd_blink_interval_mean_sec(),
                        "mmd_blink_morph": self._get_mmd_blink_morph(),
                        "mmd_blink_mode": self._get_mmd_blink_mode(),
                    },
                }
            )

    def _send_remote_plugin_settings(self) -> None:
        """Send MDF Remote plugin settings to Audio Output Worker（MMD bridge は worker 側のみ）。"""
        settings = self._build_remote_plugin_tcp_settings_dict()
        digest = hashlib.sha256(
            json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if digest == self._remote_plugin_settings_digest_sent:
            return
        if self._audio_output_worker_ready_event.is_set():
            if self._send_audio_output_command(
                {
                    "type": "remote_plugin_apply_settings",
                    "settings": dict(settings),
                }
            ):
                self._remote_plugin_settings_digest_sent = digest

    def _send_remote_plugin_stop(self) -> None:
        """Audio Output Worker の MMD bridge を停止する。"""
        self._remote_plugin_settings_digest_sent = None
        self._orch_mmd_bridge_connected = False
        self._orch_mmd_bridge_mode = ""
        self._audio_mmd_bridge_connected = False
        self._audio_mmd_bridge_mode = ""
        self._send_audio_output_command({"type": "remote_plugin_stop"})

    def _voice_dialog_check_modules_ready_for_start(self) -> str | None:
        """音声対話開始前に Orchestrator 接続が揃っているか検査する。問題なしなら None。"""
        if not self._dialog_orchestrator_ready_event.is_set():
            return "対話オーケストレーターの準備が完了していません。"
        if self._dialog_orchestrator_ipc_client is None:
            return "対話オーケストレーターのIPCが接続されていません。"
        return None

    def _voice_dialog_assert_all_modules_ready_for_start(self) -> None:
        """各 ensure 完了後、IPC/ready 反映の遅延に備え短時間リトライしてから開始可否を確定する。"""
        deadline = time.time() + 3.0
        last_msg = "音声対話開始前の依存確認に失敗しました。"
        while time.time() < deadline:
            self._pump_ui_events()
            err = self._voice_dialog_check_modules_ready_for_start()
            if err is None:
                return
            last_msg = err
            time.sleep(0.02)
        raise RuntimeError(f"音声対話開始前チェック: {last_msg}")

    def _ensure_voice_dialog_modules_ready(self, _tts_engine: str) -> None:
        self._ensure_dialog_orchestrator_worker()

    def _build_asr_config_dict(self) -> dict[str, object]:
        """ASR ワーカー起動用の設定辞書。`vds.build_asr_config_dict` に一本化（Phase 1-2）。

        旧実装は `nemotron-asr`（専用 venv のサブプロセス）用の
        `nemotron_model` / `nemotron_python` / `nemotron_device` /
        `nemotron_chunk_size_ms` / `nemotron_window_sec` / `nemotron_max_symbols` /
        `nemotron_lang` を**渡していなかった**。そのため GUI で `nemotron-asr` を選ぶと
        `asr/service.py` の「Nemotron 用 Python（専用 venv）が見つかりません」で必ず失敗した
        （設定欄には正しいパスが入っているのに渡らない）。一本化で解消する。
        """
        cfg = self._widget_config_snapshot()
        built = vds.build_asr_config_dict(cfg)
        if self._remote_backend_enabled:
            raw = str(cfg.get("asr_whisper_model_path", "") or "").strip()
            built["whisper_use_local_path"] = self._remote_whisper_use_local_path(raw)
        return built

    def _build_worker_configs(self, tts_engine: str) -> dict[str, object]:
        """Orchestrator に渡すワーカー起動設定。`vds.build_worker_configs` に一本化（Phase 1-2）。

        audio_output は MMDAgent の起動状態に依存するため明示的に渡す。
        """
        self._validate_remote_backend_constraints(tts_engine=tts_engine)
        cfg = self._widget_config_snapshot()
        built = vds.build_worker_configs(
            cfg,
            tts_engine,
            audio_output_mode=self._get_audio_output_mode(),
            assume_mmd_running=(self._remote_backend_enabled or self._avatar_output_available()),
            force_browser_mic=self._remote_backend_enabled,
        )
        if self._remote_backend_enabled:
            audio_output_cfg = built.get("audio_output")
            if isinstance(audio_output_cfg, dict):
                audio_output_cfg["remote_plugin_settings"] = (
                    self._build_remote_plugin_tcp_settings_dict()
                )
            asr_cfg = built.get("asr")
            if isinstance(asr_cfg, dict):
                raw = str(cfg.get("asr_whisper_model_path", "") or "").strip()
                asr_cfg["whisper_use_local_path"] = self._remote_whisper_use_local_path(raw)
            audio_input_cfg = built.get("audio_input")
            if isinstance(audio_input_cfg, dict) and audio_input_cfg.get("input_mode") == "wav":
                mapped = self._container_path_for_backend_value(
                    str(audio_input_cfg.get("wav_path", "") or "")
                )
                if not mapped:
                    raise RuntimeError(
                        "WAV 入力ファイルが Docker の bind mount 外です。"
                        "INPUT_WAV_HOST またはリポジトリ配下のファイルを指定してください。"
                    )
                audio_input_cfg["wav_path"] = mapped
        return built

    def _start_voice_dialog(self) -> None:
        if not self._dialog_server_resume_for_action():
            return
        if self._voice_dialog_active or self._voice_dialog_starting:
            return
        try:
            self._voice_dialog_starting = True
            self._set_voice_dialog_controls_state()
            self.voice_dialog_status_label.setText("音声対話：起動中…")
            engine = self.tts_engine_combo.currentText().strip() or "VOICEVOX"
            self._validate_remote_backend_constraints(tts_engine=engine)
            self._voice_dialog_maai_preexisting = self._maai_ready
            self._voice_dialog_asr_preexisting = self._asr_ready
            self._ensure_voice_dialog_modules_ready(engine)
            self._voice_dialog_assert_all_modules_ready_for_start()
            settings = self._build_dialog_orchestrator_settings()
            worker_configs = self._build_worker_configs(engine)
            settings["worker_configs"] = worker_configs
            # どの ASR で動いているかを控える。stop_asr はここを見て Qwen3-ASR の
            # コンテナを落とすか決めるが、これまで代入していたのは GUI 自身の
            # start_asr だけだった。メイン画面から起動したあとコンボを別エンジンへ
            # 変えて停止すると、コンテナが残ったままになる。
            asr_cfg = worker_configs.get("asr") if isinstance(worker_configs, dict) else None
            if isinstance(asr_cfg, dict) and asr_cfg:
                self._last_asr_config = dict(asr_cfg)
            settings["llm_connection"] = self._active_llm_connection_settings()
            try:
                settings["tts_stream_inputs"] = self._build_tts_inputs(
                    engine=engine,
                    utterance="",
                    input_streaming_enabled=bool(self.tts_input_streaming_check.isChecked()),
                    split_settings=self._resolve_llm_split_settings(),
                )
                if engine == "TEMPLATE":
                    settings["tts_stream_inputs"]["template"] = self._load_tts_template_dict()
            except Exception:
                settings["tts_stream_inputs"] = {}
            if not self._send_dialog_orchestrator_command(
                {"type": "start_voice_dialog", "settings": settings}
            ):
                raise RuntimeError("対話オーケストレーターへの開始コマンド送信に失敗しました。")
            self._open_console_window()
            self._open_maai_plot_window()
            self._obs_acquired_voice = False
            if self._should_obs_link_voice():
                if self._obs_remote_try_start():
                    self._obs_acquired_voice = True
            self.process_output_signal.emit(
                f"[VoiceDialog] 自動準備完了: ASR={'ON' if self._asr_ready else 'OFF'} / TTS={engine}"
            )
        except Exception as ex:
            self._voice_dialog_starting = False
            self._set_voice_dialog_controls_state()
            self.voice_dialog_status_label.setText("音声対話：停止")
            QMessageBox.warning(self, "音声対話開始エラー", str(ex))

    def _stop_voice_dialog(self) -> None:
        if not self._voice_dialog_active and not self._voice_dialog_starting:
            return
        try:
            self._send_dialog_orchestrator_command({
                "type": "stop_voice_dialog",
                "keep_maai": self._voice_dialog_maai_preexisting,
                "keep_asr": self._voice_dialog_asr_preexisting,
            })
        except Exception:
            pass
        try:
            self._send_dialog_orchestrator_tts_done(playback_aborted=True)
        except Exception:
            pass
        self._voice_dialog_active = False
        self._voice_dialog_starting = False
        if not self._voice_dialog_maai_preexisting:
            self._teardown_maai_ipc()
        # 対話が ASR を新規起動していた場合（preexisting でない）、stop は keep_asr=False で
        # ワーカーを停止する。_asr_ready を戻さないと「ready のまま」になり、次の対話開始が
        # ASR を preexisting と誤認して起動をスキップしてしまう。状態を実体へ同期する。
        if not self._voice_dialog_asr_preexisting and self._asr_ready:
            self._asr_ready = False
            self._set_asr_controls_state()
        self._obs_remote_release_voice()
        self._set_voice_dialog_controls_state()
        self.voice_dialog_status_label.setText("音声対話：停止")
        self.process_output_signal.emit("[VoiceDialog] 音声対話を終了しました。")

    def _voice_dialog_interrupt_system_speech(self) -> bool:
        # 音声対話の割り込み・TTS は dialog/worker / Duplex 側で処理する。
        return False

    def _is_maai_backchannel_enabled(self) -> bool:
        check = getattr(self, "maai_backchannel_enabled_check", None)
        if check is None:
            return False
        try:
            return bool(check.isChecked())
        except Exception:
            return False

    def _get_maai_backchannel_prob_threshold(self) -> float:
        edit = getattr(self, "maai_backchannel_prob_threshold_edit", None)
        raw = ""
        if edit is not None:
            try:
                raw = edit.text().strip()
            except Exception:
                raw = ""
        try:
            value = float(raw or "0.5")
        except Exception:
            value = 0.5
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _get_maai_nod_swing_threshold(self) -> float:
        edit = getattr(self, "maai_test_nod_swing_threshold_edit", None)
        raw = ""
        if edit is not None:
            try:
                raw = edit.text().strip()
            except Exception:
                raw = ""
        # 空欄はモデルの値（ワーカーの準備完了時に受け取る .pt の t_swing。公式モデルは 0.43）。
        fallback = 0.5
        defaults = getattr(self, "_maai_nod_threshold_defaults", None)
        if isinstance(defaults, dict):
            try:
                fallback = float(defaults["swing"])
            except (KeyError, TypeError, ValueError):
                pass
        try:
            value = float(raw) if raw else fallback
        except Exception:
            value = fallback
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _get_maai_backchannel_min_interval_sec(self) -> float:
        edit = getattr(self, "maai_backchannel_min_interval_edit", None)
        raw = ""
        if edit is not None:
            try:
                raw = edit.text().strip()
            except Exception:
                raw = ""
        try:
            value = float(raw or "1.0")
        except Exception:
            value = 1.0
        return max(0.0, value)

    def _get_maai_backchannel_consecutive_frames(self) -> int:
        edit = getattr(self, "maai_backchannel_consecutive_frames_edit", None)
        raw = ""
        if edit is not None:
            try:
                raw = edit.text().strip()
            except Exception:
                raw = ""
        try:
            value = int(raw or "1")
        except Exception:
            value = 1
        return max(1, value)

    def _get_maai_backchannel_dictionary_path(self) -> str:
        edit = getattr(self, "maai_backchannel_dictionary_path_edit", None)
        if edit is not None:
            try:
                return edit.text().strip()
            except Exception:
                return ""
        return ""

    # ---- nod_para motion helpers ----

    def _is_nod_para_enabled(self) -> bool:
        cb = getattr(self, "nod_para_enabled_check", None)
        if cb is not None:
            try:
                return bool(cb.isChecked())
            except Exception:
                pass
        return False

    def _get_nod_para_float(self, attr: str, default: float) -> float:
        edit = getattr(self, attr, None)
        if edit is None:
            return default
        try:
            return float(edit.text().strip() or str(default))
        except Exception:
            return default

    def _get_nod_para_int(self, attr: str, default: int) -> int:
        edit = getattr(self, attr, None)
        if edit is None:
            return default
        try:
            return int(edit.text().strip() or str(default))
        except Exception:
            return default

    def _get_nod_para_text(self, attr: str, default: str) -> str:
        edit = getattr(self, attr, None)
        if edit is None:
            return default
        try:
            return edit.text().strip() or default
        except Exception:
            return default

    def _get_nod_para_combo(self, attr: str, default: str) -> str:
        combo = getattr(self, attr, None)
        if combo is None:
            return default
        try:
            return combo.currentText().strip() or default
        except Exception:
            return default

    def _tts_bridge_rpc_notify(self, payload: dict) -> None:
        rid = str(payload.get("request_id", "")).strip()
        if not rid:
            return
        with self._tts_bridge_rpc_lock:
            tup = self._tts_bridge_rpc_pending.get(rid)
        if tup is None:
            return
        ev, out = tup
        out.clear()
        out.update(payload)
        ev.set()

    def _on_tts_ipc_event(self, payload: dict) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "ready":
            self._tts_worker_ready_event.set()
            self._set_module_panel_state("tts", "running", str(self._tts_worker_engine or ""))
            return
        if event_type == "log":
            message = str(payload.get("message", "")).strip()
            if message:
                self.process_output_signal.emit(f"[TTSWorker] {message}")
            return
        if event_type == "fatal":
            message = str(payload.get("message", "")).strip() or "TTSワーカーで致命的エラーが発生しました。"
            self._set_module_panel_state("tts", "error", message)
            self.process_output_signal.emit(f"[TTSWorker][FATAL] {message}")
            return
        if event_type == "bridge_status":
            self._tts_bridge_rpc_notify(payload)
            return
        if event_type == "bridge_launch_result":
            self._tts_bridge_rpc_notify(payload)
            return
        if event_type == "bridge_stop_result":
            self._tts_bridge_rpc_notify(payload)
            return

    def _build_tts_worker_config(self, engine: str) -> dict:
        # `vds.build_tts_worker_config` に一本化（Phase 1-2）。
        # 未対応エンジン（VOICEVOX / TEMPLATE）で RuntimeError を投げる契約も共通化した。
        return vds.build_tts_worker_config(self._widget_config_snapshot(), engine)


    @staticmethod
    def _tts_config_digest(config: object) -> str:
        """TTS ワーカー設定の指紋。Orchestrator 側 `_tts_config_digest` と同じ正規化。"""
        if not isinstance(config, dict) or not config:
            return ""
        blob = json.dumps(config, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _ensure_tts_worker(self, engine: str) -> None:
        config = self._build_tts_worker_config(engine)
        if (
            self._tts_worker_ready_event.is_set()
            and self._tts_worker_engine == engine
            and self._tts_worker_config_digest == self._tts_config_digest(config)
        ):
            return
        self._tts_worker_ready_event.clear()
        self._start_module_via_orchestrator(
            "tts",
            config,
            engine=engine,
            timeout_sec=180.0,
        )
        self._tts_worker_engine = engine
        self._tts_worker_config_digest = self._tts_config_digest(config)

    def _parse_float(self, text: str, name: str, min_value: float | None = None) -> float:
        try:
            value = float(text.strip())
        except ValueError as ex:
            raise ValueError(f"{name} は数値で指定してください。") from ex
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} は {min_value} 以上で指定してください。")
        return value

    @staticmethod
    def _read_input_text(widget) -> str:
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        if isinstance(widget, QPlainTextEdit):
            return widget.toPlainText().strip()
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        return ""

    @staticmethod
    def _write_input_text(widget, value: str) -> None:
        text = str(value)
        if isinstance(widget, QLineEdit):
            widget.setText(text)
            return
        if isinstance(widget, QPlainTextEdit):
            widget.setPlainText(text)
            return
        if isinstance(widget, QComboBox):
            if widget.isEditable():
                widget.setEditText(text)
            else:
                widget.setCurrentText(text)

    def _get_nemotron_finalize_mode(self) -> str:
        engine = ""
        if hasattr(self, "asr_engine_combo"):
            try:
                engine = self.asr_engine_combo.currentText().strip()
            except Exception:
                engine = ""
        onnx_combo = getattr(self, "asr_nemotron_onnx_finalize_combo", None)
        if engine == "nemotron-onnx" and onnx_combo is not None:
            try:
                return onnx_combo.currentText().strip() or "vad"
            except Exception:
                pass
        combo = getattr(self, "asr_nemotron_finalize_combo", None)
        if combo is not None:
            try:
                return combo.currentText().strip() or "vad"
            except Exception:
                pass
        return "vad"

    def _nemotron_onnx_compute_key(self) -> str:
        """nemotron-onnx 実行モード combo -> 永続キー 'gpu-fp32' / 'cpu-int8'。"""
        combo = getattr(self, "asr_nemotron_onnx_compute_combo", None)
        if combo is not None:
            try:
                if combo.currentText().strip().startswith("CPU"):
                    return "cpu-int8"
            except Exception:
                pass
        return "gpu-fp32"

    def _nemotron_onnx_provider_quant(self) -> tuple[str, str]:
        """実行モードから (provider, quant) を導出。gpu-fp32->(cuda,fp32) / cpu-int8->(cpu,int8)。"""
        if self._nemotron_onnx_compute_key() == "cpu-int8":
            return "cpu", "int8"
        return "cuda", "fp32"

    def _sherpa_offline_provider(self, combo_attr: str) -> str:
        """sherpa-onnx OfflineRecognizer 系の実行デバイス combo -> 'cuda' / 'cpu'。

        既定が cpu なのは実測に基づく（int8 は dynamic 量子化で CUDA カーネルが効かない）。
        詳細は asr/engines_sherpa_offline.py 冒頭。
        """
        combo = getattr(self, combo_attr, None)
        if combo is not None:
            try:
                if combo.currentText().strip().startswith("GPU"):
                    return "cuda"
            except Exception:  # noqa: BLE001
                pass
        return "cpu"

    def _checkbox_state(self, check_attr: str, default: bool = True) -> bool:
        check = getattr(self, check_attr, None)
        return bool(check.isChecked()) if check is not None else default

    def _sherpa_offline_worker_config(self) -> dict[str, object]:
        """sherpa-onnx OfflineRecognizer 系 4 エンジンの ASRService 引数をウィジェットから作る。

        音声対話側は `dialog/settings.build_asr_config_dict` が同じキーを config から
        組み立てる。ここを 1 メソッドにまとめてあるのは、`launch_asr`（ASR 単体起動）の
        巨大な dict へ 20 行を直書きすると、片方にだけ項目を足して取りこぼす事故
        （d50e464 で nemotron / qwen3 / chirp3 が実際に踏んだもの）を繰り返すため。
        """
        def _text(attr: str, fallback: str) -> str:
            widget = getattr(self, attr, None)
            return (widget.text().strip() if widget is not None else "") or fallback

        def _threads(attr: str) -> int:
            try:
                return max(1, min(int(_text(attr, "8")), 32))
            except ValueError:
                return 8

        def _max_new_tokens(attr: str) -> int:
            try:
                return max(16, min(int(_text(attr, "128")), 2048))
            except ValueError:
                return 128

        return {
            "parakeet_onnx_model_dir": _text(
                "asr_parakeet_onnx_dir_edit", "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"),
            "parakeet_onnx_num_threads": _threads("asr_parakeet_onnx_num_threads_edit"),
            "parakeet_onnx_provider": self._sherpa_offline_provider("asr_parakeet_onnx_provider_combo"),
            "sensevoice_onnx_model_dir": _text(
                "asr_sensevoice_onnx_dir_edit", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"),
            "sensevoice_onnx_num_threads": _threads("asr_sensevoice_onnx_num_threads_edit"),
            "sensevoice_onnx_provider": self._sherpa_offline_provider("asr_sensevoice_onnx_provider_combo"),
            "sensevoice_onnx_language": _text("asr_sensevoice_onnx_language_edit", "ja"),
            "sensevoice_onnx_use_itn": self._checkbox_state("asr_sensevoice_onnx_itn_check"),
            "funasr_nano_onnx_model_dir": _text(
                "asr_funasr_nano_onnx_dir_edit", "sherpa-onnx-funasr-nano-int8-2025-12-30"),
            "funasr_nano_onnx_num_threads": _threads("asr_funasr_nano_onnx_num_threads_edit"),
            "funasr_nano_onnx_provider": self._sherpa_offline_provider("asr_funasr_nano_onnx_provider_combo"),
            "funasr_nano_onnx_language": _text("asr_funasr_nano_onnx_language_edit", "ja"),
            "funasr_nano_onnx_use_itn": self._checkbox_state("asr_funasr_nano_onnx_itn_check"),
            "funasr_nano_onnx_max_new_tokens": _max_new_tokens("asr_funasr_nano_onnx_max_new_tokens_edit"),
            "cohere_onnx_model_dir": _text(
                "asr_cohere_onnx_dir_edit", "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01"),
            "cohere_onnx_num_threads": _threads("asr_cohere_onnx_num_threads_edit"),
            "cohere_onnx_provider": self._sherpa_offline_provider("asr_cohere_onnx_provider_combo"),
            "cohere_onnx_language": _text("asr_cohere_onnx_language_edit", "ja"),
            "cohere_onnx_use_punct": self._checkbox_state("asr_cohere_onnx_punct_check"),
            "cohere_onnx_use_itn": self._checkbox_state("asr_cohere_onnx_itn_check"),
        }

    def _qwen3_onnx_provider(self) -> str:
        """qwen3-asr-onnx 実行デバイス combo -> 'cuda' / 'cpu'。int8 は CPU が速い。"""
        combo = getattr(self, "asr_qwen3_onnx_provider_combo", None)
        if combo is not None:
            try:
                if combo.currentText().strip().startswith("GPU"):
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def _qwen3_onnx_num_threads(self) -> int:
        edit = getattr(self, "asr_qwen3_onnx_num_threads_edit", None)
        raw = "8"
        if edit is not None:
            try:
                raw = edit.text().strip() or "8"
            except Exception:
                pass
        try:
            n = int(raw)
        except Exception:
            n = 8
        return max(1, min(n, 32))

    def _on_nemotron_finalize_combo_changed(self, _text: str = "") -> None:
        sender = self.sender()
        finalize_combo = getattr(self, "asr_nemotron_finalize_combo", None)
        onnx_combo = getattr(self, "asr_nemotron_onnx_finalize_combo", None)
        if sender is finalize_combo:
            other = onnx_combo
        elif sender is onnx_combo:
            other = finalize_combo
        else:
            other = None
        if other is not None and sender is not None:
            try:
                val = sender.currentText()
                other.blockSignals(True)
                other.setCurrentText(val)
                other.blockSignals(False)
            except Exception:
                pass
        if not self._is_loading_app_config:
            self._save_app_config()

    def _get_nemotron_chunk_size_ms(self) -> int:
        engine = ""
        if hasattr(self, "asr_engine_combo"):
            try:
                engine = self.asr_engine_combo.currentText().strip()
            except Exception:
                engine = ""
        combo = getattr(self, "asr_nemotron_onnx_chunk_combo", None)
        if engine == "nemotron-onnx" and combo is not None:
            try:
                return int(self._parse_float(combo.currentText(), "Nemotron chunk size", 1.0))
            except Exception:
                pass
        combo = getattr(self, "asr_nemotron_chunk_combo", None)
        if combo is not None:
            try:
                return int(self._parse_float(combo.currentText(), "Nemotron chunk size", 1.0))
            except Exception:
                pass
        return 320

    def _on_nemotron_chunk_combo_changed(self, _text: str = "", *, source: str = "asr") -> None:
        asr_combo = getattr(self, "asr_nemotron_chunk_combo", None)
        onnx_combo = getattr(self, "asr_nemotron_onnx_chunk_combo", None)
        sender = asr_combo if source == "asr" else onnx_combo
        other = onnx_combo if source == "asr" else asr_combo
        if sender is not None and other is not None:
            try:
                val = sender.currentText()
                other.blockSignals(True)
                other.setCurrentText(val)
                other.blockSignals(False)
            except Exception:
                pass
        if not self._is_loading_app_config:
            self._save_app_config()

    def _on_asr_engine_changed(self, engine: str) -> None:
        box = getattr(self, "asr_engine_settings_box", None)
        if box is not None:
            # ページの項目名にはエンジン名を付けないので、見出しに 1 回だけ出す。
            box.setTitle(f"エンジン設定（{engine}）" if engine else "エンジン設定")
        if hasattr(self, "asr_engine_combo") and self.asr_engine_combo.currentText() != engine:
            self.asr_engine_combo.blockSignals(True)
            self.asr_engine_combo.setCurrentText(engine)
            self.asr_engine_combo.blockSignals(False)
        if hasattr(self, "asr_engine_combo_settings") and self.asr_engine_combo_settings.currentText() != engine:
            self.asr_engine_combo_settings.blockSignals(True)
            self.asr_engine_combo_settings.setCurrentText(engine)
            self.asr_engine_combo_settings.blockSignals(False)

        is_whisper = engine == "faster-whisper"
        is_sherpa = engine == "sherpa-onnx"
        is_qwen = engine == "qwen3-asr"
        is_nemotron = engine == "nemotron-asr"
        is_external = engine == "external-asr-tcp"
        is_chirp = engine == "chirp3-asr"
        is_qwen3_onnx = engine == "qwen3-asr-onnx"
        is_nemotron_onnx = engine == "nemotron-onnx"
        if hasattr(self, "asr_engine_stack"):
            # 対応表は ui.py の asr_engine_pages が唯一の定義（ページ追加時にここを直す必要はない）。
            page_index = getattr(self, "asr_engine_page_index", {}).get(engine)
            if page_index is not None:
                self.asr_engine_stack.setCurrentIndex(page_index)

        # 念のため、旧ロジック依存コードとの互換性を保つため各フィールドの表示状態も同期する。
        self.asr_device_label.setVisible(is_whisper)
        self.asr_device_combo.setVisible(is_whisper)
        self.asr_compute_type_label.setVisible(is_whisper)
        self.asr_compute_type_combo.setVisible(is_whisper)
        self.asr_whisper_hf_label.setVisible(is_whisper)
        self.asr_whisper_hf_edit.setVisible(is_whisper)
        self.asr_whisper_model_label.setVisible(is_whisper)
        self.asr_whisper_model_edit.setVisible(is_whisper)
        self.asr_whisper_path_label.setVisible(is_whisper)
        self.asr_whisper_row.setVisible(is_whisper)
        self.asr_sherpa_tokens_label.setVisible(is_sherpa)
        self.asr_sherpa_tokens_row.setVisible(is_sherpa)
        self.asr_sherpa_encoder_label.setVisible(is_sherpa)
        self.asr_sherpa_encoder_row.setVisible(is_sherpa)
        self.asr_sherpa_decoder_label.setVisible(is_sherpa)
        self.asr_sherpa_decoder_row.setVisible(is_sherpa)
        self.asr_sherpa_joiner_label.setVisible(is_sherpa)
        self.asr_sherpa_joiner_row.setVisible(is_sherpa)
        self.asr_qwen_host_label.setVisible(is_qwen)
        self.asr_qwen_host_edit.setVisible(is_qwen)
        self.asr_qwen_port_label.setVisible(is_qwen)
        self.asr_qwen_port_edit.setVisible(is_qwen)
        self.asr_external_host_label.setVisible(is_external)
        self.asr_external_host_edit.setVisible(is_external)
        self.asr_external_port_label.setVisible(is_external)
        self.asr_external_port_edit.setVisible(is_external)
        self.asr_external_audio_host_label.setVisible(is_external)
        self.asr_external_audio_host_edit.setVisible(is_external)
        self.asr_external_audio_port_label.setVisible(is_external)
        self.asr_external_audio_port_edit.setVisible(is_external)
        self.asr_external_start_script_label.setVisible(is_external)
        self.asr_external_start_script_edit.setVisible(is_external)
        self.asr_external_start_script_browse_btn.setVisible(is_external)
        self.asr_qwen_model_label.setVisible(is_qwen)
        self.asr_qwen_model_edit.setVisible(is_qwen)
        self.asr_qwen_custom_model_path_label.setVisible(is_qwen)
        self.asr_qwen_custom_model_path_edit.setVisible(is_qwen)
        self.asr_qwen_custom_model_path_browse_btn.setVisible(is_qwen)
        self.asr_qwen_gpu_util_label.setVisible(is_qwen)
        self.asr_qwen_gpu_util_edit.setVisible(is_qwen)
        self.asr_qwen_max_model_len_label.setVisible(is_qwen)
        self.asr_qwen_max_model_len_edit.setVisible(is_qwen)
        self.asr_qwen_chunk_sec_label.setVisible(is_qwen)
        self.asr_qwen_chunk_sec_edit.setVisible(is_qwen)
        self.asr_qwen_timeout_sec_label.setVisible(is_qwen)
        self.asr_qwen_timeout_sec_edit.setVisible(is_qwen)
        self.asr_qwen_ws_path_label.setVisible(is_qwen)
        self.asr_qwen_ws_path_edit.setVisible(is_qwen)
        self.asr_qwen_session_timeout_label.setVisible(is_qwen)
        self.asr_qwen_session_timeout_edit.setVisible(is_qwen)
        self.asr_qwen_no_text_flush_sec_label.setVisible(is_qwen)
        self.asr_qwen_no_text_flush_sec_edit.setVisible(is_qwen)
        self.asr_qwen_min_return_audio_sec_label.setVisible(is_qwen)
        self.asr_qwen_min_return_audio_sec_edit.setVisible(is_qwen)
        self.asr_qwen_finish_on_vad_end_label.setVisible(is_qwen)
        self.asr_qwen_finish_on_vad_end_check.setVisible(is_qwen)
        self.asr_qwen_stop_container_label.setVisible(is_qwen)
        self.asr_qwen_stop_container_check.setVisible(is_qwen)
        self.asr_qwen_enforce_eager_label.setVisible(is_qwen)
        self.asr_qwen_enforce_eager_check.setVisible(is_qwen)
        self.asr_nemotron_model_label.setVisible(is_nemotron)
        self.asr_nemotron_model_edit.setVisible(is_nemotron)
        self.asr_nemotron_chunk_label.setVisible(is_nemotron)
        self.asr_nemotron_chunk_combo.setVisible(is_nemotron)
        self.asr_nemotron_finalize_label.setVisible(is_nemotron)
        self.asr_nemotron_finalize_combo.setVisible(is_nemotron)
        self.asr_nemotron_window_sec_label.setVisible(is_nemotron)
        self.asr_nemotron_window_sec_edit.setVisible(is_nemotron)
        self.asr_nemotron_max_symbols_label.setVisible(is_nemotron)
        self.asr_nemotron_max_symbols_edit.setVisible(is_nemotron)
        self.asr_nemotron_lang_label.setVisible(is_nemotron)
        self.asr_nemotron_lang_edit.setVisible(is_nemotron)
        self.asr_nemotron_device_label.setVisible(is_nemotron)
        self.asr_nemotron_device_combo.setVisible(is_nemotron)
        self.asr_nemotron_python_label.setVisible(is_nemotron)
        self.asr_nemotron_python_row.setVisible(is_nemotron)
        self.asr_chirp_project_id_label.setVisible(is_chirp)
        self.asr_chirp_project_id_edit.setVisible(is_chirp)
        self.asr_chirp_location_label.setVisible(is_chirp)
        self.asr_chirp_location_edit.setVisible(is_chirp)
        self.asr_chirp_recognizer_label.setVisible(is_chirp)
        self.asr_chirp_recognizer_edit.setVisible(is_chirp)
        self.asr_chirp_model_label.setVisible(is_chirp)
        self.asr_chirp_model_edit.setVisible(is_chirp)
        self._save_app_config()

    def _on_asr_compare_mode_changed(self, _state: int | None = None) -> None:
        enabled = bool(self.asr_compare_mode_check.isChecked())
        self.asr_engine_combo.setEnabled(not enabled)
        self.asr_engine_combo_settings.setEnabled(not enabled)
        self.asr_compare_panel.setVisible(enabled)
        self.asr_runtime_result_label.setVisible(not enabled)
        self.asr_result_edit.setVisible(not enabled)
        # QFormLayout環境差異でhidden行の隙間が残る場合に備えて高さも明示調整する。
        self.asr_compare_panel.setMaximumHeight(16777215 if enabled else 0)

    def _update_asr_compare_result_labels(self, _text: str | None = None) -> None:
        top_engine = self.asr_compare_top_engine_combo.currentText().strip() or "top"
        bottom_engine = self.asr_compare_bottom_engine_combo.currentText().strip() or "bottom"
        self.asr_compare_top_result_label.setText(f"結果（{top_engine}）")
        self.asr_compare_bottom_result_label.setText(f"結果（{bottom_engine}）")

    def _begin_asr_starting(self, status_text: str) -> None:
        self._asr_starting = True
        self._asr_ready = False
        self._asr_stop_requested_during_start = False
        self._set_module_panel_state("asr", "starting")
        self._set_asr_controls_state()
        self.status_label.setText(status_text)
        self._open_console_window()

    def _launch_asr_worker(self, asr_config: dict, worker_script: Path | None = None) -> None:
        target_script = worker_script or self._asr_worker_script
        launch_config = dict(asr_config)
        is_compare = (target_script == self._asr_compare_worker_script)
        if not is_compare:
            launch_config.pop("compare_mode_enabled", None)
            launch_config.pop("compare_top_engine", None)
            launch_config.pop("compare_bottom_engine", None)
        module = "asr_compare" if is_compare else "asr"

        def _start_via_orch() -> None:
            try:
                from ..dialog.settings import asr_worker_ready_timeout_sec
                timeout_sec = asr_worker_ready_timeout_sec(launch_config)
                self._start_module_via_orchestrator(module, launch_config, timeout_sec=timeout_sec)
            except Exception as ex:
                self._asr_starting = False
                self.asr_start_finished_signal.emit(False, str(ex))

        threading.Thread(target=_start_via_orch, daemon=True, name="asr-start-orch").start()

    def _maai_test_line_float(self, edit, default: float, lo: float, hi: float) -> float:
        if edit is None:
            return default
        try:
            v = float(edit.text().strip())
            if v != v:
                return default
            return max(lo, min(hi, v))
        except Exception:
            return default

    def _maai_nod_para_threshold_alias_kwargs(self) -> dict[str, object]:
        d = self._maai_nod_threshold_defaults
        if not isinstance(d, dict):
            return {
                "count_thresholds_ui": None,
                "swing_threshold_ui": None,
                "count_thresholds_default": None,
                "swing_threshold_default": None,
            }
        try:
            c_def = {
                "t0": float(d["count"]["t0"]),
                "t1": float(d["count"]["t1"]),
                "t2": float(d["count"]["t2"]),
            }
            s_def = float(d["swing"])
        except (TypeError, ValueError, KeyError):
            return {
                "count_thresholds_ui": None,
                "swing_threshold_ui": None,
                "count_thresholds_default": None,
                "swing_threshold_default": None,
            }
        t0e = getattr(self, "maai_test_nod_count_t0_edit", None)
        t1e = getattr(self, "maai_test_nod_count_t1_edit", None)
        t2e = getattr(self, "maai_test_nod_count_t2_edit", None)
        swe = getattr(self, "maai_test_nod_swing_threshold_edit", None)
        if t0e is None or t1e is None or t2e is None or swe is None:
            return {
                "count_thresholds_ui": None,
                "swing_threshold_ui": None,
                "count_thresholds_default": c_def,
                "swing_threshold_default": s_def,
            }
        try:
            c_ui = {
                "t0": float(t0e.text().strip()),
                "t1": float(t1e.text().strip()),
                "t2": float(t2e.text().strip()),
            }
            s_ui = float(swe.text().strip())
        except (TypeError, ValueError):
            return {
                "count_thresholds_ui": None,
                "swing_threshold_ui": None,
                "count_thresholds_default": c_def,
                "swing_threshold_default": s_def,
            }
        return {
            "count_thresholds_ui": c_ui,
            "swing_threshold_ui": s_ui,
            "count_thresholds_default": c_def,
            "swing_threshold_default": s_def,
        }

    def _apply_maai_nod_para_prediction_aliases_with_ui(self, result: dict) -> None:
        apply_maai_nod_para_prediction_aliases(result, **self._maai_nod_para_threshold_alias_kwargs())

    def _update_maai_nod_threshold_default_labels(self) -> None:
        """MaAI テストの nod 閾値行に、ワーカー ready で得た既定値を横表示する。"""
        placeholder = "既定: —"
        d = self._maai_nod_threshold_defaults if isinstance(self._maai_nod_threshold_defaults, dict) else None
        for key, attr in (
            ("t0", "maai_test_nod_count_t0_default_label"),
            ("t1", "maai_test_nod_count_t1_default_label"),
            ("t2", "maai_test_nod_count_t2_default_label"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is None:
                continue
            if d is not None:
                try:
                    v = float(d["count"][key])
                    lbl.setText(f"既定: {v}")
                except (KeyError, TypeError, ValueError):
                    lbl.setText(placeholder)
            else:
                lbl.setText(placeholder)
        sl = getattr(self, "maai_test_nod_swing_default_label", None)
        if sl is None:
            return
        if d is not None:
            try:
                sl.setText(f"既定: {float(d['swing'])}")
            except (KeyError, TypeError, ValueError):
                sl.setText(placeholder)
        else:
            sl.setText(placeholder)

    def _build_maai_worker_config(self) -> dict[str, object]:
        # `vds.build_maai_worker_config` に一本化（Phase 1-2）。
        return vds.build_maai_worker_config(self._widget_config_snapshot())

    def _get_maai_turn_vap_frame_rate_float(self) -> float:
        edit = getattr(self, "maai_test_turn_vap_frame_rate_edit", None)
        try:
            v = float(edit.text().strip()) if edit is not None else 20.0
        except Exception:
            v = 20.0
        if v != v:  # NaN
            v = 20.0
        return float(max(0.1, min(500.0, v)))

    def _maai_turn_vap_mode(self) -> str:
        edit = getattr(self, "maai_test_turn_vap_mode_edit", None)
        s = edit.text().strip() if edit is not None else ""
        return s if s else "vap_mc"

    def _maai_turn_vap_lang(self) -> str:
        edit = getattr(self, "maai_test_turn_vap_lang_edit", None)
        s = edit.text().strip() if edit is not None else ""
        return s if s else "jp_kyoto"

    def _maai_unified_device_str(self) -> str:
        """MaAI 系に渡すデバイス文字列。**ここで torch は読まない**。

        asr_vap_device / maai_device / maai_vap_device の出所。以前ここで
        `torch.cuda.is_available()` を見ていたが、それは GUI プロセスに torch を
        ロードさせるため、エンジン管理の「適用」が

            failed to rename ... torch\\lib\\c10.dll: アクセスが拒否されました

        で落ちるようになった（Windows はロード済み DLL を置き換えられない。torch は
        cpu / cu128 の排他 extra なので、適用のたびに入れ替わる）。

        CUDA が実際に使えるかの最終判定は ASR ワーカー（asr/service.py）が行う。
        あちらは maai の load_vap_model を呼ぶ側で、torch がもともと読まれている。
        ここは環境構築時の選択（.device_extra）だけを見て、CPU 構成なら cpu を返す。
        """
        combo = getattr(self, "maai_device_combo", None)
        wanted = "cuda"
        if combo is not None:
            wanted = "cuda" if str(combo.currentText()).strip().lower() == "cuda" else "cpu"
        if wanted != "cuda":
            return "cpu"
        try:
            from .. import engine_manager

            if engine_manager.device_extra() == "cpu":
                return "cpu"
        except Exception:  # noqa: BLE001
            pass
        return "cuda"

    def _maai_turn_vap_model_type(self) -> str:
        edit = getattr(self, "maai_test_turn_vap_model_type_edit", None)
        if edit is None:
            return "normal"
        s = (edit.currentText() if hasattr(edit, "currentText") else edit.text()).strip()
        return s if s else "normal"

    def _build_audio_input_worker_config(self) -> dict[str, object]:
        # `vds.build_audio_input_worker_config` に一本化（Phase 1-2）。
        # AEC は GUI にチェックボックスが 1 個しかなく、保存時に
        # `audio_input_enable_aec` / `audio_input_wav_enable_aec` の両方へ同じ値を
        # 書いているため、vds 側の優先順は結果に影響しない。
        built = vds.build_audio_input_worker_config(
            self._widget_config_snapshot(),
            force_browser_mic=self._remote_backend_enabled,
        )
        if self._remote_backend_enabled and built.get("input_mode") == "wav":
            mapped = self._container_path_for_backend_value(
                str(built.get("wav_path", "") or "")
            )
            if not mapped:
                raise RuntimeError(
                    "WAV 入力ファイルが Docker の bind mount 外です。"
                    "INPUT_WAV_HOST またはリポジトリ配下のファイルを指定してください。"
                )
            built["wav_path"] = mapped
        return built

    def _refresh_audio_input_device_list(self) -> None:
        combo = getattr(self, "audio_input_mic_device_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        try:
            prev_data = combo.currentData()
            combo.clear()
            combo.addItem("default", -1)
            if sd is not None:
                try:
                    devs = sd.query_devices()
                    for idx, dev in enumerate(devs):
                        if int(dev.get("max_input_channels", 0) or 0) > 0:
                            name = str(dev.get("name", f"device-{idx}"))
                            combo.addItem(f"{idx}: {name}", int(idx))
                except Exception:
                    pass
            if prev_data is not None:
                i = combo.findData(prev_data)
                if i >= 0:
                    combo.setCurrentIndex(i)
        finally:
            combo.blockSignals(False)

    def _is_audio_input_wav_local_playback_enabled(self) -> bool:
        cb = getattr(self, "audio_input_wav_local_playback_check", None)
        if cb is None:
            return False
        try:
            return bool(cb.isChecked())
        except Exception:
            return False


    def _audio_input_mode_is_wav(self) -> bool:
        combo = getattr(self, "audio_input_mode_combo", None)
        if combo is None:
            return False
        return combo.currentText().strip().lower() == "wav"

    def _sync_wav_module_panel(self) -> None:
        widgets = getattr(self, "module_status_wav_widgets", None)
        if not widgets:
            return
        visible = self._audio_input_mode_is_wav()
        for w in widgets:
            w.setVisible(visible)
        if not visible:
            return
        playing = not self._audio_input_wav_paused
        busy = self._audio_input_wav_command_inflight
        start = getattr(self, "module_status_wav_start_btn", None)
        stop = getattr(self, "module_status_wav_stop_btn", None)
        state_lbl = getattr(self, "module_status_wav_state_label", None)
        lamp = getattr(self, "module_status_wav_lamp", None)
        summary = getattr(self, "module_status_wav_summary_label", None)
        if state_lbl is not None:
            state_lbl.setText("再生中" if playing else "停止")
        if lamp is not None:
            lamp.setStyleSheet(lamp_style(STATE_COLORS["running" if playing else "stopped"]))
        if start is not None:
            start.setEnabled((not playing) and (not busy))
        if stop is not None:
            stop.setEnabled(playing and (not busy))
        if summary is not None:
            path_edit = getattr(self, "audio_input_wav_path_edit", None)
            path = path_edit.text().strip() if path_edit is not None else ""
            name = Path(path).name if path else "-"
            summary.setText(name)
            summary.setToolTip(path or "")

    def _sync_audio_input_wav_transport_button(self) -> None:
        play_pause_btn = getattr(self, "audio_input_wav_play_pause_btn", None)
        if play_pause_btn is not None:
            play_pause_btn.setText("再生" if self._audio_input_wav_paused else "一時停止")
        self._set_audio_input_wav_transport_busy(self._audio_input_wav_command_inflight)

    def _set_audio_input_wav_transport_busy(self, busy: bool) -> None:
        self._audio_input_wav_command_inflight = bool(busy)
        play_pause_btn = getattr(self, "audio_input_wav_play_pause_btn", None)
        end_btn = getattr(self, "audio_input_wav_end_btn", None)
        enabled = not bool(busy)
        if play_pause_btn is not None:
            play_pause_btn.setEnabled(enabled)
        if end_btn is not None:
            end_btn.setEnabled(enabled)
        self._sync_wav_module_panel()

    def _audio_input_wav_panel_play(self) -> None:
        if not self._audio_input_wav_paused:
            return
        self._audio_input_wav_play_pause_toggle()

    def _audio_input_wav_panel_stop(self) -> None:
        if self._audio_input_wav_paused:
            return
        self._audio_input_wav_play_pause_toggle()

    def _audio_input_wav_play_pause_toggle(self) -> None:
        if self._audio_input_wav_command_inflight:
            return
        if not self._audio_input_worker_ready_event.is_set():
            self._set_audio_input_wav_transport_busy(True)
            try:
                self._ensure_audio_input_worker()
            except Exception as ex:
                self._set_audio_input_wav_transport_busy(False)
                self.process_output_signal.emit(f"[AudioInputWorker] 起動失敗: {ex}")
                return
            self._send_dialog_orchestrator_command({"type": "audio_input_send_command", "command": "wav_play"})
            self.process_output_signal.emit("[AudioInputWorker] wav 再生要求")
            return
        self._set_audio_input_wav_transport_busy(True)
        if self._audio_input_wav_paused:
            self._send_dialog_orchestrator_command({"type": "audio_input_send_command", "command": "wav_play"})
            self.process_output_signal.emit("[AudioInputWorker] wav 再生要求")
        else:
            self._send_dialog_orchestrator_command({"type": "audio_input_send_command", "command": "wav_stop"})
            self.process_output_signal.emit("[AudioInputWorker] wav 一時停止要求")

    def _audio_input_wav_end(self) -> None:
        if self._audio_input_wav_command_inflight:
            return
        if not self._audio_input_worker_ready_event.is_set():
            return
        self._set_audio_input_wav_transport_busy(True)
        self._send_dialog_orchestrator_command({"type": "audio_input_send_command", "command": "wav_end"})
        self.process_output_signal.emit("[AudioInputWorker] wav 終了要求")

    def _ensure_audio_input_worker(self) -> None:
        """Orchestrator 側 `ensure_audio_input` に委ねる（設定 JSON が同一ならプロセスを取り直さない）。"""
        cfg = self._build_audio_input_worker_config()
        self._start_module_via_orchestrator("audio_input", cfg, timeout_sec=30.0)
        if not cfg.get("start_paused", False):
            self._audio_input_processing_event.set()
        self._sync_remote_mic_capture()

    def _ensure_maai_worker(self, *, check_backend_conflict: bool = True) -> None:
        if self._maai_ready:
            return
        maai_config = self._build_maai_worker_config()
        has_any_model = (
            maai_config.get("mode") in {"bc", "nod_para", "both"}
            or bool(maai_config.get("turn_vap_enabled"))
        )
        if not has_any_model:
            raise RuntimeError("有効な MaAI モデルがありません。少なくとも1つのモデルを有効にしてください。")
        if check_backend_conflict:
            turn_vap_on = bool(maai_config.get("turn_vap_enabled"))
            vad_be = self.asr_vad_backend_combo.currentText().strip().lower() if hasattr(self, "asr_vad_backend_combo") else ""
            turn_be = self.asr_turn_backend_combo.currentText().strip().lower() if hasattr(self, "asr_turn_backend_combo") else ""
            conflicts: list[str] = []
            if vad_be == "maai" and not turn_vap_on:
                conflicts.append("VAD バックエンドが「maai」ですが、ターンテイキング VAP モデルが無効です。")
            if turn_be == "maai" and not turn_vap_on:
                conflicts.append("ターンテイキングバックエンドが「maai」ですが、ターンテイキング VAP モデルが無効です。")
            if conflicts:
                raise RuntimeError(
                    "\n".join(conflicts)
                    + "\nターンテイキング VAP を有効にするか、バックエンド設定を変更してください。"
                )
        self._maai_ready = False
        self._maai_starting = True
        # Supervisor 側 ensure_maai の ready timeout (180s) 以上待つ（先に Controller 側が
        # タイムアウトすると、ロード完了直前のワーカーを起動失敗と誤判定する）。
        wait_sec = 300.0 if bool(maai_config.get("turn_vap_enabled")) else 200.0
        self._start_module_via_orchestrator("maai", maai_config, timeout_sec=wait_sec)

    def _effective_qwen_weight_path(self) -> str:
        custom = self.asr_qwen_custom_model_path_edit.text().strip()
        if custom:
            if self._remote_backend_enabled:
                host_path = self._host_path_for_backend_value(custom)
                if host_path is not None:
                    custom = str(host_path)
            return self._host_path_to_qwen_container_model_path(custom)
        return self._read_input_text(self.asr_qwen_model_edit) or "Qwen/Qwen3-ASR-1.7B"

    def _host_path_to_qwen_container_model_path(self, path_str: str) -> str:
        from ..asr import qwen_docker

        return qwen_docker.container_model_path(path_str, _paths.REPO_ROOT)

    @staticmethod
    def _derive_qwen_served_model_name(model_ref: str) -> str:
        from ..asr import qwen_docker

        return qwen_docker.served_model_name(model_ref)


    def _start_external_asr_script(self, script_path: str) -> None:
        running = self._external_asr_start_proc
        if running is not None and running.poll() is None:
            self._log_stream("External ASR 起動ファイルは既に実行中です。")
            return
        mapped_target = self._host_path_for_backend_value(script_path)
        target = (mapped_target or Path(script_path)).expanduser()
        if not target.exists():
            raise FileNotFoundError(f"外部ASR起動ファイルが見つかりません：{target}")
        suffix = target.suffix.lower()
        if suffix == ".ps1":
            cmd = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(target),
            ]
        elif suffix in {".bat", ".cmd"}:
            cmd = ["cmd.exe", "/c", str(target)]
        else:
            cmd = [str(target)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(target.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=self._preferred_process_encoding(),
            errors="replace",
            bufsize=1,
        )
        self._external_asr_start_proc = proc
        self._attach_process_console_output("External-ASR-Start", proc)
        self._log_stream(f"External ASR 起動ファイル実行: {target}")

        def _watch() -> None:
            rc = proc.wait()
            if self._external_asr_start_proc is proc:
                self._external_asr_start_proc = None
            if rc != 0:
                self.process_output_signal.emit(
                    f"[External-ASR-Start][WARN] 起動ファイルが終了しました (exit={rc})"
                )

        threading.Thread(target=_watch, daemon=True, name="external-asr-start-watch").start()

    def _stop_external_asr_script(self) -> None:
        proc = self._external_asr_start_proc
        self._external_asr_start_proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                cp = subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    encoding=self._preferred_process_encoding(),
                    errors="replace",
                    check=False,
                )
                output = (cp.stdout or "").strip()
                err = (cp.stderr or "").strip()
                if output:
                    self.process_output_signal.emit(output)
                if err:
                    self.process_output_signal.emit(err)
                if cp.returncode != 0:
                    # taskkillに失敗した場合は通常終了にフォールバック
                    proc.terminate()
                    proc.wait(timeout=2.0)
            else:
                proc.terminate()
                proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._log_stream("External ASR 起動ファイルプロセスを停止")

    @staticmethod
    def _preferred_process_encoding() -> str:
        # Windows の外部プロセス出力は cp932 のことが多く、utf-8固定だと文字化けしやすい。
        if os.name == "nt":
            return "cp932"
        return locale.getpreferredencoding(False) or "utf-8"


    def launch_asr(self, *, start_audio_input_after_ready: bool = True) -> None:
        self._start_audio_input_after_asr_ready = start_audio_input_after_ready
        self._ensure_dialog_orchestrator_worker()
        if self._asr_starting:
            self.asr_log("ASR は起動処理中です。")
            return
        if self._asr_ready:
            if start_audio_input_after_ready:
                try:
                    self._ensure_audio_input_worker()
                except Exception as ex:
                    QMessageBox.warning(self, "音声入力の起動エラー", str(ex))
                    self.status_label.setText("状態：音声入力の起動失敗")
                    return
            engine = self.asr_engine_combo.currentText().strip() or "sherpa-onnx"
            compare_enabled = bool(self.asr_compare_mode_check.isChecked())
            compare_top = self.asr_compare_top_engine_combo.currentText().strip() or "sherpa-onnx"
            compare_bottom = self.asr_compare_bottom_engine_combo.currentText().strip() or "faster-whisper"
            should_follow_qwen_logs = (engine == "qwen3-asr") or (
                compare_enabled and ("qwen3-asr" in {compare_top, compare_bottom})
            )
            if should_follow_qwen_logs:
                self._open_console_window()
                self._start_qwen_asr_container_logs()
            self.asr_log("ASR は既に起動済みです。")
            return
        try:
            # 不正な入力値はここで知らせる（値そのものは下の _build_asr_config_dict が読む）。
            for edit, label, lo in (
                (self.asr_vad_on_threshold_edit, "VAD開始しきい値", 0.0),
                (self.asr_vad_on_sec_edit, "VAD開始継続時間（秒）", 0.0),
                (self.asr_vad_off_threshold_edit, "VAD終了しきい値", 0.0),
                (self.asr_vad_off_sec_edit, "VAD終了継続時間（秒）", 0.0),
                (self.asr_min_segment_sec_edit, "最小セグメント長（秒）", 0.0),
                (self.asr_pre_roll_buffer_sec_edit, "プリロールバッファー（秒）", 0.0),
                (self.asr_nonstream_pseudo_chunk_sec_edit, "疑似ストリーミングチャンク（秒）", 0.0),
                (self.asr_smart_turn_threshold_edit, "SmartTurnしきい値", 0.0),
                (self.asr_qwen_timeout_sec_edit, "Qwenタイムアウト（秒）", 1.0),
                (self.asr_qwen_no_text_flush_sec_edit, "Qwen無応答確定時間（秒）", 0.0),
                (self.asr_qwen_min_return_audio_sec_edit, "Qwen最小返却音声長（秒）", 0.0),
                (self.asr_nemotron_window_sec_edit, "Nemotron音声窓（秒）", 1.0),
                (self.asr_nemotron_max_symbols_edit, "Nemotron最大シンボル数", 1.0),
                (self.asr_external_port_edit, "外部ASRポート", 1.0),
                (self.asr_external_audio_port_edit, "外部ASR音声ポート", 1.0),
            ):
                self._parse_float(edit.text(), label, lo)
            self._get_nemotron_chunk_size_ms()
            # Qwen3-ASR サーバーの起動に使う値。
            qwen_port = int(self._parse_float(self.asr_qwen_port_edit.text(), "Qwenポート", 1.0))
            qwen_chunk_sec = self._parse_float(self.asr_qwen_chunk_sec_edit.text(), "Qwenチャンク長（秒）", 0.0)
            qwen_session_timeout_sec = self._parse_float(
                self.asr_qwen_session_timeout_edit.text(), "Qwen session timeout sec", 5.0
            )
        except ValueError as ex:
            QMessageBox.warning(self, "ASR設定エラー", str(ex))
            return

        engine = self.asr_engine_combo.currentText().strip() or "sherpa-onnx"
        qwen_model = self._effective_qwen_weight_path()
        qwen_gpu_util = self.asr_qwen_gpu_util_edit.text().strip() or "0.8"
        qwen_max_model_len = self.asr_qwen_max_model_len_edit.text().strip() or "4096"
        qwen_enforce_eager = self.asr_qwen_enforce_eager_check.isChecked()
        external_asr_start_script_path = self.asr_external_start_script_edit.text().strip()
        compare_mode_enabled = self.asr_compare_mode_check.isChecked()
        compare_top_engine = self.asr_compare_top_engine_combo.currentText().strip() or "sherpa-onnx"
        compare_bottom_engine = self.asr_compare_bottom_engine_combo.currentText().strip() or "faster-whisper"
        compare_has_external = (compare_top_engine == "external-asr-tcp") or (compare_bottom_engine == "external-asr-tcp")
        compare_has_qwen = (compare_top_engine == "qwen3-asr") or (compare_bottom_engine == "qwen3-asr")

        # 対話・実験の開始時と同じ関数で組み立てる。別々に組み立てると中身が食い違い、
        # 開始時の ensure_asr が「設定が変わった」と判断して起動済みの ASR を起動し直していた。
        asr_config = self._build_asr_config_dict()
        compare_config = dict(asr_config)
        if compare_mode_enabled:
            compare_config["compare_mode_enabled"] = True
            compare_config["compare_top_engine"] = compare_top_engine
            compare_config["compare_bottom_engine"] = compare_bottom_engine
            # 同一エンジン比較は不可。状態コミット(_last_asr_config 等)/プロット表示の前に弾く(#18)。
            # 後段で弾くと、起動しないのに状態が上書きされプロットだけ開いてしまう。
            if compare_top_engine == compare_bottom_engine:
                QMessageBox.warning(self, "ASR設定エラー", "比較モードでは上段／下段で異なるエンジンを選択してください。")
                self.status_label.setText("状態：ASR起動失敗")
                return
        if self.asr_maai_gui_plot_check.isChecked():
            self._open_maai_plot_window()
        self._last_asr_config = dict(compare_config if compare_mode_enabled else asr_config)
        self._pending_asr_config = compare_config if compare_mode_enabled else asr_config
        if compare_mode_enabled:
            if compare_has_qwen:
                self._begin_asr_starting("状態：Qwen3-ASRサーバー起動中…")
                self._start_qwen_asr_server_async(
                    model=qwen_model,
                    gpu_memory_utilization=qwen_gpu_util,
                    max_model_len=qwen_max_model_len,
                    port=str(qwen_port),
                    session_timeout_sec=str(qwen_session_timeout_sec),
                    chunk_sec=str(qwen_chunk_sec),
                    enforce_eager=qwen_enforce_eager,
                )
                return
            if compare_has_external and external_asr_start_script_path:
                try:
                    self._start_external_asr_script(external_asr_start_script_path)
                except Exception as ex:
                    QMessageBox.warning(self, "外部ASRの起動エラー", str(ex))
                    self.status_label.setText("状態：ASR起動失敗")
                    return
            self._begin_asr_starting("状態：ASR比較モード起動中…")
            self._launch_asr_worker(compare_config, worker_script=self._asr_compare_worker_script)
            return
        if engine == "qwen3-asr":
            self._begin_asr_starting("状態：Qwen3-ASRサーバー起動中…")
            self._start_qwen_asr_server_async(
                model=qwen_model,
                gpu_memory_utilization=qwen_gpu_util,
                max_model_len=qwen_max_model_len,
                port=str(qwen_port),
                session_timeout_sec=str(qwen_session_timeout_sec),
                chunk_sec=str(qwen_chunk_sec),
                enforce_eager=qwen_enforce_eager,
            )
            return
        if engine == "external-asr-tcp" and external_asr_start_script_path:
            try:
                self._start_external_asr_script(external_asr_start_script_path)
            except Exception as ex:
                QMessageBox.warning(self, "外部ASRの起動エラー", str(ex))
                self.status_label.setText("状態：ASR起動失敗")
                return
        self._begin_asr_starting("状態：ASR起動中…")
        self._launch_asr_worker(asr_config)

    def stop_asr(self) -> None:
        engine_in_use = str((self._last_asr_config or {}).get("asr_engine", "")).strip()
        stop_qwen_with_down = (
            self.asr_qwen_stop_container_check.isChecked()
            and (engine_in_use == "qwen3-asr" or self.asr_engine_combo.currentText().strip() == "qwen3-asr")
        )
        if self._asr_starting:
            self._asr_stop_requested_during_start = True
            self.asr_log("ASR 起動中です。起動完了後に停止します。")
            return
        if not self._asr_ready:
            self._stop_qwen_asr_container_logs()
            self._stop_external_asr_script()
            if stop_qwen_with_down and not self._fast_shutdown:
                try:
                    self._stop_qwen_asr_server()
                    self.asr_result_signal.emit("[INFO] Qwen3-ASR コンテナを停止しました。")
                except Exception as ex:
                    QMessageBox.warning(self, "Qwen3-ASRの停止エラー", str(ex))
            return
        self._stop_module_via_orchestrator("asr")
        self._stop_module_via_orchestrator("asr_compare")
        self._pending_asr_config = None
        self._asr_ready = False
        self._stop_qwen_asr_container_logs()
        self._stop_external_asr_script()
        if stop_qwen_with_down and not self._fast_shutdown:
            try:
                self._stop_qwen_asr_server()
                self.asr_result_signal.emit("[INFO] Qwen3-ASR コンテナを停止しました。")
            except Exception as ex:
                QMessageBox.warning(self, "Qwen3-ASRの停止エラー", str(ex))
        self.status_label.setText("状態：ASR停止")
        self.asr_result_signal.emit("[INFO] ASR を停止しました。")
        self._log_stream("ASR 停止")  # ターミナルにも起動/停止を出す
        self._set_asr_controls_state()

    def _on_audio_input_ipc_event(self, payload: dict) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if not event_type:
            return
        if event_type == "ready":
            self._audio_input_worker_ready_event.set()
            self._audio_input_processing_event.clear()
            self._set_audio_input_wav_transport_busy(False)
            self.process_output_signal.emit("[AudioInput] worker ready")
            return
        if event_type == "processing_started":
            self._audio_input_processing_event.set()
            if hasattr(self, "audio_input_mode_combo"):
                if str(self.audio_input_mode_combo.currentText()).strip().lower() == "wav":
                    self._audio_input_wav_paused = False
                    self._sync_audio_input_wav_transport_button()
            self.process_output_signal.emit("[AudioInput] processing started")
            self._sync_remote_mic_capture()
            return
        if event_type == "processing_stopped":
            self._audio_input_processing_event.clear()
            self._stop_remote_mic_capture()
            if hasattr(self, "audio_input_mode_combo"):
                if str(self.audio_input_mode_combo.currentText()).strip().lower() == "wav":
                    self._audio_input_wav_paused = True
                    self._sync_audio_input_wav_transport_button()
            self.process_output_signal.emit("[AudioInput] processing stopped")
            return
        if event_type == "wav_command_done":
            cmd = str(payload.get("command", "")).strip().lower()
            if cmd == "wav_stop":
                self._audio_input_wav_paused = True
            elif cmd == "wav_play":
                self._audio_input_wav_paused = False
            elif cmd == "wav_end":
                self._audio_input_wav_paused = True
            self._set_audio_input_wav_transport_busy(False)
            self._sync_audio_input_wav_transport_button()
            return
        if event_type == "wav_playback_finished":
            self._audio_input_wav_paused = True
            self._set_audio_input_wav_transport_busy(False)
            self._sync_audio_input_wav_transport_button()
            reason = str(payload.get("reason", "")).strip().lower()
            self.process_output_signal.emit(f"[AudioInput] wav 再生終了 ({reason or 'done'})")
            QTimer.singleShot(0, self._on_wav_playback_finished_auto)
            return
        if event_type == "log":
            self.process_output_signal.emit(f"[AudioInput] {str(payload.get('message', ''))}")
            return
        if event_type == "error":
            self._set_audio_input_wav_transport_busy(False)
            self.process_output_signal.emit(f"[AudioInput][ERROR] {str(payload.get('message', ''))}")
            return
        if event_type == "fatal":
            self._audio_input_processing_event.clear()
            self._stop_remote_mic_capture()
            self._set_audio_input_wav_transport_busy(False)
            self.process_output_signal.emit(f"[AudioInput][FATAL] {str(payload.get('message', ''))}")
            return
        if event_type == "stopped":
            self._audio_input_worker_ready_event.clear()
            self._audio_input_processing_event.clear()
            self._stop_remote_mic_capture()
            self._audio_input_wav_paused = True
            self._set_audio_input_wav_transport_busy(False)
            self._sync_audio_input_wav_transport_button()
            self.process_output_signal.emit("[AudioInput] worker stopped")
            return
        if event_type in {"audio_frame", "system_audio_frame"}:
            # Audio routing is fully direct (AudioInput -> Orchestrator -> workers).
            # Controller no longer processes audio frames here.
            return

    def _on_wav_playback_finished_auto(self) -> None:
        combo = getattr(self, "audio_input_mode_combo", None)
        if combo is None:
            return
        if str(combo.currentText()).strip().lower() != "wav":
            return
        if self._voice_dialog_active or self._voice_dialog_starting:
            self._stop_voice_dialog()
            return

    def _on_asr_start_finished(self, ok: bool, error_message: str) -> None:
        self._asr_starting = False
        self._pending_asr_config = None
        start_audio_input_after_ready = self._start_audio_input_after_asr_ready
        self._start_audio_input_after_asr_ready = True
        if not ok:
            self._asr_ready = False
            self._set_module_panel_state("asr", "error", error_message)
            self._stop_module_via_orchestrator("asr")
            self._teardown_asr_ipc()
            self._stop_qwen_asr_container_logs()
            # 以前は `_asr_recovering` で「自動復旧失敗」と出し分けていたが、
            # そのフラグを True にする箇所が 1 つも無く到達不能だった（自動復旧機構は無い）。
            QMessageBox.warning(self, "ASR起動エラー", error_message)
            self.status_label.setText("状態：ASR起動失敗")
            self._set_asr_controls_state()
            return

        if start_audio_input_after_ready and not self._audio_input_worker_ready_event.is_set():
            try:
                self._ensure_audio_input_worker()
            except Exception as ex:
                self._asr_ready = False
                self._set_module_panel_state("asr", "error", f"音声入力の起動待機に失敗：{ex}")
                self._stop_module_via_orchestrator("asr")
                self._teardown_asr_ipc()
                self._stop_qwen_asr_container_logs()
                QMessageBox.warning(self, "ASR起動エラー", f"音声入力の起動待機に失敗しました：\n{ex}")
                self.status_label.setText("状態：ASR起動失敗")
                self._set_asr_controls_state()
                return

        engine = self.asr_engine_combo.currentText().strip() or "sherpa-onnx"
        if engine == "chirp3-asr":
            self.status_label.setText("状態：ASR起動")
        else:
            self.status_label.setText("状態：ASR起動中…")
        self._save_app_config(sync_orchestrator=False)
        self._open_console_window()
        self.asr_result_signal.emit("[INFO] ASR を開始しました。")
        self._log_stream(f"ASR 起動 (engine={engine})")  # ターミナルにも起動/停止を出す
        self._set_asr_controls_state()
        if self._asr_stop_requested_during_start:
            self._asr_stop_requested_during_start = False
            self.stop_asr()

    def asr_log(self, message: str) -> None:
        self._asr_log(message)

    def _set_default_paths(self) -> None:
        # Web 版も同じ探索（mmd/default_paths.py）を使う。
        from ..mmd.default_paths import detect_default_exec_path, detect_default_mdf_path

        exec_path = detect_default_exec_path()
        if exec_path:
            self.exec_edit.setText(exec_path)
        mdf_path = detect_default_mdf_path()
        if mdf_path and self._remote_backend_enabled:
            # Docker backend の設定ファイルにはコンテナから見えるパスだけを書く。
            # ホストのパスのまま欄に入れると、何か保存した時点で C:\ のパスが
            # controller_config.docker.json に入り、コンテナが MDF を読めなくなる。
            mapper = self._docker_path_mapper
            mdf_path = (mapper.host_to_container(mdf_path) if mapper is not None else None) or ""
        if mdf_path:
            self.mdf_edit.setText(mdf_path)
        self.load_mdf_settings()

    def _get_tts_input_port(self) -> int | None:
        try:
            port = int(self.tts_port_edit.text().strip())
            if not (1 <= port <= 65535):
                raise ValueError
            return port
        except ValueError:
            return None

    def _sync_tts_exe_paths_from_ui(self) -> None:
        try:
            self._sync_tts_exe_paths_to_orchestrator()
        except Exception:
            pass

    def _set_tts_latency(self, seconds: float, suffix: str = "") -> None:
        ms = max(0.0, seconds * 1000.0)
        text = f"{ms:.0f} ms"
        if suffix:
            text = f"{text} ({suffix})"
        self.tts_latency_value_label.setText(text)

    def _on_llm_backend_changed(self, backend: str | None = None) -> None:
        selected = (backend or self.llm_backend_combo.currentText() or "LM Studio").strip()
        previous_backend = self._last_llm_backend
        if previous_backend in self._llm_model_by_backend and previous_backend != selected:
            self._llm_model_by_backend[previous_backend] = self._read_llm_model_for_backend(previous_backend)
        self._last_llm_backend = selected
        is_lmstudio = selected == "LM Studio"
        is_openai = selected == "OpenAI"
        is_custom = selected == "Custom"
        if hasattr(self, "llm_backend_stack"):
            if is_lmstudio:
                self.llm_backend_stack.setCurrentIndex(0)
            elif is_openai:
                self.llm_backend_stack.setCurrentIndex(1)
            else:
                self.llm_backend_stack.setCurrentIndex(2)
            self.llm_backend_stack.updateGeometry()
        if hasattr(self, "llm_connection_box"):
            self.llm_connection_box.updateGeometry()
        target_model = self._llm_model_by_backend.get(selected, "").strip()
        if not target_model:
            if selected == "OpenAI":
                target_model = _LLM_OPENAI_DEFAULT_MODEL
            elif selected == "Custom":
                target_model = _LLM_CUSTOM_DEFAULT_MODEL
            else:
                target_model = self._read_llm_model_for_backend(selected)
        self._write_llm_model_for_backend(selected, target_model)
        if is_lmstudio:
            self._llm_model_by_backend["LM Studio"] = self._read_llm_model_for_backend("LM Studio")
            self._last_lmstudio_model = self._llm_model_by_backend["LM Studio"]
            if not self._is_loading_app_config:
                self.refresh_lmstudio_models(silent=True)
        elif is_openai:
            self._llm_model_by_backend["OpenAI"] = self._read_llm_model_for_backend("OpenAI")
            if not self._is_loading_app_config:
                self.refresh_openai_models(silent=True)
        elif is_custom:
            self._llm_model_by_backend["Custom"] = self._read_llm_model_for_backend("Custom")
        if not self._is_loading_app_config:
            self._schedule_llm_warmup()

    def _on_llm_model_text_changed(self, text: str) -> None:
        source = self.sender()
        if source is self.llm_model_lmstudio_combo:
            backend = "LM Studio"
        elif source is self.llm_model_openai_combo:
            backend = "OpenAI"
        elif source is self.llm_model_custom_combo:
            backend = "Custom"
        else:
            return
        self._llm_model_by_backend[backend] = str(text).strip()

    def _llm_model_widget_for_backend(self, backend: str):
        if backend == "LM Studio":
            return self.llm_model_lmstudio_combo
        if backend == "Custom":
            return self.llm_model_custom_combo
        return self.llm_model_openai_combo

    def _read_llm_model_for_backend(self, backend: str) -> str:
        return self._read_input_text(self._llm_model_widget_for_backend(backend)).strip()

    def _write_llm_model_for_backend(self, backend: str, value: str) -> None:
        self._write_input_text(self._llm_model_widget_for_backend(backend), value)

    def _active_llm_connection_settings(self) -> dict[str, str]:
        # `vds.active_llm_connection_settings` に一本化（Phase 1-2）。
        return vds.active_llm_connection_settings(self._widget_config_snapshot())

    def _llm_custom_request_option_fields(self) -> dict[str, object]:
        if (self.llm_backend_combo.currentText().strip() or "LM Studio") != "Custom":
            return {}
        if not hasattr(self, "llm_custom_send_temperature_check"):
            return {}
        try:
            temp = self._parse_llm_custom_temperature()
        except ValueError:
            temp = 0.7
        out: dict[str, object] = {
            "send_temperature": bool(self.llm_custom_send_temperature_check.isChecked()),
            "send_max_tokens": bool(self.llm_custom_send_max_tokens_check.isChecked()),
            "temperature": float(temp),
        }
        if hasattr(self, "llm_custom_extra_body_thinking_check"):
            out["custom_extra_body_thinking"] = bool(self.llm_custom_extra_body_thinking_check.isChecked())
        else:
            out["custom_extra_body_thinking"] = False
        return out

    def _resolve_openai_api_key(self, api_key: str, api_key_env: str) -> str:
        resolved_key = (api_key or "").strip()
        if (not resolved_key) and (api_key_env or "").strip():
            resolved_key = os.environ.get(api_key_env.strip(), "").strip()
        return resolved_key

    def _schedule_llm_warmup(self, *, force: bool = False) -> bool:
        backend = self.llm_backend_combo.currentText().strip() or "LM Studio"
        model = self._llm_model_by_backend.get(backend, "").strip() or self._read_llm_model_for_backend(backend)
        if not model:
            return False
        conn = self._active_llm_connection_settings()
        kw: dict[str, object] = {
            "backend": backend,
            "host": conn["host"],
            "model": model,
            "api_key": conn["api_key"],
            "api_key_env": conn["api_key_env"],
            "force": bool(force),
        }
        if backend == "Custom":
            opt = self._llm_custom_request_option_fields()
            kw["send_temperature"] = opt.get("send_temperature")
            kw["send_max_tokens"] = opt.get("send_max_tokens")
            kw["temperature"] = opt.get("temperature")
            kw["custom_extra_body_thinking"] = bool(opt.get("custom_extra_body_thinking", False))
        try:
            self._ensure_dialog_orchestrator_worker()
        except Exception as ex:
            self._log_stream(f"LLM warmup: Dialog Orchestrator 起動失敗: {ex}")
            return False
        payload: dict[str, object] = {"type": "start_llm_warmup", **kw}
        if not self._send_dialog_orchestrator_command_sync(payload):
            self._log_stream("LLM warmup: Dialog Orchestrator への送信に失敗しました。")
            return False
        return True

    def run_llm_warmup_now(self) -> None:
        self._set_module_panel_state("llm", "starting")
        self.llm_status_label.setText("状態：LLMウォームアップ実行中")
        if not self._schedule_llm_warmup(force=True):
            self._set_module_panel_state("llm", "error", "送信失敗")
            self.llm_status_label.setText("状態：LLMウォームアップ送信失敗")
        else:
            self.llm_status_label.setText("状態：LLMウォームアップをオーケストレーターに送信しました")

    def _on_llm_model_selection_committed(self) -> None:
        # このハンドラは LM Studio のコンボ専用。現在の LLM バックエンドが LM Studio でないときは
        # OpenAI/Custom のモデル辞書を LM Studio 表示値で上書きしない。
        if (self.llm_backend_combo.currentText().strip() or "LM Studio") != "LM Studio":
            return
        current_model = self._read_llm_model_for_backend("LM Studio")
        if not current_model:
            return
        self._llm_model_by_backend["LM Studio"] = current_model
        previous_model = self._last_lmstudio_model.strip()
        self._last_lmstudio_model = current_model
        if not previous_model or previous_model == current_model:
            return
        self.unload_lmstudio_non_current_models(silent=True, reason="model_switch", unload_all=False)
        self._schedule_llm_warmup()

    def _eject_llm_models_on_shutdown(self) -> None:
        """終了時に LM Studio のロード済みモデルを全部アンロードする。

        `unload_lmstudio_non_current_models` は使えない。あれは `_llm_admin_rpc` 経由で、
        `_ensure_dialog_orchestrator_worker()` が走るため**終了処理中に
        Orchestrator ワーカーを起動し直してしまう**。また daemon スレッドに投げるので
        プロセス終了で途中で消え、timeout も 120 秒ある。

        なので IPC を通さない同期版（`llm/admin_http.unload_all_loaded_models`）を
        GUI スレッドから直接呼ぶ。合計 4 秒で打ち切るので、LM Studio が起動していない
        場合（localhost なら接続拒否が即返る）も終了は待たされない。
        """
        try:
            host_edit = getattr(self, "llm_host_edit", None)
            host = (host_edit.text().strip() if host_edit is not None else "")                 or "http://127.0.0.1:1234"
            done, skipped, err = _llm_admin_http.unload_all_loaded_models(host, budget_sec=4.0)
            if done:
                self._log_stream(f"終了時アンロード: {len(done)} 件解放 ({', '.join(done)})")
            if skipped:
                self._log_stream(f"終了時アンロード: {len(skipped)} 件は解放できず ({err})")
        except Exception as ex:  # noqa: BLE001
            # 終了処理を止めない。
            self._log_stream(f"終了時アンロードに失敗: {ex}")

    def unload_lmstudio_non_current_models(
        self,
        _checked: bool = False,
        *,
        silent: bool = False,
        reason: str = "manual",
        unload_all: bool = False,
    ) -> None:
        if self._lmstudio_bulk_unload_in_progress:
            if not silent:
                QMessageBox.information(self, "情報", "LM Studioアンロード処理を実行中です。")
            return
        backend = self.llm_backend_combo.currentText().strip() or "LM Studio"
        if (not unload_all) and backend != "LM Studio":
            if not silent:
                QMessageBox.information(self, "情報", "LM Studioモードでのみアンロードできます。")
            return

        host = self.llm_host_edit.text().strip() or "http://127.0.0.1:1234"
        current_model = self._read_llm_model_for_backend("LM Studio")
        self._lmstudio_bulk_unload_in_progress = True
        self.llm_status_label.setText("状態：LM Studioモデルをアンロード中")
        threading.Thread(
            target=self._unload_lmstudio_models_worker,
            args=(host, current_model, bool(unload_all), bool(silent), str(reason)),
            daemon=True,
            name="llm-model-bulk-unload",
        ).start()

    def _unload_lmstudio_models_worker(
        self,
        host: str,
        current_model: str,
        unload_all: bool,
        silent: bool,
        reason: str,
    ) -> None:
        res = self._llm_admin_rpc(
            "lmstudio_bulk_unload",
            timeout_sec=120.0,
            host=host,
            current_model=current_model,
            unload_all=unload_all,
            silent=silent,
            reason=reason,
        )
        if res is None:
            self.llm_model_unload_finished_signal.emit(
                {
                    "ok": False,
                    "host": host,
                    "reason": reason,
                    "silent": silent,
                    "error": "Orchestrator への接続に失敗しました。",
                    "attempted": 0,
                    "success_count": 0,
                    "failed": [],
                }
            )
            return
        if str(res.get("kind", "")) != "lmstudio_unload":
            self.llm_model_unload_finished_signal.emit(
                {
                    "ok": False,
                    "host": host,
                    "reason": reason,
                    "silent": silent,
                    "error": "不正な応答 (kind)",
                    "attempted": 0,
                    "success_count": 0,
                    "failed": [],
                }
            )
            return
        failed_raw = res.get("failed")
        failed_list = failed_raw if isinstance(failed_raw, list) else []
        self.llm_model_unload_finished_signal.emit(
            {
                "ok": bool(res.get("ok", False)),
                "host": str(res.get("host", host)),
                "reason": reason,
                "silent": silent,
                "error": str(res.get("error", "") or ""),
                "attempted": int(res.get("attempted", 0) or 0),
                "success_count": int(res.get("success_count", 0) or 0),
                "failed": failed_list,
            }
        )

    def _on_lmstudio_model_unload_finished(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        self._lmstudio_bulk_unload_in_progress = False
        ok = bool(payload.get("ok", False))
        reason = str(payload.get("reason", "")).strip()
        silent = bool(payload.get("silent", False))
        error = str(payload.get("error", "")).strip()
        attempted = int(payload.get("attempted", 0) or 0)
        success_count = int(payload.get("success_count", 0) or 0)
        failed_raw = payload.get("failed")
        failed: list[dict[str, str]] = failed_raw if isinstance(failed_raw, list) else []

        if error:
            self.llm_status_label.setText("状態：LM Studioアンロード失敗")
            self._log_stream(f"LM Studioアンロード失敗: {error}")
            if not silent:
                QMessageBox.warning(self, "エラー", f"LM Studioアンロードに失敗しました。\n\n{error}")
            return

        if ok:
            self.llm_status_label.setText(f"状態：LM Studioアンロード完了（{success_count}/{attempted}件）")
            self._log_stream(f"LM Studioアンロード完了: reason={reason}, count={success_count}/{attempted}")
            if not silent:
                if attempted == 0:
                    QMessageBox.information(self, "情報", "アンロード対象のロード済みモデルはありませんでした。")
                    return
                QMessageBox.information(
                    self,
                    "情報",
                    f"LM Studioアンロード完了\n対象: {attempted}件\n成功: {success_count}件",
                )
            return

        self.llm_status_label.setText(f"状態：LM Studioアンロード一部失敗（{success_count}/{attempted}件）")
        for item in failed:
            model = str(item.get("model", "")).strip()
            model_error = str(item.get("error", "")).strip()
            self._log_stream(f"LM Studioモデルアンロード失敗 ({model}): {model_error}")
        if not silent:
            QMessageBox.warning(
                self,
                "エラー",
                f"LM Studioアンロードで一部失敗しました。\n成功: {success_count}/{attempted}",
            )

    def refresh_lmstudio_models(self, silent: bool = False) -> None:
        backend = self.llm_backend_combo.currentText().strip() or "LM Studio"
        if backend != "LM Studio":
            if not silent:
                QMessageBox.information(self, "情報", "LM Studioモードでのみモデル一覧を取得できます。")
            return
        host = self.llm_host_edit.text().strip() or "http://127.0.0.1:1234"
        self.llm_status_label.setText("状態：LM Studioモデル一覧を取得中")
        threading.Thread(
            target=self._fetch_lmstudio_models_worker,
            args=(host, bool(silent)),
            daemon=True,
            name="llm-models-fetch",
        ).start()

    def _fetch_lmstudio_models_worker(self, host: str, silent: bool) -> None:
        res = self._llm_admin_rpc("lmstudio_fetch_models", timeout_sec=30.0, host=host, silent=silent)
        if res is None:
            self.llm_models_fetched_signal.emit(
                {
                    "host": host,
                    "models": [],
                    "error": "Orchestrator への接続に失敗しました。",
                    "silent": silent,
                }
            )
            return
        if str(res.get("kind", "")) != "lmstudio_fetch":
            self.llm_models_fetched_signal.emit(
                {"host": host, "models": [], "error": "不正な応答 (kind)", "silent": silent}
            )
            return
        models_raw = res.get("models")
        models_list = models_raw if isinstance(models_raw, list) else []
        self.llm_models_fetched_signal.emit(
            {
                "host": str(res.get("host", host)),
                "models": models_list,
                "error": str(res.get("error", "") or ""),
                "silent": silent,
            }
        )

    def _on_lmstudio_models_fetched(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        host = str(payload.get("host", "")).strip()
        models_raw = payload.get("models")
        error = str(payload.get("error", "")).strip()
        silent = bool(payload.get("silent", False))
        current_host = self.llm_host_edit.text().strip() or "http://127.0.0.1:1234"
        if host.rstrip("/") != current_host.rstrip("/"):
            return
        current_model = self._read_llm_model_for_backend("LM Studio")
        models: list[str] = []
        if isinstance(models_raw, list):
            seen: set[str] = set()
            for item in models_raw:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    models.append(text)
        if models:
            self.llm_model_lmstudio_combo.blockSignals(True)
            self.llm_model_lmstudio_combo.clear()
            for name in models:
                self.llm_model_lmstudio_combo.addItem(name)
            if current_model and current_model not in models:
                self.llm_model_lmstudio_combo.insertItem(0, current_model)
            self._write_llm_model_for_backend("LM Studio", current_model or models[0])
            self.llm_model_lmstudio_combo.blockSignals(False)
            self._llm_model_by_backend["LM Studio"] = self._read_llm_model_for_backend("LM Studio")
            self._save_app_config()
            self.llm_status_label.setText(f"状態：LM Studioモデル取得完了（{len(models)}件）")
            if not self._last_lmstudio_model.strip():
                self._last_lmstudio_model = self._read_llm_model_for_backend("LM Studio")
            self._schedule_llm_warmup()
            return
        if error:
            self.llm_status_label.setText("状態：LM Studioモデル取得失敗")
            self._log_stream(f"LM Studioモデル取得失敗: {error}")
            if not silent:
                QMessageBox.warning(self, "エラー", f"LM Studioモデル一覧の取得に失敗しました。\n\n{error}")
            return
        self.llm_status_label.setText("状態：LM Studioモデルが見つかりません")
        if not silent:
            QMessageBox.information(self, "情報", "LM Studioモデル一覧は空でした。")

    def refresh_openai_models(self, silent: bool = False) -> None:
        backend = self.llm_backend_combo.currentText().strip() or "LM Studio"
        if backend != "OpenAI":
            if not silent:
                QMessageBox.information(self, "情報", "OpenAIモードでのみモデル一覧を取得できます。")
            return
        api_key = self.llm_api_key_edit.text().strip()
        api_key_env = self.llm_api_key_env_edit.text().strip()
        self.llm_status_label.setText("状態：OpenAIモデル一覧を取得中")
        threading.Thread(
            target=self._fetch_openai_models_worker,
            args=(api_key, api_key_env, bool(silent)),
            daemon=True,
            name="openai-models-fetch",
        ).start()

    def _fetch_openai_models_worker(self, api_key: str, api_key_env: str, silent: bool) -> None:
        try:
            resolved_key = self._resolve_openai_api_key(api_key, api_key_env)
        except Exception as ex:
            self.llm_openai_models_fetched_signal.emit({"models": [], "error": str(ex), "silent": silent})
            return
        if not resolved_key:
            self.llm_openai_models_fetched_signal.emit(
                {
                    "models": [],
                    "error": "OpenAI API Key を入力するか、API Key Env の環境変数を設定してください。",
                    "silent": silent,
                }
            )
            return
        res = self._llm_admin_rpc(
            "openai_fetch_models",
            timeout_sec=120.0,
            api_key=resolved_key,
            silent=silent,
        )
        if res is None:
            self.llm_openai_models_fetched_signal.emit(
                {
                    "models": [],
                    "error": "Orchestrator への接続に失敗しました。",
                    "silent": silent,
                }
            )
            return
        if str(res.get("kind", "")) != "openai_fetch":
            self.llm_openai_models_fetched_signal.emit(
                {"models": [], "error": "不正な応答 (kind)", "silent": silent}
            )
            return
        models_raw = res.get("models")
        models_list = models_raw if isinstance(models_raw, list) else []
        self.llm_openai_models_fetched_signal.emit(
            {
                "models": models_list,
                "error": str(res.get("error", "") or ""),
                "silent": silent,
            }
        )

    def _on_openai_models_fetched(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        models_raw = payload.get("models")
        error = str(payload.get("error", "")).strip()
        silent = bool(payload.get("silent", False))
        current_model = self._read_llm_model_for_backend("OpenAI")
        models: list[str] = []
        if isinstance(models_raw, list):
            seen: set[str] = set()
            for item in models_raw:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    models.append(text)
        if models:
            models.sort(key=self._openai_model_latency_sort_key)
        if models:
            self.llm_model_openai_combo.blockSignals(True)
            self.llm_model_openai_combo.clear()
            for name in models:
                self.llm_model_openai_combo.addItem(name)
            if current_model and current_model not in models:
                self.llm_model_openai_combo.insertItem(0, current_model)
            self._write_llm_model_for_backend("OpenAI", current_model or models[0])
            self.llm_model_openai_combo.blockSignals(False)
            self._llm_model_by_backend["OpenAI"] = self._read_llm_model_for_backend("OpenAI")
            self._save_app_config()
            self.llm_status_label.setText(f"状態：OpenAIモデル取得完了（{len(models)}件）")
            self._schedule_llm_warmup()
            return
        if error:
            self.llm_status_label.setText("状態：OpenAIモデル取得失敗")
            self._log_stream(f"OpenAIモデル取得失敗: {error}")
            if not silent:
                QMessageBox.warning(self, "エラー", f"OpenAIモデル一覧の取得に失敗しました。\n\n{error}")
            return
        self.llm_status_label.setText("状態：OpenAIモデルが見つかりません")
        if not silent:
            QMessageBox.information(self, "情報", "OpenAIモデル一覧は空でした。")

    @staticmethod
    def _openai_model_latency_sort_key(model_id: str) -> tuple[int, int, str]:
        name = (model_id or "").strip().lower()
        if not name:
            return (99, 99, "")

        # 低遅延寄りモデルを先頭にするための経験則スコア（小さいほど優先）。
        primary = 5
        if "gpt-5-nano" in name:
            primary = 0
        elif "gpt-4.1-nano" in name:
            primary = 0
        elif "gpt-4o-mini" in name:
            primary = 1
        elif "gpt-4.1-mini" in name:
            primary = 1
        elif "mini" in name:
            primary = 2
        elif "gpt-4o" in name or "gpt-4.1" in name:
            primary = 3
        elif "gpt-5" in name:
            primary = 4

        secondary = 0
        # 重め/長考寄りのモデルを後ろへ。
        if "o1" in name or "o3" in name or "o4" in name:
            secondary += 3
        if "preview" in name or "realtime" in name:
            secondary += 1
        if "audio" in name or "vision" in name:
            secondary += 1

        return (primary, secondary, name)

    def _parse_llm_context_len(self) -> int:
        raw = self.llm_context_len_edit.text().strip() or "10"
        try:
            value = int(raw)
        except ValueError as ex:
            raise ValueError("Context Len は整数で指定してください。") from ex
        if value < 0:
            raise ValueError("Context Len は 0 以上で指定してください。")
        return value

    def _parse_llm_max_tokens(self) -> int:
        raw = self.llm_max_tokens_edit.text().strip() or "128"
        try:
            value = int(raw)
        except ValueError as ex:
            raise ValueError("Max Tokens は整数で指定してください。") from ex
        if value < 1:
            raise ValueError("Max Tokens は 1 以上で指定してください。")
        return value

    def _parse_llm_custom_temperature(self) -> float:
        raw = self.llm_custom_temperature_edit.text().strip() if hasattr(self, "llm_custom_temperature_edit") else "0.7"
        raw = raw or "0.7"
        try:
            return float(raw)
        except ValueError as ex:
            raise ValueError("Customのtemperatureは数値で指定してください。") from ex

    def _trim_llm_messages(self, context_len: int) -> None:
        if not self._llm_messages:
            return
        system = self._llm_messages[0]
        if system.get("role") != "system":
            system = {"role": "system", "content": ""}
        keep = max(0, int(context_len))
        non_system = [m for m in self._llm_messages[1:] if m.get("role") in {"user", "assistant"}]
        if keep == 0:
            self._llm_messages = [system]
            return
        if len(non_system) > keep:
            non_system = non_system[-keep:]
        self._llm_messages = [system] + non_system

    def _ensure_llm_history(self, system_prompt: str, context_len: int) -> None:
        prompt = system_prompt.strip() or "あなたは簡潔で丁寧なアシスタントです。"
        if not self._llm_messages:
            self._llm_messages = [{"role": "system", "content": prompt}]
            return
        first = self._llm_messages[0]
        if first.get("role") != "system" or str(first.get("content", "")) != prompt:
            self._llm_messages = [{"role": "system", "content": prompt}]
            return
        self._trim_llm_messages(context_len)

    def _sync_llm_test_tts_input_streaming_widgets(self) -> None:
        chk = getattr(self, "llm_test_tts_input_streaming_check", None)
        if chk is None:
            return
        chk.setEnabled(bool(self.llm_tts_stream_check.isChecked()))

    def reset_llm_test_history(self) -> None:
        if self._llm_test_in_progress:
            QMessageBox.information(self, "情報", "LLMテスト実行中は履歴リセット/表示クリアできません。")
            return
        prompt = self.llm_system_prompt_edit.toPlainText().strip() or "あなたは簡潔で丁寧なアシスタントです。"
        # 履歴リセットは system 以外を必ず全削除し、表示も同時にクリアする（操作を1つにまとめた）。
        self._llm_messages = [{"role": "system", "content": prompt}]
        self.llm_chat_view.clear()
        self.llm_status_label.setText("状態：履歴をリセットしました")
        self._log_stream("LLMテスト履歴をリセット")

    def send_llm_test_chat(self) -> None:
        if self._llm_test_in_progress:
            QMessageBox.information(self, "情報", "LLMテストは実行中です。完了までお待ちください。")
            return
        user_text = self.llm_user_input_edit.text().strip()
        if not user_text:
            QMessageBox.warning(self, "エラー", "LLMへの入力テキストを入力してください。")
            return
        backend = self.llm_backend_combo.currentText().strip() or "LM Studio"
        model = self._llm_model_by_backend.get(backend, "").strip() or self._read_llm_model_for_backend(backend)
        if not model:
            QMessageBox.warning(self, "エラー", "モデルを入力してください。")
            return
        self._llm_model_by_backend[backend] = model
        try:
            context_len = self._parse_llm_context_len()
        except ValueError as ex:
            QMessageBox.warning(self, "エラー", str(ex))
            return
        try:
            max_tokens = self._parse_llm_max_tokens()
        except ValueError as ex:
            QMessageBox.warning(self, "エラー", str(ex))
            return
        if backend == "Custom":
            if not getattr(self, "llm_custom_base_url_edit", None) or not self.llm_custom_base_url_edit.text().strip():
                QMessageBox.warning(self, "エラー", "CustomのベースURLを入力してください。")
                return
            try:
                self._parse_llm_custom_temperature()
            except ValueError as ex:
                QMessageBox.warning(self, "エラー", str(ex))
                return

        system_prompt = self.llm_system_prompt_edit.toPlainText().strip()
        self._ensure_llm_history(system_prompt, context_len)
        self._llm_messages.append({"role": "user", "content": user_text})
        self._trim_llm_messages(context_len)
        request_messages = [dict(m) for m in self._llm_messages]

        llm_tts_enabled = bool(self.llm_tts_stream_check.isChecked())
        tts_engine = self.tts_engine_combo.currentText().strip() or "VOICEVOX"
        tts_stream_inputs: dict[str, object] = {}
        llm_split = self._resolve_llm_split_settings()
        if llm_tts_enabled:
            try:
                tts_stream_inputs = self._build_tts_inputs(
                    engine=tts_engine,
                    utterance="",
                    input_streaming_enabled=bool(self.llm_test_tts_input_streaming_check.isChecked()),
                    split_settings=llm_split,
                )
                if tts_engine == "TEMPLATE":
                    tts_stream_inputs["template"] = self._load_tts_template_dict()
            except Exception as ex:
                QMessageBox.warning(self, "エラー", f"LLM連携用TTS設定が不正です。\n\n{ex}")
                if self._llm_messages and self._llm_messages[-1].get("role") == "user":
                    self._llm_messages.pop()
                return

        llm_conn = self._active_llm_connection_settings()
        orch_payload: dict[str, object] = {
            "type": "start_llm_gui_test",
            "backend": backend,
            "model": model,
            "context_len": context_len,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": request_messages,
            "host": llm_conn["host"],
            "api_key": llm_conn["api_key"],
            "api_key_env": llm_conn["api_key_env"],
            "tts_stream_engine": tts_engine,
            "tts_stream_inputs": tts_stream_inputs,
            "voice_dialog_inject_ch2": True,
            "llm_custom": self._llm_custom_request_option_fields(),
        }

        try:
            with self._llm_ipc_lifecycle_lock:
                self._start_module_via_orchestrator(
                    "llm",
                    {},
                    timeout_sec=40.0,
                )
        except Exception as ex:
            if self._llm_messages and self._llm_messages[-1].get("role") == "user":
                self._llm_messages.pop()
            QMessageBox.warning(self, "エラー", f"LLMワーカーの起動に失敗しました。\n\n{ex}")
            return

        if llm_tts_enabled:
            try:
                self._ensure_tts_playback_workers()
            except Exception as ex:
                if self._llm_messages and self._llm_messages[-1].get("role") == "user":
                    self._llm_messages.pop()
                QMessageBox.warning(self, "エラー", f"TTS再生ワーカーの起動に失敗しました。\n\n{ex}")
                return

        if not self._send_dialog_orchestrator_command_sync(orch_payload):
            if self._llm_messages and self._llm_messages[-1].get("role") == "user":
                self._llm_messages.pop()
            QMessageBox.warning(self, "エラー", "対話オーケストレーターへのLLMテスト送信に失敗しました。")
            return

        self._llm_test_tts_confirmed_segments.clear()
        self._llm_test_streaming_assistant_full = ""
        self._llm_test_in_progress = True
        self.llm_chat_send_btn.setEnabled(False)
        self.llm_user_input_edit.setEnabled(False)
        self.llm_status_label.setText(f"状態：{backend}へ送信中")
        self.llm_user_input_edit.clear()
        self.llm_chat_view.moveCursor(QTextCursor.MoveOperation.End)
        prefix = "\n" if self.llm_chat_view.toPlainText() else ""
        self.llm_chat_view.insertPlainText(f"{prefix}ユーザー：{user_text}\nアシスタント：")
        self._llm_test_assistant_block_start = self.llm_chat_view.textCursor().position()
        self.llm_chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self._log_stream(f"LLMテスト送信開始(Orchestrator経路): backend={backend}, model={model}")

    def _on_llm_test_chat_chunk(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        text = str(payload.get("text", ""))
        if not text:
            return
        self._llm_test_streaming_assistant_full += text
        self._redraw_llm_test_assistant_block()

    def _on_llm_test_chat_finished(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        self._llm_test_in_progress = False
        self.llm_chat_send_btn.setEnabled(True)
        self.llm_user_input_edit.setEnabled(True)
        ok = bool(payload.get("ok", False))
        backend = str(payload.get("backend", ""))
        model = str(payload.get("model", ""))
        context_len = int(payload.get("context_len", 10) or 10)
        if not ok:
            message = str(payload.get("error", "不明なエラー"))
            if self._llm_messages and self._llm_messages[-1].get("role") == "user":
                self._llm_messages.pop()
            self._redraw_llm_test_assistant_block()
            self.llm_chat_view.moveCursor(QTextCursor.MoveOperation.End)
            self.llm_chat_view.insertPlainText(f"\n[error] {message}\n")
            self.llm_chat_view.moveCursor(QTextCursor.MoveOperation.End)
            self.llm_status_label.setText("状態：LLM応答失敗")
            self._log_stream(f"LLMテスト失敗: {message}")
            self._llm_test_tts_confirmed_segments.clear()
            self._llm_test_streaming_assistant_full = ""
            self._llm_test_assistant_block_start = -1
            return

        assistant_text = str(payload.get("assistant_text", ""))
        ttft_sec = float(payload.get("ttft_sec", 0.0) or 0.0)
        total_sec = float(payload.get("total_sec", 0.0) or 0.0)
        total_overall_sec = float(payload.get("total_overall_sec", total_sec) or total_sec)
        self._llm_messages.append({"role": "assistant", "content": assistant_text})
        self._trim_llm_messages(context_len)
        self._llm_test_streaming_assistant_full = assistant_text
        self._redraw_llm_test_assistant_block()
        self.llm_chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.llm_chat_view.insertPlainText(
            f"\n(ttft={ttft_sec:.3f}s total={total_sec:.3f}s overall={total_overall_sec:.3f}s)\n"
        )
        self.llm_chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self._llm_test_tts_confirmed_segments.clear()
        self._llm_test_streaming_assistant_full = ""
        self._llm_test_assistant_block_start = -1
        self.llm_status_label.setText(f"状態：{backend}応答完了")
        self._log_stream(
            "LLMテスト完了: "
            f"backend={backend}, model={model}, ttft={ttft_sec:.3f}s, total={total_sec:.3f}s, overall={total_overall_sec:.3f}s"
        )

    def send_mmdagent_tts_test(self) -> None:
        started_at = time.perf_counter()
        engine = self.tts_engine_combo.currentText().strip() or "VOICEVOX"
        try:
            self._validate_remote_backend_constraints(tts_engine=engine)
        except Exception as ex:
            QMessageBox.warning(self, "Dockerバックエンドの制約", str(ex))
            return
        stream_mode = self._tts_stream_mode(engine)
        mmd_running = self._avatar_output_available()
        if self._tts_test_in_progress:
            QMessageBox.information(self, "情報", "TTSテストは実行中です。完了までお待ちください。")
            return
        utterance = self.tts_test_text_edit.text().strip()
        if not utterance:
            QMessageBox.warning(self, "エラー", "テスト発話を入力してください。")
            self._set_tts_latency(time.perf_counter() - started_at, "失敗")
            return

        inputs: dict[str, object] = {}
        template: dict[str, object] | None = None
        try:
            if engine == "VOICEVOX-Core":
                inputs = self._build_tts_inputs(engine=engine, utterance=utterance)
            elif engine == "VOICEVOX":
                inputs = self._build_tts_inputs(engine=engine, utterance=utterance)
                speaker_text = str(inputs.get("voice", "1")).strip() or "1"
                try:
                    int(speaker_text)
                except ValueError as ex:
                    raise RuntimeError("VOICEVOX の話者IDは整数で指定してください。") from ex
            elif engine == "Kokoro-RealtimeTTS":
                inputs = self._build_tts_inputs(engine=engine, utterance=utterance)
            elif engine == "Cartesia-Realtime":
                inputs = self._build_tts_inputs(engine=engine, utterance=utterance)
                inputs.update(
                    {
                        "api_key": self.cartesia_tts_api_key_edit.text().strip(),
                        "api_key_env": self.cartesia_tts_api_key_env_edit.text().strip() or "CARTESIA_API_KEY",
                    }
                )
            elif engine in ("AivisSpeech", "Supertonic-3", "Qwen3-TTS-CPP", "LFM2.5-Audio", "Irodori-v4", "Tsukasa"):
                inputs = self._build_tts_inputs(engine=engine, utterance=utterance)
            elif engine == "TEMPLATE":
                inputs = self._build_tts_inputs(engine=engine, utterance=utterance)
                template = self._load_tts_template_dict()
            else:
                raise RuntimeError(f"未対応のTTSエンジンです: {engine}")
        except Exception as ex:
            QMessageBox.warning(
                self,
                "エラー",
                f"{engine}の入力組み立てに失敗しました。\n\n{ex}",
            )
            self.status_label.setText(f"状態：{engine} TTSテスト失敗")
            self._log_stream(f"{engine} TTSテスト失敗: {ex}")
            self._set_tts_latency(time.perf_counter() - started_at, "失敗")
            return

        tts_worker_config: dict[str, object] | None = None
        if tps.is_worker_engine(engine):
            tts_worker_config = self._build_tts_worker_config(engine)

        try:
            self._ensure_tts_engine_for_test(engine)
        except Exception as ex:
            QMessageBox.warning(
                self,
                "エラー",
                f"{engine}の自動起動に失敗しました。\n\n{ex}",
            )
            self.status_label.setText(f"状態：TTS実行失敗（{engine}起動失敗）")
            self._log_stream(f"TTSテスト: {engine} の自動起動に失敗: {ex}")
            self._set_tts_latency(time.perf_counter() - started_at, "失敗")
            return

        try:
            self._ensure_tts_playback_workers()
        except Exception as ex:
            QMessageBox.warning(
                self,
                "エラー",
                "TTSテストの再生には音声出力経路（MMD時は対話オーケストレーター）が必要ですが、起動に失敗しました。\n\n"
                f"{ex}",
            )
            self.status_label.setText("状態：音声出力経路の起動失敗")
            self._log_stream(f"TTSテスト: 音声出力経路の起動失敗: {ex}")
            self._set_tts_latency(time.perf_counter() - started_at, "失敗")
            return

        orch: dict[str, object] = {
            "type": "start_tts_gui_test",
            "engine": engine,
            "utterance": utterance,
            "inputs": inputs,
            "voice_dialog_inject_ch2": True,
            "mmd_running": mmd_running,
            "started_at": started_at,
        }
        if template is not None:
            orch["template"] = template
        if tts_worker_config is not None:
            orch["tts_worker_config"] = tts_worker_config

        if not self._send_dialog_orchestrator_command_sync(orch):
            QMessageBox.warning(
                self,
                "エラー",
                "対話オーケストレーターへのTTSテスト送信に失敗しました。",
            )
            self.status_label.setText("状態：オーケストレーターへの送信失敗")
            self._log_stream("TTSテスト: Orchestrator 送信失敗")
            self._set_tts_latency(time.perf_counter() - started_at, "失敗")
            return

        self._tts_test_in_progress = True
        self._sync_tts_launch_test_controls()
        self.status_label.setText(f"状態：{engine}テスト実行中（モード：{stream_mode}、オーケストレーター経路）")
        self._log_stream(f"TTSテスト開始(Orchestrator経路): engine={engine}, mode={stream_mode}")

    def _on_tts_test_synthesis_finished(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        self._tts_test_in_progress = False
        self._sync_tts_launch_test_controls()
        ok = bool(payload.get("ok", False))
        engine = str(payload.get("engine", "VOICEVOX"))
        started_at = float(payload.get("started_at", time.perf_counter()) or time.perf_counter())
        if not ok:
            message = str(payload.get("error", "不明なエラー"))
            if message == "interrupt":
                self.status_label.setText(f"状態：{engine}テスト中断")
                self._log_stream(f"{engine} TTSテスト中断（割り込み）")
                self._set_tts_latency(time.perf_counter() - started_at, "失敗")
                return
            QMessageBox.warning(
                self,
                "エラー",
                f"{engine}のTTSテストに失敗しました。\n\n{message}",
            )
            self.status_label.setText(f"状態：{engine} TTSテスト失敗")
            self._log_stream(f"{engine} TTSテスト失敗: {message}")
            self._set_tts_latency(time.perf_counter() - started_at, "失敗")
            return

        if bool(payload.get("playback_via_orchestrator", False)):
            voice = str(payload.get("voice", ""))
            stream_bytes = int(payload.get("stream_bytes", 0) or 0)
            first_audible = bool(payload.get("first_audible", False))
            self._save_app_config()
            self.status_label.setText(f"状態：{engine}成功（オーケストレーター再生経路）")
            self._log_stream(
                f"{engine}成功(Orchestrator): bytes={stream_bytes} speaker={voice}"
            )
            if not first_audible:
                self._set_tts_latency(time.perf_counter() - started_at, "実測")
            else:
                self._set_tts_latency(time.perf_counter() - started_at, "推定")
            return

        QMessageBox.warning(
            self,
            "エラー",
            f"{engine}のTTSテスト応答が想定外です（オーケストレーター再生経路フラグがありません）。",
        )
        self.status_label.setText(f"状態：{engine} TTSテスト応答不正")
        self._log_stream(f"{engine} TTSテスト: 想定外の成功応答 payload={payload!r}")
        self._set_tts_latency(time.perf_counter() - started_at, "失敗")

    def launch_voicevox(self) -> None:
        try:
            self._validate_remote_backend_constraints(tts_engine="VOICEVOX")
        except Exception as ex:
            QMessageBox.warning(self, "Dockerバックエンドの制約", str(ex))
            return
        self._sync_tts_exe_paths_from_ui()
        self._begin_tts_engine_launch("VOICEVOX")
        try:
            try:
                res = self._tts_bridge_rpc("launch_voicevox", timeout_sec=90.0)
                if not res or not res.get("ok"):
                    msg = str(res.get("message", "")) if res else "Orchestrator へ接続できませんでした"
                    raise RuntimeError(msg)
                pid = res.get("pid", "")
            except Exception as ex:
                self._set_module_panel_state("tts", "error", str(ex))
                QMessageBox.warning(self, "エラー", f"VOICEVOXの起動に失敗しました：\n{ex}")
                self.status_label.setText("状態：VOICEVOX起動失敗")
                self._log_stream(f"VOICEVOX 起動失敗: {ex}")
                return
            self._set_module_panel_state("tts", "running", "VOICEVOX")
            self.status_label.setText(f"状態：VOICEVOX起動（PID：{pid}）")
            self._log_stream(f"VOICEVOX 起動 (Orchestrator 経由 PID: {pid})")
            self._open_console_window()
        finally:
            self._end_tts_engine_launch()

    def stop_voicevox(self) -> None:
        if self._remote_backend_enabled:
            QMessageBox.warning(
                self,
                "Dockerバックエンドの制約",
                "DockerバックエンドからWindows版VOICEVOXを管理できません。",
            )
            return
        try:
            res = self._tts_bridge_rpc("stop_voicevox", timeout_sec=20.0)
            if res and res.get("ok") is False:
                raise RuntimeError(str(res.get("message", "")))
        except Exception as ex:
            QMessageBox.warning(self, "エラー", f"VOICEVOXの停止に失敗しました：\n{ex}")
            self.status_label.setText("状態：VOICEVOX停止失敗")
            self._log_stream(f"VOICEVOX 停止失敗: {ex}")
            return
        self._set_module_panel_state("tts", "stopped")
        self.status_label.setText("状態：VOICEVOX停止")
        self._log_stream("VOICEVOX 停止")

    def launch_aivisspeech(self) -> None:
        try:
            self._validate_remote_backend_constraints(tts_engine="AivisSpeech")
        except Exception as ex:
            QMessageBox.warning(self, "Dockerバックエンドの制約", str(ex))
            return
        self._begin_tts_engine_launch("AivisSpeech")
        try:
            host = self.aivisspeech_host_edit.text().strip() or "127.0.0.1"
            port = int(self.aivisspeech_port_edit.text().strip() or "10104")
            res = self._tts_bridge_rpc("launch_aivisspeech", timeout_sec=75.0, host=host, port=port, use_gpu=True)
            if not res or not res.get("ok"):
                raise RuntimeError(str(res.get("message", "AivisSpeech Engine を起動できません")) if res else "Orchestratorへの接続に失敗しました")
            self.launch_worker_tts("AivisSpeech")
        except Exception as ex:
            self._set_module_panel_state("tts", "error", str(ex))
            QMessageBox.warning(self, "エラー", f"AivisSpeechの起動に失敗しました：\n{ex}")
            self.status_label.setText("状態：AivisSpeech起動失敗")
            self._log_stream(f"AivisSpeech 起動失敗: {ex}")
        finally:
            self._end_tts_engine_launch()

    def stop_aivisspeech(self) -> None:
        if self._remote_backend_enabled:
            QMessageBox.warning(
                self,
                "Dockerバックエンドの制約",
                "DockerバックエンドからWindows版AivisSpeech Engineは管理できません。",
            )
            return
        self.stop_worker_tts("AivisSpeech")
        try:
            res = self._tts_bridge_rpc("stop_aivisspeech", timeout_sec=20.0)
            if res and res.get("ok") is False:
                raise RuntimeError(str(res.get("message", "")))
        except Exception as ex:
            QMessageBox.warning(self, "エラー", f"AivisSpeechエンジンの停止に失敗しました：\n{ex}")
            return
        self.status_label.setText("状態：AivisSpeech停止")
        self._log_stream("AivisSpeech Engine 停止（アプリ起動分のみ）")

    def launch_voicevox_core(self) -> None:
        self._sync_tts_exe_paths_from_ui()
        self._begin_tts_engine_launch("VOICEVOX-Core")
        try:
            try:
                self._ensure_tts_worker("VOICEVOX-Core")
            except Exception as ex:
                self._set_module_panel_state("tts", "error", str(ex))
                QMessageBox.warning(self, "エラー", f"VOICEVOX-Coreの起動に失敗しました：\n{ex}")
                self.status_label.setText("状態：VOICEVOX-Core起動失敗")
                self._log_stream(f"VOICEVOX-Core 起動失敗: {ex}")
                return
            self.status_label.setText("状態：VOICEVOX-Core起動")
            self._log_stream("VOICEVOX-Core IPCワーカー起動")
            self._open_console_window()
        finally:
            self._end_tts_engine_launch()

    def stop_voicevox_core(self) -> None:
        if not (self._tts_worker_ready_event.is_set() and self._tts_worker_engine == "VOICEVOX-Core"):
            self.status_label.setText("状態：VOICEVOX-Coreは未起動")
            self._log_stream("VOICEVOX-Core は未起動")
            return
        self._teardown_tts_ipc()
        self.status_label.setText("状態：VOICEVOX-Core停止")
        self._log_stream("VOICEVOX-Core IPCワーカー停止")

    def launch_kokoro_realtimetts(self) -> None:
        self._open_console_window()
        self._begin_tts_engine_launch("Kokoro-RealtimeTTS")
        try:
            try:
                self._log_stream("Kokoro-RealtimeTTS 起動開始")
                self.status_label.setText("状態：Kokoro-RealtimeTTS起動中（モデル読込中）")
                self._ensure_tts_worker("Kokoro-RealtimeTTS")
                self._log_stream("Kokoro-RealtimeTTS 予熱を Dialog Orchestrator に依頼")
                warmup_inputs = self._build_tts_inputs(engine="Kokoro-RealtimeTTS", utterance="ウォームアップです。")
                warm_cfg = self._build_tts_worker_config("Kokoro-RealtimeTTS")
                try:
                    self._ensure_dialog_orchestrator_worker()
                except Exception as ex:
                    raise RuntimeError(f"対話オーケストレーターの起動に失敗しました：{ex}") from ex
                warm_payload: dict[str, object] = {
                    "type": "start_tts_worker_warmup",
                    "engine": "Kokoro-RealtimeTTS",
                    "tts_worker_config": warm_cfg,
                    "inputs": warmup_inputs,
                    "timeout_sec": 60.0,
                }
                if not self._send_dialog_orchestrator_command_sync(warm_payload):
                    raise RuntimeError("対話オーケストレーターへ予熱コマンドを送信できませんでした。")
                self._log_stream("Kokoro-RealtimeTTS 予熱コマンド送信済み（Orchestrator 内で実行）")
            except Exception as ex:
                self._set_module_panel_state("tts", "error", str(ex))
                QMessageBox.warning(self, "エラー", f"Kokoro-RealtimeTTSの起動に失敗しました：\n{ex}")
                self.status_label.setText("状態：Kokoro-RealtimeTTS起動失敗")
                self._log_stream(f"Kokoro-RealtimeTTS 起動失敗: {ex}")
                return
            self.status_label.setText("状態：Kokoro-RealtimeTTS起動（予熱はオーケストレーターが実行）")
            self._log_stream("Kokoro-RealtimeTTS IPCワーカー起動")
        finally:
            self._end_tts_engine_launch()

    def stop_kokoro_realtimetts(self) -> None:
        if not (self._tts_worker_ready_event.is_set() and self._tts_worker_engine == "Kokoro-RealtimeTTS"):
            self.status_label.setText("状態：Kokoro-RealtimeTTSは未起動")
            self._log_stream("Kokoro-RealtimeTTS は未起動")
            return
        self._teardown_tts_ipc()
        self.status_label.setText("状態：Kokoro-RealtimeTTS停止")
        self._log_stream("Kokoro-RealtimeTTS IPCワーカー停止")

    def launch_worker_tts(self, engine: str) -> None:
        """追加ローカルTTSの共通起動。モデル/接続をここで予熱しておく。"""
        self._begin_tts_engine_launch(engine)
        try:
            try:
                self._ensure_tts_worker(engine)
                self._warmup_worker_tts(engine)
            except Exception as ex:
                self._set_module_panel_state("tts", "error", str(ex))
                QMessageBox.warning(self, "エラー", f"{engine}の起動に失敗しました：\n{ex}")
                self.status_label.setText(f"状態：{engine}起動失敗")
                self._log_stream(f"{engine} 起動失敗: {ex}")
                return
            self.status_label.setText(f"状態：{engine}起動済み")
            self._log_stream(f"{engine} TTS worker 起動済み")
            self._open_console_window()
        finally:
            self._end_tts_engine_launch()

    def _warmup_worker_tts(self, engine: str) -> None:
        """起動完了をモデル読込だけで終わらせず、最初の合成コストも吸収する。"""
        inputs = self._build_tts_inputs(engine=engine, utterance="ウォームアップです。")
        worker_config = self._build_tts_worker_config(engine)
        self._ensure_dialog_orchestrator_worker()
        payload: dict[str, object] = {
            "type": "start_tts_worker_warmup",
            "engine": engine,
            "tts_worker_config": worker_config,
            "inputs": inputs,
            "timeout_sec": 180.0 if engine == "Irodori-v4" else 90.0,
        }
        if not self._send_dialog_orchestrator_command_sync(payload):
            raise RuntimeError("対話オーケストレーターへのTTSウォームアップ要求に失敗しました。")
        self._log_stream(f"{engine} 実音声ウォームアップ完了")

    def stop_worker_tts(self, engine: str) -> None:
        if not (self._tts_worker_ready_event.is_set() and self._tts_worker_engine == engine):
            self.status_label.setText(f"状態：{engine}は未起動")
            self._log_stream(f"{engine} は未起動")
            return
        self._teardown_tts_ipc()
        self._set_module_panel_state("tts", "stopped")
        self.status_label.setText(f"状態：{engine}停止")
        self._log_stream(f"{engine} TTS worker 停止")

    def _sync_maai_local_mimi_onnx_controls(self) -> None:
        """ONNX 有効かつ「ローカルファイルを使う」のときだけパス指定 UI を有効化する。"""
        onnx_on = bool(
            getattr(self, "maai_use_mimi_onnx_check", None) is not None
            and self.maai_use_mimi_onnx_check.isChecked()
        )
        local_check = getattr(self, "maai_use_local_mimi_onnx_check", None)
        if local_check is not None:
            local_check.setEnabled(onnx_on)
        local_on = bool(local_check is not None and local_check.isChecked() and onnx_on)
        for attr in (
            "maai_root_label",
            "maai_root_edit",
            "maai_root_browse_btn",
            "maai_mimi_onnx_path_label",
            "maai_mimi_onnx_path_edit",
            "maai_mimi_onnx_path_browse_btn",
            "maai_mimi_onnx_meta_path_label",
            "maai_mimi_onnx_meta_path_edit",
            "maai_mimi_onnx_meta_path_browse_btn",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.setEnabled(local_on)

    def _on_maai_mimi_onnx_options_changed(self) -> None:
        self._sync_maai_local_mimi_onnx_controls()
        self._save_app_config()

    @staticmethod
    def _guess_maai_onnx_meta_path(onnx_path: str) -> str:
        p = Path(onnx_path)
        candidates = [
            Path(str(p) + ".json"),
            p.with_suffix(".json"),
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return str(candidates[0])


    def _maai_onnx_browse_initial_dir(self) -> str:
        root_edit = getattr(self, "maai_root_edit", None)
        root = ""
        if root_edit is not None:
            root = root_edit.text().strip()
        if not root:
            root = str(_paths.assets_root())
        root = self._docker_picker_initial_path(root)
        onnx_dir = Path(root) / "onnx"
        if onnx_dir.is_dir():
            return str(onnx_dir)
        if Path(root).is_dir():
            return root
        return str(Path.cwd())

    def _parse_mdf_key_values(self, lines: list[str]) -> dict[str, str]:
        return self.mdf_service.parse_mdf_key_values(lines)

    def _widget_to_value(self, key: str) -> str:
        widget = self.field_widgets[key]
        if isinstance(widget, QCheckBox):
            return "true" if widget.isChecked() else "false"
        if isinstance(widget, QLineEdit):
            value = widget.text().strip()
            if key == "Plugin_Remote_Websocket_Directory" and value and not value.startswith("/"):
                return f"/{value}"
            return value
        return ""

    def _apply_value_to_widget(self, key: str, value: str) -> None:
        widget = self.field_widgets[key]
        if isinstance(widget, QCheckBox):
            widget.setChecked(to_bool(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(value)

    def _on_checkbox_changed(self, key: str, state: int) -> None:
        if self._suppress_checkbox_handler:
            return
        self.touched_keys.add(key)
        if state == 0:
            return
        for other_key in self.checkbox_conflicts.get(key, set()):
            other_widget = self.field_widgets.get(other_key)
            if isinstance(other_widget, QCheckBox) and other_widget.isChecked():
                self._suppress_checkbox_handler = True
                other_widget.setChecked(False)
                self._suppress_checkbox_handler = False
                self.touched_keys.add(other_key)

    def _tooltip_for_field(self, key: str, default_value: str) -> str:
        desc = self.FIELD_DESCRIPTIONS.get(key, "この設定項目の値を指定します。")
        default_text = default_value if default_value != "" else "（空）"
        return f"{desc}\nキー：{key}\n既定値：{default_text}"

    def _current_mdf_host_path(self, *, warn: bool = False) -> Path | None:
        configured = self.mdf_edit.text().strip()
        path = self._host_path_for_backend_value(configured)
        if path is None and warn:
            QMessageBox.warning(
                self,
                "Dockerパス変換エラー",
                "MDFパスをホスト側へ変換できません。"
                "MMDAGENT_ASSETS_HOSTまたはリポジトリ配下のMDFを指定してください。\n\n"
                f"{configured}",
            )
        return path

    def load_mdf_settings(self) -> None:
        mdf_path = self._current_mdf_host_path()
        if mdf_path is None or not mdf_path.exists():
            return
        lines, _encoding = self.mdf_service.read_mdf(mdf_path)
        parsed = self._parse_mdf_key_values(lines)
        self.loaded_keys = set(parsed.keys())
        self.touched_keys.clear()
        self._suppress_checkbox_handler = True
        for key in self.field_widgets:
            # 読込前の値が残らないよう、毎回いったん既定値に戻してからMDF値を適用する。
            self._apply_value_to_widget(key, self.default_values.get(key, ""))
            if key in parsed:
                self._apply_value_to_widget(key, parsed[key])
        self._suppress_checkbox_handler = False
        self.status_label.setText("状態：MDF設定を読込済み")

    def save_mdf_settings(self) -> bool:
        mdf_path = self._current_mdf_host_path(warn=True)
        if mdf_path is None:
            return False
        if not mdf_path.exists():
            QMessageBox.warning(self, "エラー", f"MDFファイルが見つかりません：\n{mdf_path}")
            return False
        ok, new_loaded = self.mdf_service.write_values(
            mdf_path=mdf_path,
            field_keys=list(self.field_widgets.keys()),
            widget_to_value=self._widget_to_value,
            loaded_keys=self.loaded_keys,
            touched_keys=self.touched_keys,
        )
        if ok:
            self.status_label.setText("状態：MDF設定を保存済み")
            self.touched_keys.clear()
            self.loaded_keys = new_loaded
            self._sync_orchestrator_audio_output_from_ui()
            self._sync_orchestrator_audio_input_from_ui()
            if self.process_service.is_running():
                self._send_remote_plugin_settings()
        return ok

    def reset_to_defaults(self) -> None:
        self._suppress_checkbox_handler = True
        for key in self.field_widgets:
            self._apply_value_to_widget(key, self.default_values.get(key, ""))
        self._suppress_checkbox_handler = False
        self.touched_keys = set(self.field_widgets.keys())
        self.status_label.setText("状態：デフォルト値にリセット済み（未保存）")

    def save_mdf_as(self) -> None:
        current = self.mdf_edit.text().strip()
        current_host = self._docker_picker_initial_path(current)
        default_dir = str(Path(current_host).parent) if current_host else str(Path.cwd())
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "MDFを新規保存",
            str(Path(default_dir) / "new_config.mdf"),
            "MDF (*.mdf)",
        )
        if not target_path:
            return
        configured_target = self._docker_backend_path_from_picker(
            target_path, label="MDF",
        )
        if configured_target is None:
            return
        mdf_path = Path(target_path)
        if mdf_path.exists():
            result = QMessageBox.question(
                self,
                "上書き確認",
                f"既存ファイルを上書きしますか？\n{mdf_path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return
        else:
            mdf_path.parent.mkdir(parents=True, exist_ok=True)
            mdf_path.write_text("", encoding="utf-8")
        ok, new_loaded = self.mdf_service.write_values(
            mdf_path=mdf_path,
            field_keys=list(self.field_widgets.keys()),
            widget_to_value=self._widget_to_value,
            loaded_keys=self.loaded_keys,
            touched_keys=self.touched_keys,
        )
        if ok:
            self.touched_keys.clear()
            self.loaded_keys = new_loaded
            self.mdf_edit.setText(configured_target)
            self.status_label.setText(f"状態：新規MDFを保存済み（{mdf_path.name}）")
            self._save_app_config()

    def launch_mmdagent(self) -> None:
        if self._avatar_in_browser():
            port = vds.avatar_browser_ws_port(self._widget_config_snapshot())
            QMessageBox.information(
                self,
                "ブラウザ表示",
                "アバターはブラウザ（MMDAgent-EX Web）に表示する設定なので、MMDAgent-EX は起動しません。\n\n"
                "MMDAgent-EX Web でコンテンツのフォルダーを選び、maai_sds_web.mdf を開始して、"
                f"ws://localhost:{port}/chat に接続してください。",
            )
            return
        exec_path = Path(self.exec_edit.text().strip())
        mdf_path = self._current_mdf_host_path(warn=True)
        if mdf_path is None:
            return
        if not exec_path.exists():
            QMessageBox.warning(self, "エラー", f"実行ファイルが見つかりません：\n{exec_path}")
            return
        if not mdf_path.exists():
            QMessageBox.warning(self, "エラー", f"MDFファイルが見つかりません：\n{mdf_path}")
            return
        if self.process_service.is_running():
            QMessageBox.information(self, "情報", "MMDAgent-EXはすでに起動中です。")
            return
        if not self.save_mdf_settings():
            return
        ok_fst, fst_msg = self.mdf_service.ensure_fst_for_mdf(mdf_path)
        if not ok_fst:
            QMessageBox.warning(
                self,
                "エラー",
                f"FSTファイルの自動生成に失敗しました：\n{mdf_path.with_suffix('.fst')}\n\n{fst_msg}",
            )
            return
        if fst_msg:
            self.status_label.setText(fst_msg)

        ws_host = self._widget_to_value("Plugin_Remote_Websocket_Host") or "localhost"
        ws_port = self._widget_to_value("Plugin_Remote_Websocket_Port") or "9001"
        ws_dir = self._widget_to_value("Plugin_Remote_Websocket_Directory") or "/chat"
        try:
            port_num = int(ws_port)
            if not (1 <= port_num <= 65535):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "エラー", "WebSocketポート番号が不正です。")
            return

        env = os.environ.copy()
        env["WEBSOCKET_USE_CA_WEBSOCKET"] = "false"
        env["WEBSOCKET_SERVER"] = ws_host
        env["WEBSOCKET_PORT"] = str(port_num)
        env["WEBSOCKET_CHANNEL"] = ws_dir
        self._send_remote_plugin_stop()
        try:
            proc = self.process_service.start(exec_path=exec_path, mdf_path=mdf_path, env=env)
            self._attach_process_console_output("MMDAgent", proc)
            self._send_remote_plugin_settings()
            self._save_app_config()
            self.status_label.setText(f"状態：起動中（PID：{proc.pid}）")
            self._teardown_fallback_audio_output_if_unused()
            if self._maai_ready:
                self._maai_mmd_av_preamble_wait_for_mmd = False
                self._try_send_maai_mmd_av_preamble_once()
        except Exception as ex:
            self._send_remote_plugin_stop()
            QMessageBox.critical(self, "起動失敗", str(ex))
            self.status_label.setText("状態：起動失敗")

    def stop_mmdagent(self) -> None:
        was_running = self.process_service.is_running()
        if was_running:
            self.process_service.stop()
        self._send_remote_plugin_stop()
        if was_running:
            self.status_label.setText("状態：停止")
        else:
            self.status_label.setText("状態：未起動")

    def closeEvent(self, event) -> None:  # noqa: N802
        # 窓の位置・大きさ・開いていた画面を覚える（子の窓を閉じる前に。開き直したときに元へ戻す）
        try:
            from . import window_state

            window_state.save(self)
        except Exception:  # noqa: BLE001
            pass
        self._controller_closing = True
        self._fast_shutdown = True
        self._stop_remote_mic_capture()
        # ワーカーを落とす前に LM Studio のモデルを解放する（VRAM を握ったままにしない）。
        if not self._remote_backend_enabled:
            self._eject_llm_models_on_shutdown()
        self._shutdown_controller_threads()

        hub = getattr(self, "settings_hub_window", None)
        if hub is not None:
            hub.close()
        popout = getattr(self, "chat_popout", None)
        if popout is not None:
            popout.shutdown()  # 別ウィンドウだったことは覚えたまま閉じる
        if hasattr(self, "settings_window") and self.settings_window is not None:
            self.settings_window.close()
        if hasattr(self, "audio_input_settings_window") and self.audio_input_settings_window is not None:
            self.audio_input_settings_window.close()
        if hasattr(self, "audio_output_settings_window") and self.audio_output_settings_window is not None:
            self.audio_output_settings_window.close()
        if hasattr(self, "tts_template_auth_window") and self.tts_template_auth_window is not None:
            self.tts_template_auth_window.close()
        if hasattr(self, "obs_settings_window") and self.obs_settings_window is not None:
            self.obs_settings_window.close()
        if hasattr(self, "tts_test_window") and self.tts_test_window is not None:
            self.tts_test_window.close()
        if hasattr(self, "asr_test_window") and self.asr_test_window is not None:
            self.asr_test_window.close()
        if hasattr(self, "llm_test_window") and self.llm_test_window is not None:
            self.llm_test_window.close()
        if hasattr(self, "llm_s2s_test_window") and self.llm_s2s_test_window is not None:
            self.llm_s2s_test_window.close()
        if hasattr(self, "maai_test_window") and self.maai_test_window is not None:
            self.maai_test_window.close()
        if hasattr(self, "maai_plot_window") and self.maai_plot_window is not None:
            self._maai_plot_pending = []
            self.maai_plot_window.close()
        if hasattr(self, "nod_motion_chart_window") and self.nod_motion_chart_window is not None:
            self.nod_motion_chart_window.close()
        if hasattr(self, "asr_settings_window") and self.asr_settings_window is not None:
            self.asr_settings_window.close()
        if hasattr(self, "console_window") and self.console_window is not None:
            self.console_window.close()
        self._close_console_log()
        if hasattr(self, "engine_manager_window") and self.engine_manager_window is not None:
            self.engine_manager_window.close()
        # Ctrl+C 終了の体感を速くするため、MMDAgent 停止の graceful 待ちは短くする。
        self.process_service.stop(timeout_sec=1)
        if self._remote_backend_enabled:
            # Disconnect only.  The persistent container orchestrator and its
            # workers remain alive so another Qt instance can reattach.
            self._teardown_dialog_orchestrator_ipc()
            event.accept()
            return
        self._send_remote_plugin_stop()
        self._stop_voice_dialog()
        self.stop_asr()
        self._stop_qwen_asr_container_logs()
        self._teardown_maai_ipc()
        self._teardown_audio_input_ipc()
        self._teardown_tts_ipc()
        self._teardown_llm_ipc()
        self._teardown_dialog_orchestrator_ipc()
        event.accept()

