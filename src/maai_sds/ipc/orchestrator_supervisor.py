"""子ワーカーのライフサイクルと IPC を管理する Supervisor。

`dialog/worker.py` から分離。挙動・ログ文言・スレッド名・IPC は
完全に維持しており、TTS Bridge / LLM admin だけ専用ハンドラへ委譲する。
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .jsonline import JsonLineDuplexServer
from .process_service import (
    ASRWorkerProcessService,
    AudioInputWorkerProcessService,
    AudioOutputWorkerProcessService,
    LLMWorkerProcessService,
    MaAIWorkerProcessService,
    TTSWorkerProcessService,
)
from ..llm.admin_http import LlmAdminHttp
from ..mmd.blink import (
    DEFAULT_BLINK_INTERVAL_MEAN_SEC,
    normalize_blink_enable,
    normalize_blink_interval_mean_sec,
    normalize_blink_mode,
    sanitize_blink_morph,
)
from ..tts.bridge_handler import TtsBridgeHandler

from .. import paths as _paths


def _script(*parts: str) -> Path:
    """Worker スクリプトのフルパスを解決する（`_script("llm", "worker.py")`）。

    パッケージ内の配置基準（`paths.PACKAGE_DIR`）で解決するので、
    このファイル自身がどこに置かれても同じファイルを指す。
    """
    return _paths.worker_script(*parts)


def _audio_output_restart_digest(config: dict[str, Any]) -> str:
    """プロセス再起動が必要な設定だけを正規化して比較する。

    MMD のバースト・まばたき設定は worker 起動後に `update_settings` で
    変えられるためダイジェストに含めない（毎回の ensure でプロセスを落とさない）。
    """
    rp = config.get("remote_plugin_settings")
    if not isinstance(rp, dict):
        rp = {}
    try:
        sr = int(config.get("sample_rate", 16000) or 16000)
    except (TypeError, ValueError):
        sr = 16000
    try:
        tcp_port = int(config.get("tcp_output_port", 15601) or 15601)
    except (TypeError, ValueError):
        tcp_port = 15601
    tcp_port = max(1, min(65535, tcp_port))
    sub = {
        "output_mode": str(config.get("output_mode", "local") or "local").strip().lower(),
        "sample_rate": sr,
        "remote_plugin_settings": json.dumps(rp, sort_keys=True, ensure_ascii=False, default=str),
        "tcp_output_host": str(config.get("tcp_output_host", "127.0.0.1") or "127.0.0.1").strip(),
        "tcp_output_port": tcp_port,
    }
    return json.dumps(sub, sort_keys=True, ensure_ascii=False)


class OrchestratorSupervisor:
    """子ワーカー群のライフサイクルと IPC を管理する。

    Orchestrator プロセス（dialog/worker.py）内で使用される。
    GUI プロセス (Controller) は不要。
    """

    def __init__(
        self,
        log_fn: Callable[[str], None] | None = None,
        on_llm_event: Callable[[dict], None] | None = None,
        on_tts_event: Callable[[dict], None] | None = None,
        on_audio_output_event: Callable[[dict], None] | None = None,
        on_llm_admin_event: Callable[[dict], None] | None = None,
    ) -> None:
        self._log = log_fn or (lambda msg: None)
        self._on_llm_event = on_llm_event
        self._on_tts_event = on_tts_event
        self._on_audio_output_event = on_audio_output_event
        self._on_llm_admin_event = on_llm_admin_event

        self._python_exe = Path(sys.executable)

        # ProcessService instances
        self.asr = ASRWorkerProcessService()
        self.asr_compare = ASRWorkerProcessService()
        self.audio_input = AudioInputWorkerProcessService()
        self.audio_output = AudioOutputWorkerProcessService()
        self.tts = TTSWorkerProcessService()
        self.llm = LLMWorkerProcessService()
        self.maai = MaAIWorkerProcessService()

        # IPC servers (owned by this supervisor)
        self._llm_ipc: JsonLineDuplexServer | None = None
        self._tts_ipc: JsonLineDuplexServer | None = None
        self._audio_output_ipc: JsonLineDuplexServer | None = None

        # Ready events
        self._asr_ready = threading.Event()
        self._asr_compare_ready = threading.Event()
        self._llm_ready = threading.Event()
        self._tts_ready = threading.Event()
        self._maai_ready = threading.Event()
        self._audio_input_ready = threading.Event()
        self._audio_output_ready = threading.Event()
        self._asr_fatal: str = ""
        self._asr_compare_fatal: str = ""
        self._llm_fatal: str = ""
        self._tts_fatal: str = ""
        self._maai_fatal: str = ""
        self._audio_input_fatal: str = ""
        self._audio_output_fatal: str = ""

        # Worker state
        self._tts_engine: str = ""
        self._tts_config_json: str = ""
        self._audio_output_config_json: str = ""
        self._audio_input_config_json: str = ""
        self._asr_config_json: str = ""
        self._asr_route: tuple[str, int] | None = None

        # RLock: ensure_*() は _lock 保持中に _teardown_*() を呼ぶ。teardown 自身も _lock を
        # 取得して ensure のクリティカルセクションと直列化する(#11)ため、再入可能ロックが必要。
        # モジュールごとに分けるのが要点。起動〜ready 待ちを直列化したいが、
        # 共通の _lock で待つと他モジュールの ensure が最大 180 秒止まる。
        self._llm_ensure_lock = threading.RLock()
        self._tts_ensure_lock = threading.RLock()
        self._asr_compare_ensure_lock = threading.RLock()
        self._asr_ensure_lock = threading.RLock()
        self._maai_ensure_lock = threading.RLock()
        self._audio_output_ensure_lock = threading.RLock()
        self._audio_input_ensure_lock = threading.RLock()

        # Sub-handlers (TTS Bridge / LLM admin)
        self._tts_bridge_handler = TtsBridgeHandler(
            on_event=self._emit_tts_bridge_event,
            log_fn=self._log,
        )
        self._llm_admin = LlmAdminHttp(on_event=self._emit_llm_admin_event)

    def ui_state_snapshot(self) -> list[dict[str, object]]:
        """Describe live child processes for a newly attached UI client.

        Worker lifecycle events are normally edge-triggered.  A persistent
        orchestrator can outlive Qt, so a replacement client needs a compact
        level-triggered snapshot instead of waiting for events that already
        happened.
        """
        services = (
            ("asr", self.asr),
            ("asr_compare", self.asr_compare),
            ("llm", self.llm),
            ("tts", self.tts),
            ("maai", self.maai),
            ("audio_input", self.audio_input),
            ("audio_output", self.audio_output),
        )
        events: list[dict[str, object]] = []
        running: dict[str, bool] = {}
        for module, service in services:
            try:
                is_running = bool(service.is_running())
            except Exception:
                is_running = False
            running[module] = is_running
            # UI（Qt の _set_module_panel_state / Web の applyModuleEvent）は asr_compare を
            # ASR 行に集約する。止まっている asr_compare を asr の後に送ると、動いている
            # ASR の表示が「停止」で上書きされるので、動いているときだけ送る。
            if module == "asr_compare" and not is_running:
                continue
            payload: dict[str, object] = {
                "type": "module_state",
                "module": module,
                "state": "running" if is_running else "stopped",
                "source": "ui_reconnect",
            }
            if module == "tts" and is_running and self._tts_engine:
                payload["engine"] = self._tts_engine
            events.append(payload)

        if running.get("audio_output"):
            output_mode = "local"
            try:
                config = json.loads(self._audio_output_config_json or "{}")
                if isinstance(config, dict):
                    output_mode = str(config.get("output_mode", "local") or "local")
            except Exception:
                pass
            events.append(
                {"type": "ready", "worker": "audio_output", "output_mode": output_mode}
            )

        if running.get("audio_input"):
            try:
                config = json.loads(self._audio_input_config_json or "{}")
            except Exception:
                config = {}
            if isinstance(config, dict) and not bool(config.get("start_paused", False)):
                events.append({"type": "processing_started", "source": "ui_reconnect"})
        return events

    # -- TTS Bridge (delegated) -----------------------------------------------

    def _emit_tts_bridge_event(self, payload: dict[str, Any]) -> None:
        if self._on_tts_event:
            try:
                self._on_tts_event(payload)
            except Exception:
                pass

    def handle_tts_bridge_command(self, payload: dict[str, Any]) -> None:
        self._tts_bridge_handler.handle(payload)

    def ensure_voicevox(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50021,
        exe_path: str = "",
    ) -> None:
        self._tts_bridge_handler.launch_voicevox_blocking(
            host=host, port=port, exe_path=exe_path,
        )

    # -- LLM admin HTTP (delegated) -------------------------------------------

    def _emit_llm_admin_event(self, payload: dict[str, Any]) -> None:
        if self._on_llm_admin_event:
            try:
                self._on_llm_admin_event(payload)
            except Exception:
                pass

    def _handle_worker_disconnect(
        self,
        name: str,
        ready_event: threading.Event,
        on_event: Callable[[dict], None] | None,
    ) -> None:
        """ワーカー IPC が予期せず切断されたときの共通処理。

        ready をクリアして死活状態を実体に合わせる（放置すると ready がセットの
        まま残り、死んだワーカーへ send し続けて対話が無言でハングする）。次の
        `ensure_*` で再起動できるようになる。ログと上流イベントで可視化もする。
        """
        ready_event.clear()
        self._log(f"[{name}] IPC disconnected")
        if on_event is not None:
            try:
                on_event({"type": "worker_disconnected", "worker": name})
            except Exception:
                pass

    def handle_llm_admin_http_command(self, payload: dict[str, Any]) -> None:
        self._llm_admin.handle(payload)

    # -- LLM ------------------------------------------------------------------

    def ensure_llm(
        self,
        timeout_sec: float = 40.0,
    ) -> None:
        with self._llm_ensure_lock:
            if (
                self.llm.is_running()
                and self._llm_ipc is not None
                and self._llm_ipc.is_connected()
                and self._llm_ready.is_set()
            ):
                return
            self._teardown_llm()
            self._llm_ready.clear()
            self._llm_fatal = ""
            ipc = JsonLineDuplexServer("127.0.0.1", 0)
            ipc.start(
                on_event=self._handle_llm_event,
                on_disconnected=lambda: self._handle_worker_disconnect(
                    "LLM", self._llm_ready, self._on_llm_event
                ),
            )
            self._llm_ipc = ipc
            proc = self.llm.start(
                python_exe=self._python_exe,
                worker_script=_script("llm", "worker.py"),
                host=ipc.host,
                port=ipc.port,
                forward_console=True,
                extra_args=[],
            )
            self._log(f"[LLM] started pid={proc.pid}")
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if self._llm_ready.is_set():
                    return
                if self._llm_fatal:
                    raise RuntimeError(self._llm_fatal)
                if proc.poll() is not None:
                    error = self._worker_exited_before_ready(proc, "LLM", "_llm_fatal")
                    raise error
                time.sleep(0.05)
            raise RuntimeError("LLM worker ready timeout")

    def send_llm(self, payload: dict) -> bool:
        ipc = self._llm_ipc
        if ipc is None or not ipc.is_connected():
            return False
        try:
            ipc.send(payload)
            return True
        except Exception:
            return False

    def _handle_llm_event(self, payload: dict) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "ready":
            self._llm_ready.set()
        elif event_type == "fatal":
            self._llm_fatal = str(payload.get("message", "")) or "LLM worker reported a fatal error."
        if self._on_llm_event:
            try:
                self._on_llm_event(payload)
            except Exception:
                pass

    def _teardown_llm(self) -> None:
        with self._llm_ensure_lock:
            if self._llm_ipc is not None:
                self._llm_ipc.stop()
                self._llm_ipc = None
            self.llm.stop()
            self._llm_ready.clear()

    # -- TTS ------------------------------------------------------------------

    def ensure_tts(
        self,
        engine: str,
        config: dict,
        timeout_sec: float = 180.0,
    ) -> None:
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        with self._tts_ensure_lock:
            if (
                self.tts.is_running()
                and self._tts_engine == engine
                and self._tts_config_json == config_json
                and self._tts_ipc is not None
                and self._tts_ipc.is_connected()
                and self._tts_ready.is_set()
            ):
                return
            self._teardown_tts()
            self._tts_ready.clear()
            self._tts_fatal = ""
            ipc = JsonLineDuplexServer("127.0.0.1", 0)
            ipc.start(
                on_event=self._handle_tts_event,
                on_disconnected=lambda: self._handle_worker_disconnect(
                    "TTS", self._tts_ready, self._on_tts_event
                ),
            )
            self._tts_ipc = ipc
            self._tts_engine = engine
            self._tts_config_json = config_json
            proc = self.tts.start(
                python_exe=self._python_exe,
                worker_script=_script("tts", "worker.py"),
                host=ipc.host,
                port=ipc.port,
                engine=engine,
                config_json=config_json,
                forward_console=True,
            )
            self._log(f"[TTS] started pid={proc.pid} engine={engine}")
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if self._tts_ready.is_set():
                    return
                if self._tts_fatal:
                    raise RuntimeError(self._tts_fatal)
                if proc.poll() is not None:
                    error = self._worker_exited_before_ready(proc, "TTS", "_tts_fatal")
                    raise error
                time.sleep(0.05)
            raise RuntimeError("TTS worker ready timeout")

    def send_tts(self, payload: dict) -> bool:
        ipc = self._tts_ipc
        if ipc is None or not ipc.is_connected():
            return False
        try:
            ipc.send(payload)
            return True
        except Exception:
            return False

    def _handle_tts_event(self, payload: dict) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "ready":
            self._tts_ready.set()
        elif event_type == "fatal":
            self._tts_fatal = str(payload.get("message", "")) or "TTS worker reported a fatal error."
        if self._on_tts_event:
            try:
                self._on_tts_event(payload)
            except Exception:
                pass

    def _teardown_tts(self) -> None:
        with self._tts_ensure_lock:
            if self._tts_ipc is not None:
                self._tts_ipc.stop()
                self._tts_ipc = None
            self.tts.stop()
            self._tts_ready.clear()
            self._tts_engine = ""
            self._tts_config_json = ""

    # -- Audio Output ----------------------------------------------------------

    def ensure_audio_output(
        self,
        config: dict,
        timeout_sec: float = 45.0,
    ) -> bool:
        """Audio Output ワーカーを起動し ready を待つ。

        起動〜ready 待ちを 1 つの RLock で直列化する。ロックの外で待つと、
        start_module を複数スレッドから呼んだときに起動直後に別スレッドが teardown し、
        ready 待ちが永久に満たされないレースが起き得る（全 ensure_* が同じ形）。

        **再起動が必要な設定**（output_mode / sample_rate / remote_plugin / tcp 出力先）が
        変わったときだけプロセスを取り直す。`mmd_burst_snd_frames` だけの差分は
        `update_settings` で反映する。
        """
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        new_rd = _audio_output_restart_digest(config)
        with self._audio_output_ensure_lock:
            if (
                self.audio_output.is_running()
                and self._audio_output_ipc is not None
                and self._audio_output_ipc.is_connected()
                and self._audio_output_ready.is_set()
                and self._audio_output_config_json
            ):
                try:
                    old_cfg = json.loads(self._audio_output_config_json)
                except Exception:
                    old_cfg = {}
                if not isinstance(old_cfg, dict):
                    old_cfg = {}
                if _audio_output_restart_digest(old_cfg) == new_rd:
                    try:
                        ob = int(old_cfg.get("mmd_burst_snd_frames", -999999))
                    except (TypeError, ValueError):
                        ob = -999999
                    try:
                        nb = int(config.get("mmd_burst_snd_frames", 20) or 20)
                    except (TypeError, ValueError):
                        nb = 20
                    nb = max(0, min(500, nb))
                    live: dict[str, object] = {}
                    if ob != nb:
                        live["mmd_burst_snd_frames"] = nb
                    old_blink_enable = normalize_blink_enable(
                        old_cfg.get("mmd_blink_enable"), True
                    )
                    new_blink_enable = normalize_blink_enable(
                        config.get("mmd_blink_enable"), True
                    )
                    if old_blink_enable != new_blink_enable:
                        live["mmd_blink_enable"] = new_blink_enable
                    old_blink_interval = normalize_blink_interval_mean_sec(
                        old_cfg.get("mmd_blink_interval_mean_sec", DEFAULT_BLINK_INTERVAL_MEAN_SEC)
                    )
                    new_blink_interval = normalize_blink_interval_mean_sec(
                        config.get("mmd_blink_interval_mean_sec", DEFAULT_BLINK_INTERVAL_MEAN_SEC)
                    )
                    if old_blink_interval != new_blink_interval:
                        live["mmd_blink_interval_mean_sec"] = new_blink_interval
                    old_blink_morph = sanitize_blink_morph(
                        old_cfg.get("mmd_blink_morph", "まばたき")
                    )
                    new_blink_morph = sanitize_blink_morph(
                        config.get("mmd_blink_morph", "まばたき")
                    )
                    if old_blink_morph != new_blink_morph:
                        live["mmd_blink_morph"] = new_blink_morph
                    old_blink_mode = normalize_blink_mode(old_cfg.get("mmd_blink_mode"))
                    new_blink_mode = normalize_blink_mode(config.get("mmd_blink_mode"))
                    if old_blink_mode != new_blink_mode:
                        live["mmd_blink_mode"] = new_blink_mode
                    if live:
                        self.send_audio_output({"type": "update_settings", "settings": live})
                    self._audio_output_config_json = config_json
                    return False
            self._teardown_audio_output_impl()
            self._audio_output_ready.clear()
            self._audio_output_fatal = ""
            ipc = JsonLineDuplexServer("127.0.0.1", 0)
            ipc.start(
                on_event=self._handle_audio_output_event,
                on_disconnected=lambda: self._handle_worker_disconnect(
                    "AudioOutput", self._audio_output_ready, self._on_audio_output_event
                ),
            )
            self._audio_output_ipc = ipc
            proc = self.audio_output.start(
                python_exe=self._python_exe,
                worker_script=_script("audio", "output_worker.py"),
                host=ipc.host,
                port=ipc.port,
                config_json=config_json,
                forward_console=True,
            )
            self._log(f"[AudioOutput] started pid={proc.pid}")
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if self._audio_output_ready.is_set():
                    self._audio_output_config_json = config_json
                    return True
                if self._audio_output_fatal:
                    self._teardown_audio_output_impl()
                    raise RuntimeError(self._audio_output_fatal)
                if proc.poll() is not None:
                    error = self._worker_exited_before_ready(proc, "AudioOutput", "_audio_output_fatal")
                    self._teardown_audio_output_impl()
                    raise error
                time.sleep(0.05)
            self._teardown_audio_output_impl()
            raise RuntimeError("AudioOutput worker ready timeout")

    def send_audio_output(self, payload: dict) -> bool:
        ipc = self._audio_output_ipc
        if ipc is None or not ipc.is_connected():
            return False
        try:
            ipc.send(payload)
            return True
        except Exception:
            return False

    def _handle_audio_output_event(self, payload: dict) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "ready":
            self._audio_output_ready.set()
        elif event_type == "fatal":
            self._audio_output_fatal = (
                str(payload.get("message", "")) or "AudioOutput worker reported a fatal error."
            )
        if self._on_audio_output_event:
            try:
                self._on_audio_output_event(payload)
            except Exception:
                pass

    def _teardown_audio_output_impl(self) -> None:
        self._audio_output_config_json = ""
        if self._audio_output_ipc is not None:
            self._audio_output_ipc.stop()
            self._audio_output_ipc = None
        self.audio_output.stop()
        self._audio_output_ready.clear()

    def _teardown_audio_output(self) -> None:
        with self._audio_output_ensure_lock:
            self._teardown_audio_output_impl()

    # -- ASR -------------------------------------------------------------------

    def ensure_asr(
        self,
        config: dict,
        ipc_host: str,
        ipc_port: int,
        timeout_sec: float | None = None,
    ) -> None:
        """ASR は Orchestrator 内の JsonLineEventServer（1本の Duplex）に接続する。"""
        if timeout_sec is None:
            from ..dialog.settings import asr_worker_ready_timeout_sec
            timeout_sec = asr_worker_ready_timeout_sec(config)
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        route = (ipc_host, ipc_port)
        with self._asr_ensure_lock:
            if (
                self.asr.is_running()
                and self._asr_ready.is_set()
                and self._asr_config_json == config_json
                and self._asr_route == route
            ):
                return
            self._teardown_asr()
            self._asr_ready.clear()
            self._asr_fatal = ""
            proc = self.asr.start(
                python_exe=self._python_exe,
                worker_script=_script("asr", "worker.py"),
                host=ipc_host,
                port=ipc_port,
                config_json=config_json,
                forward_console=True,
            )
            self._log(f"[ASR] started pid={proc.pid}")
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if self._asr_ready.is_set():
                    self._asr_config_json = config_json
                    self._asr_route = route
                    return
                if self._asr_fatal:
                    self._teardown_asr()
                    raise RuntimeError(self._asr_fatal)
                if proc.poll() is not None:
                    error = self._worker_exited_before_ready(proc, "ASR", "_asr_fatal")
                    self._teardown_asr()
                    raise error
                time.sleep(0.05)
            self._teardown_asr()
            raise RuntimeError("ASR worker ready timeout")

    def mark_asr_ready(self) -> None:
        self._asr_ready.set()

    def mark_asr_fatal(self, message: str) -> None:
        self._asr_fatal = message or "ASR worker reported a fatal error."

    def _teardown_asr(self) -> None:
        with self._asr_ensure_lock:
            self.asr.stop()
            self._asr_ready.clear()
            self._asr_config_json = ""
            self._asr_route = None

    # -- ASR Compare -----------------------------------------------------------

    def ensure_asr_compare(
        self,
        config: dict,
        ipc_host: str,
        ipc_port: int,
        timeout_sec: float = 90.0,
    ) -> None:
        with self._asr_compare_ensure_lock:
            if self.asr_compare.is_running() and self._asr_compare_ready.is_set():
                return
            self._teardown_asr_compare()
            self._asr_compare_ready.clear()
            self._asr_compare_fatal = ""
            config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
            proc = self.asr_compare.start(
                python_exe=self._python_exe,
                worker_script=_script("asr", "compare_worker.py"),
                host=ipc_host,
                port=ipc_port,
                config_json=config_json,
                forward_console=True,
            )
            self._log(f"[ASRCompare] started pid={proc.pid}")
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if self._asr_compare_ready.is_set():
                    return
                if self._asr_compare_fatal:
                    raise RuntimeError(self._asr_compare_fatal)
                if proc.poll() is not None:
                    error = self._worker_exited_before_ready(proc, "ASRCompare", "_asr_compare_fatal")
                    raise error
                time.sleep(0.05)
            raise RuntimeError("ASR compare worker ready timeout")

    def mark_asr_compare_ready(self) -> None:
        self._asr_compare_ready.set()

    def mark_asr_compare_fatal(self, message: str) -> None:
        self._asr_compare_fatal = message or "ASR compare worker reported a fatal error."

    def _teardown_asr_compare(self) -> None:
        with self._asr_compare_ensure_lock:
            self.asr_compare.stop()
            self._asr_compare_ready.clear()

    # -- MaAI ------------------------------------------------------------------

    def ensure_maai(
        self,
        config: dict,
        ipc_host: str,
        ipc_port: int,
        # モデルロード（bc + nod_para + turn_vap 併用や CPU 実行）で 60s を超えることが
        # あるため余裕を持たせる。失敗時は module_start_failed が返るので長くても実害は薄い。
        timeout_sec: float = 180.0,
    ) -> None:
        with self._maai_ensure_lock:
            if self.maai.is_running() and self._maai_ready.is_set():
                return
            self._teardown_maai()
            self._maai_ready.clear()
            self._maai_fatal = ""
            config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
            proc = self.maai.start(
                python_exe=self._python_exe,
                worker_script=_script("maai", "worker.py"),
                host=ipc_host,
                port=ipc_port,
                config_json=config_json,
                forward_console=True,
            )
            self._log(f"[MaAI] started pid={proc.pid}")
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if self._maai_ready.is_set():
                    return
                if self._maai_fatal:
                    raise RuntimeError(self._maai_fatal)
                if proc.poll() is not None:
                    error = self._worker_exited_before_ready(proc, "MaAI", "_maai_fatal")
                    raise error
                time.sleep(0.05)
            raise RuntimeError("MaAI worker ready timeout")

    def _worker_exited_before_ready(self, proc, label: str, fatal_attr: str) -> RuntimeError:
        """ready を待っている間にワーカーが終わったときのエラー。

        以前は終了に気づかず、ready の時間切れ（MaAI は 180 秒）まで待ってから失敗にしていた。
        落ちる直前に送られた fatal（理由つき）が届くかもしれないので少しだけ待ち、あればそれを使う。
        """
        deadline = time.time() + 0.5
        while time.time() < deadline and not getattr(self, fatal_attr, ""):
            time.sleep(0.05)
        fatal = getattr(self, fatal_attr, "")
        if fatal:
            return RuntimeError(fatal)
        return RuntimeError(
            f"{label} ワーカーが準備完了の前に終了しました（終了コード {proc.returncode}）。"
            f"プロセスコンソールの [{label}] の出力に理由が出ています。"
        )

    def mark_maai_ready(self) -> None:
        self._maai_ready.set()

    def mark_maai_fatal(self, message: str) -> None:
        self._maai_fatal = message or "MaAI worker reported a fatal error."

    def _teardown_maai(self) -> None:
        with self._maai_ensure_lock:
            self.maai.stop()
            self._maai_ready.clear()

    # -- Audio Input -----------------------------------------------------------

    def ensure_audio_input(
        self,
        config: dict,
        ipc_host: str,
        ipc_port: int,
        timeout_sec: float = 15.0,
    ) -> bool:
        """起動済みかつ設定 JSON が同一なら何もしない。変わったときだけプロセスを取り直す。"""
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        with self._audio_input_ensure_lock:
            if (
                self.audio_input.is_running()
                and self._audio_input_ready.is_set()
                and self._audio_input_config_json == config_json
            ):
                return False
            self._teardown_audio_input()
            self._audio_input_ready.clear()
            self._audio_input_fatal = ""
            proc = self.audio_input.start(
                python_exe=self._python_exe,
                worker_script=_script("audio", "input_worker.py"),
                host=ipc_host,
                port=ipc_port,
                config_json=config_json,
                forward_console=True,
            )
            self._log(f"[AudioInput] started pid={proc.pid}")
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if self._audio_input_ready.is_set():
                    self._audio_input_config_json = config_json
                    return True
                if self._audio_input_fatal:
                    self._teardown_audio_input()
                    raise RuntimeError(self._audio_input_fatal)
                if proc.poll() is not None:
                    error = self._worker_exited_before_ready(proc, "AudioInput", "_audio_input_fatal")
                    self._teardown_audio_input()
                    raise error
                time.sleep(0.05)
            self._teardown_audio_input()
            raise RuntimeError("AudioInput worker ready timeout")

    def mark_audio_input_ready(self) -> None:
        self._audio_input_ready.set()

    def mark_audio_input_fatal(self, message: str) -> None:
        self._audio_input_fatal = message or "AudioInput worker reported a fatal error."

    def send_audio_input_command(self, command: str) -> None:
        """AudioInput ワーカーの stdin にコマンドを送る。"""
        self.audio_input.send_command(command)

    def _teardown_audio_input(self) -> None:
        with self._audio_input_ensure_lock:
            self._audio_input_config_json = ""
            self.audio_input.stop()
            self._audio_input_ready.clear()

    # -- Lifecycle -------------------------------------------------------------

    def stop_all(self) -> None:
        # 終了は速さ優先。各ワーカーへ stop を投げて短時間だけ待ち即 terminate する処理を
        # 並列実行する（直列の graceful 待ち(各3秒)が累積して Ctrl+C 終了が遅くなる/
        # orchestrator が終われず controller に強制終了されてワーカーが孤児化するのを防ぐ）。
        _FAST_STOP_SEC = 0.6
        workers = (
            self.llm, self.tts, self.audio_output, self.asr,
            self.asr_compare, self.maai, self.audio_input,
        )

        def _safe_fast_stop(w: Any) -> None:
            try:
                w.stop(timeout_sec=_FAST_STOP_SEC)
            except Exception:
                pass

        threads = [
            threading.Thread(target=_safe_fast_stop, args=(w,), daemon=True)
            for w in workers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_FAST_STOP_SEC + 1.5)

        # 後始末（フラグ/ロック等のクリア）。.stop() は既に完了しており process=None で即 return。
        self._teardown_llm()
        self._teardown_tts()
        self._tts_bridge_handler.shutdown()
        self._teardown_audio_output()
        self._teardown_asr()
        self._teardown_asr_compare()
        self._teardown_maai()
        self._teardown_audio_input()

    def stop_voice_dialog_workers(self, *, keep_maai: bool = False, keep_asr: bool = False) -> None:
        """音声対話終了時に呼ぶ。ASR / ASR Compare / MaAI を停止する。

        Audio Input / Output は MMD 接続や先行起動を維持するため止めない。
        """
        if not keep_asr:
            self._teardown_asr()
            self._teardown_asr_compare()
        if not keep_maai:
            self._teardown_maai()

    def stop_module(self, name: str) -> None:
        """モジュール名を指定して個別に停止する。"""
        fn = {
            "asr": self._teardown_asr,
            "asr_compare": self._teardown_asr_compare,
            "llm": self._teardown_llm,
            "tts": self._teardown_tts,
            "maai": self._teardown_maai,
            "audio_input": self._teardown_audio_input,
            "audio_output": self._teardown_audio_output,
        }.get(name.strip().lower())
        if fn is not None:
            fn()
