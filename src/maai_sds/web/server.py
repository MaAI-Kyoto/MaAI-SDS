"""FastAPI bridge between the browser UI and the dialog orchestrator."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:
    # `web` is an optional uv extra (the GUI and the headless Docker image both run
    # without it), so a plain `uv sync` does not install FastAPI/uvicorn. Same treatment
    # as main.py gives PySide6: fail with the fix instead of a raw ModuleNotFoundError.
    sys.stderr.write(
        "FastAPI / uvicorn がインストールされていません。Web UI には 'web' extra が必要です。\n"
        "次を実行してから、もう一度お試しください:\n"
        "  uv sync --extra web\n"
    )
    sys.exit(1)

from .. import device as device_rules, dialog_server_mode, engine_manager, paths
from ..asr import qwen_docker
from ..dialog_server_docker import DockerComposeTarget
from ..ipc.worker_runtime import PARENT_PID_ENV
from ..config_store import ConfigStore
from ..dialog import settings as vds
from ..gui.app_config_spec import APP_CONFIG_FIELDS
from ..gui.settings import BOOL_KEYS, FIELD_SPECS
from ..ipc.jsonline import JsonLineDuplexClient
from ..ipc.process_service import MMDProcessService
from . import parity as web_parity
from .obs_link import ObsRecordingSessions
from .mmdagent import (
    is_docker_environment,
    DOCKER_MDF_READ_ONLY_REASON,
    MmdAgentConfigurationError,
    MmdAgentLaunchUnavailable,
    WebMmdAgentController,
    WebRuntimePolicy,
)


WEB_APP_DIR = paths.REPO_ROOT / "web" / "app"
# PySide のウィジェット木から吸い出した画面定義。生成は scripts/extract_web_ui_manifest.py。
# ここで読むだけなので、この配信経路自体は PySide を必要としない。
UI_MANIFEST_PATH = Path(__file__).resolve().parent / "ui_manifest.json"
_HOST_RELEASE = Path(r"C:\MMDAgent-EX\Release")
_DOCKER_RELEASE = Path("/app/mmdagent_assets")


def _without_mdf_keys(data: dict[str, Any]) -> dict[str, Any]:
    mdf_keys = set(web_parity.mdf_field_keys())
    return {k: v for k, v in data.items() if k not in mdf_keys}


def resolve_mmd_assets_root(cfg: dict[str, Any] | None = None) -> Path:
    """MDF 編集とファイル参照で許可する MMDAgent-EX の Release ディレクトリ。"""
    env = os.environ.get("MAAI_SDS_MMD_ASSETS", "").strip()
    if env:
        return Path(env)
    mdf = str((cfg or {}).get("mdf_path") or "").strip()
    if mdf:
        parent = Path(mdf).parent
        if parent.is_dir():
            return parent
    candidates = (_HOST_RELEASE, _DOCKER_RELEASE) if os.name == "nt" else (_DOCKER_RELEASE, _HOST_RELEASE)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return _HOST_RELEASE if os.name == "nt" else _DOCKER_RELEASE


def resolve_mdf_path(cfg: dict[str, Any] | None, assets_root: Path) -> Path:
    mdf = str((cfg or {}).get("mdf_path") or "").strip()
    if mdf:
        path = Path(mdf)
        if path.is_file():
            return path
    # 設定が無いか見つからないときの候補。自分で用意した gene の MDF と、公式のサンプル集
    # （MMDAGENT_ASSETS_HOST=C:/MMDAgent-EX/example）に入れた MaAI-SDS 用の MDF、サンプル集の WebSocket の例。
    gene = assets_root / "MMDAgent-EX_gene.mdf"
    for candidate in (
        gene,
        assets_root / "maai_sds" / "maai_sds.mdf",
        assets_root / "example_websocket" / "example_websocket.mdf",
    ):
        if candidate.is_file():
            return candidate
    return Path(mdf) if mdf else gene


class HostPathView:
    """Docker の対話サーバーを使うときの、コンテナのパス ⇔ この PC のパスの変換。

    設定（controller_config.docker.json）にはコンテナの中のパスが入る（対話サーバーが読むため）。
    この Web サーバーが自分でファイルを読み書きするとき（MDF・ファイル参照・WAV の再生）は
    この PC のパスに直し、参照ダイアログで選んだファイルはコンテナのパスに直して設定に入れる。
    Qt の Docker モード（controller._docker_backend_path_from_picker など）と同じ考え方。
    """

    _HIDDEN_ROOTS = frozenset({"/app/scripts", "/app/web", "/app/src"})

    def __init__(self, mapper: Any | None) -> None:
        self.mapper = mapper

    @property
    def active(self) -> bool:
        return self.mapper is not None

    def to_host(self, value: str) -> str:
        text = str(value or "").strip()
        if not text or self.mapper is None:
            return text
        mapped = self.mapper.container_to_host(text)
        return str(mapped) if mapped is not None else text

    def to_backend(self, value: str) -> str | None:
        """この PC のパスを、対話サーバー（コンテナ）から読めるパスにする。読めなければ None。"""
        text = str(value or "").strip()
        if self.mapper is None:
            return text
        if text.startswith("/") and self.mapper.container_to_host(text) is not None:
            return text
        return self.mapper.host_to_container(text)

    def host_cfg(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """設定のうちコンテナのパスになっている値を、この PC のパスにした写し。"""
        if self.mapper is None:
            return cfg
        out = dict(cfg)
        for key, value in cfg.items():
            if isinstance(value, str) and value.startswith("/"):
                out[key] = self.to_host(value)
        return out

    def mount_roots(self) -> list[str]:
        """参照ダイアログのルート（コンテナのパス）。この PC に実体があるマウントだけ。"""
        if self.mapper is None:
            return []
        roots: list[str] = []
        for mount in getattr(self.mapper, "mounts", []):
            host = Path(getattr(mount, "host", ""))
            container = getattr(mount, "container", None)
            if container is None or getattr(mount, "is_file", False) or not host.is_dir():
                continue
            text = container.as_posix() if hasattr(container, "as_posix") else str(container)
            if text in self._HIDDEN_ROOTS:
                continue  # コードのマウント（選ぶ場面がない）
            if text not in roots:
                roots.append(text)
        return roots


class _HostViewConfig:
    """ConfigStore の読み出しだけを HostPathView に通す（MMDAgent-EX の起動用）。"""

    def __init__(self, store: ConfigStore, view: HostPathView) -> None:
        self._store = store
        self._view = view

    def load(self) -> dict[str, Any]:
        return self._view.host_cfg(self._store.load())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


class OrchestratorBridge:
    """Owns the worker process and safely relays every IPC event to WebSockets."""

    _PCM_EVENT_TYPES = frozenset({"audio_frame", "system_audio_frame", "raw_audio"})

    def __init__(
        self,
        config: ConfigStore,
        *,
        assume_mmd_running: Callable[[], bool] | None = None,
        remote_endpoint: tuple[str, int] | None = None,
        compose_target: Any | None = None,
    ) -> None:
        self.config = config
        self._assume_mmd_running = assume_mmd_running or (lambda: True)
        # Docker の対話サーバー（compose の orchestrator、--listen）につなぐとき。
        # None なら従来どおり、この Web サーバーの子プロセスとして対話サーバーを起動する。
        self.remote_endpoint = remote_endpoint
        # remote のとき ▶ / ■ で docker compose up / stop する（gui/start_qt_docker.ps1 と同じ compose）。
        self.compose_target = compose_target
        self._reconnect_stop = threading.Event()
        self._reconnect_thread: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.client: JsonLineDuplexClient | None = None
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sockets: set[WebSocket] = set()
        self.wav_sockets: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.broadcast_task: asyncio.Task[None] | None = None
        self._console_threads: list[threading.Thread] = []
        self.engine_apply_running = False
        # モジュールごとの稼働状態。エンジン管理は全停止時だけ許すのに使う。
        # `uv sync` は exact sync なので、ワーカーが動いている間の適用は
        # それらがロードしている DLL を置き換えられず失敗する（Windows の torch で実際に
        # 落ちた）。PySide 側の _module_panel_states と同じ役割。
        self.module_states: dict[str, str] = {}
        self.module_errors: dict[str, str] = {}
        self.voice_state = "stopped"
        self.mmd_remote_connected = False
        self.mmd_remote_mode = ""
        self.wav_playing = False
        self.maai_nod_threshold_defaults: dict[str, Any] | None = None
        self.latest_history: dict[str, Any] = {}
        self.startup_error: str | None = None
        # 対話サーバー（dialog worker）の状態。画面の ▶ / ■ で起動・停止できる。
        # running / starting / stopping / stopped / error
        self.server_state = "stopped"
        self.server_error = ""
        self._worker_lock = threading.Lock()
        self._worker_stopping = False
        # 音声対話・モジュールが（操作以外の理由も含めて）止まったときの通知先。
        # OBS 録画の連動解除に使う（PySide の _stop_voice_dialog / _teardown_maai_ipc 相当）。
        self.on_voice_stopped: Callable[[], None] | None = None
        self.on_module_stopped: Callable[[str], None] | None = None

    def _plot_audio_wanted(self) -> bool:
        return any(getattr(socket.state, "plot_audio", False) for socket in self.sockets)

    @staticmethod
    def _compact_plot_audio_event(event: dict[str, Any]) -> dict[str, Any]:
        """Reduce plot PCM to its visible peak envelope before JSON encoding."""
        audio = event.get("audio")
        if not isinstance(audio, list) or not audio:
            return event
        try:
            t0 = float(event.get("ts_start"))
            t1 = float(event.get("ts_end"))
            duration = max(0.0, t1 - t0)
        except (TypeError, ValueError):
            rate = max(1, int(event.get("sample_rate", 16000) or 16000))
            duration = len(audio) / float(rate)
        bins = max(1, min(64, int(round(duration * 80.0))))
        peaks: list[float] = []
        n = len(audio)
        for b in range(bins):
            start = (b * n) // bins
            end = max(start + 1, ((b + 1) * n) // bins)
            peak = 0.0
            for raw in audio[start:min(n, end)]:
                try:
                    peak = max(peak, abs(float(raw)))
                except (TypeError, ValueError):
                    continue
            peaks.append(peak)
        compact = {key: value for key, value in event.items() if key != "audio"}
        compact["audio_peaks"] = peaks
        return compact

    def start_engine_apply(self, ids: list[str], device: str = "") -> None:
        """エンジン適用を別スレッドで走らせ、出力を events としてブラウザへ流す。

        `uv sync` とモデル取得で数分かかることがあるため、HTTP を掴んだままにせず
        既存の WebSocket に相乗りさせる。GUI の `_run_engine_install_worker` と同じ形。
        """
        self.engine_apply_running = True

        def run() -> None:
            try:
                ok, message = engine_manager.set_enabled_engines(
                    ids,
                    # web extra を落とすと、この API を配信しているサーバ自身が壊れる。
                    # GUI が gui extra を always_extras に渡しているのと同じ理由。
                    always_extras=("web",),
                    device=device or None,
                    on_output=lambda line: self._on_event(
                        {"type": "engine_apply_output", "line": line}),
                )
            except Exception as ex:  # noqa: BLE001
                ok, message = False, f"{type(ex).__name__}: {ex}"
            finally:
                self.engine_apply_running = False
            self._on_event({"type": "engine_apply_done", "ok": ok, "message": message})

        threading.Thread(target=run, daemon=True, name="web-engine-apply").start()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        if self.broadcast_task is None:
            self.broadcast_task = loop.create_task(self._broadcast_events())
        if self.remote_endpoint is not None:
            # Docker の対話サーバーはコンテナの起動を待つことがあるので、画面を先に出す
            # （つながるまで「起動中」。失敗したら異常にして ▶ でやり直せる）。
            def run() -> None:
                try:
                    self.start_worker()
                except Exception as ex:  # noqa: BLE001
                    self.startup_error = f"{type(ex).__name__}: {ex}"

            threading.Thread(target=run, daemon=True, name="web-dialog-server-start").start()
            return
        self.start_worker()

    def _set_server_state(self, state: str, error: str = "") -> None:
        self.server_state = state
        self.server_error = error
        self._on_event({"type": "dialog_server", "state": state, "error": error})

    def worker_running(self) -> bool:
        if self.remote_endpoint is not None:
            alive = getattr(self.client, "reader_alive", None)
            return self.client is not None and (alive() if callable(alive) else True)
        return self.client is not None and self.process is not None and self.process.poll() is None

    def start_worker(self) -> None:
        """対話サーバーを起動してつなぐ（ブロックする。HTTP からはスレッドで呼ぶ）。"""
        self._reconnect_stop.set()  # 自動のつなぎ直しは止め、ここでつなぐ
        with self._worker_lock:
            if self.worker_running():
                return
            self._set_server_state("starting")
            try:
                if self.remote_endpoint is not None:
                    if self.compose_target is not None:
                        self._run_compose("start")
                    self.client = self._connect_remote(deadline_sec=120.0)
                else:
                    self.process, port = self._spawn_worker()
                    self.client = JsonLineDuplexClient("127.0.0.1", port)
                    self.client.connect()
                    self.client.start_reader(self._on_event, self._on_disconnected)
            except Exception as ex:
                message = f"{type(ex).__name__}: {ex}"
                self._discard_worker(timeout_sec=2.0)
                self._set_server_state("error", message)
                raise
            self.startup_error = None
            self._set_server_state("running")
            self._on_event({"type": "connection", "state": "connected"})

    def stop_worker(self, timeout_sec: float = 10.0) -> None:
        """対話サーバーを止める。IPC を閉じるとワーカーは各モジュールを止めてから自分で終わる。"""
        self._reconnect_stop.set()
        with self._worker_lock:
            if self.client is None and self.process is None and self.remote_endpoint is None:
                self._set_server_state("stopped")
                return
            self._worker_stopping = True
            self._reconnect_stop.set()
            self._set_server_state("stopping")
            try:
                self._discard_worker(timeout_sec=timeout_sec)
                if self.remote_endpoint is not None and self.compose_target is not None:
                    self._run_compose("stop")
            finally:
                self._worker_stopping = False
            # ワーカーと一緒にモジュールと音声対話も止まっている。画面の表示を合わせる。
            for module, state in list(self.module_states.items()):
                if state != "stopped":
                    self._on_event({"type": "module_state", "module": module, "state": "stopped",
                                    "source": "dialog_server_stopped"})
            if self.voice_state != "stopped":
                self._on_event({"type": "status", "state": "stopped"})
            if self.wav_playing:
                self._on_event({"type": "processing_stopped"})
            self._set_server_state("stopped")

    def _run_compose(self, action: str) -> None:
        verb = "up -d" if action == "start" else "stop"
        self._on_event({"type": "log", "message": f"[対話サーバー] docker compose {verb} orchestrator"})
        ok, output = self.compose_target.run(action)
        for line in output.splitlines()[-5:]:
            self._on_event({"type": "log", "message": f"[対話サーバー] {line}"})
        if not ok:
            raise RuntimeError(f"docker compose {verb} に失敗しました: {output[-300:]}")

    def _connect_remote(self, *, deadline_sec: float | None, stop: threading.Event | None = None) -> Any:
        """Docker の対話サーバーにつなぎ、ready が届いた接続を返す。

        Docker Desktop のポート転送は、コンテナの中がまだ待ち受けていなくても接続を受け付けて
        すぐ切る。切られたら待たずにやり直す（Qt の _connect_remote_dialog_orchestrator と同じ）。
        """
        host, port = self.remote_endpoint  # type: ignore[misc]
        deadline = None if deadline_sec is None else time.monotonic() + deadline_sec
        last_error: Exception | None = None
        while True:
            if stop is not None and stop.is_set():
                raise RuntimeError("つなぎ直しを中止しました")
            client = JsonLineDuplexClient(host, port)
            ready = threading.Event()
            try:
                client.connect(attempts=1, sleep_sec=0.05)

                def on_event(event: dict[str, Any], _client=client, _ready=ready) -> None:
                    if event.get("type") == "ready":
                        _ready.set()
                    # 試しにつないだだけの接続（相手がすぐ切った等）のイベントは流さない
                    if self.client is _client or _ready.is_set():
                        self._on_event(event)

                client.start_reader(on_event, lambda _client=client: self._on_remote_disconnected(_client))
                wait_until = time.monotonic() + 5.0
                while time.monotonic() < wait_until and not ready.is_set() and client.reader_alive():
                    time.sleep(0.02)
                if ready.is_set():
                    return client
                last_error = RuntimeError(
                    "接続がすぐに切れました（対話サーバーの準備中）" if not client.reader_alive()
                    else "接続後の ready 待機がタイムアウトしました"
                )
            except Exception as ex:  # noqa: BLE001
                last_error = ex
            client.close()
            if deadline is not None and time.monotonic() > deadline:
                raise RuntimeError(f"対話サーバー {host}:{port} につながりません: {last_error}")
            time.sleep(0.3)

    def _on_remote_disconnected(self, client: Any) -> None:
        if client is not self.client:
            return  # 試しにつないだ接続が切れただけ
        if not self._worker_stopping:
            self.client = None  # 切れた接続は捨てる（つなぎ直しが新しい接続を入れる）
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        self._on_disconnected()

    def _start_auto_reconnect(self) -> None:
        """Docker の対話サーバーとの接続が切れたら、■ で止めるまでつなぎ直し続ける（Qt と同じ）。"""
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return
        self._reconnect_stop = threading.Event()
        stop = self._reconnect_stop

        def run() -> None:
            # つなぎ直しの間は _worker_lock を持たない（■ / ▶ がすぐ効くように）。
            try:
                client = self._connect_remote(deadline_sec=None, stop=stop)
            except Exception:  # noqa: BLE001
                return
            with self._worker_lock:
                if stop.is_set() or self.client is not None:
                    client.close()
                    return
                self.client = client
                self.startup_error = None
                self._set_server_state("running")
                self._on_event({"type": "connection", "state": "connected"})

        self._reconnect_thread = threading.Thread(target=run, daemon=True, name="web-dialog-server-reconnect")
        self._reconnect_thread.start()

    def _discard_worker(self, *, timeout_sec: float) -> None:
        client, process = self.client, self.process
        self.client = None
        self.process = None
        if client is not None:
            client.close()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    def _spawn_worker(self) -> tuple[subprocess.Popen[str], int]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "maai_sds.dialog.worker",
            "--config",
            str(self.config.path),
        ]
        # Docker/非 Windows は従来どおりホスト側 MMDAgent を待つ。native Windows は
        # Web サーバ自身の MMDProcessService が実際に起動中のときだけ mmd を維持する。
        if self._assume_mmd_running():
            command.append("--assume-mmd-running")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        # このサーバが落ちたら orchestrator も落ちるようにする。ここは
        # WorkerProcessService を通らないので、同じ env を自分で渡す。
        env = os.environ.copy()
        env[PARENT_PID_ENV] = str(os.getpid())
        process = subprocess.Popen(
            command,
            cwd=str(paths.REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            env=env,
        )
        if process.stdout is None:
            raise RuntimeError("Dialog worker stdout is unavailable")

        # PySide と同じく stderr は起動直後からコンソールへ。stdout は ui_port を
        # 取ってから（readline の競合を避ける）。
        self._start_stream_pump(process.stderr, "STDERR", thread_name="web-dialog-worker-stderr")

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = (process.stdout.read() or "").strip()[-1024:]
                raise RuntimeError(f"Dialog worker exited early (rc={process.returncode}): {tail}")
            line = process.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            text = line.rstrip("\r\n")
            try:
                info = json.loads(text) if text else None
            except json.JSONDecodeError:
                self._emit_worker_console_line(text, "STDOUT")
                continue
            if isinstance(info, dict) and int(info.get("ui_port", 0) or 0) > 0:
                self._start_stream_pump(process.stdout, "STDOUT", thread_name="web-dialog-worker-stdout")
                return process, int(info["ui_port"])
            self._emit_worker_console_line(text, "STDOUT")
        process.terminate()
        raise RuntimeError("Timed out waiting for dialog worker ui_port")

    # PySide の controller._ensure_dialog_orchestrator_worker / _orch_ev_log と同じ前置。
    _WORKER_PREFIX_RE = re.compile(r"^\[.+\]\[(STDOUT|STDERR)\] ")
    _CONSOLE_IPC_TYPES = frozenset({"log", "error", "interrupt_condition"})

    def _console_line_from_worker_stream(self, raw: str, stream_name: str) -> str | None:
        text = raw.rstrip("\r\n")
        if not text:
            return None
        if self._WORKER_PREFIX_RE.match(text):
            return text
        return f"[DialogOrchestrator][{stream_name}] {text}"

    def _emit_worker_console_line(self, raw: str, stream_name: str) -> None:
        text = raw.rstrip("\r\n")
        print(f"[dialog-worker] {text}", flush=True)
        msg = self._console_line_from_worker_stream(raw, stream_name)
        if msg:
            self._on_event({"type": "log", "message": msg})

    def _start_stream_pump(
        self,
        stream: Any,
        stream_name: str,
        *,
        thread_name: str,
    ) -> None:
        if stream is None:
            return

        def pump() -> None:
            for line in stream:
                self._emit_worker_console_line(line, stream_name)

        thread = threading.Thread(target=pump, daemon=True, name=thread_name)
        self._console_threads.append(thread)
        thread.start()

    _MODULE_EVENT_STATES = {
        "module_started": "running",
        "module_stopped": "stopped",
        "module_start_failed": "error",
    }

    def _notify(self, callback: Callable[..., None] | None, *args: Any) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as ex:  # noqa: BLE001 - 通知先の失敗でイベント中継を止めない
            print(f"[web] state listener failed: {type(ex).__name__}: {ex}", flush=True)

    def _note_module_state(self, event: dict[str, Any]) -> None:
        """ワーカーのモジュール状態イベントを覚える。"""
        before = dict(self.module_states)
        self._note_module_state_impl(event)
        for module, state in self.module_states.items():
            if state in {"stopped", "error"} and before.get(module) not in {None, "stopped", "error"}:
                self._notify(self.on_module_stopped, module)

    def _note_module_state_impl(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        module = str(event.get("module") or "").strip()
        if kind == "fatal" and not module:
            module = str(event.get("_src") or event.get("worker") or "").strip().lower()
            module = module.replace("_worker", "").replace("worker", "").strip(" _-")
        if not module:
            return
        if kind == "fatal":
            self.module_states[module] = "error"
            self.module_errors[module] = str(
                event.get("message") or event.get("error") or "worker reported a fatal error"
            )
            return
        if kind == "module_state":
            state = str(event.get("state") or "").strip().lower()
            if state:
                self.module_states[module] = state
                error = str(event.get("error") or "").strip()
                if state == "error" and error:
                    self.module_errors[module] = error
                elif state != "error":
                    self.module_errors.pop(module, None)
            return
        state = self._MODULE_EVENT_STATES.get(str(kind))
        if state:
            self.module_states[module] = state
            error = str(event.get("error") or "").strip()
            if state == "error" and error:
                self.module_errors[module] = error
            elif state != "error":
                self.module_errors.pop(module, None)

    def _note_runtime_state(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "status":
            state = str(event.get("state") or "").strip().lower()
            if state:
                previous = self.voice_state
                self.voice_state = state
                if state == "stopped" and previous != "stopped":
                    self._notify(self.on_voice_stopped)
        elif kind == "history_snapshot":
            history = event.get("history")
            if isinstance(history, dict):
                self.latest_history = dict(history)
        elif kind == "wav_command_done":
            command = str(event.get("command") or "").strip().lower()
            if command == "wav_play":
                self.wav_playing = True
            elif command in {"wav_stop", "wav_end"}:
                self.wav_playing = False
        elif kind in {"wav_playback_finished", "processing_stopped"}:
            self.wav_playing = False
        elif kind == "maai_nod_threshold_defaults":
            defaults = event.get("defaults")
            self.maai_nod_threshold_defaults = dict(defaults) if isinstance(defaults, dict) else None
        elif kind == "module_stopped" and str(event.get("module") or "").strip().lower() == "maai":
            self.maai_nod_threshold_defaults = None
        elif kind == "remote_status":
            self.mmd_remote_connected = bool(event.get("connected", False))
            self.mmd_remote_mode = str(event.get("mode") or "")

    def runtime_snapshot(self) -> dict[str, Any]:
        """Return the latest UI state for a newly connected/reloaded browser."""
        return {
            "type": "runtime_snapshot",
            "modules": {
                module: {"state": state, "error": self.module_errors.get(module, "")}
                for module, state in self.module_states.items()
            },
            "voice_state": self.voice_state,
            "mmd_remote": {
                "connected": self.mmd_remote_connected,
                "mode": self.mmd_remote_mode,
            },
            "wav_playing": self.wav_playing,
            "maai_nod_threshold_defaults": self.maai_nod_threshold_defaults,
            "history": dict(self.latest_history),
            "dialog_server": {"state": self.server_state, "error": self.server_error},
        }

    def running_modules(self) -> list[str]:
        return sorted(m for m, st in self.module_states.items()
                      if st in ("running", "starting"))

    # PySide の _orch_ev_worker_disconnected と同じく、LLM / TTS のワーカー IPC が
    # 切れたらモジュールを error 表示にする（放置すると「動作中」のまま残る）。
    _DISCONNECT_MODULES = {"llm": "llm", "tts": "tts"}

    def _on_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "worker_disconnected":
            worker = str(event.get("worker") or "").strip().lower()
            module = self._DISCONNECT_MODULES.get(worker)
            if module:
                self._on_event({
                    "type": "module_state",
                    "module": module,
                    "state": "error",
                    "source": "worker_disconnected",
                    "error": "ワーカー IPC が切断されました",
                })
                self._on_event({"type": "log", "message": f"[Orchestrator] {worker} worker IPC 切断"})
            return
        self._note_module_state(event)
        self._note_runtime_state(event)
        kind = str(event.get("type") or "")
        # PySide の _orch_ev_log: log/error/interrupt_condition をコンソールへ。
        # stdout 中継行は既に最終形なので二重前置しない。
        if kind in self._CONSOLE_IPC_TYPES:
            msg = str(event.get("message") or "").strip()
            if not msg:
                msg = json.dumps(event, ensure_ascii=False)
            if not self._WORKER_PREFIX_RE.match(msg) and not msg.startswith("[DialogOrchestrator]"):
                msg = f"[DialogOrchestrator] {msg}"
            event = {**event, "type": "log", "message": msg}
        # PySide はプロット用 PCM をプロセス内で描く。Web は同じ audio_frame を
        # JSON でブラウザへ流すと、uvicorn のループがマイク WS と取り合いになり
        # MaAI への供給が遅れる。プロットを開いているときだけ中継する。
        if event.get("type") in self._PCM_EVENT_TYPES and not self._plot_audio_wanted():
            return
        if self.loop is not None and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.events.put(event), self.loop)

    def _on_disconnected(self) -> None:
        self._on_event({"type": "remote_status", "connected": False, "mode": ""})
        if self._worker_stopping:
            return  # ■ で止めたとき。状態は stop_worker が更新する。
        self._on_event({"type": "connection", "state": "disconnected"})
        self._set_server_state("error", "対話サーバーとの接続が切れました")
        if self.remote_endpoint is not None:
            self._start_auto_reconnect()

    async def _broadcast_events(self) -> None:
        while True:
            event = await self.events.get()
            if event.get("type") == "wav_monitor_audio":
                if self.wav_sockets:
                    b64 = str(event.get("audio_b64") or "")
                    if b64:
                        try:
                            data = base64.b64decode(b64)
                        except Exception:
                            continue
                        for socket in tuple(self.wav_sockets):
                            try:
                                await socket.send_bytes(data)
                            except Exception:
                                self.wav_sockets.discard(socket)
                continue
            pcm = event.get("type") in self._PCM_EVENT_TYPES
            outgoing = self._compact_plot_audio_event(event) if pcm else event
            targets = tuple(
                socket for socket in self.sockets
                if (not pcm) or getattr(socket.state, "plot_audio", False)
            )
            for socket in targets:
                try:
                    await socket.send_json(outgoing)
                except Exception:
                    self.sockets.discard(socket)

    def send(self, command: dict[str, Any]) -> None:
        if self.client is None:
            raise RuntimeError("対話サーバーが停止しています。モジュール状態の「対話サーバー」の ▶ で起動してください。")
        self.client.send(command)

    def emit_event(self, event: dict[str, Any]) -> None:
        """Publish a server-owned event through the same WebSocket/log path."""
        self._on_event(dict(event))

    async def close(self) -> None:
        if self.broadcast_task is not None:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        self._worker_stopping = True
        self._reconnect_stop.set()
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


_SECURE_SCHEMES = {"https", "wss"}


def _request_server_exit() -> None:
    """uvicorn に Ctrl+C と同じ終了をさせる（lifespan の後片付けが走る）。"""
    import signal

    signal.raise_signal(signal.SIGINT)

# PySide 側で echoMode=Password になっている 5 つのフィールド（ui_manifest.json の
# `"password": true` と同じ集合）。この UI には認証が無いので、`GET /api/config` が
# これらを平文で返すと、到達できる相手に API キーがそのまま渡る。
_SECRET_CONFIG_KEYS = frozenset({
    "llm_api_key",
    "llm_custom_api_key",
    "cartesia_tts_api_key",
    "obs_remote_password",
    "hf_token",
})
# 「値が入っている」ことだけを伝える番兵。これが保存で返ってきたら既存値を維持する。
# 空文字を返してしまうと、UI が全ウィジェット値を POST し直す作りなので
# （app.js の flushAppConfig）、画面を開いただけで秘密が消える。
_SECRET_MASK = "********"


def _mask_secrets(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key in _SECRET_CONFIG_KEYS:
        if str(out.get(key) or ""):
            out[key] = _SECRET_MASK
    return out


# PySide の _save_app_config が数値（int/float）で書くキー。それ以外は文字列で保存する。
_QT_NUMERIC_CONFIG_KEYS: dict[str, type] = {
    "mmd_blink_interval_mean_sec": float,
    "mmd_burst_snd_frames": int,
    "voice_dialog_idle_take_turn_sec": float,
    "voice_dialog_scheduled_exit_sec": float,
}


def _unmask_secrets(incoming: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    """マスクのまま返ってきた秘密は保存せず、既存値を残す。

    利用者が本当に消したいときは空文字が来るので、その場合は素直に消す。
    """
    out = dict(incoming)
    for key in _SECRET_CONFIG_KEYS:
        if key in out and out[key] == _SECRET_MASK:
            if key in stored:
                out[key] = stored[key]
            else:
                out.pop(key)
    return out


def _is_same_origin(origin: str, request: "Request | WebSocket") -> bool:
    """`Origin` ヘッダがこのサーバ自身を指しているか。

    比較相手は **そのリクエスト自身の `Host` ヘッダ**。ブラウザは同一オリジンの
    リクエストで `Origin` を `scheme://<Host と同じ host:port>` として送るので、
    `localhost` / `127.0.0.1` / LAN の IP / `host.docker.internal` といった
    到達経路の違いを許可リストで列挙しなくても、そのまま一致する。

    ポートだけを比較する実装にはしないこと（一度そう書いて塞げていなかった）:
    攻撃者は自分のページを任意のポートで配信できるので、`http://evil.example.com:8000`
    がそのまま一致してしまう。ホストまで見て初めて意味がある。
    """
    try:
        parsed = urlparse(origin)
    except ValueError:
        return False
    if not parsed.scheme or not parsed.hostname:
        return False
    host_header = (request.headers.get("host") or "").strip()
    if not host_header:
        return False
    secure = parsed.scheme in _SECURE_SCHEMES
    origin_port = parsed.port or (443 if secure else 80)

    if host_header.startswith("["):  # IPv6 リテラル [::1]:8000
        closing = host_header.rfind("]")
        host_name = host_header[: closing + 1] if closing > 0 else host_header
        rest = host_header[closing + 1 :] if closing > 0 else ""
        host_port = int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else None
    elif ":" in host_header:
        host_name, _, raw_port = host_header.rpartition(":")
        host_port = int(raw_port) if raw_port.isdigit() else None
    else:
        host_name, host_port = host_header, None
    if host_port is None:
        request_secure = request.url.scheme in _SECURE_SCHEMES
        host_port = 443 if request_secure else 80

    origin_host = parsed.hostname.strip("[]").lower()
    return origin_host == host_name.strip("[]").lower() and origin_port == host_port


def _require_config_file(path: Path) -> None:
    """設定がディレクトリになっていたら、黙って空の設定で動かずに止める。

    Docker の compose は controller_config.docker.json を 1 ファイルでバインドマウント
    する。ホスト側に無いまま起動すると Docker は同名の**ディレクトリ**を作り、
    ConfigStore.load は読み込みの例外を握りつぶして {} を返す。保存も黙って失敗する
    ので、全設定が既定値のまま動き、変更も残らない、という気づきにくい壊れ方になる。
    """
    if path.is_dir():
        raise SystemExit(
            f"[maai-sds] {path} がディレクトリになっています。\n"
            "  Docker の場合: ホスト側で controller_config.docker.example.json を\n"
            "  controller_config.docker.json にコピーしてから起動してください\n"
            "  （同名のディレクトリが作られていたら先に消す）。"
        )


def create_app(
    *,
    config_store: ConfigStore | None = None,
    bridge_override: Any | None = None,
    mmd_process_service: MMDProcessService | None = None,
    platform: str | None = None,
    docker: bool | None = None,
    watch_mmd_process: bool = True,
    remote_orchestrator: str | None = None,
    path_mapper: Any | None = None,
    qwen_local_compose_target: Any | None = None,
) -> FastAPI:
    # この PC の Web サーバーが Docker の対話サーバーにつなぐか（maai_sds/dialog_server_mode.py）。
    # main() が保存した選択から MAAI_SDS_REMOTE_ORCHESTRATOR などを用意する。
    remote_text = (
        str(remote_orchestrator).strip()
        if remote_orchestrator is not None
        else os.environ.get("MAAI_SDS_REMOTE_ORCHESTRATOR", "").strip()
    )
    in_container = is_docker_environment() if docker is None else bool(docker)
    remote_endpoint = dialog_server_mode.parse_endpoint(remote_text) if remote_text and not in_container else None
    env_config = os.environ.get("MAAI_SDS_CONTROLLER_CONFIG", "").strip()
    config_path = (
        config_store.path if config_store is not None
        else Path(env_config) if env_config
        else paths.config_path()
    )
    _require_config_file(config_path)
    config = config_store or ConfigStore(config_path)
    process_service = mmd_process_service or MMDProcessService()
    runtime_policy = WebRuntimePolicy.detect(
        process_service,
        platform=platform,
        docker=docker,
        remote_backend=remote_endpoint is not None,
    )
    if remote_endpoint is not None and path_mapper is None:
        from ..docker_paths import DockerPathMapper

        try:
            path_mapper = DockerPathMapper.from_compose_project(paths.REPO_ROOT)
        except Exception as ex:  # noqa: BLE001
            print(f"[web] Docker のマウントを読めません（パスは変換しません）: {ex}", flush=True)
    # この PC の対話サーバーで Qwen3-ASR を使うときの compose（Qt の scripts/start_qwen3_asr.ps1 と同じファイル）
    qwen_local_compose = qwen_local_compose_target or qwen_docker.local_compose_target(paths.REPO_ROOT)
    path_view = HostPathView(path_mapper if remote_endpoint is not None else None)
    bridge = bridge_override or OrchestratorBridge(
        config,
        assume_mmd_running=runtime_policy.assume_mmd_running,
        remote_endpoint=remote_endpoint,
        compose_target=DockerComposeTarget.from_env() if remote_endpoint is not None else None,
    )
    mmdagent = WebMmdAgentController(
        _HostViewConfig(config, path_view) if path_view.active else config,
        bridge,
        runtime_policy,
        watch_process=watch_mmd_process,
    )
    if in_container:
        dialog_server_where = "container"
    elif remote_endpoint is not None:
        dialog_server_where = "docker"
    else:
        dialog_server_where = "local"

    def _console_log(message: str) -> None:
        print(message, flush=True)
        relay = getattr(bridge, "_on_event", None)
        if callable(relay):
            relay({"type": "log", "message": message})

    obs_sessions = ObsRecordingSessions(lambda: _merged_config(), _console_log)
    if hasattr(bridge, "on_voice_stopped"):
        bridge.on_voice_stopped = lambda: obs_sessions.release_in_background("voice")
    if hasattr(bridge, "on_module_stopped"):
        bridge.on_module_stopped = (
            lambda module: obs_sessions.release_in_background("maai") if module == "maai" else None
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            bridge.start(asyncio.get_running_loop())
        except Exception as ex:  # noqa: BLE001
            # ワーカーが上がらなくてもサーバは起動する。設定もエンジン管理も
            # ワーカー抜きで使える画面で、ここで落とすと「モデルが無いので直したいのに
            # 直す画面が開けない」状態になる。接続状態は UI 側へ流す。
            # イベントの配信は残す（直したあと、画面の ▶ で対話サーバーを起動できる）。
            if getattr(bridge, "broadcast_task", None) is None:
                close = getattr(bridge, "close", None)
                if callable(close):
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            print(f"[web] dialog worker を起動できませんでした: {type(ex).__name__}: {ex}", flush=True)
            bridge.startup_error = f"{type(ex).__name__}: {ex}"
        yield
        mmdagent.close()
        close = getattr(bridge, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result

    app = FastAPI(title="MaAI-SDS Web", lifespan=lifespan)
    # Also expose these without running lifespan so API tests can use fakes without
    # starting the dialog worker, opening audio devices, or binding bridge ports.
    app.state.bridge = bridge
    app.state.mmdagent = mmdagent
    app.state.obs_sessions = obs_sessions
    app.state.runtime_policy = runtime_policy
    app.state.path_view = path_view
    app.state.dialog_server_where = dialog_server_where

    @app.middleware("http")
    async def _block_cross_origin(request: Request, call_next):
        """他サイトから叩かれる書き込みリクエストを落とす（CSRF 対策）。

        この UI には認証が無い（main() 参照）。ブラウザは同一オリジンでなくても
        「単純リクエスト」なら preflight 無しで POST を通すので、127.0.0.1 に
        bind していても、利用者が悪意あるページを開いただけで設定書き換えや
        対話プロセス起動まで到達できてしまう（レスポンスは読めなくても書き込みは成立する）。

        GET/HEAD は副作用が無い前提なので素通しし、状態を変える verb だけ弾く。
        `Origin` が無いリクエスト（curl や同一オリジンの一部ブラウザ挙動）は
        ブラウザ発のクロスオリジンではないので通す。
        """
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and not _is_same_origin(origin, request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "cross-origin request refused"},
                )
        return await call_next(request)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(
            WEB_APP_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    # アイコン（Qt の窓と同じ絵。gui/app_icon.py）
    @app.get("/favicon.ico")
    async def favicon() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "favicon.ico", media_type="image/x-icon")

    @app.get("/icon.svg")
    async def icon_svg() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "icon.svg", media_type="image/svg+xml")

    @app.get("/app.js")
    async def app_js() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "app.js", media_type="application/javascript")

    @app.get("/app.css")
    async def app_css() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "app.css", media_type="text/css")

    @app.get("/plot.js")
    async def plot_js() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "plot.js", media_type="application/javascript")

    @app.get("/plot-view")
    async def plot_view() -> FileResponse:
        return FileResponse(
            WEB_APP_DIR / "plot.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/plot-page.js")
    async def plot_page_js() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "plot-page.js", media_type="application/javascript")

    @app.get("/mic-resampler.js")
    async def mic_resampler_js() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "mic-resampler.js", media_type="application/javascript")

    @app.get("/audio-player.js")
    async def audio_player_js() -> FileResponse:
        return FileResponse(WEB_APP_DIR / "audio-player.js", media_type="application/javascript")

    assets_root = resolve_mmd_assets_root(
        web_parity.merge_app_ui_defaults(config.load(), fill_mmdagent_paths=not runtime_policy.docker)
    )
    app.state.mmd_assets_root = assets_root

    @app.get("/api/config/schema")
    async def config_schema() -> dict[str, Any]:
        return {
            "app_config_fields": [list(row) for row in APP_CONFIG_FIELDS],
            "mdf_field_specs": {category: [list(row) for row in rows] for category, rows in FIELD_SPECS.items()},
            "bool_keys": sorted(BOOL_KEYS),
        }

    @app.get("/api/ui/manifest")
    async def ui_manifest() -> dict[str, Any]:
        if not UI_MANIFEST_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail="ui_manifest.json がありません。"
                " scripts/extract_web_ui_manifest.py を実行して生成してください。",
            )
        manifest = json.loads(UI_MANIFEST_PATH.read_text(encoding="utf-8"))
        web_parity.customize_ui_manifest(manifest)
        # 「この環境で選ばせない選択肢」は生成物に焼かず、ここで足す。マニフェストへ
        # 入れると抽出した人のマシンが GPU かどうかが全ユーザーへ配られてしまう
        # （同種の機械依存を先に 2 件直している）。
        manifest["disabled_choices"] = {
            attr: list(values) for attr, values in device_rules.disabled_choices().items()
        }
        manifest["disabled_choices_reason"] = device_rules.GPU_ONLY_REASON
        return manifest

    @app.get("/api/engines")
    async def list_engines() -> dict[str, Any]:
        return {
            "engines": [
                {
                    "id": e.id, "label": e.label, "extra": e.extra,
                    "installed": e.installed,
                    "has_setup": bool(e.setup_steps or e.setup_venv_script),
                    "kind": engine_manager.engine_kind(e),
                    "download_gb": e.download_gb,
                }
                for e in engine_manager.list_engines()
            ],
            "enabled": engine_manager.get_enabled_engines(),
            "setup_supported": engine_manager.setup_is_supported(),
            "platform": sys.platform,
            # torch と onnxruntime の CUDA ランタイムは cpu / cu128 の排他 extra。
            # setup_env.ps1 を通らず uv だけで入れた場合、ここが唯一の選択場所になる。
            "device_extra": engine_manager.device_extra(),
            "device_extra_locked": engine_manager.device_extra_is_locked(),
        }

    @app.post("/api/engines/plan")
    async def plan_engines(request: Request) -> dict[str, Any]:
        """チェックの状態から「適用」で入れるもの・外すものと取得量を返す（Qt と同じ計算）。"""
        payload = await request.json()
        ids = payload.get("ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list):
            raise HTTPException(status_code=422, detail="ids must be a list of engine ids")
        return engine_manager.plan_engine_changes(engine_manager.list_engines(), [str(i) for i in ids])

    @app.post("/api/engines/apply")
    async def apply_engines(request: Request) -> dict[str, Any]:
        payload = await request.json()
        ids = payload.get("ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list):
            raise HTTPException(status_code=422, detail="ids must be a list of engine ids")
        if runtime_policy.remote_backend:
            raise HTTPException(
                status_code=409,
                detail="Docker の対話サーバーを使っている間は、エンジンはコンテナの中にあるため、"
                "ここからは入れ替えられません。ローカルに切り替えるか、コンテナのイメージ側で変えてください。",
            )
        if not engine_manager.setup_is_supported():
            raise HTTPException(
                status_code=501,
                detail=f"この環境ではエンジンのセットアップを実行できません（{sys.platform}）。",
            )
        if bridge.engine_apply_running:
            raise HTTPException(status_code=409, detail="エンジン適用はすでに実行中です。")
        busy = bridge.running_modules()
        if busy:
            raise HTTPException(
                status_code=409,
                detail=(
                    "エンジン管理は全モジュールが停止しているときだけ使えます。"
                    f"稼働中: {', '.join(busy)}。"
                    "対話を停止してからもう一度お試しください。"
                ),
            )
        device = str(payload.get("device", "") or "").strip()
        bridge.start_engine_apply([str(i) for i in ids], device)
        return {"ok": True}

    def _to_display_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
        """保存値を「画面に出す形」へ変換する。**保存はしない表示専用の変換**。

        Docker ではコンテナから見えるパス/ホストへ読み替える必要があるが、UI は
        ほぼ全コマンドの前に全ウィジェット値を POST し直す（app.js の flushAppConfig）。
        そのため、この変換結果をそのまま保存すると往復のたびに元の値が壊れる。
        特に `resolve_docker_local_file` はコンテナから見えない Windows 絶対パスを
        **空文字にする** ので、モデルパス 6 件が消える。しかも
        `controller_config.docker.json` はバインドマウントなのでホスト側が壊れる。

        保存側（save_config）はこの関数をもう一度呼び、「送られてきた値が
        表示用変換の結果と同じ＝利用者は触っていない」ものを元の値へ戻す。
        """
        cfg = dict(raw_cfg)
        if runtime_policy.docker:
            from ..llm.common import normalize_docker_loopback_host, normalize_lmstudio_base_url
            from ..paths import posixify_venv_python

            cfg["llm_host"] = normalize_lmstudio_base_url(
                str(cfg.get("llm_host") or "http://127.0.0.1:1234")
            )
            custom = str(cfg.get("llm_custom_base_url") or "").strip()
            if custom:
                cfg["llm_custom_base_url"] = normalize_lmstudio_base_url(custom)
            for key, venv_dir in (
                ("irodori_python", ".venv_irodori"),
                ("asr_nemotron_python", ".venv_nemotron"),
            ):
                cfg[key] = posixify_venv_python(str(cfg.get(key) or ""), venv_dir)
            for key in (
                "aivisspeech_host",
                "tts_host",
                "asr_qwen_host",
                "asr_external_host",
                "asr_external_audio_host",
                "asr_nemotron_remote_host",
                "qwen3_tts_cpp_host",
                "supertonic_host",
                "lfm2_audio_host",
                "irodori_audiocpp_host",
            ):
                raw = str(cfg.get(key) or "").strip()
                if raw:
                    cfg[key] = normalize_docker_loopback_host(raw)
        wav_raw = str(cfg.get("audio_input_wav_path") or "").strip()
        if wav_raw:
            from ..paths import resolve_runtime_wav_path

            resolved_wav = resolve_runtime_wav_path(wav_raw)
            if Path(resolved_wav).is_file():
                cfg["audio_input_wav_path"] = resolved_wav
        for key in (
            "maai_test_bc_local_model",
            "maai_test_turn_vap_local_model",
            "maai_mimi_onnx_path",
            "maai_mimi_onnx_meta_path",
            "maai_backchannel_dictionary_path",
        ):
            cfg[key] = paths.resolve_docker_local_file(str(cfg.get(key) or ""))
        return cfg

    def _merged_config(raw_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        return web_parity.merge_app_ui_defaults(
            config.load() if raw_cfg is None else raw_cfg,
            fill_mmdagent_paths=not runtime_policy.docker,
        )

    def _runtime_options(cfg: dict[str, Any]) -> tuple[bool, bool]:
        return (
            runtime_policy.assume_mmd_running(),
            runtime_policy.force_browser_mic(cfg),
        )

    def _push_runtime_settings(cfg: dict[str, Any]) -> None:
        """Mirror the desktop controller's live Audio I/O settings sync."""
        merged = _merged_config(cfg)
        assume_mmd_running, force_browser_mic = _runtime_options(merged)
        settings = vds.build_start_voice_dialog_payload(
            merged,
            assume_mmd_running=assume_mmd_running,
            force_browser_mic=force_browser_mic,
        )
        bridge.send({"type": "update_settings", "settings": settings})
        workers = settings.get("worker_configs") if isinstance(settings, dict) else None
        worker_configs = workers if isinstance(workers, dict) else {}
        audio_output = worker_configs.get("audio_output")
        if isinstance(audio_output, dict) and audio_output:
            bridge.send(
                {
                    "type": "start_module",
                    "module": "audio_output",
                    "config": audio_output,
                    "source": "web-settings",
                }
            )

        # AudioInput is intentionally on demand.  Re-ensure only when a Web
        # module/voice dialog already owns it, so merely opening settings never
        # opens a local microphone.
        module_states = getattr(bridge, "module_states", {})
        input_running = isinstance(module_states, dict) and str(
            module_states.get("audio_input", "")
        ).lower() in {"starting", "running"}
        voice_running = str(getattr(bridge, "voice_state", "")).lower() in {
            "starting",
            "running",
        }
        if input_running or voice_running:
            audio_input = worker_configs.get("audio_input")
            if isinstance(audio_input, dict) and audio_input:
                bridge.send(
                    {
                        "type": "start_module",
                        "module": "audio_input",
                        "config": audio_input,
                        "source": "web-settings",
                    }
                )

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        merged = _merged_config(_without_mdf_keys(config.load()))
        return _mask_secrets(_to_display_config(merged))

    @app.post("/api/config")
    async def save_config(request: Request) -> dict[str, bool]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Configuration must be a JSON object")
        # UI の MDF 欄は json に混ぜない。Docker 用に既に入っている Plugin_Remote_* は残す。
        stored = config.load()
        incoming = web_parity.canonical_combo_values(
            _unmask_secrets(_without_mdf_keys(payload), stored)
        )
        if "audio_output_mode" in incoming:
            incoming["audio_output_mode"] = web_parity.canonical_audio_output_mode(
                incoming.get("audio_output_mode")
            )
        # 表示用に書き換えた値が、そのまま返ってきただけなら保存しない（元の値を維持）。
        # UI は利用者が触っていないウィジェットも含めて全部送り返してくるので、
        # これが無いと Docker で画面を開いてボタンを 1 回押すだけで設定が壊れる。
        # 元のファイルに無いキーは守る値が無いので、送られてきた値をそのまま保存する。
        displayed = _to_display_config(stored)
        for key, shown in displayed.items():
            if key in incoming and key in stored and incoming[key] == shown and stored[key] != shown:
                incoming[key] = stored[key]
        payload = {**stored, **incoming}
        # Qt が数値で保存するキーは数値に戻す（Web の入力欄は文字列で返す）。
        for key, cast in _QT_NUMERIC_CONFIG_KEYS.items():
            raw = payload.get(key)
            if isinstance(raw, str) and raw.strip():
                try:
                    payload[key] = cast(float(raw)) if cast is int else cast(raw)
                except ValueError:
                    pass
        # 話者は Qt と同じくエンジンごとのキーにも書く。実行時はそちらが優先されるので、
        # tts_voice だけ書くと Web で話者を変えても反映されない。
        engine_voice_key = vds.TTS_VOICE_CONFIG_KEYS_SAVED_BY_UI.get(
            str(payload.get("tts_engine") or "").strip()
        )
        voice = str(payload.get("tts_voice") or "").strip()
        if engine_voice_key and voice and "tts_voice" in incoming:
            payload[engine_voice_key] = voice
        try:
            config.save_raising(payload)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        # A newly selected Release directory must immediately become available
        # to the MDF/file APIs; this used to remain frozen at server startup.
        app.state.mmd_assets_root = resolve_mmd_assets_root(_merged_config(payload))
        try:
            _push_runtime_settings(payload)
        except Exception:
            # Persisting configuration remains useful if the worker is unavailable.
            pass
        return {"ok": True}

    @app.get("/api/mmdagent")
    async def mmdagent_status() -> dict[str, Any]:
        return mmdagent.status()

    @app.post("/api/mmdagent/{action}")
    async def mmdagent_action(action: str) -> dict[str, Any]:
        if action not in {"start", "stop"}:
            raise HTTPException(status_code=404, detail="unknown MMDAgent action")
        available, reason = runtime_policy.mmd_launch_availability()
        if not available:
            raise HTTPException(status_code=501, detail=reason)
        try:
            return mmdagent.start() if action == "start" else mmdagent.stop()
        except MmdAgentLaunchUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except MmdAgentConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"MMDAgent-EX の操作に失敗しました: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"MMDAgent-EX の操作に失敗しました: {type(exc).__name__}: {exc}",
            ) from exc

    def _qwen_service_plan(cfg: dict[str, Any]) -> tuple[Any | None, str, dict[str, str]]:
        """Qwen3-ASR のコンテナをどう用意するか: (起動・停止に使う compose, /health の URL, compose に足す変数)。

        - この PC の対話サーバー: Qt と同じ docker-compose.qwen-asr.yml（container_name qwen3-asr、設定のポートで公開）
        - この PC の Web から Docker の対話サーバーを使う: 対話サーバーと同じ compose の qwen3-asr（ホストの 18000 番）
        - Web 自身がコンテナの中: compose を操作できない（None）。設定の asr_qwen_host:asr_qwen_port に届くかだけ見る
        """
        if in_container:
            host = str(cfg.get("asr_qwen_host") or "qwen3-asr").strip() or "qwen3-asr"
            port = str(cfg.get("asr_qwen_port") or "80").strip() or "80"
            return None, f"http://{host}:{port}/health", {}
        if remote_endpoint is not None:
            url = f"http://{remote_endpoint[0] or '127.0.0.1'}:{qwen_docker.HOST_PORT}/health"
            return getattr(bridge, "compose_target", None), url, {}
        host = str(cfg.get("asr_qwen_host") or "127.0.0.1").strip() or "127.0.0.1"
        port = str(cfg.get("asr_qwen_port") or "8000").strip() or "8000"
        return qwen_local_compose, f"http://{host}:{port}/health", {"QWEN_ASR_HOST_PORT": port}

    def _needs_qwen_service(kind: str, module: str, cfg: dict[str, Any]) -> bool:
        """これから ASR を起動し、その ASR が Qwen3-ASR（vLLM）を使うか。"""
        if not qwen_docker.uses_qwen(cfg):
            return False
        if kind in {"start_module", "module_start"}:
            return module.strip().lower() in {"asr", "asr_compare"}
        if kind == "start_voice_dialog":
            states = getattr(bridge, "module_states", {}) or {}
            return str(states.get("asr", "")).lower() != "running"
        return False

    def _qwen_log(message: str) -> None:
        on_event = getattr(bridge, "_on_event", None)
        if callable(on_event):
            on_event({"type": "log", "message": f"[Qwen3-ASR] {message}"})

    def _ensure_qwen_service(cfg: dict[str, Any]) -> None:
        """ASR を起動する前に、Qwen3-ASR のコンテナを用意する（Qt の _start_qwen_asr_server と同じ役目）。

        compose を操作できるときは起動して /health が返るまで待つ。操作できない（Web 自身がコンテナの中・
        compose の場所を知らない）ときは届くかだけ確かめる。届かないまま ASR を起動すると、ワーカーが接続の失敗を
        繰り返すだけで、画面からは理由が分からない。
        """
        target, url, extra_env = _qwen_service_plan(cfg)
        if qwen_docker.health_ok(url, timeout_sec=3.0):
            return
        if target is None:
            raise RuntimeError(qwen_docker.START_HINT)
        if any(Path(f).name == "docker-compose.cpu.yml" for f in target.files):
            raise RuntimeError("Qwen3-ASR（vLLM）は GPU が要ります。CPU 版の Docker（docker-compose.cpu.yml）では使えません。")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Qwen3-ASR（vLLM）は Docker で動きます。Docker Desktop をインストールして起動してください: "
                "https://www.docker.com/products/docker-desktop/"
            )
        env = qwen_docker.compose_env_from_config(
            cfg,
            repo_root=paths.REPO_ROOT,
            host_path_of=path_mapper.host_equivalent if remote_endpoint is not None and path_mapper is not None else None,
        )
        env.update(extra_env)
        _qwen_log(f"docker compose で qwen3-asr を起動します（{env['QWEN_ASR_MODEL']}。初回はイメージの取得に時間がかかります）")
        ok, output = target.run(
            "start", service=qwen_docker.SERVICE, profile=qwen_docker.PROFILE, env=env, timeout_sec=3600.0
        )
        if not ok:
            raise RuntimeError(f"Qwen3-ASR のコンテナを起動できませんでした: {output[-300:]}")
        qwen_docker.wait_health(url, on_progress=lambda m: _qwen_log(m.removeprefix("[Qwen3-ASR] ")))
        _qwen_log("起動しました")

    def _stop_qwen_service_if_asked(kind: str, module: str, payload: dict[str, Any], cfg: dict[str, Any]) -> None:
        """「停止時にQwenコンテナも停止」が有効なら、ASR を止めたあと Qwen3-ASR のコンテナも止める（Qt と同じ）。"""
        if not qwen_docker.uses_qwen(cfg):
            return
        if str(cfg.get("asr_qwen_stop_with_compose_down", False)).strip().lower() not in {"1", "true", "yes", "on"}:
            return
        stopping_asr = (kind in {"stop_module", "module_stop"} and module.strip().lower() in {"asr", "asr_compare"}) or (
            kind == "stop_voice_dialog" and not payload.get("keep_asr", False)
        )
        target = _qwen_service_plan(cfg)[0]
        if not stopping_asr or target is None:
            return
        ok, output = target.run("stop", service=qwen_docker.SERVICE, profile=qwen_docker.PROFILE)
        _qwen_log("コンテナを停止しました" if ok else f"コンテナを停止できませんでした: {output[-300:]}")

    @app.post("/api/command")
    async def command(request: Request) -> dict[str, bool]:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
            raise HTTPException(status_code=422, detail="Command must be a JSON object with a type")
        payload = dict(payload)
        kind = str(payload.get("type") or "")
        # PySide と同じく、音声対話の開始前から動いていた ASR / MaAI は停止時に残す
        # （controller._voice_dialog_asr_preexisting / _maai_preexisting）。
        if kind == "start_voice_dialog":
            states = getattr(bridge, "module_states", {}) or {}
            app.state.voice_preexisting = {
                module: str(states.get(module, "")).lower() == "running"
                for module in ("asr", "maai")
            }
        elif kind == "stop_voice_dialog":
            preexisting = getattr(app.state, "voice_preexisting", {}) or {}
            payload.setdefault("keep_asr", bool(preexisting.get("asr", False)))
            payload.setdefault("keep_maai", bool(preexisting.get("maai", False)))
        try:
            cfg = _merged_config(config.load())
            if _needs_qwen_service(kind, str(payload.get("module") or ""), cfg):
                await asyncio.to_thread(_ensure_qwen_service, cfg)
            assume_mmd_running, force_browser_mic = _runtime_options(cfg)
            commands = vds.expand_web_ui_commands(
                cfg,
                dict(payload),
                assume_mmd_running=assume_mmd_running,
                force_browser_mic=force_browser_mic,
            )
            for command_payload in commands:
                bridge.send(command_payload)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        await asyncio.to_thread(_stop_qwen_service_if_asked, kind, str(payload.get("module") or ""), payload, cfg)
        # OBS 録画の連動（PySide: 音声対話の開始/終了、MaAI テストの起動/停止）。
        module = str(payload.get("module") or "")
        if kind == "start_voice_dialog":
            await asyncio.to_thread(obs_sessions.acquire_if_linked, "voice")
        elif kind == "stop_voice_dialog":
            await asyncio.to_thread(obs_sessions.release, "voice")
        elif kind in {"start_module", "module_start"} and module == "maai":
            await asyncio.to_thread(obs_sessions.acquire_if_linked, "maai")
        elif kind in {"stop_module", "module_stop"} and module == "maai":
            await asyncio.to_thread(obs_sessions.release, "maai")
        return {"ok": True}

    async def _reject_cross_origin_ws(socket: WebSocket) -> bool:
        """他サイトから張られた WebSocket なら閉じて True を返す。

        WebSocket は同一オリジンポリシーの対象外で preflight も無いため、任意のページが
        `ws://127.0.0.1:8000/...` に接続できてしまう。`/ws/events` は ASR 文字起こしと
        LLM 応答が丸ごと流れ、`/ws/mic` は音声を注入できるので、ここは HTTP 側の
        CSRF ガードとは別に塞ぐ必要がある（localhost bind でも防げない）。
        """
        origin = socket.headers.get("origin")
        if origin and not _is_same_origin(origin, socket):
            await socket.close(code=1008)
            return True
        return False

    @app.websocket("/ws/events")
    async def event_socket(socket: WebSocket) -> None:
        if await _reject_cross_origin_ws(socket):
            return
        await socket.accept()
        socket.state.plot_audio = False
        # A browser reload only replaces the WebSocket.  Send the state owned by
        # the still-running backend before subscribing the new socket to events.
        await socket.send_json(bridge.runtime_snapshot())
        bridge.sockets.add(socket)
        if bridge.startup_error:
            await socket.send_json({"type": "connection", "state": "disconnected",
                                    "message": bridge.startup_error})
        try:
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "ui_subscribe":
                    socket.state.plot_audio = bool(msg.get("plot_audio"))
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            pass
        finally:
            bridge.sockets.discard(socket)

    # 対話サーバー（dialog worker）の起動・停止。Qt のモジュール状態の「対話サーバー」行と同じ。
    @app.post("/api/dialog-server/{action}")
    async def dialog_server_action(action: str) -> dict[str, Any]:
        if action not in {"start", "stop"}:
            raise HTTPException(status_code=404, detail="unknown action")
        if action == "start":
            if not hasattr(bridge, "start_worker"):
                raise HTTPException(status_code=409, detail="この構成では対話サーバーを起動できません")
            if getattr(bridge, "loop", None) is None:
                bridge.loop = asyncio.get_running_loop()
            if getattr(bridge, "broadcast_task", None) is None:
                bridge.broadcast_task = asyncio.get_running_loop().create_task(bridge._broadcast_events())
            try:
                await asyncio.to_thread(bridge.start_worker)
            except Exception as ex:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"対話サーバーを起動できませんでした: {ex}") from ex
        else:
            if not hasattr(bridge, "stop_worker"):
                raise HTTPException(status_code=409, detail="この構成では対話サーバーを停止できません")
            await asyncio.to_thread(bridge.stop_worker)
        return {"ok": True, "state": getattr(bridge, "server_state", "")}

    @app.get("/api/runtime")
    async def runtime() -> dict[str, Any]:
        cfg = _merged_config(config.load())
        mic_source = runtime_policy.mic_source(cfg)
        return {
            "docker": runtime_policy.docker,
            "browser_mic": mic_source == "browser",
            "mic_source": mic_source,
            "local_mic_available": not (runtime_policy.docker or runtime_policy.remote_backend),
            "mmdagent": mmdagent.status(),
            # local: この Web サーバーの子プロセス / docker: Docker の対話サーバー / container: コンテナの中
            "dialog_server_where": dialog_server_where,
            "dialog_server_docker_unavailable": (
                "" if dialog_server_where != "local"
                else dialog_server_mode.docker_unavailable_reason(paths.REPO_ROOT)
            ),
            "dialog_server_notice": os.environ.get("MAAI_SDS_DIALOG_SERVER_NOTICE", ""),
        }

    # ローカル / Docker の切り替え。選んだモードで Web サーバーを開き直す（Qt と同じ）。
    @app.post("/api/dialog-server-mode")
    async def dialog_server_mode_switch(request: Request) -> dict[str, Any]:
        payload = await request.json()
        mode = str((payload or {}).get("mode") or "") if isinstance(payload, dict) else ""
        if dialog_server_where == "container":
            raise HTTPException(status_code=409, detail="コンテナの中の Web サーバーでは切り替えられません")
        if mode not in dialog_server_mode.MODES:
            raise HTTPException(status_code=422, detail="mode must be local or docker")
        if mode == dialog_server_where:
            return {"ok": True, "restarting": False}
        if mode == "docker":
            reason = dialog_server_mode.docker_unavailable_reason(paths.REPO_ROOT)
            if reason:
                raise HTTPException(status_code=409, detail=reason)
        dialog_server_mode.save_mode(mode)
        dialog_server_mode.spawn_relaunch(mode, module="maai_sds.web.server")
        # 返事を返してから終わる（lifespan の後片付けで対話サーバーも止まる）。
        # 新しい Web サーバーはこのプロセスが終わるのを待ってから同じポートで起動する。
        threading.Timer(0.5, _request_server_exit).start()
        return {"ok": True, "restarting": True}

    @app.websocket("/ws/wav-monitor")
    async def wav_monitor_socket(socket: WebSocket) -> None:
        """AudioInputWorker が読んだ 16kHz s16le モノラル PCM をブラウザへ送る。"""
        if await _reject_cross_origin_ws(socket):
            return
        await socket.accept()
        bridge.wav_sockets.add(socket)
        try:
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            pass
        finally:
            bridge.wav_sockets.discard(socket)

    @app.websocket("/ws/mic")
    async def mic_socket(socket: WebSocket) -> None:
        """ブラウザの 16kHz s16le モノラル PCM を AudioInputWorker へ中継する。"""
        if await _reject_cross_origin_ws(socket):
            return
        if not runtime_policy.force_browser_mic(_merged_config(config.load())):
            await socket.close(code=1008, reason="browser mic is not selected")
            return
        await socket.accept()
        try:
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if not data:
                    continue
                try:
                    bridge.send({
                        "type": "inject_user_audio",
                        "audio_b64": base64.b64encode(data).decode("ascii"),
                    })
                except Exception:
                    pass
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            pass

    @app.get("/api/browse")
    async def browse(path: str = "", write: int = 0, host: int = 0) -> dict[str, Any]:
        extra = Path(str(app.state.mmd_assets_root or "")) if getattr(app.state, "mmd_assets_root", None) else None
        try:
            if path_view.active and not host:
                # 対話サーバー（コンテナ）が読めるのはマウントしたフォルダーだけなので、
                # そこだけをコンテナのパスで見せる（MMDAgent-EX の実行ファイルなど、
                # この PC で使うものは host=1 で普通に参照する）。
                roots = path_view.mount_roots()
                target = path_view.to_backend(path) if path else None
                if not target or path_view.to_host(target) == target:
                    target = roots[0] if roots else ""
                host_target = path_view.to_host(target)
                if Path(host_target).is_file():  # ファイルを渡されたら、そのフォルダーを開く
                    host_target = str(Path(host_target).parent)
                    target = target.rsplit("/", 1)[0] or target
                listing = web_parity.list_dir(host_target, extra, for_write=bool(write))
                listing["path"] = target
                listing["roots"] = roots
                return listing
            return web_parity.list_dir(path, extra, for_write=bool(write))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/input-wav")
    async def upload_input_wav(request: Request, name: str = "") -> dict[str, Any]:
        """ホストで選んだ WAV を `input_wav/` に置き、コンテナから読めるパスを返す。"""
        raw_name = name or request.headers.get("x-filename") or "input.wav"
        safe = Path(str(raw_name).replace("\\", "/")).name
        safe = "".join(ch for ch in safe if ch.isprintable() and ch not in '<>:"|?*')
        safe = safe.strip(" .") or "input.wav"
        if not safe.lower().endswith(".wav"):
            safe = f"{safe}.wav"
        dest_dir = paths.input_wav_dir()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"input_wav を作成できません: {exc}") from exc
        dest = dest_dir / safe
        max_bytes = 512 * 1024 * 1024
        written = 0
        try:
            with dest.open("wb") as out:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(status_code=413, detail="WAV が 512MB を超えています")
                    out.write(chunk)
        except HTTPException:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        except OSError as exc:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            hint = ""
            if "Read-only" in str(exc) or getattr(exc, "errno", None) == 30:
                hint = " docker-compose の input_wav マウントを書き込み可にして再作成してください。"
            raise HTTPException(status_code=500, detail=f"{exc}.{hint}".strip()) from exc
        if written == 0:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=400, detail="空のファイルです")
        backend = path_view.to_backend(str(dest.resolve()))
        if backend is None:
            raise HTTPException(status_code=500, detail=f"{dest} は Docker の対話サーバーから読めません（マウントの外）")
        return {"path": backend, "name": safe, "bytes": written}

    @app.post("/api/host-upload")
    async def upload_host_file(request: Request, name: str = "") -> dict[str, Any]:
        """ホストで選んだファイルを `host_uploads/` に置き、コンテナから読めるパスを返す。"""
        raw_name = name or request.headers.get("x-filename") or "upload.bin"
        safe = Path(str(raw_name).replace("\\", "/")).name
        safe = "".join(ch for ch in safe if ch.isprintable() and ch not in '<>:"|?*')
        safe = safe.strip(" .") or "upload.bin"
        dest_dir = paths.host_uploads_dir()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"host_uploads を作成できません: {exc}") from exc
        dest = dest_dir / safe
        max_bytes = 4 * 1024 * 1024 * 1024
        written = 0
        try:
            with dest.open("wb") as out:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(status_code=413, detail="ファイルが 4GB を超えています")
                    out.write(chunk)
        except HTTPException:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        except OSError as exc:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if written == 0:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=400, detail="空のファイルです")
        backend = path_view.to_backend(str(dest.resolve()))
        if backend is None:
            raise HTTPException(status_code=500, detail=f"{dest} は Docker の対話サーバーから読めません（マウントの外）")
        return {"path": backend, "name": safe, "bytes": written}

    @app.get("/api/input-wav-media")
    async def input_wav_media(path: str = "") -> FileResponse:
        """設定中の入力 WAV をブラウザ再生用に返す。"""
        extra = Path(str(app.state.mmd_assets_root or "")) if getattr(app.state, "mmd_assets_root", None) else None
        raw = str(path or "").strip() or str(config.load().get("audio_input_wav_path") or "")
        resolved = paths.resolve_runtime_wav_path(path_view.to_host(raw))
        try:
            target = web_parity.resolve_sandboxed(resolved, extra)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"WAV がありません: {target}")
        return FileResponse(target, media_type="audio/wav")

    @app.get("/api/file")
    async def read_file(path: str) -> dict[str, str]:
        extra = Path(str(app.state.mmd_assets_root or "")) if getattr(app.state, "mmd_assets_root", None) else None
        try:
            return {"path": path, "content": web_parity.read_text_file(path_view.to_host(path), extra)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.put("/api/file")
    async def write_file(request: Request) -> dict[str, bool]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be an object")
        extra = Path(str(app.state.mmd_assets_root or "")) if getattr(app.state, "mmd_assets_root", None) else None
        try:
            web_parity.write_text_file(
                path_view.to_host(str(payload.get("path") or "")),
                str(payload.get("content") or ""),
                extra,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"ok": True}

    def _current_mdf_path(cfg: dict[str, Any] | None = None, *, for_write: bool = False) -> Path:
        """設定の `mdf_path` を、必ずサンドボックス検証を通したうえで返す。

        `resolve_mdf_path` は設定値をそのまま `Path` にして返すだけで検証しない。
        `mdf_path` は無認証の `POST /api/config` で任意に書き換えられるので、
        ここを素通しにすると `/api/mdf` が任意ファイルの読み書きに化ける
        （`KEY=value` 形式のファイルなら .env でも中身が JSON で返っていた）。
        """
        loaded = path_view.host_cfg(cfg if cfg is not None else config.load())
        resolved = resolve_mdf_path(loaded, resolve_mmd_assets_root(loaded))
        extra = Path(str(app.state.mmd_assets_root or "")) if getattr(app.state, "mmd_assets_root", None) else None
        return web_parity.resolve_sandboxed(str(resolved), extra, for_write=for_write)

    def _reject_docker_mdf_write() -> None:
        if runtime_policy.docker:
            raise HTTPException(status_code=403, detail=DOCKER_MDF_READ_ONLY_REASON)

    @app.get("/api/mdf")
    async def get_mdf(path: str = "") -> dict[str, Any]:
        cfg = config.load()
        extra = Path(str(app.state.mmd_assets_root or ""))
        try:
            mdf_path = web_parity.resolve_sandboxed(path, extra) if path.strip() else _current_mdf_path(cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        values = web_parity.load_mdf(mdf_path)
        return {
            "path": str(mdf_path),
            "exists": mdf_path.is_file(),
            "values": values,
            "defaults": web_parity.mdf_defaults(),
            "writable": False if runtime_policy.docker else (
                os.access(mdf_path, os.W_OK)
                if mdf_path.is_file()
                else os.access(mdf_path.parent, os.W_OK)
            ),
            "read_only_reason": DOCKER_MDF_READ_ONLY_REASON if runtime_policy.docker else "",
        }

    @app.post("/api/mdf")
    async def post_mdf(request: Request) -> dict[str, Any]:
        _reject_docker_mdf_write()
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be an object")
        extra = Path(str(app.state.mmd_assets_root or ""))
        raw_path = str(payload.get("path") or "").strip()
        try:
            mdf_path = (
                web_parity.resolve_sandboxed(raw_path, extra, for_write=True)
                if raw_path
                else _current_mdf_path(for_write=True)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        values = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        try:
            written = web_parity.save_mdf(
                mdf_path, values,
                create=bool(payload.get("create")),
                touched=payload.get("touched") if isinstance(payload.get("touched"), list) else None,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        try:
            # Plugin_Remote_* lives in the MDF, not controller_config.json.
            # Re-ensure AudioOutput so a transport/host/port edit takes effect.
            mmdagent.sync_audio_output()
        except Exception:
            pass
        return {"ok": True, "path": str(mdf_path), "values": written}

    @app.post("/api/mdf/save-as")
    async def mdf_save_as(request: Request) -> dict[str, Any]:
        _reject_docker_mdf_write()
        payload = await request.json()
        if not isinstance(payload, dict) or not str(payload.get("path") or "").strip():
            raise HTTPException(status_code=422, detail="path is required")
        extra = Path(str(app.state.mmd_assets_root or ""))
        try:
            mdf_path = web_parity.resolve_sandboxed(str(payload.get("path")), extra, for_write=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        values = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        try:
            # 名前を付けて保存は新規ファイルなので、全キーを明示的に書く（PySide 版も同じ）。
            written = web_parity.save_mdf(
                mdf_path, values, create=True, touched=list(values.keys()),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"ok": True, "path": str(mdf_path), "values": written}

    @app.get("/api/mdf/defaults")
    async def mdf_defaults() -> dict[str, Any]:
        return {"values": web_parity.mdf_defaults()}

    @app.post("/api/obs/{action}")
    async def obs_action(action: str, request: Request) -> dict[str, Any]:
        if action not in {"test", "start", "stop"}:
            raise HTTPException(status_code=404, detail="unknown OBS action")
        body = {}
        try:
            raw = await request.json()
            if isinstance(raw, dict):
                body = raw
        except Exception:
            body = {}
        stored = config.load()
        # 画面のパスワード欄はマスク（********）のまま送られてくるので保存値へ戻す。
        cfg = {**stored, **_unmask_secrets(body, stored)}
        if action == "test":
            ok, err = web_parity.obs_test(cfg)
        elif not ObsRecordingSessions.enabled(cfg):
            ok, err = False, "先に「遠隔録画を有効にする（WebSocket）」をオンにしてください。"
        elif action == "start":
            # PySide と同じ参照カウント。音声対話の連動録画中に押しても二重に始めない。
            ok, err = await asyncio.to_thread(obs_sessions.acquire, "manual", cfg)
        elif not obs_sessions.is_acquired("manual"):
            ok, err = False, "手動による参照がありません。"
        else:
            ok, err = await asyncio.to_thread(obs_sessions.release, "manual", cfg)
        hint = ""
        host, _port, _password = web_parity.obs_connection(cfg)
        if host in {"127.0.0.1", "localhost"} and Path("/.dockerenv").is_file():
            hint = "Docker 内の 127.0.0.1 はホストの OBS ではありません。host.docker.internal を指定してください。"
        return {"ok": ok, "error": err, "hint": hint}

    @app.post("/api/llm-admin")
    async def llm_admin(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be an object")
        op = str(payload.get("op") or "").strip()
        cfg = config.load()
        if op == "lmstudio_fetch_models":
            from ..llm.common import normalize_lmstudio_base_url

            requested = str(payload.get("host") or cfg.get("llm_host") or "http://127.0.0.1:1234")
            host = normalize_lmstudio_base_url(requested)
            models, err = web_parity.fetch_lmstudio_models(host)
            hint = ""
            if Path("/.dockerenv").is_file() and host != requested.rstrip("/"):
                hint = "Docker 内の 127.0.0.1 / localhost はコンテナ自身のため、host.docker.internal に付け替えました。"
            return {
                "ok": not err,
                "kind": "lmstudio_fetch",
                "models": models,
                "error": err,
                "host": host,
                "hint": hint,
            }
        if op == "openai_fetch_models":
            api_key = str(payload.get("api_key") or cfg.get("llm_api_key") or "")
            env_name = str(payload.get("api_key_env") or cfg.get("llm_api_key_env") or "")
            if not api_key and env_name:
                api_key = os.environ.get(env_name, "")
            models, err = web_parity.fetch_openai_models(api_key)
            return {"ok": not err, "kind": "openai_fetch", "models": models, "error": err}
        if op == "lmstudio_bulk_unload":
            from ..llm.common import normalize_lmstudio_base_url

            host = normalize_lmstudio_base_url(
                str(payload.get("host") or cfg.get("llm_host") or "http://127.0.0.1:1234")
            )
            current = str(payload.get("current_model") or cfg.get("llm_model_lmstudio") or cfg.get("llm_model") or "")
            result = web_parity.unload_lmstudio_others(host, current)
            result["kind"] = "lmstudio_unload"
            result["host"] = host
            return result
        raise HTTPException(status_code=422, detail=f"unknown op: {op}")

    @app.get("/api/audio-devices")
    async def audio_devices() -> dict[str, Any]:
        return {"devices": web_parity.list_audio_input_devices()}

    @app.post("/api/s2s-preview")
    async def s2s_preview(request: Request) -> dict[str, str]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be an object")
        from ..llm.s2s_preview import build_preview_lines

        cfg = config.load()
        try:
            chunk_size = int(payload.get("chunk_size") or 2)
        except (TypeError, ValueError):
            chunk_size = 2
        text = str(payload.get("text") or "")
        enabled = bool(payload.get("enabled", vds.cfg_bool(cfg, "llm_test_tts_input_streaming_enabled", True)))
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else vds.resolve_llm_split_settings(cfg)
        return {"text": build_preview_lines(text, enabled=enabled, chunk_size=chunk_size, settings=settings)}

    return app


app = create_app()


def _apply_dialog_server_mode(argv: list[str]) -> None:
    """--dialog-server / 保存した選択から、使う対話サーバーの環境変数を用意する（Qt の main と同じ）。"""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dialog-server", choices=dialog_server_mode.MODES)
    parser.add_argument("--wait-pid", type=int, default=0)
    args, _rest = parser.parse_known_args(argv)
    if args.wait_pid:
        dialog_server_mode.wait_for_pid_exit(args.wait_pid)
    if is_docker_environment():
        return  # コンテナの中の Web サーバーは、いつも同じコンテナの対話サーバー
    _mode, env, notice = dialog_server_mode.resolve_launch(
        args.dialog_server, None, dict(os.environ), paths.REPO_ROOT, dialog_server_mode.load_saved_mode(),
    )
    env.pop("MAAI_SDS_DIALOG_SERVER_NOTICE", None)
    if notice:
        env["MAAI_SDS_DIALOG_SERVER_NOTICE"] = notice
        print(f"[web] {notice}", flush=True)
    dialog_server_mode.apply_env(env)


def main() -> None:
    _apply_dialog_server_mode(sys.argv[1:])
    # 最初の起動では設定が無い（Git 管理外）。例から始める（Qt の main と同じ）。uvicorn が
    # このモジュールを読み込み直して app を作るので、それより先に用意する。
    if not os.environ.get("MAAI_SDS_CONTROLLER_CONFIG", "").strip() and paths.ensure_config_file():
        print("[maai-sds] controller_config.json が無いので controller_config.example.json から作りました", flush=True)
    # 既定は 127.0.0.1。この UI には認証が無く、対話の開始停止から設定書き換えまで
    # できてしまうので、外向きに開くのは明示的な指定があるときだけにする。
    # コンテナでは 0.0.0.0 でないとポート公開しても届かないため、compose が環境変数で渡す。
    host = os.environ.get("MAAI_SDS_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.environ.get("MAAI_SDS_WEB_PORT", "8000"))
    except ValueError:
        port = 8000
    uvicorn.run("maai_sds.web.server:app", host=host, port=port)


if __name__ == "__main__":
    main()
