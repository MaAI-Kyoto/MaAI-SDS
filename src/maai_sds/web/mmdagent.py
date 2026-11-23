"""Native MMDAgent-EX process control and Web-runtime policy.

The desktop controller owns an :class:`MMDProcessService`, while the Web UI
historically assumed that an MMDAgent process existed outside its container.
This module keeps that container behaviour, but lets a native Windows Web
server own the same process service as the desktop controller.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config_store import ConfigStore
from ..dialog import settings as vds
from ..ipc.process_service import MMDProcessService
from ..mmd.console_filter import is_per_frame_line
from . import parity as web_parity


WEB_MIC_SOURCE_KEY = "web_mic_source"
WEB_MIC_BROWSER = "browser"
WEB_MIC_LOCAL = "local"
DOCKER_MDF_READ_ONLY_REASON = (
    "Dockerでは/app/mmdagent_assetsのMDFは読み取り専用です。"
    "MMDAGENT_ASSETS_HOSTのMDFをWindowsホスト側で編集し、Web UIで再読み込みしてください。"
)


def is_docker_environment() -> bool:
    return Path("/.dockerenv").is_file()


@dataclass(frozen=True)
class WebRuntimePolicy:
    """Platform decisions shared by server bootstrap and request handlers."""

    platform: str
    docker: bool
    process_service: MMDProcessService
    # この PC の Web サーバーが Docker の対話サーバーにつないでいるとき。
    # 対話サーバーはコンテナの中なので、この PC のマイクは直接使えない（ブラウザのマイクを使う）。
    remote_backend: bool = False

    @classmethod
    def detect(
        cls,
        process_service: MMDProcessService,
        *,
        platform: str | None = None,
        docker: bool | None = None,
        remote_backend: bool = False,
    ) -> "WebRuntimePolicy":
        return cls(
            platform=sys.platform if platform is None else str(platform),
            docker=is_docker_environment() if docker is None else bool(docker),
            process_service=process_service,
            remote_backend=bool(remote_backend),
        )

    def mmd_launch_availability(self) -> tuple[bool, str]:
        if self.docker:
            return (
                False,
                "Dockerコンテナ内からWindowsのMMDAgent-EX.exeは起動できません。"
                "MMDAgent-EXはWindowsホスト側で起動してください。",
            )
        if self.platform != "win32":
            return (
                False,
                "MMDAgent-EX.exeの起動はネイティブWindowsのWebサーバーでのみ利用できます。"
                "MMDAgent-EXはWindowsホスト側で起動してください。",
            )
        return True, ""

    def assume_mmd_running(self) -> bool:
        """Whether Web worker configs may keep ``audio_output_mode=mmd``.

        Non-Windows/container deployments retain the historical Web behaviour:
        they prepare the Remote bridge for an MMDAgent started outside the
        process/container.  Native Windows follows the process actually owned
        by this Web server, matching the desktop controller.
        """

        available, _reason = self.mmd_launch_availability()
        if not available:
            return True
        return bool(self.process_service.is_running())

    def mic_source(self, cfg: dict[str, Any] | None) -> str:
        if self.docker or self.remote_backend:
            return WEB_MIC_BROWSER
        raw = str((cfg or {}).get(WEB_MIC_SOURCE_KEY, WEB_MIC_BROWSER) or "").strip().lower()
        return WEB_MIC_LOCAL if raw == WEB_MIC_LOCAL else WEB_MIC_BROWSER

    def force_browser_mic(self, cfg: dict[str, Any] | None) -> bool:
        return self.mic_source(cfg) == WEB_MIC_BROWSER


class MmdAgentLaunchUnavailable(RuntimeError):
    """The current platform cannot launch the Windows executable."""


class MmdAgentConfigurationError(RuntimeError):
    """Configured executable or MDF cannot be used."""


class WebMmdAgentController:
    """Own a native MMDAgent process and keep the orchestrator output in sync."""

    def __init__(
        self,
        config: ConfigStore,
        bridge: Any,
        policy: WebRuntimePolicy,
        *,
        watch_process: bool = True,
    ) -> None:
        self.config = config
        self.bridge = bridge
        self.policy = policy
        self.process_service = policy.process_service
        self.watch_process = bool(watch_process)
        self._lock = threading.RLock()
        self._generation = 0
        self._stopping = False
        self._last_exit_code: int | None = None
        self._last_bridge_error = ""

    def _config(self) -> dict[str, Any]:
        return web_parity.merge_app_ui_defaults(
            self.config.load(), fill_mmdagent_paths=not self.policy.docker
        )

    def _transport_status(self, cfg: dict[str, Any]) -> dict[str, Any]:
        merged = vds.merge_plugin_remote_from_mdf_into_cfg(cfg)
        remote = vds.build_remote_plugin_tcp_settings_dict(
            merged,
            running_in_docker=self.policy.docker,
            log_docker_substitution=False,
        )
        enable_server = bool(remote.get("tcp_enable_server"))
        enable_client = bool(remote.get("tcp_enable_client"))
        both_enabled = enable_server and enable_client

        if enable_server:
            port = int(remote.get("tcp_listen_port", 60001) or 60001)
            endpoint = f"{remote.get('tcp_client_host') or '127.0.0.1'}:{port}"
            return {
                "mode": "tcp_client",
                "label": "TCPクライアント（コンテナ→MMDAgent-EXサーバー）",
                "endpoint": endpoint,
                "listen_address": endpoint,
                "mdf_settings": (
                    "Plugin_Remote_EnableServer=true / Plugin_Remote_EnableClient=false / "
                    f"Plugin_Remote_ListenPort={port}"
                ),
                "instruction": (
                    f"MMDAgent-EXをTCPサーバーとして待受ポート{port}で待ち受けます。"
                    f"コンテナが{endpoint}へ接続します。"
                ),
                "both_enabled": both_enabled,
            }

        if enable_client:
            port = int(remote.get("tcp_port", 60001) or 60001)
            host = vds.cfg_str(merged, "Plugin_Remote_Hostname", "localhost") or "localhost"
            endpoint = f"{host}:{port}"
            return {
                "mode": "tcp_server",
                "label": "TCPサーバー（MMDAgent-EXクライアント→コンテナ）",
                "endpoint": endpoint,
                "listen_address": f"{remote.get('tcp_server_host') or '0.0.0.0'}:{port}",
                "mdf_settings": (
                    "Plugin_Remote_EnableServer=false / Plugin_Remote_EnableClient=true / "
                    f"Plugin_Remote_Hostname={host} / Plugin_Remote_Port={port}"
                ),
                "instruction": f"MMDAgent-EXが{endpoint}へ接続します。",
                "both_enabled": False,
            }

        port = int(remote.get("ws_port", 80) or 80)
        directory = str(remote.get("ws_directory") or "/")
        if not directory.startswith("/"):
            directory = f"/{directory}"
        host = vds.cfg_str(merged, "Plugin_Remote_Websocket_Host", "localhost") or "localhost"
        endpoint = f"ws://{host}:{port}{directory}"
        return {
            "mode": "websocket",
            "label": "WebSocketサーバー（MMDAgent-EXクライアント→コンテナ）",
            "endpoint": endpoint,
            "listen_address": f"{remote.get('ws_host') or '0.0.0.0'}:{port}{directory}",
            "mdf_settings": (
                "Plugin_Remote_EnableServer=false / Plugin_Remote_EnableClient=false / "
                f"Plugin_Remote_Websocket_Host={host} / "
                f"Plugin_Remote_Websocket_Port={port} / "
                f"Plugin_Remote_Websocket_Directory={directory}"
            ),
            "instruction": f"MMDAgent-EXが{endpoint}へ接続します。",
            "both_enabled": False,
        }

    def status(self) -> dict[str, Any]:
        cfg = self._config()
        available, reason = self.policy.mmd_launch_availability()
        running = bool(self.process_service.is_running())
        process = getattr(self.process_service, "process", None)
        pid = getattr(process, "pid", None) if running else None
        mdf_path = str(cfg.get("mdf_path") or "")
        mdf_name = Path(mdf_path).name or "<MDFファイル>"
        return {
            "docker": self.policy.docker,
            "launch_available": available,
            "running": running,
            "pid": pid,
            "reason": reason,
            "exec_path": str(cfg.get("exec_path") or ""),
            "mdf_path": mdf_path,
            "host_mdf_name": mdf_name,
            "host_command": f'.\\MMDAgent-EX.exe ".\\{mdf_name}"',
            "transport": self._transport_status(cfg),
            "remote_connected": bool(getattr(self.bridge, "mmd_remote_connected", False)),
            "remote_mode": str(getattr(self.bridge, "mmd_remote_mode", "") or ""),
            "mdf_read_only": self.policy.docker,
            "mdf_read_only_reason": DOCKER_MDF_READ_ONLY_REASON if self.policy.docker else "",
            "last_exit_code": self._last_exit_code,
            "bridge_error": self._last_bridge_error,
        }

    def _emit_event(self, event: dict[str, Any]) -> None:
        emit = getattr(self.bridge, "emit_event", None)
        if callable(emit):
            try:
                emit(dict(event))
            except Exception:
                pass

    def _log(self, message: str) -> None:
        print(message, flush=True)
        self._emit_event({"type": "log", "message": message})

    def _send(self, command: dict[str, Any]) -> bool:
        try:
            self.bridge.send(dict(command))
            self._last_bridge_error = ""
            return True
        except Exception as ex:  # the process action itself is still useful
            self._last_bridge_error = f"{type(ex).__name__}: {ex}"
            self._log(f"[MMDAgent][WARN] Dialog worker への設定同期に失敗: {self._last_bridge_error}")
            return False

    def _send_remote_stop(self) -> None:
        self._send(
            {
                "type": "send_to_audio_output",
                "payload": {"type": "remote_plugin_stop"},
            }
        )

    def sync_audio_output(self, *, assume_mmd_running: bool | None = None) -> bool:
        cfg = self._config()
        assume = self.policy.assume_mmd_running() if assume_mmd_running is None else bool(assume_mmd_running)
        output_cfg = vds.build_audio_output_worker_config_dict(
            cfg,
            assume_mmd_running=assume,
        )
        return self._send(
            {
                "type": "start_module",
                "module": "audio_output",
                "config": output_cfg,
                "source": "web-mmdagent",
            }
        )

    def _attach_process_output(self, process: Any) -> None:
        def pump(stream: Any, stream_name: str) -> None:
            if stream is None:
                return
            try:
                for raw in iter(stream.readline, ""):
                    line = str(raw).rstrip("\r\n").replace("\x00", "")
                    # まばたき・頷きの値を毎フレーム受けるたびに出る 2 行は出さない（mmd/console_filter.py）。
                    if line.strip() and not is_per_frame_line(line):
                        self._log(f"[MMDAgent][{stream_name}] {line}")
            except Exception:
                pass

        for stream, name in (
            (getattr(process, "stdout", None), "STDOUT"),
            (getattr(process, "stderr", None), "STDERR"),
        ):
            if stream is not None:
                threading.Thread(
                    target=pump,
                    args=(stream, name),
                    daemon=True,
                    name=f"web-mmdagent-{name.lower()}",
                ).start()

    def _watch(self, process: Any, generation: int) -> None:
        try:
            return_code = process.wait()
        except Exception:
            return
        with self._lock:
            if generation != self._generation or self._stopping:
                return
            self._last_exit_code = int(return_code) if return_code is not None else None
        self._log(f"[MMDAgent] プロセスが終了しました (exit={self._last_exit_code})")
        self._send_remote_stop()
        self.sync_audio_output(assume_mmd_running=False)
        self._emit_event({"type": "mmdagent_status", **self.status()})

    def start(self) -> dict[str, Any]:
        available, reason = self.policy.mmd_launch_availability()
        if not available:
            raise MmdAgentLaunchUnavailable(reason)

        cfg = self._config()
        exec_path = Path(str(cfg.get("exec_path") or "").strip())
        mdf_path = Path(str(cfg.get("mdf_path") or "").strip())
        if not exec_path.is_file():
            raise MmdAgentConfigurationError(f"MMDAgent-EX実行ファイルが見つかりません：{exec_path}")
        if not mdf_path.is_file():
            raise MmdAgentConfigurationError(f"MDFファイルが見つかりません：{mdf_path}")

        with self._lock:
            if self.process_service.is_running():
                result = self.status()
                result["message"] = "MMDAgent-EXはすでに起動しています。"
                return result

            ok_fst, fst_message = web_parity.mdf_service().ensure_fst_for_mdf(mdf_path)
            if not ok_fst:
                raise MmdAgentConfigurationError(
                    f"FSTファイルを準備できません：{mdf_path.with_suffix('.fst')}（{fst_message}）"
                )

            merged = vds.merge_plugin_remote_from_mdf_into_cfg(cfg)
            remote = vds.build_remote_plugin_tcp_settings_dict(merged)
            try:
                ws_port = int(remote.get("ws_port", 0) or 0)
            except (TypeError, ValueError) as ex:
                raise MmdAgentConfigurationError("Plugin_RemoteのWebSocketポートが不正です。") from ex
            if not 1 <= ws_port <= 65535:
                raise MmdAgentConfigurationError("Plugin_RemoteのWebSocketポートが不正です。")

            env = os.environ.copy()
            env["WEBSOCKET_USE_CA_WEBSOCKET"] = "false"
            env["WEBSOCKET_SERVER"] = str(remote.get("ws_host") or "localhost")
            env["WEBSOCKET_PORT"] = str(ws_port)
            env["WEBSOCKET_CHANNEL"] = str(remote.get("ws_directory") or "/")

            self._generation += 1
            generation = self._generation
            self._stopping = False
            self._last_exit_code = None
            self._last_bridge_error = ""
            self._send_remote_stop()
            try:
                process = self.process_service.start(exec_path=exec_path, mdf_path=mdf_path, env=env)
            except Exception:
                self._send_remote_stop()
                raise
            self._attach_process_output(process)
            if self.watch_process:
                threading.Thread(
                    target=self._watch,
                    args=(process, generation),
                    daemon=True,
                    name="web-mmdagent-watch",
                ).start()

        self.sync_audio_output(assume_mmd_running=True)
        pid = getattr(process, "pid", None)
        self._log(f"[MMDAgent] 起動しました (PID={pid}) exe={exec_path} mdf={mdf_path}")
        result = self.status()
        result["message"] = f"MMDAgent-EXを起動しました（PID：{pid}）"
        if fst_message:
            result["fst_message"] = fst_message
        self._emit_event({"type": "mmdagent_status", **result})
        return result

    def stop(self, *, sync_audio_output: bool = True, emit: bool = True) -> dict[str, Any]:
        with self._lock:
            was_running = bool(self.process_service.is_running())
            self._generation += 1
            self._stopping = True
            if was_running:
                self.process_service.stop()
            self._stopping = False
        self._send_remote_stop()
        if sync_audio_output:
            self.sync_audio_output(assume_mmd_running=False)
        message = "MMDAgent-EXを停止しました。" if was_running else "MMDAgent-EXは起動していません。"
        self._log(f"[MMDAgent] {message}")
        result = self.status()
        result["message"] = message
        if emit:
            self._emit_event({"type": "mmdagent_status", **result})
        return result

    def close(self) -> None:
        if self.process_service.is_running():
            self.stop(sync_audio_output=False, emit=False)
