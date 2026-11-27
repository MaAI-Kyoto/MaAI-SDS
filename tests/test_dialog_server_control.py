"""対話サーバー（dialog worker / orchestrator）の起動・停止（Qt・Web・Docker）。"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from maai_sds.dialog_server_docker import DockerComposeTarget


class DockerComposeTargetTests(unittest.TestCase):
    def test_not_available_without_env(self) -> None:
        self.assertIsNone(DockerComposeTarget.from_env({}))

    def test_builds_up_and_stop_for_the_orchestrator_service(self) -> None:
        env = {
            "MAAI_SDS_QT_COMPOSE_FILES": os.pathsep.join(["/r/docker-compose.yml", "/r/docker-compose.gpu.yml"]),
            "MAAI_SDS_QT_COMPOSE_DIR": "/r",
            "MAAI_SDS_QT_COMPOSE_PROJECT": "maai",
        }
        target = DockerComposeTarget.from_env(env)
        assert target is not None
        self.assertEqual(target.workdir, "/r")
        self.assertEqual(
            target.command("start"),
            ["docker", "compose", "-f", "/r/docker-compose.yml", "-f", "/r/docker-compose.gpu.yml",
             "-p", "maai", "--profile", "qt", "up", "-d", "orchestrator"],
        )
        self.assertEqual(target.command("stop")[-2:], ["stop", "orchestrator"])


class DialogServerModeTests(unittest.TestCase):
    """GUI で選んだ対話サーバー（ローカル / Docker）で開く・開き直す。"""

    def _repo(self, tmp: str, *, docker_config: bool = True) -> Path:
        root = Path(tmp)
        (root / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
        (root / "docker-compose.gpu.yml").write_text("services: {}\n", encoding="utf-8")
        if docker_config:
            (root / "controller_config.docker.json").write_text("{}", encoding="utf-8")
        return root

    def test_saved_mode_round_trip(self) -> None:
        from maai_sds import dialog_server_mode as m

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "sub" / "launch.json"
            self.assertEqual(m.load_saved_mode(store), "local")
            m.save_mode("docker", store)
            self.assertEqual(m.load_saved_mode(store), "docker")
            store.write_text('{"dialog_server": "bogus"}', encoding="utf-8")
            self.assertEqual(m.load_saved_mode(store), "local")
            store.write_text("not json", encoding="utf-8")
            self.assertEqual(m.load_saved_mode(store), "local")
            self.assertEqual(m.prefs_path({"MAAI_SDS_LAUNCH_PREFS": str(store)}), store)

    def test_docker_mode_sets_the_same_env_as_the_launch_script(self) -> None:
        from maai_sds import dialog_server_mode as m

        with tempfile.TemporaryDirectory() as tmp:
            root = self._repo(tmp)
            mode, env, notice = m.resolve_launch(None, None, {"PATH": "x"}, root, "docker", which=lambda _n: "docker")
            self.assertEqual((mode, notice), ("docker", ""))
            self.assertEqual(env["MAAI_SDS_REMOTE_ORCHESTRATOR"], "127.0.0.1:8765")
            self.assertEqual(env["MAAI_SDS_CONTROLLER_CONFIG"], str(root / "controller_config.docker.json"))
            self.assertEqual(
                env["MAAI_SDS_QT_COMPOSE_FILES"].split(os.pathsep),
                [str(root / "docker-compose.yml"), str(root / "docker-compose.gpu.yml")],
            )
            self.assertEqual(env["MAAI_SDS_QT_COMPOSE_UP_ON_START"], "1")
            (root / ".env").write_text("MAAI_SDS_QT_ORCHESTRATOR_PORT=9876\n", encoding="utf-8")
            (root / ".device_extra").write_text("cpu\n", encoding="utf-8")
            _mode, env, _n = m.resolve_launch("docker", None, {}, root, "local", which=lambda _n: "docker")
            self.assertEqual(env["MAAI_SDS_REMOTE_ORCHESTRATOR"], "127.0.0.1:9876")
            self.assertTrue(env["MAAI_SDS_QT_COMPOSE_FILES"].endswith("docker-compose.cpu.yml"))

    def test_falls_back_to_local_when_docker_is_unavailable(self) -> None:
        from maai_sds import dialog_server_mode as m

        with tempfile.TemporaryDirectory() as tmp:
            root = self._repo(tmp, docker_config=False)
            mode, env, notice = m.resolve_launch(None, None, {}, root, "docker", which=lambda _n: "docker")
            self.assertEqual(mode, "local")
            self.assertIn("controller_config.docker.json", notice)
            self.assertNotIn("MAAI_SDS_REMOTE_ORCHESTRATOR", env)
            mode, _env, notice = m.resolve_launch(None, None, {}, self._repo(tmp), "docker", which=lambda _n: None)
            self.assertEqual(mode, "local")
            self.assertIn("docker コマンド", notice)

    def test_explicit_remote_wins_and_relaunch_resets_mode_env(self) -> None:
        from maai_sds import dialog_server_mode as m

        with tempfile.TemporaryDirectory() as tmp:
            root = self._repo(tmp)
            ps1_env = {
                "MAAI_SDS_REMOTE_ORCHESTRATOR": "127.0.0.1:8765",
                "MAAI_SDS_CONTROLLER_CONFIG": "docker.json",
                "MAAI_SDS_QT_COMPOSE_FILES": "a.yml",
                "MAAI_SDS_QT_LOCAL_CONFIG": "local.json",
            }
            # start_qt_docker.ps1 から起動したとき（保存した選択より優先）
            mode, env, _n = m.resolve_launch(None, None, ps1_env, root, "local", which=lambda _n: "docker")
            self.assertEqual((mode, env), ("docker", ps1_env))
            # GUI で「ローカル」を選んで開き直す: Docker の接続先は消え、ローカル用の設定ファイルを使う。
            # compose の場所は残す（次に Docker へ戻したとき同じ compose を使うため）
            relaunched = m.relaunch_env("local", ps1_env)
            self.assertNotIn("MAAI_SDS_REMOTE_ORCHESTRATOR", relaunched)
            self.assertEqual(relaunched["MAAI_SDS_QT_COMPOSE_FILES"], "a.yml")
            mode, env, _n = m.resolve_launch("local", None, relaunched, root, "docker", which=lambda _n: "docker")
            self.assertEqual(mode, "local")
            self.assertEqual(env["MAAI_SDS_CONTROLLER_CONFIG"], "local.json")
            # Docker に開き直すときは compose の場所を引き継ぐ
            relaunched = m.relaunch_env("docker", ps1_env)
            self.assertEqual(relaunched["MAAI_SDS_QT_COMPOSE_FILES"], "a.yml")
            _mode, env, _n = m.resolve_launch("docker", None, relaunched, root, "local", which=lambda _n: "docker")
            self.assertEqual(env["MAAI_SDS_QT_COMPOSE_FILES"], "a.yml")
            self.assertEqual(m.relaunch_command("docker", 123)[-4:], ["--dialog-server", "docker", "--wait-pid", "123"])

    def test_wait_for_pid_exit(self) -> None:
        import subprocess
        import sys

        from maai_sds import dialog_server_mode as m

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
        self.assertTrue(m._pid_alive(proc.pid))
        # POSIX では、終わった子プロセスは親が回収（wait）するまでゾンビとして残り、os.kill(pid, 0) が
        # 通ってしまう。実際に待つ相手（前の GUI）は自分の子ではないので、テストでも別のスレッドで回収しておく。
        reaper = threading.Thread(target=proc.wait, daemon=True)
        reaper.start()
        self.assertTrue(m.wait_for_pid_exit(proc.pid, timeout_sec=10))
        reaper.join(timeout=10)


class QtDialogServerStateTests(unittest.TestCase):
    def _fake(self, **kw):
        from maai_sds.gui.module_status_mixin import ModuleStatusPanelMixin

        ready = threading.Event()
        fake = SimpleNamespace(
            _dialog_orchestrator_ready_event=ready,
            _dialog_server_busy="",
            _dialog_server_user_stopped=False,
            _dialog_server_error="",
            _remote_backend_enabled=False,
        )
        fake.__dict__.update(kw)
        fake._dialog_server_state = lambda: ModuleStatusPanelMixin._dialog_server_state(fake)
        return fake, ready

    def test_state_follows_ready_stop_and_busy(self) -> None:
        from maai_sds.gui.module_status_mixin import ModuleStatusPanelMixin

        fake, ready = self._fake()
        self.assertEqual(fake._dialog_server_state(), "waiting")
        ready.set()
        self.assertEqual(fake._dialog_server_state(), "running")
        ready.clear()
        fake._dialog_server_user_stopped = True
        self.assertEqual(fake._dialog_server_state(), "stopped")
        fake._dialog_server_busy = "starting"
        self.assertEqual(fake._dialog_server_state(), "starting")
        fake._dialog_server_busy = ""
        fake._dialog_server_user_stopped = False
        fake._dialog_server_error = "boom"
        self.assertEqual(fake._dialog_server_state(), "error")
        # 止めていなければ、操作の前に起動し直す必要はない
        self.assertTrue(ModuleStatusPanelMixin._dialog_server_resume_for_action(fake))

    def test_stopped_server_is_not_restarted_automatically(self) -> None:
        from maai_sds.gui.controller import MMDAgentController

        calls: list[str] = []
        fake, _ready = self._fake(_dialog_server_user_stopped=True, _controller_closing=False)
        fake._ensure_dialog_orchestrator_worker = lambda: calls.append("ensure")
        fake._stop_remote_mic_capture = lambda: None
        fake._clear_remote_orchestrator_state_cache = lambda: None
        fake.process_output_signal = SimpleNamespace(emit=lambda *_a: calls.append("log"))
        MMDAgentController._ensure_dialog_orchestrator_worker_startup(fake)
        MMDAgentController._on_dialog_orchestrator_ipc_disconnected(fake)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, "停止しています"):
            MMDAgentController._ensure_dialog_orchestrator_worker(fake)


class JsonLineCloseTests(unittest.TestCase):
    def test_close_returns_while_the_peer_stays_silent(self) -> None:
        """Windows では shutdown だけだと読み取り中の recv が戻らず close が止まっていた。"""
        import socket
        import time

        from maai_sds.ipc.jsonline import JsonLineDuplexClient

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        peers: list = []
        threading.Thread(target=lambda: peers.append(server.accept()[0]), daemon=True).start()
        client = JsonLineDuplexClient("127.0.0.1", server.getsockname()[1])
        client.connect(attempts=1)
        client.start_reader(lambda _e: None)
        time.sleep(0.2)
        t0 = time.monotonic()
        client.close()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertFalse(client.reader_alive())
        for peer in peers:
            peer.close()
        server.close()


class RemoteConnectTests(unittest.TestCase):
    def test_gives_up_quickly_when_the_port_forwarder_drops_the_connection(self) -> None:
        """Docker Desktop は中がまだ待ち受けていなくても接続を受けてすぐ切る。5 秒待たずにやり直す。"""
        import socket
        import time

        from maai_sds.gui.controller import MMDAgentController

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def drop() -> None:
            conn, _ = server.accept()
            conn.close()

        threading.Thread(target=drop, daemon=True).start()
        fake = SimpleNamespace(
            _remote_orchestrator_endpoint=("127.0.0.1", port),
            _dialog_orchestrator_ipc_client=None,
            _dialog_orchestrator_ready_event=threading.Event(),
            _pump_ui_events=lambda: None,
            _on_dialog_orchestrator_ipc_event=lambda _p: None,
            dialog_orchestrator_ipc_disconnected_signal=SimpleNamespace(emit=lambda: None),
        )

        def teardown() -> None:
            if fake._dialog_orchestrator_ipc_client is not None:
                fake._dialog_orchestrator_ipc_client.close()
                fake._dialog_orchestrator_ipc_client = None

        fake._teardown_dialog_orchestrator_ipc = teardown
        t0 = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "すぐに切れました"):
            MMDAgentController._connect_remote_dialog_orchestrator(fake)
        self.assertLess(time.monotonic() - t0, 2.0)
        server.close()


class _FakeListenServer:
    """Docker の対話サーバー（--listen）の代わり。最初の接続は Docker Desktop のように受けてすぐ切る。"""

    def __init__(self, drop_first: int = 1) -> None:
        import json
        import socket

        self._json = json
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.drop_first = drop_first
        self.conns: list = []
        self.closed = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self.closed.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            if self.drop_first > 0:
                self.drop_first -= 1
                conn.close()
                continue
            self.conns.append(conn)
            conn.sendall((self._json.dumps({"type": "ready"}) + "\n").encode("utf-8"))

    def drop_all(self) -> None:
        import socket

        for conn in self.conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        self.conns.clear()

    def close(self) -> None:
        self.closed.set()
        self.drop_all()
        self.sock.close()


class WebRemoteDialogServerTests(unittest.TestCase):
    """この PC の Web サーバーから Docker の対話サーバーにつなぐ。"""

    def _bridge(self, port: int, compose=None):
        from maai_sds.config_store import ConfigStore
        from maai_sds.web.server import OrchestratorBridge

        bridge = OrchestratorBridge(
            ConfigStore(Path(tempfile.gettempdir()) / "unused_remote.json"),
            remote_endpoint=("127.0.0.1", port),
            compose_target=compose,
        )
        events: list[dict] = []
        original = bridge._on_event
        bridge._on_event = lambda ev: (events.append(dict(ev)), original(ev))[1]  # type: ignore[method-assign]
        return bridge, events

    def test_connects_stops_with_compose_and_reconnects_after_a_drop(self) -> None:
        import time

        server = _FakeListenServer(drop_first=1)
        compose_calls: list[str] = []

        class _Compose:
            def run(self, action):
                compose_calls.append(action)
                return True, ""

        bridge, events = self._bridge(server.port, _Compose())
        try:
            bridge.start_worker()
            self.assertEqual(bridge.server_state, "running")
            self.assertTrue(bridge.worker_running())
            self.assertEqual(compose_calls, ["start"])
            self.assertIn("ready", [e.get("type") for e in events])
            # 相手に切られたら異常にして、自動でつなぎ直す
            server.drop_all()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not (bridge.server_state == "running" and server.conns):
                time.sleep(0.05)
            self.assertEqual(bridge.server_state, "running")
            self.assertIn("error", [e.get("state") for e in events if e.get("type") == "dialog_server"])
            # ■: 切って、compose stop。自動ではつなぎ直さない
            bridge.stop_worker(timeout_sec=1)
            self.assertEqual(bridge.server_state, "stopped")
            self.assertEqual(compose_calls, ["start", "stop"])
            time.sleep(0.5)
            self.assertEqual(bridge.server_state, "stopped")
            self.assertFalse(bridge.worker_running())
        finally:
            bridge._reconnect_stop.set()
            server.close()

    def test_path_view_and_browse_use_container_paths(self) -> None:
        from fastapi.testclient import TestClient

        from maai_sds.config_store import ConfigStore
        from maai_sds.docker_paths import DockerPathMapper, MountMapping
        from maai_sds.web.server import HostPathView, create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            (assets / "sub").mkdir(parents=True)
            (assets / "a.mdf").write_text("x=1\n", encoding="utf-8")
            from pathlib import PurePosixPath

            mapper = DockerPathMapper(root, [MountMapping(assets, PurePosixPath("/app/mmdagent_assets"))])
            view = HostPathView(mapper)
            self.assertEqual(Path(view.to_host("/app/mmdagent_assets/a.mdf")), assets / "a.mdf")
            self.assertEqual(view.to_backend(str(assets / "a.mdf")), "/app/mmdagent_assets/a.mdf")
            self.assertIsNone(view.to_backend(str(root / "elsewhere.txt")))
            self.assertEqual(view.host_cfg({"mdf_path": "/app/mmdagent_assets/a.mdf", "n": 1})["mdf_path"],
                             str(assets / "a.mdf"))
            self.assertEqual(view.mount_roots(), ["/app/mmdagent_assets"])

            store = ConfigStore(root / "controller_config.docker.json")
            store.path.write_text("{}", encoding="utf-8")

            class _Bridge:
                loop = None
                broadcast_task = object()
                server_state = "stopped"
                startup_error = None
                sockets: set = set()
                wav_sockets: set = set()

            os.environ["MAAI_SDS_MMD_ASSETS"] = str(assets)
            try:
                app = create_app(config_store=store, bridge_override=_Bridge(), docker=False,
                                 watch_mmd_process=False, remote_orchestrator="127.0.0.1:1", path_mapper=mapper)
                client = TestClient(app)
                try:
                    res = client.get("/api/browse", params={"path": "/app/mmdagent_assets/a.mdf"})
                    self.assertEqual(res.status_code, 200, res.text)
                    listing = res.json()
                    self.assertEqual(listing["roots"], ["/app/mmdagent_assets"])
                    self.assertIn("a.mdf", [e["name"] for e in listing["entries"]])
                    content = client.get("/api/file", params={"path": "/app/mmdagent_assets/a.mdf"}).json()
                    self.assertEqual(content["content"], "x=1\n")
                    runtime = client.get("/api/runtime").json()
                    self.assertEqual(runtime["dialog_server_where"], "docker")
                    self.assertFalse(runtime["local_mic_available"])
                    self.assertEqual(client.post("/api/engines/apply", json={"ids": []}).status_code, 409)
                finally:
                    client.close()
            finally:
                os.environ.pop("MAAI_SDS_MMD_ASSETS", None)

    def test_mode_switch_saves_and_restarts_the_web_server(self) -> None:
        from unittest import mock

        from fastapi.testclient import TestClient

        from maai_sds.config_store import ConfigStore
        from maai_sds.web import server as web_server

        class _Bridge:
            loop = None
            broadcast_task = object()
            server_state = "running"
            startup_error = None
            sockets: set = set()
            wav_sockets: set = set()

        with tempfile.TemporaryDirectory() as tmp:
            store = ConfigStore(Path(tmp) / "controller_config.json")
            store.path.write_text("{}", encoding="utf-8")
            app = web_server.create_app(config_store=store, bridge_override=_Bridge(), docker=False,
                                        watch_mmd_process=False, remote_orchestrator="")
            client = TestClient(app)
            calls: list = []
            try:
                with mock.patch.object(web_server.dialog_server_mode, "docker_unavailable_reason", return_value=""), \
                        mock.patch.object(web_server.dialog_server_mode, "save_mode", lambda m: calls.append(("save", m))), \
                        mock.patch.object(web_server.dialog_server_mode, "spawn_relaunch",
                                          lambda m, module: calls.append(("spawn", m, module))), \
                        mock.patch.object(web_server, "_request_server_exit", lambda: calls.append(("exit",))):
                    self.assertFalse(client.post("/api/dialog-server-mode", json={"mode": "local"}).json()["restarting"])
                    self.assertTrue(client.post("/api/dialog-server-mode", json={"mode": "docker"}).json()["restarting"])
                    import time

                    time.sleep(0.8)
            finally:
                client.close()
        self.assertEqual(calls[:2], [("save", "docker"), ("spawn", "docker", "maai_sds.web.server")])
        self.assertIn(("exit",), calls)


class WebDialogServerTests(unittest.TestCase):
    def _bridge(self):
        from maai_sds.config_store import ConfigStore
        from maai_sds.web.server import OrchestratorBridge

        bridge = OrchestratorBridge(ConfigStore(Path(tempfile.gettempdir()) / "unused_dialog_server.json"))
        events: list[dict] = []
        original = bridge._on_event

        def capture(ev):
            events.append(dict(ev))
            return original(ev)

        bridge._on_event = capture  # type: ignore[method-assign]
        return bridge, events

    def test_stop_worker_closes_ipc_and_resets_states(self) -> None:
        bridge, events = self._bridge()
        closed: list[str] = []

        class _Client:
            def close(self_inner):
                closed.append("client")
                # 実物は IPC を閉じると reader が切断を知らせる。■ で止めたときは異常にしない。
                bridge._on_disconnected()

        class _Proc:
            returncode = None

            def poll(self_inner):
                return self_inner.returncode

            def wait(self_inner, timeout=None):
                self_inner.returncode = 0
                closed.append("wait")
                return 0

        bridge.client = _Client()
        bridge.process = _Proc()
        bridge.server_state = "running"
        bridge.module_states["asr"] = "running"
        bridge.voice_state = "running"
        bridge.stop_worker(timeout_sec=1)
        self.assertEqual(closed, ["client", "wait"])
        self.assertEqual(bridge.server_state, "stopped")
        self.assertEqual(bridge.module_states["asr"], "stopped")
        self.assertEqual(bridge.voice_state, "stopped")
        states = [e["state"] for e in events if e.get("type") == "dialog_server"]
        self.assertEqual(states, ["stopping", "stopped"])
        self.assertEqual(bridge.runtime_snapshot()["dialog_server"]["state"], "stopped")
        with self.assertRaisesRegex(RuntimeError, "停止しています"):
            bridge.send({"type": "start_module", "module": "asr"})

    def test_unexpected_disconnect_is_an_error(self) -> None:
        bridge, _events = self._bridge()
        bridge._on_disconnected()
        self.assertEqual(bridge.server_state, "error")

    def test_api_starts_and_stops_through_the_bridge(self) -> None:
        from fastapi.testclient import TestClient

        from maai_sds.config_store import ConfigStore
        from maai_sds.web.server import create_app

        seen: list[str] = []

        class _Bridge:
            loop = None
            broadcast_task = object()
            server_state = "stopped"
            startup_error = None
            sockets: set = set()
            wav_sockets: set = set()

            def start_worker(self_inner):
                seen.append("start")
                self_inner.server_state = "running"

            def stop_worker(self_inner):
                seen.append("stop")
                self_inner.server_state = "stopped"

        with tempfile.TemporaryDirectory() as tmp:
            store = ConfigStore(Path(tmp) / "controller_config.json")
            app = create_app(config_store=store, bridge_override=_Bridge(), docker=False, watch_mmd_process=False)
            client = TestClient(app)
            try:
                self.assertEqual(client.post("/api/dialog-server/start").json()["state"], "running")
                self.assertEqual(client.post("/api/dialog-server/stop").json()["state"], "stopped")
                self.assertEqual(client.post("/api/dialog-server/restart").status_code, 404)
            finally:
                client.close()
        self.assertEqual(seen, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
