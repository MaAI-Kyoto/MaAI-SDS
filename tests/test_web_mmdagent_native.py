from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from maai_sds.config_store import ConfigStore
from maai_sds.web import parity
from maai_sds.web.mmdagent import (
    MmdAgentLaunchUnavailable,
    WebMmdAgentController,
    WebRuntimePolicy,
)
from maai_sds.web.server import OrchestratorBridge, create_app


class _FakeProcess:
    def __init__(self, pid: int = 42420) -> None:
        self.pid = pid
        self.returncode = None
        self.stdout = None
        self.stderr = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _FakeProcessService:
    def __init__(self) -> None:
        self.process = None
        self.starts = []
        self.stop_calls = 0

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, exec_path: Path, mdf_path: Path, env: dict):
        self.starts.append((exec_path, mdf_path, dict(env)))
        self.process = _FakeProcess()
        return self.process

    def stop(self, timeout_sec: int = 3) -> None:
        self.stop_calls += 1
        if self.process is not None:
            self.process.terminate()
        self.process = None


class _FakeBridge:
    def __init__(self) -> None:
        self.commands = []
        self.events = []
        self.module_states = {}
        self.voice_state = "stopped"
        self.mmd_remote_connected = False
        self.mmd_remote_mode = ""

    def send(self, command: dict) -> None:
        self.commands.append(dict(command))

    def emit_event(self, event: dict) -> None:
        self.events.append(dict(event))


def _find_widget(manifest: dict, attr: str) -> dict:
    pending = [manifest]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item.get("attr") == attr:
                return item
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    raise AssertionError(f"widget not found: {attr}")


class WebRuntimePolicyTests(unittest.TestCase):
    def test_launch_availability_and_assume_mmd_running_are_platform_dependent(self) -> None:
        process = _FakeProcessService()
        native = WebRuntimePolicy.detect(process, platform="win32", docker=False)
        self.assertTrue(native.mmd_launch_availability()[0])
        self.assertFalse(native.assume_mmd_running())
        process.process = _FakeProcess()
        self.assertTrue(native.assume_mmd_running())

        docker = WebRuntimePolicy.detect(_FakeProcessService(), platform="linux", docker=True)
        self.assertFalse(docker.mmd_launch_availability()[0])
        self.assertIn("Windowsホスト", docker.mmd_launch_availability()[1])
        self.assertTrue(docker.assume_mmd_running())

        linux = WebRuntimePolicy.detect(_FakeProcessService(), platform="linux", docker=False)
        self.assertFalse(linux.mmd_launch_availability()[0])
        self.assertTrue(linux.assume_mmd_running())

    def test_mic_source_is_selectable_natively_and_forced_in_docker(self) -> None:
        native = WebRuntimePolicy.detect(_FakeProcessService(), platform="win32", docker=False)
        self.assertTrue(native.force_browser_mic({}))
        self.assertTrue(native.force_browser_mic({"web_mic_source": "browser"}))
        self.assertFalse(native.force_browser_mic({"web_mic_source": "local"}))

        docker = WebRuntimePolicy.detect(_FakeProcessService(), platform="linux", docker=True)
        self.assertTrue(docker.force_browser_mic({"web_mic_source": "local"}))


class WebMmdAgentControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.exe = root / "MMDAgent-EX.exe"
        self.mdf = root / "MMDAgent-EX_test.mdf"
        self.exe.write_bytes(b"fake")
        self.mdf.write_text(
            "\n".join(
                (
                    "Plugin_Remote_EnableServer=false",
                    "Plugin_Remote_EnableClient=false",
                    "Plugin_Remote_Websocket_Host=127.0.0.1",
                    "Plugin_Remote_Websocket_Port=49123",
                    "Plugin_Remote_Websocket_Directory=/native-test",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.store = ConfigStore(root / "controller_config.json")
        self.store.save_raising(
            {
                "exec_path": str(self.exe),
                "mdf_path": str(self.mdf),
                "audio_output_mode": "mmd",
                "audio_input_mode": "mic",
            }
        )
        self.process = _FakeProcessService()
        self.bridge = _FakeBridge()
        self.policy = WebRuntimePolicy.detect(self.process, platform="win32", docker=False)
        self.controller = WebMmdAgentController(
            self.store,
            self.bridge,
            self.policy,
            watch_process=False,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_start_and_stop_use_process_service_and_resync_audio_output(self) -> None:
        started = self.controller.start()
        self.assertTrue(started["running"])
        self.assertEqual(started["pid"], 42420)
        self.assertTrue(self.mdf.with_suffix(".fst").is_file())
        self.assertEqual(len(self.process.starts), 1)
        exec_path, mdf_path, env = self.process.starts[0]
        self.assertEqual(exec_path, self.exe)
        self.assertEqual(mdf_path, self.mdf)
        self.assertEqual(env["WEBSOCKET_PORT"], "49123")
        self.assertEqual(env["WEBSOCKET_CHANNEL"], "/native-test")

        start_audio = [
            cmd for cmd in self.bridge.commands
            if cmd.get("type") == "start_module" and cmd.get("module") == "audio_output"
        ][-1]
        self.assertEqual(start_audio["config"]["output_mode"], "mmd")
        self.assertEqual(
            start_audio["config"]["remote_plugin_settings"]["ws_port"],
            49123,
        )

        stopped = self.controller.stop()
        self.assertFalse(stopped["running"])
        self.assertEqual(self.process.stop_calls, 1)
        stop_audio = [
            cmd for cmd in self.bridge.commands
            if cmd.get("type") == "start_module" and cmd.get("module") == "audio_output"
        ][-1]
        self.assertEqual(stop_audio["config"]["output_mode"], "local")
        self.assertTrue(
            any(
                cmd.get("type") == "send_to_audio_output"
                and cmd.get("payload", {}).get("type") == "remote_plugin_stop"
                for cmd in self.bridge.commands
            )
        )

    def test_unavailable_platform_never_calls_process_service(self) -> None:
        policy = WebRuntimePolicy.detect(self.process, platform="linux", docker=True)
        controller = WebMmdAgentController(
            self.store,
            self.bridge,
            policy,
            watch_process=False,
        )
        with self.assertRaises(MmdAgentLaunchUnavailable):
            controller.start()
        self.assertEqual(self.process.starts, [])


class WebMmdAgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.exe = root / "MMDAgent-EX.exe"
        self.mdf = root / "MMDAgent-EX_test.mdf"
        self.exe.write_bytes(b"fake")
        self.mdf.write_text(
            "Plugin_Remote_Websocket_Port=49124\n"
            "Plugin_Remote_Websocket_Directory=/api-test\n",
            encoding="utf-8",
        )
        config_path = root / "controller_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "exec_path": str(self.exe),
                    "mdf_path": str(self.mdf),
                    "audio_output_mode": "mmd",
                    "web_mic_source": "local",
                    "mmd_burst_snd_frames": 17,
                    # 削除した「口パク方式」の古い保存値。読み込みでは無視される。
                    "mmd_lipsync_mode": "Julius音素 (本家)",
                    "maai_enable_bc": False,
                    "maai_enable_nod": True,
                    "maai_enable_turn_vap": False,
                }
            ),
            encoding="utf-8",
        )
        self.store = ConfigStore(config_path)
        self.process = _FakeProcessService()
        self.bridge = _FakeBridge()
        self.app = create_app(
            config_store=self.store,
            bridge_override=self.bridge,
            mmd_process_service=self.process,
            platform="win32",
            docker=False,
            watch_mmd_process=False,
        )
        # Deliberately do not use TestClient as a context manager: lifespan would
        # start the real dialog worker.  Requests still exercise the ASGI routes.
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.tmp.cleanup()

    def test_runtime_manifest_and_native_start_stop_endpoints(self) -> None:
        runtime = self.client.get("/api/runtime")
        self.assertEqual(runtime.status_code, 200)
        self.assertEqual(runtime.json()["mic_source"], "local")
        self.assertFalse(runtime.json()["browser_mic"])
        self.assertTrue(runtime.json()["mmdagent"]["launch_available"])

        manifest_response = self.client.get("/api/ui/manifest")
        self.assertEqual(manifest_response.status_code, 200)
        manifest = manifest_response.json()
        output = _find_widget(manifest, "audio_output_mode_combo")
        self.assertEqual([item["value"] for item in output["items"]], ["mmd", "local", "tcp"])
        mic = _find_widget(manifest, "audio_input_mic_device_combo")
        self.assertEqual(mic["config_key"], "audio_input_mic_device_index")
        self.assertEqual(_find_widget(manifest, "web_mic_source_combo")["value"], "browser")

        started = self.client.post("/api/mmdagent/start", json={})
        self.assertEqual(started.status_code, 200, started.text)
        self.assertTrue(started.json()["running"])
        stopped = self.client.post("/api/mmdagent/stop", json={})
        self.assertEqual(stopped.status_code, 200, stopped.text)
        self.assertFalse(stopped.json()["running"])

    def test_container_start_endpoint_reports_host_requirement(self) -> None:
        app = create_app(
            config_store=self.store,
            bridge_override=_FakeBridge(),
            mmd_process_service=_FakeProcessService(),
            platform="linux",
            docker=True,
            watch_mmd_process=False,
        )
        client = TestClient(app)
        try:
            response = client.post("/api/mmdagent/start", json={})
        finally:
            client.close()
        self.assertEqual(response.status_code, 501)
        self.assertIn("Windowsホスト", response.json()["detail"])

    def test_container_status_explains_host_ws_and_mdf_is_read_only(self) -> None:
        bridge = _FakeBridge()
        bridge.mmd_remote_connected = True
        bridge.mmd_remote_mode = "websocket"
        app = create_app(
            config_store=self.store,
            bridge_override=bridge,
            mmd_process_service=_FakeProcessService(),
            platform="linux",
            docker=True,
            watch_mmd_process=False,
        )
        client = TestClient(app)
        try:
            runtime = client.get("/api/runtime")
            mdf = client.get("/api/mdf")
            save = client.post(
                "/api/mdf",
                json={"path": str(self.mdf), "values": {}, "touched": []},
            )
        finally:
            client.close()

        self.assertEqual(runtime.status_code, 200, runtime.text)
        status = runtime.json()["mmdagent"]
        self.assertEqual(status["transport"]["mode"], "websocket")
        self.assertEqual(status["transport"]["endpoint"], "ws://localhost:49124/api-test")
        self.assertEqual(status["transport"]["listen_address"], "0.0.0.0:49124/api-test")
        self.assertIn("MMDAgent-EX_test.mdf", status["host_command"])
        self.assertTrue(status["remote_connected"])
        self.assertEqual(status["remote_mode"], "websocket")
        self.assertTrue(status["mdf_read_only"])

        self.assertEqual(mdf.status_code, 200, mdf.text)
        self.assertEqual(mdf.json()["values"]["Plugin_Remote_Websocket_Port"], "49124")
        self.assertFalse(mdf.json()["writable"])
        self.assertIn("読み取り専用", mdf.json()["read_only_reason"])
        self.assertEqual(save.status_code, 403, save.text)
        self.assertIn("読み取り専用", save.json()["detail"])

    def test_docker_voice_dialog_keeps_mmd_snd_and_nod_config(self) -> None:
        bridge = _FakeBridge()
        app = create_app(
            config_store=self.store,
            bridge_override=bridge,
            mmd_process_service=_FakeProcessService(),
            platform="linux",
            docker=True,
            watch_mmd_process=False,
        )
        client = TestClient(app)
        try:
            with patch("maai_sds.dialog.settings._paths.running_in_docker", return_value=True):
                response = client.post("/api/command", json={"type": "start_voice_dialog"})
        finally:
            client.close()
        self.assertEqual(response.status_code, 200, response.text)
        start = next(command for command in bridge.commands if command.get("type") == "start_voice_dialog")
        workers = start["settings"]["worker_configs"]
        output = workers["audio_output"]
        self.assertEqual(output["output_mode"], "mmd")
        self.assertEqual(output["mmd_burst_snd_frames"], 17)
        # 古い設定に残る mmd_lipsync_mode は無視し、ワーカーへも渡さない。
        self.assertNotIn("mmd_lipsync_mode", output)
        self.assertEqual(output["remote_plugin_settings"]["ws_host"], "0.0.0.0")
        self.assertEqual(output["remote_plugin_settings"]["ws_port"], 49124)
        self.assertEqual(workers["maai"]["mode"], "nod_para")

    def test_remote_status_is_retained_for_reload_snapshot(self) -> None:
        bridge = OrchestratorBridge(self.store)
        bridge._on_event({"type": "remote_status", "connected": True, "mode": "tcp_server"})
        snapshot = bridge.runtime_snapshot()
        self.assertEqual(
            snapshot["mmd_remote"],
            {"connected": True, "mode": "tcp_server"},
        )

    def test_nod_threshold_defaults_are_retained_until_maai_stops(self) -> None:
        bridge = OrchestratorBridge(self.store)
        self.assertIsNone(bridge.runtime_snapshot()["maai_nod_threshold_defaults"])
        defaults = {"count": {"t0": 0.4, "t1": 0.5, "t2": 0.6}, "swing": 0.3}
        bridge._on_event({"type": "maai_nod_threshold_defaults", "defaults": defaults})
        self.assertEqual(bridge.runtime_snapshot()["maai_nod_threshold_defaults"], defaults)
        bridge._on_event({"type": "module_stopped", "module": "maai"})
        self.assertIsNone(bridge.runtime_snapshot()["maai_nod_threshold_defaults"])

    def test_module_fatal_reason_is_retained_for_reload_snapshot(self) -> None:
        bridge = OrchestratorBridge(self.store)
        bridge._on_event(
            {"type": "fatal", "_src": "tts", "message": "reference audio is missing"}
        )

        self.assertEqual(
            bridge.runtime_snapshot()["modules"]["tts"],
            {"state": "error", "error": "reference audio is missing"},
        )


class WebParityTests(unittest.TestCase):
    def test_audio_output_display_labels_are_canonicalized(self) -> None:
        self.assertEqual(parity.canonical_audio_output_mode("MMD (Remote)"), "mmd")
        self.assertEqual(parity.canonical_audio_output_mode("ローカルスピーカー"), "local")
        self.assertEqual(parity.canonical_audio_output_mode("TCP 送出"), "tcp")


@unittest.skipUnless(shutil.which("node"), "Node.js が無い")
class WebPlotAudioClockTests(unittest.TestCase):
    def test_viewport_uses_monotonic_audio_clock_after_wall_clock_step(self) -> None:
        root = Path(__file__).resolve().parent.parent
        module_url = (root / "web" / "app" / "plot.js").as_uri()
        script = f"""
import {{ createAudioViewportClock }} from {json.dumps(module_url)};
let wall = 1000;
let monotonic = 10;
const clock = createAudioViewportClock({{
  wallNow: () => wall,
  monotonicNow: () => monotonic,
}});
const beforeAudio = clock.now();
clock.note(500);
wall = 2400;
monotonic = 10.4;
const afterForwardStep = clock.now();
wall = 200;
monotonic = 10.9;
const afterBackwardStep = clock.now();
console.log(JSON.stringify([beforeAudio, afterForwardStep, afterBackwardStep]));
"""
        result = subprocess.run(
            ["node", "--input-type=module", "--eval", script],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        values = json.loads(result.stdout.strip())
        self.assertEqual(values[0], 1000)
        self.assertAlmostEqual(values[1], 500.4)
        self.assertAlmostEqual(values[2], 500.9)


if __name__ == "__main__":
    unittest.main()
