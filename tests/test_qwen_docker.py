"""Docker 版の Qwen3-ASR（vLLM）を compose の中だけで動かす部品（asr/qwen_docker.py と Web の起動前の確認）のテスト。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from maai_sds.asr import qwen_docker
from maai_sds.config_store import ConfigStore
from maai_sds.dialog_server_docker import DockerComposeTarget
from maai_sds.web.server import create_app


class _FakeProcessService:
    def find_running(self, *_args, **_kwargs):
        return None


class _FakeBridge:
    def __init__(self, compose_target=None) -> None:
        self.commands: list[dict] = []
        self.logs: list[str] = []
        self.module_states: dict[str, str] = {}
        self.voice_state = "stopped"
        self.mmd_remote_connected = False
        self.mmd_remote_mode = ""
        self.compose_target = compose_target

    def send(self, command: dict) -> None:
        self.commands.append(dict(command))

    def emit_event(self, event: dict) -> None:
        pass

    def _on_event(self, event: dict) -> None:
        if event.get("type") == "log":
            self.logs.append(str(event.get("message")))


class _FakeCompose:
    files = ("D:/repo/docker-compose.yml", "D:/repo/docker-compose.gpu.yml")

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict | None]] = []

    def run(self, action, *, timeout_sec=180.0, service="orchestrator", profile="qt", env=None):
        self.calls.append((action, service, profile, env))
        return True, ""


class QwenDockerHelpersTest(unittest.TestCase):
    def test_uses_qwen_looks_at_the_engine_or_both_compare_rows(self) -> None:
        self.assertTrue(qwen_docker.uses_qwen({"asr_engine": "qwen3-asr"}))
        self.assertFalse(qwen_docker.uses_qwen({"asr_engine": "sherpa-onnx"}))
        compare = {"asr_compare_mode_enabled": True, "asr_engine": "qwen3-asr",
                   "asr_compare_top_engine": "sherpa-onnx", "asr_compare_bottom_engine": "faster-whisper"}
        self.assertFalse(qwen_docker.uses_qwen(compare))
        self.assertTrue(qwen_docker.uses_qwen({**compare, "asr_compare_bottom_engine": "qwen3-asr"}))

    def test_compose_env_from_config_maps_a_custom_model_inside_the_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            env = qwen_docker.compose_env_from_config(
                {"asr_qwen_custom_model_path": "/app/models/my-qwen", "asr_qwen_vllm_enforce_eager": True,
                 "asr_qwen_max_model_len": "1024"},
                repo_root=root,
                host_path_of=lambda p: root / "models" / "my-qwen" if p == "/app/models/my-qwen" else None,
            )
        self.assertEqual(env["QWEN_ASR_MODEL"], "/data/shared/Qwen3-ASR/models/my-qwen")
        self.assertEqual(env["QWEN_ASR_SERVED_MODEL_NAME"], "my-qwen")
        self.assertEqual(env["QWEN_ASR_MAX_MODEL_LEN"], "1024")
        self.assertEqual(env["QWEN_ASR_ENFORCE_EAGER"], "1")
        plain = qwen_docker.compose_env_from_config({}, repo_root=root)
        self.assertEqual(plain["QWEN_ASR_MODEL"], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(plain["QWEN_ASR_SERVED_MODEL_NAME"], "Qwen3-ASR-1.7B")
        self.assertEqual(plain["QWEN_ASR_ENFORCE_EAGER"], "")

    def test_compose_target_starts_and_stops_other_services(self) -> None:
        target = DockerComposeTarget(files=("a.yml", "b.yml"), workdir=".", project="p")
        self.assertEqual(
            target.command("start", service="qwen3-asr", profile="qwen3"),
            ["docker", "compose", "-f", "a.yml", "-f", "b.yml", "-p", "p", "--profile", "qwen3", "up", "-d", "qwen3-asr"],
        )
        self.assertEqual(target.command("stop")[-3:], ["qt", "stop", "orchestrator"])

    def test_compose_start_creates_the_engine_list_file_and_refuses_a_directory(self) -> None:
        """clone 直後は enabled_engines.json が無い。無いまま compose を起動すると Docker が同名のフォルダーを作る。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = DockerComposeTarget(files=(str(root / "docker-compose.yml"),), workdir=str(root))
            self.assertEqual(target.prepare_mounts(), "")
            self.assertEqual((root / "enabled_engines.json").read_text(encoding="utf-8"), "[]\n")
            (root / "enabled_engines.json").write_text('["asr-sherpa"]', encoding="utf-8")
            self.assertEqual(target.prepare_mounts(), "")  # 既にあるものは触らない
            self.assertEqual((root / "enabled_engines.json").read_text(encoding="utf-8"), '["asr-sherpa"]')

            (root / "enabled_engines.json").unlink()
            (root / "enabled_engines.json").mkdir()
            with patch("subprocess.run", side_effect=AssertionError("docker を呼ばない")):
                ok, message = target.run("start")
            self.assertFalse(ok)
            self.assertIn("フォルダー", message)

    def test_wait_health_returns_when_ready_and_times_out_otherwise(self) -> None:
        answers = iter([False, False, True])
        qwen_docker.wait_health("http://x/health", timeout_sec=5, interval_sec=0.0, probe=lambda _u: next(answers))
        with self.assertRaises(RuntimeError):
            qwen_docker.wait_health("http://x/health", timeout_sec=0.05, interval_sec=0.01, probe=lambda _u: False)


class WebQwenServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "controller_config.json"
        path.write_text(json.dumps({
            "asr_engine": "qwen3-asr", "asr_qwen_host": "qwen3-asr", "asr_qwen_port": "80",
            "asr_qwen_model": "Qwen/Qwen3-ASR-0.6B", "asr_qwen_stop_with_compose_down": True,
        }), encoding="utf-8")
        self.store = ConfigStore(path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _post(self, bridge, payload, **app_kwargs):
        app = create_app(config_store=self.store, bridge_override=bridge, mmd_process_service=_FakeProcessService(),
                         platform="linux", watch_mmd_process=False, **app_kwargs)
        client = TestClient(app)
        try:
            return client.post("/api/command", json=payload)
        finally:
            client.close()

    def test_web_in_a_container_tells_how_to_start_the_qwen_service(self) -> None:
        bridge = _FakeBridge()
        with patch.object(qwen_docker, "health_ok", return_value=False) as probe:
            response = self._post(bridge, {"type": "start_module", "module": "asr"}, docker=True)
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("--profile qwen3", response.json()["detail"])
        self.assertEqual(probe.call_args.args[0], "http://qwen3-asr:80/health")
        self.assertEqual(bridge.commands, [])  # 届かないまま ASR を起動しない

        with patch.object(qwen_docker, "health_ok", return_value=True):
            response = self._post(bridge, {"type": "start_module", "module": "asr"}, docker=True)
        self.assertEqual(response.status_code, 200, response.text)

    def test_host_web_with_a_docker_dialog_server_starts_and_stops_qwen_through_compose(self) -> None:
        compose = _FakeCompose()
        bridge = _FakeBridge(compose)
        kwargs = {"docker": False, "remote_orchestrator": "127.0.0.1:18765", "path_mapper": None}
        with patch.object(qwen_docker, "health_ok", return_value=False), \
                patch.object(qwen_docker, "wait_health") as wait, \
                patch("maai_sds.web.server.shutil.which", return_value="docker"):
            response = self._post(bridge, {"type": "start_module", "module": "asr"}, **kwargs)
        self.assertEqual(response.status_code, 200, response.text)
        action, service, profile, env = compose.calls[0]
        self.assertEqual((action, service, profile), ("start", "qwen3-asr", "qwen3"))
        self.assertEqual(env["QWEN_ASR_MODEL"], "Qwen/Qwen3-ASR-0.6B")
        self.assertNotIn("QWEN_ASR_HOST_PORT", env)  # docker-compose.yml は 18000 番で固定
        self.assertEqual(wait.call_args.args[0], "http://127.0.0.1:18000/health")

        response = self._post(bridge, {"type": "stop_module", "module": "asr"}, **kwargs)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(compose.calls[-1][:3], ("stop", "qwen3-asr", "qwen3"))

    def _use_native_qwen_config(self) -> None:
        self.store.save({**self.store.load(), "asr_qwen_host": "127.0.0.1", "asr_qwen_port": "8123"})

    def test_native_web_starts_and_stops_qwen_with_the_local_compose_file(self) -> None:
        """この PC の対話サーバーの Web も、Qt と同じ docker-compose.qwen-asr.yml でコンテナを用意する。"""
        self._use_native_qwen_config()
        compose = _FakeCompose()
        kwargs = {"docker": False, "remote_orchestrator": "", "qwen_local_compose_target": compose}
        bridge = _FakeBridge()
        with patch.object(qwen_docker, "health_ok", return_value=False), \
                patch.object(qwen_docker, "wait_health") as wait, \
                patch("maai_sds.web.server.shutil.which", return_value="docker"):
            response = self._post(bridge, {"type": "start_module", "module": "asr"}, **kwargs)
        self.assertEqual(response.status_code, 200, response.text)
        action, service, _profile, env = compose.calls[0]
        self.assertEqual((action, service), ("start", "qwen3-asr"))
        self.assertEqual(env["QWEN_ASR_HOST_PORT"], "8123")
        self.assertEqual(wait.call_args.args[0], "http://127.0.0.1:8123/health")
        self.assertTrue(any(c.get("module") == "asr" for c in bridge.commands))

        response = self._post(bridge, {"type": "stop_module", "module": "asr"}, **kwargs)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(compose.calls[-1][:2], ("stop", "qwen3-asr"))

    def test_native_web_skips_compose_when_qwen_is_already_up(self) -> None:
        self._use_native_qwen_config()
        compose = _FakeCompose()
        with patch.object(qwen_docker, "health_ok", return_value=True):
            response = self._post(_FakeBridge(), {"type": "start_module", "module": "asr"},
                                  docker=False, remote_orchestrator="", qwen_local_compose_target=compose)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(compose.calls, [])

    def test_native_web_without_docker_explains_what_is_needed(self) -> None:
        self._use_native_qwen_config()
        compose = _FakeCompose()
        bridge = _FakeBridge()
        with patch.object(qwen_docker, "health_ok", return_value=False), \
                patch("maai_sds.web.server.shutil.which", return_value=None):
            response = self._post(bridge, {"type": "start_module", "module": "asr"},
                                  docker=False, remote_orchestrator="", qwen_local_compose_target=compose)
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("Docker Desktop", response.json()["detail"])
        self.assertEqual((compose.calls, bridge.commands), ([], []))

    def test_local_compose_target_points_at_the_qwen_compose_file(self) -> None:
        target = qwen_docker.local_compose_target(Path("D:/repo"))
        self.assertEqual(Path(target.files[0]), Path("D:/repo/docker-compose.qwen-asr.yml"))
        self.assertEqual(target.command("start", service="qwen3-asr", profile="qwen3")[-2:], ["-d", "qwen3-asr"])

    def test_other_engines_skip_the_qwen_check(self) -> None:
        self.store.save({**self.store.load(), "asr_engine": "sherpa-onnx"})
        with patch.object(qwen_docker, "health_ok", return_value=False) as probe:
            response = self._post(_FakeBridge(), {"type": "start_module", "module": "asr"}, docker=True)
        self.assertEqual(response.status_code, 200, response.text)
        probe.assert_not_called()


class QtDockerModeQwenTest(unittest.TestCase):
    """Qt が Docker の対話サーバーを使うとき、PC 側のスクリプトではなく compose で qwen3-asr を起動・停止する。"""

    def _controller(self, target):
        from maai_sds.gui.qwen_asr_mixin import QwenAsrServerMixin

        class _Signal:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def emit(self, message: str) -> None:
                self.messages.append(message)

        class _Controller(QwenAsrServerMixin):
            _remote_backend_enabled = True
            _remote_orchestrator_endpoint = ("127.0.0.1", 18765)

            def __init__(self) -> None:
                self.process_output_signal = _Signal()
                self.logs_started = 0

            def _dialog_server_docker_target(self):
                return target

            def _derive_qwen_served_model_name(self, model):
                return qwen_docker.served_model_name(model)

            def _start_qwen_asr_container_logs(self) -> None:
                self.logs_started += 1

        return _Controller()

    def _start(self, c) -> None:
        c._start_qwen_asr_server(model="Qwen/Qwen3-ASR-1.7B", gpu_memory_utilization="0.7", max_model_len="2048",
                                 port="80", session_timeout_sec="60", chunk_sec="0.1", enforce_eager=False)

    def test_starts_the_compose_service_with_the_gui_values_and_waits_for_health(self) -> None:
        compose = _FakeCompose()
        c = self._controller(compose)
        with patch("maai_sds.gui.qwen_asr_mixin._require_docker_available"),                 patch.object(qwen_docker, "wait_health") as wait,                 patch("subprocess.Popen", side_effect=AssertionError("PC 側のスクリプトは使わない")):
            self._start(c)
        action, service, profile, env = compose.calls[0]
        self.assertEqual((action, service, profile), ("start", "qwen3-asr", "qwen3"))
        self.assertEqual((env["QWEN_ASR_GPU_MEMORY_UTILIZATION"], env["QWEN_ASR_MAX_MODEL_LEN"]), ("0.7", "2048"))
        self.assertEqual(env["QWEN_ASR_SERVED_MODEL_NAME"], "Qwen3-ASR-1.7B")
        self.assertEqual(wait.call_args.args[0], "http://127.0.0.1:18000/health")
        self.assertEqual(c.logs_started, 1)

        c._stop_qwen_asr_server()
        self.assertEqual(compose.calls[-1][:3], ("stop", "qwen3-asr", "qwen3"))

    def test_without_compose_it_only_checks_and_explains(self) -> None:
        c = self._controller(None)
        with patch.object(qwen_docker, "health_ok", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                self._start(c)
        self.assertIn("--profile qwen3", str(ctx.exception))
        c._stop_qwen_asr_server()  # 止めない（compose の場所を知らない）が、失敗にもしない

    def test_cpu_compose_is_refused(self) -> None:
        compose = _FakeCompose()
        compose.files = ("docker-compose.yml", "docker-compose.cpu.yml")
        c = self._controller(compose)
        with patch.object(qwen_docker, "health_ok", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                self._start(c)
        self.assertIn("GPU", str(ctx.exception))
        self.assertEqual(compose.calls, [])


if __name__ == "__main__":
    unittest.main()
