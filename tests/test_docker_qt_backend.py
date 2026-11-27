from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from maai_sds.dialog import settings as dialog_settings
from maai_sds.dialog.worker import parse_listen_endpoint
from maai_sds.docker_paths import DockerPathMapper
from maai_sds.gui.remote_mic import RemoteMicSender
from maai_sds.ipc.jsonline import JsonLineDuplexClient, JsonLineDuplexServer


class DockerPathMapperTests(unittest.TestCase):
    def test_maps_compose_mounts_both_ways_and_rejects_outside_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            input_dir = base / "recordings"
            mmd_dir = base / "mmd"
            (repo / "models").mkdir(parents=True)
            input_dir.mkdir()
            mmd_dir.mkdir()
            model = repo / "models" / "asr.onnx"
            wav = input_dir / "sample.wav"
            mdf = mmd_dir / "avatar.mdf"
            model.write_bytes(b"model")
            wav.write_bytes(b"wav")
            mdf.write_text("", encoding="utf-8")

            mapper = DockerPathMapper.from_compose_project(
                repo,
                env={
                    "INPUT_WAV_HOST": str(input_dir),
                    "MMDAGENT_ASSETS_HOST": str(mmd_dir),
                },
            )

            self.assertEqual(mapper.host_to_container(model), "/host_repo/models/asr.onnx")
            self.assertEqual(mapper.host_to_container(wav), "/app/input_wav/sample.wav")
            self.assertEqual(mapper.host_to_container(mdf), "/app/mmdagent_assets/avatar.mdf")
            self.assertEqual(mapper.container_to_host("/host_repo/models/asr.onnx"), model)
            self.assertEqual(mapper.container_to_host("/app/input_wav/sample.wav"), wav)
            self.assertEqual(mapper.container_to_host("/app/mmdagent_assets/avatar.mdf"), mdf)
            self.assertEqual(
                mapper.container_to_host("voicevox_runtime/models/0.vvm"),
                repo / "voicevox_runtime_linux" / "models" / "0.vvm",
            )
            self.assertIsNone(mapper.host_to_container(base / "outside" / "secret.bin"))
            self.assertIsNone(mapper.container_to_host("/not-mounted/file.bin"))

    def test_dotenv_values_and_explicit_environment_override_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".env").write_text(
                "INPUT_WAV_HOST=dotenv-input\nMMDAGENT_ASSETS_HOST=dotenv-mmd\n",
                encoding="utf-8",
            )
            dotenv_mapper = DockerPathMapper.from_compose_project(repo, env={})
            self.assertEqual(
                dotenv_mapper.container_to_host("/app/input_wav/a.wav"),
                repo / "dotenv-input" / "a.wav",
            )
            override_mapper = DockerPathMapper.from_compose_project(
                repo,
                env={"INPUT_WAV_HOST": "override-input", "MMDAGENT_ASSETS_HOST": "override-mmd"},
            )
            self.assertEqual(
                override_mapper.container_to_host("/app/input_wav/a.wav"),
                repo / "override-input" / "a.wav",
            )


class ListenModeTests(unittest.TestCase):
    def test_parse_listen_endpoint(self) -> None:
        self.assertEqual(parse_listen_endpoint("0.0.0.0:8765"), ("0.0.0.0", 8765))
        with self.assertRaises(ValueError):
            parse_listen_endpoint("8765")
        with self.assertRaises(ValueError):
            parse_listen_endpoint("localhost:99999")

    def test_jsonline_server_replays_ready_to_second_client(self) -> None:
        server = JsonLineDuplexServer("127.0.0.1", 0)
        connection_count = 0
        count_lock = threading.Lock()

        def on_connected() -> None:
            nonlocal connection_count
            with count_lock:
                connection_count += 1
                generation = connection_count
            server.send({"type": "ready", "generation": generation})

        server.start(lambda _payload: None, on_connected=on_connected)
        self.addCleanup(server.stop)

        def connect_once(expected_generation: int) -> None:
            client = JsonLineDuplexClient("127.0.0.1", server.port)
            received: list[dict] = []
            ready = threading.Event()
            client.connect(attempts=20, sleep_sec=0.01)
            client.start_reader(
                lambda payload: (received.append(dict(payload)), ready.set())
            )
            self.assertTrue(ready.wait(2.0))
            self.assertEqual(received[0]["type"], "ready")
            self.assertEqual(received[0]["generation"], expected_generation)
            client.close()
            deadline = time.time() + 2.0
            while server.is_connected() and time.time() < deadline:
                time.sleep(0.01)
            self.assertFalse(server.is_connected())

        connect_once(1)
        connect_once(2)
        self.assertEqual(connection_count, 2)


class _FakeInputStream:
    def __init__(self, owner, **kwargs) -> None:
        self.owner = owner
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True
        self.owner.last_stream = self
        # 20 ms at 48 kHz, mono s16le. The sender must resample this to
        # exactly 320 samples / 640 bytes at 16 kHz.
        samples = b"\x10\x00" * 960
        self.kwargs["callback"](samples, 960, None, None)

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        return None


class _FakeSoundDevice:
    def __init__(self) -> None:
        self.last_stream = None

    def query_devices(self, *args, **kwargs):
        return {"default_samplerate": 48000.0, "max_input_channels": 1}

    def RawInputStream(self, **kwargs):  # noqa: N802
        return _FakeInputStream(self, **kwargs)


class RemoteMicSenderTests(unittest.TestCase):
    def test_fake_input_is_sent_as_20ms_16khz_s16le_messages(self) -> None:
        payloads: list[dict[str, object]] = []
        sent = threading.Event()

        def send(payload: dict[str, object]) -> bool:
            payloads.append(dict(payload))
            sent.set()
            return True

        fake_sd = _FakeSoundDevice()
        sender = RemoteMicSender(send, sounddevice_module=fake_sd)
        try:
            sender.start(3)
            self.assertTrue(sent.wait(2.0))
        finally:
            sender.stop()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["type"], "inject_user_audio")
        raw = base64.b64decode(str(payloads[0]["audio_b64"]))
        self.assertEqual(len(raw), 640)
        self.assertEqual(fake_sd.last_stream.kwargs["device"], 3)
        self.assertEqual(fake_sd.last_stream.kwargs["channels"], 1)


class QtRemoteBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_remote_controller_connects_without_spawning_local_orchestrator(self) -> None:
        from maai_sds.gui import controller as controller_module
        from maai_sds.gui.controller import MMDAgentController
        from maai_sds.ipc.process_service import DialogOrchestratorProcessService

        server = JsonLineDuplexServer("127.0.0.1", 0)
        commands: list[dict] = []
        server.start(
            lambda payload: commands.append(dict(payload)),
            on_connected=lambda: server.send({"type": "ready"}),
        )
        self.addCleanup(server.stop)

        class _NoDeviceSoundDevice:
            @staticmethod
            def query_devices(*_args, **_kwargs):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "controller_config.docker.json"
            config_path.write_text(
                json.dumps(
                    {
                        "audio_input_mode": "mic",
                        "audio_output_mode": "mmd",
                        "tts_engine": "VOICEVOX-Core",
                        "asr_whisper_model_path": "/opt/models/whisper-local",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                DialogOrchestratorProcessService,
                "start",
                side_effect=AssertionError("remote Qt must not spawn a local orchestrator"),
            ) as start_mock, mock.patch.object(
                DialogOrchestratorProcessService,
                "stop",
                side_effect=AssertionError("remote Qt must not stop the remote orchestrator"),
            ) as stop_mock, mock.patch.object(
                MMDAgentController,
                "_sync_orchestrator_audio_io_after_ready",
                lambda _self: None,
            ), mock.patch.object(controller_module, "sd", _NoDeviceSoundDevice()):
                window = MMDAgentController(
                    config_path=config_path,
                    remote_orchestrator=f"127.0.0.1:{server.port}",
                )
                try:
                    deadline = time.time() + 5.0
                    while not window._dialog_orchestrator_ready_event.is_set() and time.time() < deadline:
                        self.app.processEvents()
                        time.sleep(0.01)
                    self.assertTrue(window._dialog_orchestrator_ready_event.is_set())
                    self.assertTrue(window.remote_backend_enabled)
                    self.assertFalse(start_mock.called)
                    mic_cfg = window._build_audio_input_worker_config()
                    self.assertEqual(mic_cfg["input_mode"], "browser")
                    bridge_cfg = window._build_remote_plugin_tcp_settings_dict()
                    self.assertEqual(bridge_cfg["ws_host"], "0.0.0.0")
                    output_cfg = window._build_audio_output_worker_config_dict()
                    self.assertEqual(
                        output_cfg["remote_plugin_settings"]["ws_host"],
                        "0.0.0.0",
                    )
                    self.assertEqual(
                        window.asr_whisper_model_path_edit.text(),
                        "/opt/models/whisper-local",
                    )
                    self.assertTrue(window._build_asr_config_dict()["whisper_use_local_path"])
                finally:
                    window.close()
                    self.app.processEvents()
                self.assertFalse(stop_mock.called)


class AudioInputConfigTests(unittest.TestCase):
    def test_force_browser_mic_does_not_change_wav_mode(self) -> None:
        mic = dialog_settings.build_audio_input_worker_config(
            {"audio_input_mode": "mic"}, force_browser_mic=True,
        )
        wav = dialog_settings.build_audio_input_worker_config(
            {"audio_input_mode": "wav", "audio_input_wav_path": "/app/input_wav/a.wav"},
            force_browser_mic=True,
        )
        self.assertEqual(mic["input_mode"], "browser")
        self.assertEqual(wav["input_mode"], "wav")
        self.assertEqual(wav["wav_path"], "/app/input_wav/a.wav")


class UiStateSnapshotTests(unittest.TestCase):
    """つながり直したときの状態の送り直し（asr_compare が ASR 行を上書きしない）。"""

    def _snapshot(self, **running: bool) -> list[dict]:
        from maai_sds.ipc.orchestrator_supervisor import OrchestratorSupervisor

        fake = mock.Mock()
        for name in ("asr", "asr_compare", "llm", "tts", "maai", "audio_input", "audio_output"):
            service = mock.Mock()
            service.is_running.return_value = running.get(name, False)
            setattr(fake, name, service)
        fake._tts_engine = "VOICEVOX-Core"
        fake._audio_output_config_json = json.dumps({"output_mode": "mmd"})
        fake._audio_input_config_json = json.dumps({"start_paused": False})
        return OrchestratorSupervisor.ui_state_snapshot(fake)

    def _panel_asr_state(self, events: list[dict]) -> str:
        state = ""
        for ev in events:  # Qt / Web と同じく asr_compare は ASR 行へ集約
            if ev.get("type") == "module_state" and ev.get("module") in {"asr", "asr_compare"}:
                state = str(ev.get("state"))
        return state

    def test_running_asr_is_not_overwritten_by_stopped_compare(self) -> None:
        events = self._snapshot(asr=True)
        self.assertEqual(self._panel_asr_state(events), "running")
        self.assertNotIn("asr_compare", [ev.get("module") for ev in events])

    def test_running_compare_shows_running(self) -> None:
        self.assertEqual(self._panel_asr_state(self._snapshot(asr_compare=True)), "running")


# MMDAgent-EX を置いたホストの場所。Windows 以外では C:\\ のパスが相対パス扱いになるので OS で分ける。
_MMD_HOST = r"C:\MMDAgent-EX" if os.name == "nt" else "/opt/MMDAgent-EX"
_MMD_SEP = "\\" if os.name == "nt" else "/"


def _mmd_host(*parts: str) -> str:
    return _MMD_SEP.join((_MMD_HOST, *parts))


class DefaultMmdPathTests(unittest.TestCase):
    """Docker backend では、自動検出した MDF もコンテナのパスで欄に入れる。"""

    class _Edit:
        def __init__(self) -> None:
            self.text_value = ""

        def setText(self, value: str) -> None:  # noqa: N802 - Qt と同じ名前
            self.text_value = value

    def _run(self, *, remote: bool, mapper) -> tuple[str, str]:
        from maai_sds.gui.controller import MMDAgentController

        fake = mock.Mock()
        fake.exec_edit, fake.mdf_edit = self._Edit(), self._Edit()
        fake._remote_backend_enabled = remote
        fake._docker_path_mapper = mapper
        with mock.patch("maai_sds.mmd.default_paths.detect_default_exec_path",
                        return_value=_mmd_host("Release", "MMDAgent-EX.exe")), \
                mock.patch("maai_sds.mmd.default_paths.detect_default_mdf_path",
                           return_value=_mmd_host("example", "maai_sds", "maai_sds.mdf")):
            MMDAgentController._set_default_paths(fake)
        return fake.exec_edit.text_value, fake.mdf_edit.text_value

    def test_native_keeps_host_paths(self) -> None:
        exe, mdf = self._run(remote=False, mapper=None)
        self.assertEqual(mdf, _mmd_host("example", "maai_sds", "maai_sds.mdf"))
        self.assertTrue(exe.endswith("MMDAgent-EX.exe"))

    def test_remote_maps_mdf_into_container_or_leaves_it_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mapper = DockerPathMapper.from_compose_project(
                Path(tmp), env={"MMDAGENT_ASSETS_HOST": _mmd_host("example")}, env_file=Path(tmp) / "none.env",
            )
            exe, mdf = self._run(remote=True, mapper=mapper)
            self.assertEqual(mdf, "/app/mmdagent_assets/maai_sds/maai_sds.mdf")
            self.assertTrue(exe.endswith("MMDAgent-EX.exe"))  # EXE はホストで起動するのでホストのまま
            unmapped = DockerPathMapper.from_compose_project(Path(tmp), env={}, env_file=Path(tmp) / "none.env")
            _exe, mdf = self._run(remote=True, mapper=unmapped)
            self.assertEqual(mdf, "")  # コンテナから見えない MDF は入れない


if __name__ == "__main__":
    unittest.main()
