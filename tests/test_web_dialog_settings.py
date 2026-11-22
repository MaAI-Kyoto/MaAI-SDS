from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from maai_sds.dialog import settings as vds
from maai_sds.mmd.remote_bridge import MmdRemoteBridge


class WebDialogSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.mdf = Path(self.tmp.name) / "MMDAgent-EX_test.mdf"
        self.mdf.write_text(
            "\n".join(
                (
                    "Plugin_Remote_EnableServer=false",
                    "Plugin_Remote_EnableClient=false",
                    "Plugin_Remote_Websocket_Host=127.0.0.1",
                    "Plugin_Remote_Websocket_Port=49131",
                    "Plugin_Remote_Websocket_Directory=/settings-test",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.cfg = {
            "mdf_path": str(self.mdf),
            "tts_engine": "TEMPLATE",
            "audio_output_mode": "mmd",
            "audio_input_mode": "mic",
            "audio_input_mic_device_index": "7",
            "maai_enable_bc": False,
            "maai_enable_nod": False,
            "maai_enable_turn_vap": False,
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_mdf(self, *lines: str) -> None:
        self.mdf.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _voice_worker_configs(commands: list[dict]) -> dict:
        start = commands[0]
        return start["settings"]["worker_configs"]

    def test_native_stopped_mmd_and_local_mic_match_desktop_fallback(self) -> None:
        commands = vds.expand_web_ui_commands(
            self.cfg,
            {"type": "start_voice_dialog"},
            assume_mmd_running=False,
            force_browser_mic=False,
        )
        workers = self._voice_worker_configs(commands)
        self.assertEqual(workers["audio_output"]["output_mode"], "local")
        self.assertEqual(workers["audio_input"]["input_mode"], "mic")
        self.assertEqual(workers["audio_input"]["mic_device_index"], 7)

    def test_running_mmd_and_browser_mic_keep_mmd_and_use_mdf_transport(self) -> None:
        commands = vds.expand_web_ui_commands(
            self.cfg,
            {"type": "start_voice_dialog"},
            assume_mmd_running=True,
            force_browser_mic=True,
        )
        workers = self._voice_worker_configs(commands)
        output = workers["audio_output"]
        self.assertEqual(output["output_mode"], "mmd")
        self.assertEqual(output["remote_plugin_settings"]["ws_host"], "127.0.0.1")
        self.assertEqual(output["remote_plugin_settings"]["ws_port"], 49131)
        self.assertEqual(output["remote_plugin_settings"]["ws_directory"], "/settings-test")
        self.assertEqual(workers["audio_input"]["input_mode"], "browser")

    def test_standalone_asr_module_uses_selected_native_mic_source(self) -> None:
        commands = vds.expand_web_ui_commands(
            self.cfg,
            {"type": "start_module", "module": "asr"},
            assume_mmd_running=False,
            force_browser_mic=False,
        )
        audio = next(command for command in commands if command.get("module") == "audio_input")
        self.assertEqual(audio["config"]["input_mode"], "mic")
        self.assertEqual(audio["config"]["mic_device_index"], 7)

    def test_tts_test_reports_real_mmd_state(self) -> None:
        stopped = vds.expand_web_ui_commands(
            self.cfg,
            {"type": "start_tts_gui_test", "utterance": "test", "engine": "TEMPLATE"},
            assume_mmd_running=False,
            force_browser_mic=False,
        )
        stopped_command = next(command for command in stopped if command.get("type") == "start_tts_gui_test")
        self.assertFalse(stopped_command["mmd_running"])

        running = vds.expand_web_ui_commands(
            self.cfg,
            {"type": "start_tts_gui_test", "utterance": "test", "engine": "TEMPLATE"},
            assume_mmd_running=True,
            force_browser_mic=True,
        )
        running_command = next(command for command in running if command.get("type") == "start_tts_gui_test")
        self.assertTrue(running_command["mmd_running"])

    def test_legacy_web_audio_output_label_is_canonical_at_bootstrap(self) -> None:
        cfg = {**self.cfg, "audio_output_mode": "MMD (Remote)"}
        output = vds.build_audio_output_worker_config_dict(
            cfg,
            assume_mmd_running=True,
        )
        self.assertEqual(output["output_mode"], "mmd")

    def test_docker_websocket_listener_is_reachable_without_forcing_transport(self) -> None:
        for host in ("", "localhost", "127.0.0.1"):
            with self.subTest(host=host):
                self._write_mdf(
                    "Plugin_Remote_EnableServer=false",
                    "Plugin_Remote_EnableClient=false",
                    f"Plugin_Remote_Websocket_Host={host}",
                    "Plugin_Remote_Websocket_Port=49131",
                    "Plugin_Remote_Websocket_Directory=/settings-test",
                )
                with patch.object(vds._paths, "running_in_docker", return_value=True):
                    commands = vds.expand_web_ui_commands(
                        self.cfg,
                        {"type": "start_voice_dialog"},
                        assume_mmd_running=True,
                        force_browser_mic=True,
                    )
                workers = self._voice_worker_configs(commands)
                output = workers["audio_output"]
                self.assertEqual(output["output_mode"], "mmd")
                self.assertFalse(output["remote_plugin_settings"]["tcp_enable_server"])
                self.assertFalse(output["remote_plugin_settings"]["tcp_enable_client"])
                self.assertEqual(output["remote_plugin_settings"]["ws_host"], "0.0.0.0")
                self.assertEqual(output["remote_plugin_settings"]["ws_port"], 49131)

    def test_docker_tcp_server_listener_uses_all_interfaces(self) -> None:
        self._write_mdf(
            "Plugin_Remote_EnableServer=false",
            "Plugin_Remote_EnableClient=true",
            "Plugin_Remote_Hostname=localhost",
            "Plugin_Remote_Port=49132",
        )
        for host in ("", "localhost", "127.0.0.1"):
            with self.subTest(host=host):
                cfg = {**self.cfg, "Plugin_Remote_Bind_Host": host}
                with patch.object(vds._paths, "running_in_docker", return_value=True):
                    output = vds.build_audio_output_worker_config_dict(
                        cfg,
                        assume_mmd_running=True,
                    )
                remote = output["remote_plugin_settings"]
                self.assertTrue(remote["tcp_enable_client"])
                self.assertFalse(remote["tcp_enable_server"])
                self.assertEqual(remote["tcp_server_host"], "0.0.0.0")
                self.assertEqual(remote["tcp_port"], 49132)

    def test_docker_tcp_client_targets_host_and_logs_loopback_substitution(self) -> None:
        remote = {}
        for host in ("localhost", "127.0.0.1"):
            with self.subTest(host=host):
                self._write_mdf(
                    "Plugin_Remote_EnableServer=true",
                    "Plugin_Remote_EnableClient=false",
                    f"Plugin_Remote_Hostname={host}",
                    "Plugin_Remote_ListenPort=49133",
                )
                stdout = StringIO()
                with (
                    patch.object(vds._paths, "running_in_docker", return_value=True),
                    redirect_stdout(stdout),
                ):
                    output = vds.build_audio_output_worker_config_dict(
                        self.cfg,
                        assume_mmd_running=True,
                    )
                remote = output["remote_plugin_settings"]
                self.assertTrue(remote["tcp_enable_server"])
                self.assertEqual(remote["tcp_client_host"], "host.docker.internal")
                self.assertEqual(remote["tcp_listen_port"], 49133)
                self.assertIn("host.docker.internal", stdout.getvalue())

        bridge = MmdRemoteBridge()
        with patch.object(bridge, "_start_tcp_client") as start_client:
            bridge.configure(remote)
        start_client.assert_called_once_with("host.docker.internal", 49133)

    def test_native_tcp_client_keeps_historical_loopback_target(self) -> None:
        self._write_mdf(
            "Plugin_Remote_EnableServer=true",
            "Plugin_Remote_Hostname=remote.example",
            "Plugin_Remote_ListenPort=49134",
        )
        with patch.object(vds._paths, "running_in_docker", return_value=False):
            output = vds.build_audio_output_worker_config_dict(
                self.cfg,
                assume_mmd_running=True,
            )
        self.assertEqual(
            output["remote_plugin_settings"]["tcp_client_host"],
            "127.0.0.1",
        )


if __name__ == "__main__":
    unittest.main()
