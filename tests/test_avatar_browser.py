"""アバターの表示先がブラウザ（MMDAgent-EX Web）のとき：WebSocket で待ち受け、音声はいつでも送る。"""

from __future__ import annotations

import unittest

from maai_sds.dialog import settings as vds


class AvatarBrowserTest(unittest.TestCase):
    TCP_MDF = {"Plugin_Remote_EnableClient": "true", "Plugin_Remote_Port": "39001"}

    def test_mmdagent_keeps_the_mdf_tcp_settings(self) -> None:
        s = vds.build_remote_plugin_tcp_settings_dict(dict(self.TCP_MDF), running_in_docker=False)
        self.assertTrue(s["tcp_enable_client"])

    def test_browser_opens_a_websocket_instead_of_tcp(self) -> None:
        cfg = {**self.TCP_MDF, "avatar_display": "browser"}
        s = vds.build_remote_plugin_tcp_settings_dict(cfg, running_in_docker=False)
        self.assertFalse(s["tcp_enable_client"])
        self.assertFalse(s["tcp_enable_server"])
        self.assertEqual((s["ws_host"], s["ws_port"], s["ws_directory"]), ("localhost", 9001, "/chat"))
        cfg["avatar_browser_ws_port"] = "19001"
        self.assertEqual(vds.build_remote_plugin_tcp_settings_dict(cfg, running_in_docker=False)["ws_port"], 19001)
        cfg["avatar_browser_ws_port"] = "99999"
        self.assertEqual(vds.build_remote_plugin_tcp_settings_dict(cfg, running_in_docker=False)["ws_port"], 9001)

    def test_browser_in_docker_listens_on_all_interfaces(self) -> None:
        cfg = {"avatar_display": "browser"}
        s = vds.build_remote_plugin_tcp_settings_dict(cfg, running_in_docker=True, log_docker_substitution=False)
        self.assertEqual(s["ws_host"], "0.0.0.0")

    def test_audio_goes_to_the_avatar_without_a_running_mmdagent(self) -> None:
        cfg = {"audio_output_mode": "mmd"}
        self.assertEqual(vds.build_audio_output_worker_config_dict(cfg)["output_mode"], "local")
        cfg["avatar_display"] = "browser"
        self.assertEqual(vds.build_audio_output_worker_config_dict(cfg)["output_mode"], "mmd")
        self.assertFalse(vds.build_audio_output_worker_config_dict(cfg)["remote_plugin_settings"]["tcp_enable_client"])


if __name__ == "__main__":
    unittest.main()
