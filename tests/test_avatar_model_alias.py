"""音声と口パクを渡すモデルの別名（`__AV_SETMODEL,<別名>`）を FST から読む。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from maai_sds.dialog import settings as vds
from maai_sds.mmd.remote_bridge import MmdRemoteBridge


class AvatarModelAliasTest(unittest.TestCase):
    def _mdf_with_fst(self, tmp: str, fst: str) -> str:
        mdf = Path(tmp) / "example.mdf"
        mdf.write_text("Plugin_Remote_Websocket_Port=9001\n", encoding="utf-8")
        (Path(tmp) / "example.fst").write_text(fst, encoding="utf-8")
        return str(mdf)

    def test_block_format_fst_of_the_official_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mdf = self._mdf_with_fst(tmp, "0 LOOP:\n    <eps> STAGE|a.png,b.png\n    <eps> MODEL_ADD|gene|../gene/gene.pmd\n")
            self.assertEqual(vds.avatar_model_alias_from_mdf(mdf), "gene")
            self.assertEqual(vds.build_remote_plugin_tcp_settings_dict({"mdf_path": mdf}, running_in_docker=False)["model_alias"], "gene")

    def test_classic_four_column_fst_and_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mdf = self._mdf_with_fst(tmp, "# 0 10 <eps> MODEL_ADD|old|x.pmd\n10 20  <eps> MODEL_ADD|0|gene/gene.pmd\n")
            self.assertEqual(vds.avatar_model_alias_from_mdf(mdf), "0")

    def test_gui_alias_survives_when_mdf_path_is_removed(self) -> None:
        # GUI は音声出力の設定を作る前に mdf_path を外すので、先に読んだ別名が使われること
        cfg = {"avatar_model_alias": "gene", "Plugin_Remote_Websocket_Port": "9001"}
        built = vds.build_audio_output_worker_config_dict(cfg, output_mode="mmd", assume_mmd_running=True)
        self.assertEqual(built["remote_plugin_settings"]["model_alias"], "gene")

    def test_bundled_mdf_is_tcp_on_loopback_outside_dynamic_ports(self) -> None:
        # 同梱の maai_sds.mdf：TCP サーバー方式、ホストでは loopback で待ち受け、ポートは動的ポート（49152〜）の外
        mdf = Path(__file__).resolve().parents[1] / "assets" / "mmdagent" / "maai_sds" / "maai_sds.mdf"
        cfg = vds.merge_plugin_remote_from_mdf_into_cfg({"mdf_path": str(mdf)})
        rs = vds.build_remote_plugin_tcp_settings_dict(cfg, running_in_docker=False)
        self.assertTrue(rs["tcp_enable_client"])
        self.assertFalse(rs["tcp_enable_server"])
        self.assertEqual(rs["tcp_server_host"], "127.0.0.1")
        self.assertLess(rs["tcp_port"], 49152)
        self.assertEqual(rs["model_alias"], "gene")

    def test_missing_fst_falls_back_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mdf = Path(tmp) / "none.mdf"
            mdf.write_text("", encoding="utf-8")
            self.assertEqual(vds.avatar_model_alias_from_mdf(str(mdf)), "0")
        self.assertEqual(vds.avatar_model_alias_from_mdf(""), "0")

    def test_bridge_sends_the_alias_and_resends_when_it_changes(self) -> None:
        bridge = MmdRemoteBridge()
        sent: list[str] = []
        bridge.send_text = lambda text: sent.append(text) or True  # type: ignore[method-assign]
        bridge.is_connected = lambda: True  # type: ignore[method-assign]
        # 接続の設定は不正なポートにして、実際には待ち受けない
        bridge.configure({"model_alias": "gene", "ws_port": 0})
        bridge.ensure_snd_segment_preamble()
        self.assertIn("__AV_SETMODEL,gene\n", sent)
        sent.clear()
        bridge.configure({"model_alias": "0", "ws_port": 0})
        bridge.ensure_av_started()
        self.assertIn("__AV_SETMODEL,0\n", sent)


if __name__ == "__main__":
    unittest.main()
