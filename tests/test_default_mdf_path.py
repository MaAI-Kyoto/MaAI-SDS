"""MDF の欄が空のときに選ぶ既定の MDF（mmd/default_paths.py）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from maai_sds.mmd.default_paths import detect_default_exec_path, detect_default_mdf_path


class DefaultMdfPathTest(unittest.TestCase):
    def test_files_that_only_exist_in_a_private_build_are_not_picked(self) -> None:
        """自作の MDF（非公開版の環境のもの）は候補にしない。公式のサンプル集の MaAI-SDS 用の MDF を選ぶ。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "Release").mkdir()
            (base / "Release" / "MMDAgent-EX_gene.mdf").write_text("", encoding="utf-8")
            self.assertEqual(detect_default_mdf_path(base), "")
            (base / "example" / "maai_sds").mkdir(parents=True)
            (base / "example" / "maai_sds" / "maai_sds.mdf").write_text("", encoding="utf-8")
            self.assertEqual(detect_default_mdf_path(base), str(base / "example" / "maai_sds" / "maai_sds.mdf"))

    def test_default_fst_is_the_bundled_one_and_uses_only_official_assets(self) -> None:
        """MDF と同名の FST が無いときに書き出す FST は、同梱の maai_sds.fst と同じ（公式のサンプル集の素材だけ）。"""
        from maai_sds.gui.settings import DEFAULT_FST_CONTENT

        repo = Path(__file__).resolve().parents[1]
        bundled = (repo / "assets" / "mmdagent" / "maai_sds" / "maai_sds.fst").read_text(encoding="utf-8")
        body = [ln for ln in bundled.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        self.assertEqual(DEFAULT_FST_CONTENT.splitlines(), body)
        for name in ("ms_generic", "MS_Generic", "suna2", "background-red"):
            self.assertNotIn(name, DEFAULT_FST_CONTENT)

    def test_official_release_layout_and_example_repo(self) -> None:
        """公式の zip を C:\MMDAgent-EX に展開し、サンプル集を example に clone した置き方（README の手順）。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "MMDAgent-EX.exe").write_text("", encoding="utf-8")
            (base / "MMDAgent-EX.mdf").write_text("", encoding="utf-8")  # 配布物の既定の MDF（対話サーバーにつながらない）
            ws = base / "example" / "example_websocket"
            ws.mkdir(parents=True)
            (ws / "example_websocket.mdf").write_text("", encoding="utf-8")
            self.assertEqual(detect_default_exec_path(base), str(base / "MMDAgent-EX.exe"))
            self.assertEqual(detect_default_mdf_path(base), str(ws / "example_websocket.mdf"))
            # README の手順どおり MaAI-SDS 用の MDF をサンプル集に入れたら、そちらを使う
            ours = base / "example" / "maai_sds"
            ours.mkdir()
            (ours / "maai_sds.mdf").write_text("", encoding="utf-8")
            self.assertEqual(detect_default_mdf_path(base), str(ours / "maai_sds.mdf"))


if __name__ == "__main__":
    unittest.main()
