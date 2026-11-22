"""MDF の欄が空のときに選ぶ既定の MDF（mmd/default_paths.py）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from maai_sds.mmd.default_paths import detect_default_exec_path, detect_default_mdf_path


class DefaultMdfPathTest(unittest.TestCase):
    def test_tcp_gene_mdf_is_preferred_over_the_websocket_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "Release").mkdir()
            for name in ("MMDAgent-EX_gene_websocket.mdf", "MMDAgent-EX_gene.mdf"):
                (base / "Release" / name).write_text("", encoding="utf-8")
            self.assertEqual(detect_default_mdf_path(base), str(base / "Release" / "MMDAgent-EX_gene.mdf"))
            (base / "Release" / "MMDAgent-EX_gene.mdf").unlink()
            self.assertEqual(detect_default_mdf_path(base), str(base / "Release" / "MMDAgent-EX_gene_websocket.mdf"))

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


if __name__ == "__main__":
    unittest.main()
