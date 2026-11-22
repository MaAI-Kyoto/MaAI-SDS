"""Web の画面の定義（web/ui_manifest.json）が、Qt の画面から作り直したものと一致している。

Qt 側だけ直して `scripts/extract_web_ui_manifest.py` を実行し忘れると、Web の画面が古いまま残る
（実際に、Mimi のチェックボックスの初期値を変えたときに残った）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class WebManifestInSyncTest(unittest.TestCase):
    def test_manifest_matches_the_qt_ui(self) -> None:
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, "-B", str(REPO / "scripts" / "extract_web_ui_manifest.py"), "--check"],
            cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
