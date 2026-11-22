"""コンソールの行を logs/console/ に残すデバッグ用ログのテスト。

MMDAgentController を起動せず、書き出しメソッドだけを最小ホストで検証する。
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

try:
    from maai_sds.gui import controller as C
except Exception:  # noqa: BLE001
    C = None


@unittest.skipIf(C is None, "PySide6 が必要（プロジェクト venv で実行）")
class ConsoleLogFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = C._paths.logs_dir
        C._paths.logs_dir = lambda: self.tmp
        self.host = types.SimpleNamespace(
            _CONSOLE_LOG_KEEP=C.MMDAgentController._CONSOLE_LOG_KEEP
        )
        self.write = C.MMDAgentController._write_console_log.__get__(self.host)
        self.close = C.MMDAgentController._close_console_log.__get__(self.host)

    def tearDown(self) -> None:
        self.close()
        C._paths.logs_dir = self._orig

    def test_lines_are_written_with_timestamps(self) -> None:
        self.write("[LLM][START] path=test")
        self.write("second")
        self.close()
        files = list((self.tmp / "console").glob("console_*.log"))
        self.assertEqual(len(files), 1)
        lines = files[0].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(
            lines[0],
            r"^\d\d:\d\d:\d\d\.\d{3} \[LLM\]\[START\] path=test$",
        )

    def test_only_newest_files_are_kept(self) -> None:
        folder = self.tmp / "console"
        folder.mkdir()
        for i in range(25):
            f = folder / f"console_old{i:02d}.log"
            f.write_text("x", encoding="utf-8")
            os.utime(f, (i, i))
        self.write("new session")
        kept = list(folder.glob("console_*.log"))
        self.assertEqual(len(kept), C.MMDAgentController._CONSOLE_LOG_KEEP)
        self.assertFalse((folder / "console_old00.log").exists())
        self.assertTrue((folder / "console_old24.log").exists())


if __name__ == "__main__":
    unittest.main()
