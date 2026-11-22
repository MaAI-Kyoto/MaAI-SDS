"""MDF の保存は、値が変わらないときにファイルを書かない（mmd/mdf_service.py）。"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from maai_sds.web.parity import mdf_service

MDF = (
    "Plugin_Remote_EnableClient=true\r\n"
    "Plugin_Remote_Hostname=localhost\r\n"
    "Plugin_Remote_Port=60001\r\n"
    "Plugin_Remote_RotationRateHead=0.5\r\n"
)


class MdfWriteUnchangedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "test.mdf"
        self.path.write_bytes(MDF.encode("ascii"))
        old = time.time() - 3600
        os.utime(self.path, (old, old))
        self.mtime = self.path.stat().st_mtime
        self.service = mdf_service()
        self.loaded = set(self.service.parse_mdf_key_values(self.service.read_mdf(self.path)[0]).keys())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, values: dict[str, str]) -> tuple[bool, set[str]]:
        keys = sorted(self.loaded)
        return self.service.write_values(self.path, keys, lambda k: values.get(k, ""), self.loaded, set())

    def test_same_values_leave_the_file_untouched(self) -> None:
        same = self.service.parse_mdf_key_values(self.service.read_mdf(self.path)[0])
        ok, keys = self._write(same)
        self.assertTrue(ok)
        self.assertEqual(keys, self.loaded)
        self.assertEqual(self.path.read_bytes(), MDF.encode("ascii"))  # CRLF もそのまま
        self.assertEqual(self.path.stat().st_mtime, self.mtime)  # 更新日時も変わらない

    def test_changed_value_is_written(self) -> None:
        values = self.service.parse_mdf_key_values(self.service.read_mdf(self.path)[0])
        values["Plugin_Remote_Port"] = "46001"
        ok, _ = self._write(values)
        self.assertTrue(ok)
        self.assertIn("Plugin_Remote_Port=46001", self.path.read_text(encoding="utf-8"))
        self.assertNotEqual(self.path.stat().st_mtime, self.mtime)


if __name__ == "__main__":
    unittest.main()
