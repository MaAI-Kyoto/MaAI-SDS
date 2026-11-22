"""clone した直後（controller_config.json が無い）の起動のテスト。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from maai_sds import paths
from maai_sds.dialog import settings as vds


class FirstRunConfigTest(unittest.TestCase):
    def test_missing_config_is_seeded_from_the_example_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "controller_config.json"
            self.assertTrue(paths.ensure_config_file(target))
            example = paths.repo_path("controller_config.example.json").read_bytes()
            self.assertEqual(target.read_bytes(), example)
            target.write_text('{"tts_engine": "AivisSpeech"}', encoding="utf-8")
            self.assertFalse(paths.ensure_config_file(target))  # 既にある設定は上書きしない
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"tts_engine": "AivisSpeech"})

    def test_example_config_runs_maai_without_local_mimi_onnx(self) -> None:
        """セットアップは Mimi の ONNX を置かない。例でも、項目が無い設定でも、HF から取る形になる。"""
        example = json.loads(paths.repo_path("controller_config.example.json").read_text(encoding="utf-8"))
        for cfg in (example, {}):
            out = vds.build_maai_worker_config(cfg)
            self.assertFalse(out["use_local_mimi_onnx"])
            self.assertNotIn("mimi_onnx_path", out)


if __name__ == "__main__":
    unittest.main()
