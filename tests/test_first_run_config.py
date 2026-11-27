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

    def test_gui_built_without_main_starts_from_the_example(self) -> None:
        """setup_env.ps1 の検証のように main() を通らずに GUI を作っても、画面の初期値ではなく例から始める。"""
        import os
        from unittest import mock

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from PySide6.QtWidgets import QApplication

        from maai_sds.gui import controller as controller_module
        from maai_sds.gui.controller import MMDAgentController
        from maai_sds.ipc.process_service import DialogOrchestratorProcessService

        app = QApplication.instance() or QApplication([])

        class _NoDeviceSoundDevice:
            @staticmethod
            def query_devices(*_args, **_kwargs):
                return []

        example = json.loads(paths.repo_path("controller_config.example.json").read_text(encoding="utf-8"))
        env = {k: v for k, v in os.environ.items() if k != "MAAI_SDS_CONTROLLER_CONFIG"}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "controller_config.json"
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(MMDAgentController, "APP_CONFIG_PATH", target), \
                    mock.patch.object(DialogOrchestratorProcessService, "start", lambda *_a, **_k: None), \
                    mock.patch.object(controller_module, "sd", _NoDeviceSoundDevice()):
                window = MMDAgentController()
                try:
                    self.assertTrue(target.exists())
                    saved = json.loads(target.read_text(encoding="utf-8"))
                    for key in ("llm_system_prompt", "tts_engine", "asr_vap_model_type", "nod_para_enabled",
                                "voice_dialog_opening_utterance"):
                        self.assertEqual(saved.get(key), example[key], key)
                finally:
                    window.close()
                    window.deleteLater()
                    app.processEvents()

    def test_example_config_runs_maai_without_local_mimi_onnx(self) -> None:
        """セットアップは Mimi の ONNX を置かない。例でも、項目が無い設定でも、HF から取る形になる。"""
        example = json.loads(paths.repo_path("controller_config.example.json").read_text(encoding="utf-8"))
        for cfg in (example, {}):
            out = vds.build_maai_worker_config(cfg)
            self.assertFalse(out["use_local_mimi_onnx"])
            self.assertNotIn("mimi_onnx_path", out)


if __name__ == "__main__":
    unittest.main()
