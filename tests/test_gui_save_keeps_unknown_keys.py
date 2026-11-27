"""画面で保存しても、設定ファイルに手で書いた項目が消えないこと。

以前の画面の保存は画面の部品にある項目だけを書き出していたので、画面に無い項目
（speechcpp_exe など）を手で書いても、何か 1 つ保存した時点で消えていた。
Web の保存（{**stored, **incoming}）とそろえる。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from maai_sds import paths
from maai_sds.config_store import example_path_for
from maai_sds.dialog import settings as vds


class _NoDeviceSoundDevice:
    @staticmethod
    def query_devices(*_args, **_kwargs):
        return []


class GuiSaveKeepsUnknownKeysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _save_through_gui(self, initial: dict) -> dict:
        from maai_sds.gui import controller as controller_module
        from maai_sds.gui.controller import MMDAgentController
        from maai_sds.ipc.process_service import DialogOrchestratorProcessService

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "controller_config.json"
            shutil.copyfile(paths.repo_path("controller_config.example.json"), example_path_for(target))
            target.write_text(json.dumps(initial), encoding="utf-8")
            with mock.patch.object(DialogOrchestratorProcessService, "start", lambda *_a, **_k: None), \
                    mock.patch.object(controller_module, "sd", _NoDeviceSoundDevice()):
                window = MMDAgentController(config_path=target)
                try:
                    window._save_app_config()
                    return json.loads(target.read_text(encoding="utf-8"))
                finally:
                    window.close()
                    window.deleteLater()
                    self.app.processEvents()

    def test_hand_written_keys_survive_a_gui_save(self) -> None:
        saved = self._save_through_gui({"speechcpp_exe": "C:/tools/speech.exe", "my_note": {"a": 1}})
        self.assertEqual(saved.get("speechcpp_exe"), "C:/tools/speech.exe")
        self.assertEqual(saved.get("my_note"), {"a": 1})

    def test_gui_values_still_win_over_the_file(self) -> None:
        # 画面に部品がある項目は、画面の値で保存する（ファイルの古い値を残さない）
        saved = self._save_through_gui({"supertonic_steps": "5", "supertonic_backend": "vulkan",
                                        "supertonic_threads": "2"})
        self.assertEqual(saved.get("supertonic_steps"), "5")
        self.assertEqual(saved.get("supertonic_backend"), "vulkan")
        self.assertEqual(saved.get("supertonic_threads"), "2")

    def test_supertonic_backend_and_threads_reach_the_worker(self) -> None:
        saved = self._save_through_gui({"supertonic_backend": "cpu", "supertonic_threads": "3"})
        out = vds.build_tts_worker_config(saved, "Supertonic-3")
        self.assertEqual(out["backend"], "cpu")
        self.assertEqual(out["threads"], 3)
        auto = vds.build_tts_worker_config(self._save_through_gui({}), "Supertonic-3")
        self.assertEqual((auto["backend"], auto["threads"]), ("", 0))


if __name__ == "__main__":
    unittest.main()
