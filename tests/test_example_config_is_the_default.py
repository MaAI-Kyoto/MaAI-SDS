"""既定値の置き場は見本（controller_config.example.json）1 か所。

設定を読むときは見本を下に敷く（ConfigStore）。設定ファイルが無い状態で GUI を作って保存すると、
見本の全項目がそのままの値で保存される＝画面がどの見本の値も表せる、ことを確かめる。
見本を変えたのに画面側の選択肢や読み込みがついてこない、といったずれをここで捕まえる。
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
from maai_sds.config_store import ConfigStore, example_path_for


class ConfigStoreLayersTheExampleTest(unittest.TestCase):
    def test_example_fills_missing_keys_and_the_file_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "controller_config.json"
            example_path_for(target).write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
            self.assertEqual(ConfigStore(target).load(), {"a": 1, "b": 2})  # 設定ファイルが無い
            target.write_text(json.dumps({"b": 3, "c": 4}), encoding="utf-8")
            self.assertEqual(ConfigStore(target).load(), {"a": 1, "b": 3, "c": 4})

    def test_without_an_example_only_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.json"
            self.assertEqual(ConfigStore(target).load(), {})
            target.write_text(json.dumps({"b": 3}), encoding="utf-8")
            self.assertEqual(ConfigStore(target).load(), {"b": 3})

    def test_example_names(self) -> None:
        self.assertEqual(example_path_for(Path("controller_config.json")).name, "controller_config.example.json")
        self.assertEqual(example_path_for(Path("controller_config.docker.json")).name,
                         "controller_config.docker.example.json")


class GuiKeepsEveryExampleValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_saving_without_a_config_writes_the_example_values(self) -> None:
        from maai_sds import engine_manager
        from maai_sds.gui import controller as controller_module
        from maai_sds.gui.controller import MMDAgentController
        from maai_sds.ipc.process_service import DialogOrchestratorProcessService
        from maai_sds.mmd import default_paths

        class _NoDeviceSoundDevice:
            @staticmethod
            def query_devices(*_args, **_kwargs):
                return []

        example_src = paths.repo_path("controller_config.example.json")
        example = json.loads(example_src.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "controller_config.json"
            shutil.copyfile(example_src, example_path_for(target))
            # GPU 構成として作る（CPU のマシンでは GPU の選択肢が CPU 側へ寄せられる）。
            # MMDAgent-EX の自動探索も止める（見本は空欄で、PC によって埋まる値が変わる）。
            with mock.patch.object(DialogOrchestratorProcessService, "start", lambda *_a, **_k: None), \
                    mock.patch.object(controller_module, "sd", _NoDeviceSoundDevice()), \
                    mock.patch.object(engine_manager, "device_extra", lambda: "cu128"), \
                    mock.patch.object(default_paths, "detect_default_exec_path", lambda *_a, **_k: ""), \
                    mock.patch.object(default_paths, "detect_default_mdf_path", lambda *_a, **_k: ""):
                window = MMDAgentController(config_path=target)
                try:
                    window._save_app_config()
                    saved = json.loads(target.read_text(encoding="utf-8"))
                finally:
                    window.close()
                    window.deleteLater()
                    self.app.processEvents()
        # 空欄なら GUI がその PC の場所で埋める項目（MMDAgent-EX の EXE / MDF と同じ扱い）。
        auto_filled = {"maai_root"}
        mismatched = {
            key: (value, saved.get(key, "<保存されない>"))
            for key, value in example.items()
            if key not in auto_filled and saved.get(key, "<保存されない>") != value
        }
        self.assertEqual(mismatched, {}, "見本の値（左）が画面を通すと変わる（右）")


if __name__ == "__main__":
    unittest.main()
