"""macOS（Apple Silicon）で動かすための分岐。どの OS でも確かめられるよう、OS の判定を差し替える。"""

from __future__ import annotations

import unittest
from unittest import mock

from maai_sds import device
from maai_sds.tts import engines

_DLL = "voicevox_runtime/onnxruntime/lib/voicevox_onnxruntime_v.dll"  # 見本の設定の値（Windows 用）
_DYLIB = "voicevox_runtime/onnxruntime/lib/libvoicevox_onnxruntime.dylib"
_SO = "voicevox_runtime/onnxruntime/lib/libvoicevox_onnxruntime.so"


class VoicevoxOrtFileTest(unittest.TestCase):
    def test_example_dll_becomes_the_dylib_on_macos(self) -> None:
        with mock.patch.object(engines, "_DEFAULT_VOICEVOX_ORT_FILE", _DYLIB):
            self.assertEqual(engines._voicevox_ort_file_for_this_os(_DLL), _DYLIB)
            self.assertEqual(engines._voicevox_ort_file_for_this_os(_SO), _DYLIB)
            # この OS 用のファイルを指していればそのまま（版付きの名前・別の場所も）
            own = "/opt/vv/lib/libvoicevox_onnxruntime.1.17.3.dylib"
            self.assertEqual(engines._voicevox_ort_file_for_this_os(own), own)

    def test_windows_keeps_its_dll(self) -> None:
        with mock.patch.object(engines, "_DEFAULT_VOICEVOX_ORT_FILE", "x/voicevox_onnxruntime.dll"):
            self.assertEqual(engines._voicevox_ort_file_for_this_os(_DLL), _DLL)
            self.assertEqual(engines._voicevox_ort_file_for_this_os(""), "x/voicevox_onnxruntime.dll")

    def test_linux_versioned_so_is_kept(self) -> None:
        with mock.patch.object(engines, "_DEFAULT_VOICEVOX_ORT_FILE", _SO):
            versioned = "voicevox_runtime/onnxruntime/lib/libvoicevox_onnxruntime.so.1.23.2"
            self.assertEqual(engines._voicevox_ort_file_for_this_os(versioned), versioned)
            self.assertEqual(engines._voicevox_ort_file_for_this_os(_DLL), _SO)


class VoicevoxThreadsTest(unittest.TestCase):
    def test_capped_at_performance_cores_on_macos(self) -> None:
        with mock.patch.object(engines, "_macos_performance_cores", return_value=12), \
                mock.patch.object(engines.multiprocessing, "cpu_count", return_value=16):
            self.assertEqual(engines._resolve_voicevox_cpu_threads(None), 12)
            self.assertEqual(engines._resolve_voicevox_cpu_threads("16"), 12)
            self.assertEqual(engines._resolve_voicevox_cpu_threads("8"), 8)

    def test_other_os_unchanged(self) -> None:
        with mock.patch.object(engines, "_macos_performance_cores", return_value=None), \
                mock.patch.object(engines.multiprocessing, "cpu_count", return_value=24):
            self.assertEqual(engines._resolve_voicevox_cpu_threads(None), 16)
            self.assertEqual(engines._resolve_voicevox_cpu_threads("20"), 20)


class CpuBuildTest(unittest.TestCase):
    def test_macos_is_always_a_cpu_build(self) -> None:
        # .device_extra が無くても、GPU（CUDA・DirectML）の選択肢を塞ぐ
        with mock.patch.object(device.sys, "platform", "darwin"), \
                mock.patch("maai_sds.engine_manager.device_extra", return_value=""):
            self.assertTrue(device.is_cpu_build())
            self.assertIn("GPU (DirectML)", device.disabled_choices()["voicevox_core_device_combo"])

    def test_other_os_follows_the_marker(self) -> None:
        with mock.patch.object(device.sys, "platform", "win32"), \
                mock.patch("maai_sds.engine_manager.device_extra", return_value="cu128"):
            self.assertFalse(device.is_cpu_build())

    def test_worker_settings_use_the_same_cpu_build_check(self) -> None:
        # GUI を通さないヘッドレス・Web でも、macOS では見本の cuda をワーカーへ渡さない
        from maai_sds.dialog import settings as vds

        with mock.patch.object(device.sys, "platform", "darwin"), \
                mock.patch("maai_sds.engine_manager.device_extra", return_value=""):
            self.assertTrue(vds._is_cpu_build())
            self.assertEqual(vds._device("cuda", label="test"), "cpu")
            self.assertEqual(vds._compute_type("int8_float16", label="test"), "int8")
        with mock.patch.object(device, "is_cpu_build", return_value=False):
            self.assertEqual(vds._device("cuda", label="test"), "cuda")


class CpuBuildGuiStartupTest(unittest.TestCase):
    """CPU 構成で GUI を作ると GPU の値を CPU に寄せて保存する。その間に対話サーバーを待たない。"""

    @classmethod
    def setUpClass(cls) -> None:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_gpu_values_are_saved_as_cpu_without_starting_modules(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from maai_sds.gui import controller as controller_module
        from maai_sds.gui.controller import MMDAgentController
        from maai_sds.ipc.process_service import DialogOrchestratorProcessService
        from maai_sds.mmd import default_paths

        class _NoDeviceSoundDevice:
            @staticmethod
            def query_devices(*_args, **_kwargs):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "controller_config.json"
            target.write_text(
                json.dumps({"asr_device": "cuda", "voicevox_core_device": "GPU (DirectML)"}), encoding="utf-8"
            )
            # 対話サーバーにつながった状態で作る（Docker などリモートの対話サーバーでは、GUI の初期化中に
            # もうつながっている）。以前はこの状態で保存が音声出力モジュールの起動を頼み、返事を待った。
            with mock.patch.object(device, "is_cpu_build", lambda: True), \
                    mock.patch.object(MMDAgentController, "_is_dialog_orchestrator_runtime_active", lambda _self: True), \
                    mock.patch.object(DialogOrchestratorProcessService, "start", lambda *_a, **_k: None), \
                    mock.patch.object(controller_module, "sd", _NoDeviceSoundDevice()), \
                    mock.patch.object(default_paths, "detect_default_exec_path", lambda *_a, **_k: ""), \
                    mock.patch.object(default_paths, "detect_default_mdf_path", lambda *_a, **_k: ""), \
                    mock.patch.object(
                        MMDAgentController, "_start_module_via_orchestrator",
                        side_effect=AssertionError("GUI の初期化中に対話サーバーのモジュール起動を待たない"),
                    ) as start_module:
                window = MMDAgentController(config_path=target)
                try:
                    saved = json.loads(target.read_text(encoding="utf-8"))
                finally:
                    window.close()
                    window.deleteLater()
                    self.app.processEvents()
            start_module.assert_not_called()
            self.assertEqual(saved.get("asr_device"), "cpu")
            self.assertEqual(saved.get("voicevox_core_device"), "CPU")


if __name__ == "__main__":
    unittest.main()
