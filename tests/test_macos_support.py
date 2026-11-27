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


if __name__ == "__main__":
    unittest.main()
