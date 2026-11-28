"""macOS（Apple Silicon）で各エンジンを動かすための分岐。どの OS でも確かめられるよう、OS や torch を差し替える。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from maai_sds import engine_manager, paths
from maai_sds.tts import audiocpp_client, irodori_server

_MAC_SERVER = "audio.cpp/build/macos-metal-release/bin/audiocpp_server"
_WIN_SERVER = "audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe"  # 見本の設定の値


def _fake_torch(*, cuda: bool, mps: bool | None) -> SimpleNamespace:
    backends = SimpleNamespace() if mps is None else SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda), backends=backends)


class IrodoriDeviceTest(unittest.TestCase):
    def test_cuda_keeps_bf16(self) -> None:
        self.assertEqual(irodori_server.pick_device_and_precision(_fake_torch(cuda=True, mps=True)), ("cuda", "bf16"))

    def test_mps_uses_fp32(self) -> None:
        # irodori_tts は bf16 を CUDA・XPU にしか許さない
        self.assertEqual(irodori_server.pick_device_and_precision(_fake_torch(cuda=False, mps=True)), ("mps", "fp32"))

    def test_cpu_when_neither(self) -> None:
        self.assertEqual(irodori_server.pick_device_and_precision(_fake_torch(cuda=False, mps=False)), ("cpu", "fp32"))
        # MPS の無い古い torch
        self.assertEqual(irodori_server.pick_device_and_precision(_fake_torch(cuda=False, mps=None)), ("cpu", "fp32"))


class AudioCppBackendTest(unittest.TestCase):
    def test_macos_defaults_to_metal(self) -> None:
        # Mac 版のサーバーを cuda で起動すると exit 1 で落ちる
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(audiocpp_client.backend_for_server(Path("/x/bin/audiocpp_server")), "metal")
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(audiocpp_client.backend_for_server(Path("/x/bin/audiocpp_server")), "cuda")

    def test_metal_folder_or_flavor_file(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(audiocpp_client.backend_for_server(Path("/r") / _MAC_SERVER), "metal")
            with tempfile.TemporaryDirectory() as tmp:
                exe = Path(tmp) / "bin" / "audiocpp_server"
                exe.parent.mkdir()
                (exe.parent / "FLAVOR.txt").write_text("metal audio.cpp 0.9.6", encoding="ascii")
                self.assertEqual(audiocpp_client.backend_for_server(exe), "metal")

    def test_windows_flavors_unchanged(self) -> None:
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(audiocpp_client.backend_for_server(Path("C:/r") / _WIN_SERVER), "cuda")
            self.assertEqual(
                audiocpp_client.backend_for_server(Path("C:/r/audio.cpp/build/windows-vulkan-release/bin/a.exe")),
                "vulkan")


class AudioCppServerPathTest(unittest.TestCase):
    def test_macos_uses_its_own_server(self) -> None:
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(paths.default_audiocpp_server(), _MAC_SERVER)
            # 見本の設定の Windows の .exe は無いので、macOS の置き場になる（置かれていなくてもエラーにはその場所が出る）
            self.assertEqual(paths.resolve_audiocpp_server(_WIN_SERVER), paths.resolve_under_repo(_MAC_SERVER))
            self.assertEqual(paths.resolve_audiocpp_server(""), paths.resolve_under_repo(_MAC_SERVER))

    def test_other_os_never_picks_the_mac_server(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            self.assertNotEqual(paths.resolve_audiocpp_server(_WIN_SERVER), paths.resolve_under_repo(_MAC_SERVER))
            self.assertNotEqual(paths.default_audiocpp_server(), _MAC_SERVER)

    def test_engine_manager_probes_the_same_place(self) -> None:
        with mock.patch.object(sys, "platform", "darwin"), mock.patch.object(engine_manager.os, "name", "posix"):
            self.assertEqual(engine_manager._posixify_probe_path(_WIN_SERVER), _MAC_SERVER)
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(engine_manager.os, "name", "posix"):
            self.assertEqual(engine_manager._posixify_probe_path(_WIN_SERVER),
                             "audio.cpp/build/linux-cuda-release/bin/audiocpp_server")


class MacosTtsEngineRunTest(unittest.TestCase):
    """アプリ版 VOICEVOX・AivisSpeech：macOS では setup_assets.sh が置いたエンジンの run を使う。"""

    def test_probe_paths_on_macos(self) -> None:
        aivis = "%LOCALAPPDATA%/Programs/AivisSpeech/AivisSpeech-Engine/run.exe"
        voicevox = "%LOCALAPPDATA%/Programs/VOICEVOX/VOICEVOX.exe"
        with mock.patch.object(sys, "platform", "darwin"), mock.patch.object(engine_manager.os, "name", "posix"):
            self.assertEqual(engine_manager._posixify_probe_path(aivis), "aivisspeech_engine/run")
            self.assertEqual(engine_manager._posixify_probe_path(voicevox), "voicevox_engine/run")
        # Linux はホストの HTTP エンジンへ繋ぐので、本体は確かめない（以前と同じ）
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(engine_manager.os, "name", "posix"):
            self.assertIsNone(engine_manager._posixify_probe_path(aivis))
            self.assertIsNone(engine_manager._posixify_probe_path(voicevox))

    def test_configured_path_wins_then_the_macos_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            own = Path(tmp) / "my_run"
            own.write_bytes(b"")
            installed = Path(tmp) / "voicevox_engine" / "run"
            installed.parent.mkdir()
            installed.write_bytes(b"")
            with mock.patch.object(sys, "platform", "darwin"), \
                    mock.patch.object(paths, "resolve_under_repo", lambda rel: Path(tmp) / rel):
                self.assertEqual(paths.resolve_tts_engine_exe(own, "voicevox"), own)
                self.assertEqual(paths.resolve_tts_engine_exe(None, "voicevox"), installed)
                self.assertEqual(paths.resolve_tts_engine_exe(r"%LOCALAPPDATA%\VOICEVOX\VOICEVOX.exe", "voicevox"),
                                 installed)
                # 置かれていなければ設定の値のまま（呼ぶ側が「見つからない」を出す）
                self.assertEqual(paths.resolve_tts_engine_exe("missing.exe", "aivisspeech"), Path("missing.exe"))
                self.assertIsNone(paths.resolve_tts_engine_exe("", "aivisspeech"))
            with mock.patch.object(sys, "platform", "win32"), \
                    mock.patch.object(paths, "resolve_under_repo", lambda rel: Path(tmp) / rel):
                self.assertIsNone(paths.resolve_tts_engine_exe(None, "voicevox"))


if __name__ == "__main__":
    unittest.main()
