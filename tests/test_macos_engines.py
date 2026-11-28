"""macOS（Apple Silicon）で各エンジンを動かすための分岐。どの OS でも確かめられるよう、OS や torch を差し替える。"""

from __future__ import annotations

import queue
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from maai_sds import engine_manager, paths
from maai_sds.asr.service import ASRService
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


class InferenceQueueKeepsFinalsTest(unittest.TestCase):
    """推論が遅いエンジン（Mac の CPU の faster-whisper）で、疑似ストリーミングの partial が final を押し出さない。"""

    def _service(self, size: int = 4) -> ASRService:
        service = object.__new__(ASRService)  # キューだけを使う
        service.inference_queue = queue.Queue(maxsize=size)
        return service

    @staticmethod
    def _p(name: str) -> dict:
        return {"result_mode": "partial", "name": name}

    @staticmethod
    def _f(name: str) -> dict:
        return {"result_mode": "final", "name": name}

    @staticmethod
    def _names(service: ASRService) -> list[str]:
        return [item["name"] for item in list(service.inference_queue.queue)]

    def test_final_drops_waiting_partials(self) -> None:
        service = self._service()
        for name in ("p1", "p2", "p3"):
            service._put_inference(self._p(name))
        service._put_inference(self._f("f1"))
        self.assertEqual(self._names(service), ["f1"])

    def test_partials_of_the_next_utterance_do_not_push_the_final_out(self) -> None:
        service = self._service()
        service._put_inference(self._f("f1"))
        for i in range(10):  # 推論が追いつかない間に次の発話の partial が続く
            service._put_inference(self._p(f"q{i}"))
        names = self._names(service)
        self.assertEqual(names[0], "f1")
        self.assertEqual(len(names), 4)
        self.assertEqual(names[-1], "q9")  # partial は新しいものが残る

    def test_full_of_finals_keeps_them_and_skips_the_partial(self) -> None:
        service = self._service(size=2)
        service._put_inference(self._f("f1"))
        service._put_inference(self._f("f2"))
        self.assertFalse(service._put_inference(self._p("p")))
        self.assertEqual(self._names(service), ["f1", "f2"])
        # final どうしは以前と同じく古いほうを捨てる
        self.assertTrue(service._put_inference(self._f("f3")))
        self.assertEqual(self._names(service), ["f2", "f3"])

    def test_order_is_kept_when_not_full(self) -> None:
        service = self._service()
        service._put_inference(self._f("f1"))
        service._put_inference(self._p("p1"))
        self.assertEqual(self._names(service), ["f1", "p1"])


if __name__ == "__main__":
    unittest.main()
