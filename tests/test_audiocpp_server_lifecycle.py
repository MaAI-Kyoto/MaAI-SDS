"""audio.cpp のサーバーの後始末（使い直してよいか、backend とスレッド数の決め方）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from maai_sds.tts import audiocpp_client as ac
from maai_sds.tts.engines import SupertonicEngine, _audiocpp_backend_for


class ReuseRunningServerTest(unittest.TestCase):
    def _files(self, tmp: str, stored: dict | None, pid: int | None) -> Path:
        cfg = Path(tmp) / "maai-x-1.json"
        if stored is not None:
            cfg.write_text(json.dumps(stored), encoding="utf-8")
        if pid is not None:
            cfg.with_suffix(".pid").write_text(str(pid), encoding="utf-8")
        return cfg

    def test_without_a_record_the_server_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._files(tmp, None, None)
            self.assertTrue(ac.reuse_running_server(base_url="http://127.0.0.1:1", config_file=cfg, config={"a": 1}, name="t"))

    def test_same_config_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._files(tmp, {"a": 1}, 12345)
            with mock.patch.object(ac, "_stop_pid") as stop:
                self.assertTrue(ac.reuse_running_server(base_url="http://127.0.0.1:1", config_file=cfg, config={"a": 1}, name="t"))
            stop.assert_not_called()

    def test_different_config_from_our_server_is_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._files(tmp, {"threads": 8}, 12345)
            with mock.patch.object(ac, "_is_audiocpp_process", return_value=True), \
                    mock.patch.object(ac, "_stop_pid") as stop:
                reuse = ac.reuse_running_server(base_url="http://127.0.0.1:1", config_file=cfg, config={"threads": 2}, name="t")
            self.assertFalse(reuse)
            stop.assert_called_once_with(12345)
            self.assertFalse(cfg.with_suffix(".pid").exists())

    def test_a_reused_pid_of_another_program_is_not_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._files(tmp, {"threads": 8}, os.getpid())  # この Python は audio.cpp ではない
            with mock.patch.object(ac, "_stop_pid") as stop:
                self.assertTrue(ac.reuse_running_server(base_url="http://127.0.0.1:1", config_file=cfg, config={"threads": 2}, name="t"))
            stop.assert_not_called()


class SupertonicBackendAndThreadsTest(unittest.TestCase):
    def test_backend_from_the_server_folder(self) -> None:
        self.assertEqual(_audiocpp_backend_for(Path("audio.cpp/build/windows-vulkan-release/bin/audiocpp_server.exe")), "vulkan")
        self.assertEqual(_audiocpp_backend_for(Path("x/build/windows-cpu/bin/audiocpp_server.exe")), "cpu")
        self.assertEqual(_audiocpp_backend_for(Path("C:/nowhere/build/windows-cuda-release/bin/audiocpp_server.exe")), "cuda")

    def test_backend_from_flavor_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "build" / "windows-cuda-release" / "bin" / "audiocpp_server.exe"
            exe.parent.mkdir(parents=True)
            (exe.parent / "FLAVOR.txt").write_text("cpu (hand placed)", encoding="utf-8")
            self.assertEqual(_audiocpp_backend_for(exe), "cpu")

    def test_threads_default_by_backend(self) -> None:
        self.assertEqual(SupertonicEngine({"backend": "vulkan"}).threads, 2)
        self.assertEqual(SupertonicEngine({"backend": "cuda"}).threads, 8)
        self.assertLessEqual(SupertonicEngine({"backend": "cpu"}).threads, 4)
        self.assertEqual(SupertonicEngine({"backend": "vulkan", "threads": 3}).threads, 3)


class OtherAudioCppEnginesTest(unittest.TestCase):
    """LFM2.5-Audio・Irodori（audio.cpp 版）も、置いてあるサーバーの種類で backend とスレッド数を決める。

    以前は cuda・8 スレッド固定で、NVIDIA の GPU が無い PC では動かず、4 コアの PC では合成が止まった。
    """

    def _exe(self, tmp: str, flavor: str | None) -> Path:
        exe = Path(tmp) / "build" / "windows-cuda-release" / "bin" / "audiocpp_server.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
        if flavor is not None:
            (exe.parent / "FLAVOR.txt").write_text(f"{flavor} audio.cpp 0.9.6", encoding="utf-8")
        return exe

    def test_server_follows_the_flavor_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = ac.AudioCppServer(host="", port=1, models=[], server_exe=str(self._exe(tmp, "vulkan")))
            self.assertEqual((server.backend, server.threads), ("vulkan", 2))

    def test_old_setup_without_flavor_file_stays_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = ac.AudioCppServer(host="", port=1, models=[], server_exe=str(self._exe(tmp, None)))
            self.assertEqual((server.backend, server.threads), ("cuda", 8))

    def test_explicit_values_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(self._exe(tmp, "vulkan"))
            server = ac.AudioCppServer(host="", port=1, models=[], server_exe=exe, backend="CPU", threads=3)
            self.assertEqual((server.backend, server.threads), ("cpu", 3))

    def test_lfm2_and_irodori_pass_auto_to_the_server(self) -> None:
        from maai_sds.tts.engines_audiocpp import IrodoriAudioCppEngine, Lfm2AudioEngine

        with tempfile.TemporaryDirectory() as tmp:
            exe = str(self._exe(tmp, "cpu"))
            for engine_cls in (Lfm2AudioEngine, IrodoriAudioCppEngine):
                engine = engine_cls({"server_exe": exe})
                self.assertEqual(engine.server.backend, "cpu", engine_cls.__name__)
                self.assertLessEqual(engine.server.threads, 4, engine_cls.__name__)
                forced = engine_cls({"server_exe": exe, "audiocpp_backend": "cuda", "audiocpp_threads": 6})
                self.assertEqual((forced.server.backend, forced.server.threads), ("cuda", 6))


if __name__ == "__main__":
    unittest.main()
