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


if __name__ == "__main__":
    unittest.main()
