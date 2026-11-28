"""audio.cpp のサーバーの後始末（使い直してよいか、backend とスレッド数の決め方）。"""

from __future__ import annotations

import json
import os
import sys
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
        # 手がかりが無ければ cuda なのは Windows・Linux。macOS では metal になる（test_macos_engines）
        with mock.patch.object(sys, "platform", "win32"):
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
            # Windows の配置。macOS で走らせても Windows として確かめる（macOS の既定は metal）
            with mock.patch.object(sys, "platform", "win32"):
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

    def test_qwen3_tts_cpp_uses_the_same_rules(self) -> None:
        from maai_sds.tts.engines import Qwen3TTSCppEngine

        with tempfile.TemporaryDirectory() as tmp:
            engine = Qwen3TTSCppEngine({"server_exe": str(self._exe(tmp, "vulkan"))})
            self.assertEqual((engine.backend, engine.threads), ("vulkan", 2))


def _device(index: int, kind: str, total_gb: int, backend: str = "vulkan") -> dict:
    return {"backend": backend, "index": index, "type": kind,
            "memory": {"free_bytes": 0, "total_bytes": total_gb * 1024 ** 3}}


class PickDeviceTest(unittest.TestCase):
    """Vulkan 版で使う GPU を、サーバーの --list-devices --json から選ぶ。"""

    def _pick(self, devices: list[dict] | None, backend: str = "vulkan", stdout: str | None = None) -> int:
        ac._device_cache.clear()
        text = stdout if stdout is not None else json.dumps({"schema_version": 1, "devices": devices or []})
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "audiocpp_server.exe"
            exe.write_bytes(b"")
            done = mock.Mock(stdout=text)
            with mock.patch.object(ac.subprocess, "run", return_value=done) as run:
                picked = ac.pick_device(exe, backend)
                self.called = run.called
            return picked

    def test_discrete_gpu_wins_over_the_integrated_one(self) -> None:
        # RTX 5070 Ti の PC の実際の並び（0 番が Intel の内蔵 GPU）
        devices = [_device(0, "igpu", 18), _device(1, "gpu", 12), _device(0, "cpu", 32, backend="cpu")]
        self.assertEqual(self._pick(devices), 1)

    def test_bigger_memory_among_the_same_kind(self) -> None:
        self.assertEqual(self._pick([_device(0, "gpu", 8), _device(1, "gpu", 16)]), 1)
        self.assertEqual(self._pick([_device(0, "gpu", 16), _device(1, "gpu", 16)]), 0)

    def test_only_integrated_gpu(self) -> None:
        self.assertEqual(self._pick([_device(0, "igpu", 8)]), 0)

    def test_log_lines_before_the_json_are_ignored(self) -> None:
        body = json.dumps({"devices": [_device(0, "igpu", 8), _device(1, "gpu", 8)]})
        self.assertEqual(self._pick(None, stdout="ggml_vulkan: Found 2 Vulkan devices:\n" + body), 1)

    def test_falls_back_to_zero(self) -> None:
        self.assertEqual(self._pick(None, stdout="unknown option --list-devices"), 0)  # 古い版
        self.assertEqual(self._pick([]), 0)

    def test_cuda_and_cpu_do_not_list(self) -> None:
        for backend in ("cuda", "cpu"):
            self.assertEqual(self._pick([_device(1, "gpu", 8, backend=backend)], backend=backend), 0)
            self.assertFalse(self.called, backend)


if __name__ == "__main__":
    unittest.main()
