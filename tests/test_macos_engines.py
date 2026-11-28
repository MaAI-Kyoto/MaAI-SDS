"""macOS（Apple Silicon）で各エンジンを動かすための分岐。どの OS でも確かめられるよう、OS や torch を差し替える。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from maai_sds.tts import irodori_server


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


if __name__ == "__main__":
    unittest.main()
