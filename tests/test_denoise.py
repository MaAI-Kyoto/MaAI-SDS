"""マイク入力の雑音抑制（audio/denoise.py）。"""

from __future__ import annotations

import unittest

import numpy as np

from maai_sds.audio.denoise import make_noise_suppressor
from maai_sds.dialog import settings as vds

_SR = 16000


def _noisy(seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.02 * rng.standard_normal(int(_SR * seconds))).astype(np.float32)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


class DenoiseTest(unittest.TestCase):
    def _run(self, mode: str, chunk: int = 320) -> tuple[np.ndarray, np.ndarray]:
        proc = make_noise_suppressor(mode, _SR)
        self.assertIsNotNone(proc, mode)
        x = _noisy()
        out = []
        for i in range(0, x.size, chunk):
            y = proc(x[i:i + chunk])
            # WAV の 2 ch 目などと並べて送るので、長さは必ず入力と同じ
            self.assertEqual(y.size, x[i:i + chunk].size)
            out.append(y)
        return x, np.concatenate(out)

    def test_off_and_unknown_modes_do_nothing(self) -> None:
        self.assertIsNone(make_noise_suppressor("off", _SR))
        self.assertIsNone(make_noise_suppressor("", _SR))
        self.assertIsNone(make_noise_suppressor("something", _SR))

    def test_webrtc_lowers_steady_noise(self) -> None:
        x, y = self._run("webrtc")
        tail = slice(_SR // 2, None)  # 最初の適応の間は除く
        self.assertLess(_rms(y[tail]), _rms(x[tail]) * 0.5)

    def test_rnnoise_lowers_steady_noise_with_odd_chunk_sizes(self) -> None:
        x, y = self._run("rnnoise", chunk=333)  # 10 ms の倍数でない大きさでも長さは変わらない
        tail = slice(_SR // 2, None)
        self.assertLess(_rms(y[tail]), _rms(x[tail]) * 0.1)

    def test_setting_reaches_the_audio_input_worker(self) -> None:
        for value, want in (("rnnoise", "rnnoise"), ("WebRTC", "webrtc"), ("", "off")):
            cfg = {"audio_input_noise_suppression": value}
            self.assertEqual(vds.build_audio_input_worker_config(cfg)["noise_suppression"], want)
        self.assertEqual(vds.build_audio_input_worker_config({})["noise_suppression"], "off")


if __name__ == "__main__":
    unittest.main()
