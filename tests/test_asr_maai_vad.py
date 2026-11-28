"""VAD を MaAI にしたとき、MaAI が結果を返さないチャンクがあっても VAD のループが止まらない。

Mimi を ONNX で動かすモデル（normal-ver2）は最初のエンコーダー出力を捨て、そのチャンクの結果を出さない。
以前は Maai.get_result() で待ち続けてループが止まり、ASR が区間を 1 つも切り出さなかった。
"""

from __future__ import annotations

import queue
import time
import unittest

import numpy as np

from maai_sds.asr.service import ASRService

_CHUNK = 4


class _FakeMaai:
    """チャンクの平均を vad として返す。skip_first なら最初のチャンクの結果を出さない（Mimi・ONNX と同じ）。"""

    def __init__(self, *, skip_first: bool = False, queue_attr: bool = True) -> None:
        self._skip_first = skip_first
        self.result_dict_queue: queue.Queue[dict] = queue.Queue()
        self.held: list[dict] = []  # 遅れて届く結果
        if not queue_attr:
            del self.result_dict_queue
            self._fallback: queue.Queue[dict] = queue.Queue()

    def feed(self, chunk: np.ndarray) -> None:
        if self._skip_first:
            self._skip_first = False
            return
        result = {"vad": [float(np.mean(chunk)), 0.0]}
        target = getattr(self, "result_dict_queue", None) or self._fallback
        target.put(result)

    def get_result(self) -> dict:
        # 本物と同じく、結果が来るまで待ち続ける
        return (getattr(self, "result_dict_queue", None) or self._fallback).get()


class _FakeChunk:
    def __init__(self, model: _FakeMaai | None) -> None:
        self._model = model

    def put_chunk(self, chunk: np.ndarray) -> None:
        if self._model is not None:
            self._model.feed(chunk)


def _service(model: _FakeMaai) -> tuple[ASRService, list[dict]]:
    service = object.__new__(ASRService)  # VAD に要るものだけを持たせる
    emitted: list[dict] = []
    service.vad_backend = "maai"
    service._maai_instance = model
    service._maai_mic_chunk = _FakeChunk(model)
    service._maai_zero_chunk = _FakeChunk(None)
    service._vap_maai_vad_buffer = np.empty(0, dtype=np.float32)
    service._vap_maai_chunk_size = _CHUNK
    service.MAAI_VAD_RESULT_TIMEOUT_SEC = 0.05
    service._get_system_audio_chunk = lambda n: np.zeros(n, dtype=np.float32)
    service._emit_maai_result = lambda result, **_kw: emitted.append(result)
    service._error = lambda message: (_ for _ in ()).throw(AssertionError(message))
    return service, emitted


def _chunk(value: float) -> np.ndarray:
    return np.full(_CHUNK, value, dtype=np.float32)


class MaaiVadResultTest(unittest.TestCase):
    def test_missing_first_result_does_not_block(self) -> None:
        service, emitted = _service(_FakeMaai(skip_first=True))
        started = time.monotonic()
        self.assertIsNone(service._get_vad_score(_chunk(0.1)))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(emitted, [])
        # 2 つ目からはチャンクごとに結果が出て、そのチャンクの値になる
        self.assertAlmostEqual(service._get_vad_score(_chunk(0.9)), 0.9, places=5)
        self.assertAlmostEqual(service._get_vad_score(_chunk(0.2)), 0.2, places=5)
        self.assertEqual(len(emitted), 2)

    def test_one_result_per_chunk(self) -> None:
        service, emitted = _service(_FakeMaai())
        scores = [service._get_vad_score(_chunk(v)) for v in (0.3, 0.8, 0.0)]
        self.assertEqual([round(s, 5) for s in scores], [0.3, 0.8, 0.0])
        self.assertEqual(len(emitted), 3)

    def test_late_result_is_taken_with_the_next_one(self) -> None:
        model = _FakeMaai()
        service, emitted = _service(model)
        # 1 つ目の結果が上限までに届かなかった
        model.feed = lambda chunk: model.held.append({"vad": [float(np.mean(chunk)), 0.0]})
        self.assertIsNone(service._get_vad_score(_chunk(0.4)))
        # 次のチャンクの結果と一緒に受け取り、新しいほうを返す（どちらもプロットへ送る）
        del model.feed
        for late in model.held:
            model.result_dict_queue.put(late)
        self.assertAlmostEqual(service._get_vad_score(_chunk(0.7)), 0.7, places=5)
        self.assertEqual([round(r["vad"][0], 5) for r in emitted], [0.4, 0.7])
        self.assertTrue(model.result_dict_queue.empty())

    def test_several_chunks_in_one_call(self) -> None:
        service, emitted = _service(_FakeMaai(skip_first=True))
        audio = np.concatenate([_chunk(0.1), _chunk(0.6), _chunk(0.5)])
        self.assertAlmostEqual(service._get_vad_score(audio), 0.5, places=5)
        self.assertEqual(len(emitted), 2)

    def test_model_without_result_queue_uses_get_result(self) -> None:
        service, emitted = _service(_FakeMaai(queue_attr=False))
        self.assertAlmostEqual(service._get_vad_score(_chunk(0.6)), 0.6, places=5)
        self.assertEqual(len(emitted), 1)


if __name__ == "__main__":
    unittest.main()
