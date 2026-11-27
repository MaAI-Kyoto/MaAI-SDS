"""VOICEVOX-Core の文節単位の分割合成（tts/voicevox_chunked.py）。

voicevox_core を使わず、AudioQuery と同じ形のデータクラスと「各フレームの値がそのモーラだけで決まる」
偽の合成関数で確かめる。偽の合成は前後の文脈を見ないので、分割してつないだ結果は 1 文まとめて
合成した結果とサンプル単位で一致するはず（クロスフェードは同じ値どうしを混ぜるだけ）。
"""

from __future__ import annotations

import copy
import dataclasses
import unittest

import numpy as np

from maai_sds.dialog import settings as vds
from maai_sds.tts import voicevox_chunked as vc
from maai_sds.tts.playback_service import is_binary_stream_engine


@dataclasses.dataclass
class Mora:
    text: str
    vowel: str
    vowel_length: float
    pitch: float
    consonant: str | None = None
    consonant_length: float | None = None


@dataclasses.dataclass
class AccentPhrase:
    moras: list
    accent: int
    pause_mora: Mora | None = None
    is_interrogative: bool = False


@dataclasses.dataclass
class AudioQuery:
    accent_phrases: list
    speed_scale: float = 1.0
    pre_phoneme_length: float = 0.0
    post_phoneme_length: float = 0.1


def _mora(text: str, c: float | None, v: float, pitch: float = 5.5) -> Mora:
    return Mora(text=text, vowel="a", vowel_length=v, pitch=pitch, consonant="k" if c else None, consonant_length=c)


def _query(n_phrases: int = 6, *, interrogative: bool = False, speed: float = 1.0) -> AudioQuery:
    phrases = []
    for i in range(n_phrases):
        moras = [_mora(f"p{i}m{j}", 0.05 + 0.01 * j if j % 2 else None, 0.08 + 0.013 * i + 0.007 * j) for j in range(3 + i % 3)]
        pause = Mora(text="、", vowel="pau", vowel_length=0.21, pitch=0.0) if i == 2 else None
        phrases.append(AccentPhrase(moras=moras, accent=2, pause_mora=pause))
    phrases[-1].is_interrogative = interrogative
    return AudioQuery(accent_phrases=phrases, speed_scale=speed)


def _code(text: str) -> float:
    return float(sum(ord(ch) for ch in text) % 997) + 1.0


def _fake_synth(query: AudioQuery) -> np.ndarray:
    """フレームごとに、そのモーラの文字列から決まる値を置く（前後の文脈を見ない合成）。"""
    assert query.speed_scale == 1.0, "prepare_query 済みのものだけを合成する"
    vals: list[float] = [0.0] * vc._frames(query.pre_phoneme_length)
    for ap in query.accent_phrases:
        for m in ap.moras:
            if m.consonant_length is not None:
                vals += [-_code(m.text)] * vc._frames(m.consonant_length)
            vals += [_code(m.text)] * vc._frames(m.vowel_length)
        if ap.pause_mora is not None:
            vals += [0.5] * vc._frames(ap.pause_mora.vowel_length)
    vals += [0.0] * vc._frames(query.post_phoneme_length)
    return np.repeat(np.asarray(vals, dtype=np.float32), vc.HOP).astype(np.int16)


def _run_chunked(q: AudioQuery, **kw) -> np.ndarray:
    out: list[np.ndarray] = []
    vc.synthesize_chunked(q, _fake_synth, out.append, **kw)
    return np.concatenate(out)


class PlanChunksTests(unittest.TestCase):
    def test_short_text_is_not_split(self) -> None:
        self.assertEqual(vc.plan_chunks([20, 20]), [(0, 2)])  # 計 0.43 秒 < 1 秒
        self.assertEqual(vc.plan_chunks([200]), [(0, 1)])
        self.assertEqual(vc.plan_chunks([]), [])

    def test_first_chunk_is_small_and_later_chunks_grow(self) -> None:
        plan = vc.plan_chunks([50] * 20)  # 1 文節 0.53 秒
        self.assertEqual(plan[0], (0, 1))
        sizes = [b - a for a, b in plan]
        # 最後は「残り全部」なので小さくてよい。それ以外は前より大きくなる。
        self.assertTrue(all(x < y for x, y in zip(sizes[:-2], sizes[1:-1])), sizes)
        self.assertEqual(plan[0][0], 0)
        self.assertEqual(plan[-1][1], 20)
        self.assertTrue(all(plan[i][1] == plan[i + 1][0] for i in range(len(plan) - 1)))

    def test_short_first_phrase_is_merged(self) -> None:
        self.assertEqual(vc.plan_chunks([10, 10, 60, 60, 60])[0], (0, 3))  # 0.3 秒に届くまで足す


def _simulate(counts, lctx, rctx, cost, chunks, margin=1.3):
    """分け方を順に合成・再生したときの、締め切りに対する最大の遅れ（秒）。負なら途切れない。"""
    secs = [c / vc.FRAME_RATE for c in counts]
    n = len(counts)

    def t(a, b):
        audio = sum(secs[a:b]) + (lctx[a] if a > 0 else 0) / vc.FRAME_RATE + (rctx[b] if b < n else 0) / vc.FRAME_RATE
        return cost.predict(audio) * margin

    done = t(*chunks[0])
    play_end = done + sum(secs[slice(*chunks[0])])
    worst = -1.0
    for a, b in chunks[1:]:
        done += t(a, b)
        worst = max(worst, done - play_end)
        play_end += sum(secs[a:b])
    return worst


class AdaptivePlanTests(unittest.TestCase):
    counts = [50, 40, 60, 45, 55, 50, 40, 60, 50, 45]  # 1 文節 0.4〜0.6 秒、計 約 5.2 秒
    lctx = [0] + [25] * 9
    rctx = [25] * 10 + [0]

    def _plan(self, cost):
        return vc.plan_chunks_adaptive(self.counts, self.lctx, self.rctx, cost)

    def test_fast_machine_starts_with_one_phrase_and_never_underruns(self) -> None:
        cost = vc.CostModel()
        plan = self._plan(cost)
        self.assertEqual(plan[0], (0, 1))
        self.assertLessEqual(_simulate(self.counts, self.lctx, self.rctx, cost, plan), 0.0)
        self.assertEqual(plan[-1][1], len(self.counts))

    def test_slow_machine_takes_a_bigger_first_chunk(self) -> None:
        fast = self._plan(vc.CostModel())
        slow_cost = vc.CostModel(scale=2.5)
        slow = self._plan(slow_cost)
        self.assertGreater(slow[0][1], fast[0][1])
        if len(slow) > 1:
            self.assertLessEqual(_simulate(self.counts, self.lctx, self.rctx, slow_cost, slow), 0.0)

    def test_very_slow_machine_still_splits_only_when_safe(self) -> None:
        # 8 倍遅くても「最初に大きく作り、残りをその再生中に作る」分け方なら途切れない
        cost = vc.CostModel(scale=8.0)
        plan = self._plan(cost)
        self.assertGreater(plan[0][1], 1)
        if len(plan) > 1:
            self.assertLessEqual(_simulate(self.counts, self.lctx, self.rctx, cost, plan), 0.0)

    def test_too_slow_machine_does_not_split(self) -> None:
        # 1 文節の合成が再生より長くかかるほど遅いと、どう分けても途切れるので分けない
        self.assertEqual(self._plan(vc.CostModel(scale=30.0)), [(0, len(self.counts))])

    def test_short_text_is_not_split(self) -> None:
        self.assertEqual(vc.plan_chunks_adaptive([30, 30], [0, 10], [10, 10, 0], vc.CostModel()), [(0, 2)])

    def test_cost_model_tracks_measured_speed(self) -> None:
        cost = vc.CostModel()
        for _ in range(10):
            cost.observe(1.0, 3 * (cost.fixed_sec + cost.per_audio_sec))
        self.assertAlmostEqual(cost.scale, 3.0, delta=0.05)


class ChunkedSynthesisTests(unittest.TestCase):
    def test_matches_whole_sentence_sample_for_sample(self) -> None:
        for speed in (1.0, 0.9, 1.25):
            for interrogative in (False, True):
                with self.subTest(speed=speed, interrogative=interrogative):
                    q = vc.prepare_query(_query(7, interrogative=interrogative, speed=speed))
                    full = _fake_synth(q)
                    chunked = _run_chunked(q)
                    self.assertEqual(len(chunked), len(full))
                    np.testing.assert_array_equal(chunked, full)

    def test_context_and_crossfade_settings(self) -> None:
        q = vc.prepare_query(_query(6))
        full = _fake_synth(q)
        for ctx in (0, 1, 3, 10):
            with self.subTest(ctx=ctx):
                np.testing.assert_array_equal(_run_chunked(q, context_moras=ctx), full)

    def test_first_audio_comes_before_whole_sentence(self) -> None:
        q = vc.prepare_query(_query(8))
        sizes: list[int] = []
        vc.synthesize_chunked(q, _fake_synth, lambda x: sizes.append(x.size))
        self.assertGreater(len(sizes), 1)
        self.assertLess(sizes[0], len(_fake_synth(q)) // 3)

    def test_length_mismatch_on_first_chunk_raises(self) -> None:
        q = vc.prepare_query(_query(6))
        bad = lambda sub: np.zeros(_fake_synth(sub).size + vc.HOP, dtype=np.int16)  # noqa: E731
        sent: list[np.ndarray] = []
        with self.assertRaises(vc.ChunkedSynthesisError):
            vc.synthesize_chunked(q, bad, sent.append)
        self.assertEqual(sent, [])

    def test_length_mismatch_later_falls_back_without_gap(self) -> None:
        q = vc.prepare_query(_query(8))
        calls = [0]

        def flaky(sub: AudioQuery) -> np.ndarray:
            calls[0] += 1
            audio = _fake_synth(sub)
            return np.concatenate([audio, np.zeros(vc.HOP, np.int16)]) if calls[0] == 2 else audio

        out = _run_chunked_with(q, flaky)
        self.assertEqual(len(out), len(_fake_synth(q)))


def _run_chunked_with(q: AudioQuery, synth) -> np.ndarray:
    out: list[np.ndarray] = []
    vc.synthesize_chunked(q, synth, out.append)
    return np.concatenate(out)


class PrepareQueryTests(unittest.TestCase):
    def test_upspeak_adds_rising_mora_and_clears_flag(self) -> None:
        q = _query(3, interrogative=True)
        p = vc.prepare_query(q)
        last = p.accent_phrases[-1]
        self.assertFalse(last.is_interrogative)
        self.assertEqual(len(last.moras), len(q.accent_phrases[-1].moras) + 1)
        self.assertAlmostEqual(last.moras[-1].vowel_length, 0.15)
        self.assertAlmostEqual(last.moras[-1].pitch, q.accent_phrases[-1].moras[-1].pitch + 0.3)
        self.assertIsNone(last.moras[-1].consonant)
        self.assertTrue(q.accent_phrases[-1].is_interrogative, "元の AudioQuery は変えない")

    def test_upspeak_is_skipped_for_unvoiced_last_mora(self) -> None:
        q = _query(3, interrogative=True)
        q.accent_phrases[-1].moras[-1].pitch = 0.0
        self.assertEqual(len(vc.prepare_query(q).accent_phrases[-1].moras), len(q.accent_phrases[-1].moras))

    def test_speed_is_baked_into_whole_frames(self) -> None:
        q = _query(4, speed=0.8)
        p = vc.prepare_query(q)
        self.assertEqual(p.speed_scale, 1.0)
        self.assertEqual(q.speed_scale, 0.8)
        for ap_src, ap in zip(q.accent_phrases, p.accent_phrases):
            for m_src, m in zip(ap_src.moras, ap.moras):
                frames = m.vowel_length * vc.FRAME_RATE
                self.assertAlmostEqual(frames, round(frames), places=6)
                # VOICEVOX と同じ: f32 の長さをフレーム数に丸め、f32 の話速で割ってもう一度丸める
                expected = int(np.floor(vc._frames(m_src.vowel_length) / float(np.float32(0.8)) + 0.5))
                self.assertEqual(round(frames), expected)


class WiringTests(unittest.TestCase):
    def test_voicevox_core_is_a_streaming_engine(self) -> None:
        self.assertTrue(is_binary_stream_engine("VOICEVOX-Core"))

    def test_worker_config_carries_chunk_settings(self) -> None:
        cfg = vds.build_tts_worker_config({"tts_engine": "VOICEVOX-Core"}, "VOICEVOX-Core")
        self.assertEqual(cfg["chunked"], "auto")
        self.assertEqual(cfg["chunk_context_moras"], 3)
        cfg = vds.build_tts_worker_config(
            {"voicevox_core_chunked": "off", "voicevox_core_chunk_context_moras": "5"}, "VOICEVOX-Core"
        )
        self.assertEqual(cfg["chunked"], "off")
        self.assertEqual(cfg["chunk_context_moras"], 5)

    def test_chunked_mode_values(self) -> None:
        from maai_sds.tts.engines import _voicevox_chunked_mode

        for raw, want in ((None, "auto"), ("", "auto"), ("auto", "auto"), ("ON", "on"), (True, "on"),
                          ("true", "on"), ("off", "off"), (False, "off"), ("False", "off")):
            self.assertEqual(_voicevox_chunked_mode(raw), want, raw)

    def test_auto_splits_only_off_the_gpu(self) -> None:
        from types import SimpleNamespace

        from maai_sds.tts.engines import VoicevoxCoreEngine

        for mode, gpu, want in (("auto", False, True), ("auto", True, False), ("on", True, True),
                                ("off", False, False)):
            engine = VoicevoxCoreEngine({"chunked": mode})
            engine._synthesizer = SimpleNamespace(is_gpu_mode=gpu)
            self.assertIs(engine._use_chunked(), want, (mode, gpu))


if __name__ == "__main__":
    unittest.main()
