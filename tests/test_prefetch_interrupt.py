"""先読み（p_future）と割り込みの回帰テスト。

ユーザー発話中は request_interrupt が毎フレーム来る。未確定の先読みが保持用の TTS を
合成している間にそれが来ても、先読みを取り消してはいけない（実害: p_now で再生する
音声が消え、LLM を頼み直して返答が遅れる）。確定後や他のターンの合成は従来どおり止める。
"""

from __future__ import annotations

import queue
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from maai_sds.dialog.runtime import VoiceDialogRuntime  # noqa: E402


def _make_runtime() -> VoiceDialogRuntime:
    rt = VoiceDialogRuntime(
        emit=lambda ev: None,
        event_queue=queue.Queue(),
        supervisor=MagicMock(),
        mmd=MagicMock(),
        audio_input_server=MagicMock(),
        asr_audio_server=MagicMock(),
        asr_compare_audio_server=MagicMock(),
        maai_audio_server=MagicMock(),
    )
    rt._llm_settings["interrupt_on_user_speech"] = True
    rt._orchestrator = MagicMock()
    # セグメントワーカーを動かさず、受け取ったセグメントを合成中（inflight）のまま保つ。
    rt._ensure_vd_tts_seg_worker = lambda: None
    return rt


def _segment(rt: VoiceDialogRuntime, rid: str, text: str = "そうなんですね。") -> None:
    rt.handle_llm_worker_event(
        {"type": "tts_segment", "request_id": rid, "segment_text": text, "is_final": False}
    )


def _llm_cancelled(rt: VoiceDialogRuntime, rid: str) -> bool:
    return any(
        c.args and c.args[0].get("type") == "cancel" and c.args[0].get("request_id") == rid
        for c in rt._supervisor.send_llm.call_args_list
    )


class TestPrefetchSurvivesUserSpeechFrames(unittest.TestCase):
    def test_uncommitted_prefetch_synthesis_is_not_interrupted(self) -> None:
        rt = _make_runtime()
        rt._llm_pending["pf"] = {"prefetch": True, "committed": False, "tts_engine": "dummy"}
        _segment(rt, "pf")
        with rt._vd_tts_jobs_lock:
            self.assertEqual(rt._vd_tts_jobs["inflight"], 1)

        self.assertFalse(rt.execute_user_interrupt("user_speech_detected"))
        self.assertIn("pf", rt._llm_pending)
        self.assertFalse(_llm_cancelled(rt, "pf"))
        self.assertFalse(rt._vd_interrupt_stop.is_set())

    def test_committed_prefetch_is_interrupted(self) -> None:
        rt = _make_runtime()
        rt._llm_pending["pf"] = {"prefetch": True, "committed": False, "tts_engine": "dummy"}
        _segment(rt, "pf")
        rt._llm_pending["pf"]["committed"] = True

        self.assertTrue(rt.execute_user_interrupt("user_speech_detected"))
        self.assertNotIn("pf", rt._llm_pending)
        self.assertTrue(_llm_cancelled(rt, "pf"))

    def test_other_turn_synthesis_still_interrupts_and_drops_prefetch(self) -> None:
        rt = _make_runtime()
        # 前のターンの合成が残っている（受け取り後に LLM 要求は完了し、保留リストから外れた）。
        rt._llm_pending["old-turn"] = {"tts_engine": "dummy"}
        _segment(rt, "old-turn")
        rt._llm_pending.pop("old-turn")
        rt._llm_pending["pf"] = {"prefetch": True, "committed": False, "tts_engine": "dummy"}
        _segment(rt, "pf")

        self.assertTrue(rt.execute_user_interrupt("user_speech_detected"))
        self.assertNotIn("pf", rt._llm_pending)
        self.assertTrue(_llm_cancelled(rt, "pf"))
        with rt._vd_tts_jobs_lock:
            self.assertEqual(rt._vd_tts_jobs["inflight"], 0)
            self.assertEqual(rt._vd_tts_inflight_by_rid, {})

    def test_worker_keeps_per_request_count_in_step(self) -> None:
        rt = _make_runtime()
        rt._llm_play_tts_segment = MagicMock()
        rt._llm_pending["pf"] = {"prefetch": True, "committed": False, "tts_engine": "dummy"}
        _segment(rt, "pf", "一文目。")
        _segment(rt, "pf", "二文目。")
        with rt._vd_tts_jobs_lock:
            self.assertEqual(rt._vd_tts_inflight_by_rid, {"pf": 2})
        rt._vd_tts_seg_queue.put(None)  # 2 件処理したらループを抜ける
        rt._vd_tts_seg_worker_loop()
        with rt._vd_tts_jobs_lock:
            self.assertEqual(rt._vd_tts_jobs["inflight"], 0)
            self.assertEqual(rt._vd_tts_inflight_by_rid, {})


class TestBackchannelIsNotInterruptedByUserSpeech(unittest.TestCase):
    """相槌の再生開始直後に毎フレームの割り込みが実行され、相槌と先読みが消えていた（実機ログで確認）。"""

    def test_backchannel_playback_alone_is_not_interrupted(self) -> None:
        rt = _make_runtime()
        rt.nonstream_begin(backchannel=True)

        self.assertFalse(rt.execute_user_interrupt("user_speech_detected"))
        self.assertFalse(rt._vd_interrupt_stop.is_set())
        rt._orchestrator.clear_prefetch.assert_not_called()

    def test_prefetch_survives_backchannel_playback(self) -> None:
        rt = _make_runtime()
        rt._llm_pending["pf"] = {"prefetch": True, "committed": False, "tts_engine": "dummy"}
        rt.nonstream_begin(backchannel=True)

        self.assertFalse(rt.execute_user_interrupt("user_speech_detected"))
        self.assertIn("pf", rt._llm_pending)

    def test_other_nonstream_speech_is_still_interrupted(self) -> None:
        rt = _make_runtime()
        rt.nonstream_begin(backchannel=True)
        rt.nonstream_begin()  # 冒頭・終了などの通常の発話

        self.assertTrue(rt.execute_user_interrupt("user_speech_detected"))

    def test_counts_return_to_zero(self) -> None:
        rt = _make_runtime()
        rt.nonstream_begin(backchannel=True)
        rt.nonstream_begin()
        rt.nonstream_end(backchannel=True)
        rt.nonstream_end()
        self.assertEqual(int(rt._vd_nonstream_refs.get("n") or 0), 0)
        self.assertEqual(int(rt._vd_nonstream_refs.get("bc") or 0), 0)


class TestPrefetchLlmBandEndsAtFirstSegment(unittest.TestCase):
    """先読みも通常ターンと同じく、最初の文を TTS に渡した時点で LLM の緑帯を閉じる。"""

    def test_band_closes_when_first_prefetch_segment_arrives(self) -> None:
        emitted: list[dict] = []
        rt = _make_runtime()
        rt._emit = emitted.append
        rt.begin_llm_turn_for_latency()
        rt._llm_pending["pf"] = {"prefetch": True, "committed": False, "tts_engine": "dummy"}

        _segment(rt, "pf", "一文目。")
        _segment(rt, "pf", "二文目。")

        bands = [e for e in emitted if e.get("type") == "latency" and e.get("kind") == "llm" and e.get("value") is not None]
        self.assertEqual(len(bands), 1, "最初の文で 1 回だけ閉じる")
        with rt._vd_latency_lock:
            self.assertIsNotNone(rt._vd_turn_latency_state["tts_req_perf"], "同じ時点から TTS 帯を測り始める")


_AUDIBLE_PCM = bytes([0x00, 0x10]) * 3200  # 0.2 s、振幅 4096（可聴判定のしきい値 120 を超える）


def _tts_bands(emitted: list[dict]) -> list[dict]:
    return [e for e in emitted if e.get("type") == "latency" and e.get("kind") == "tts" and e.get("value") is not None]


class TestPrefetchTtsBand(unittest.TestCase):
    """先読みのマゼンタ（TTS）帯は、最初の文を TTS に渡した時点から、確定前に合成が終わればその時点まで、
    合成中に確定されれば最初の音が出た時点まで。どちらでも 1 本だけ出る。"""

    def _runtime(self) -> tuple[VoiceDialogRuntime, list[dict]]:
        emitted: list[dict] = []
        rt = _make_runtime()
        rt._emit = emitted.append
        rt.set_tts_router(MagicMock())
        rt.begin_llm_turn_for_latency()
        rt._llm_pending["pf"] = {"prefetch": True, "committed": False, "tts_engine": "dummy"}
        _segment(rt, "pf", "一文目。")  # 最初の文を TTS に渡す（緑帯が閉じ、マゼンタ帯が始まる）
        return rt, emitted

    def _llm_band_end(self, emitted: list[dict]) -> float:
        llm = [e for e in emitted if e.get("type") == "latency" and e.get("kind") == "llm" and e.get("value") is not None]
        self.assertEqual(len(llm), 1)
        return float(llm[0]["end_ts"])

    def test_band_ends_when_synthesis_finishes_before_commit(self) -> None:
        rt, emitted = self._runtime()
        rt._synthesize_segment_pcm_l16le = lambda *a, **k: _AUDIBLE_PCM
        rt._llm_play_tts_segment("pf", "一文目。", is_final=False)  # 合成完了 → 保持
        bands = _tts_bands(emitted)
        self.assertEqual(len(bands), 1, "確定前でも合成が終わった時点で帯を出す")
        self.assertGreaterEqual(bands[0]["start_ts"], self._llm_band_end(emitted) - 1e-6, "緑帯の終わりから始まる")
        rt.commit_prefetch()  # 確定 → 保持していた音声を再生（帯は増えない）
        self.assertEqual(len(_tts_bands(emitted)), 1)

    def test_band_runs_to_first_sound_when_committed_during_synthesis(self) -> None:
        rt, emitted = self._runtime()
        committed_at: list[float] = []

        def synth_while_user_finishes(*_a, **_k) -> bytes:
            rt.commit_prefetch()  # 合成の途中でターンを取る
            committed_at.append(__import__("time").time())
            return _AUDIBLE_PCM

        rt._synthesize_segment_pcm_l16le = synth_while_user_finishes
        rt._llm_play_tts_segment("pf", "一文目。", is_final=False)
        bands = _tts_bands(emitted)
        self.assertEqual(len(bands), 1, "合成中に確定しても TTS 帯が出る")
        self.assertLessEqual(bands[0]["start_ts"], committed_at[0], "最初の文を渡した時点から")
        self.assertGreaterEqual(bands[0]["end_ts"], committed_at[0], "最初の音が出るまで続く")


if __name__ == "__main__":
    unittest.main()
