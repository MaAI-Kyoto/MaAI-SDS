"""Voice-dialog LLM watchdog regression tests."""

from __future__ import annotations

import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from maai_sds.dialog.runtime import VoiceDialogRuntime  # noqa: E402


def _make_runtime(watchdog_sec: float) -> tuple[VoiceDialogRuntime, list[dict]]:
    emitted: list[dict] = []
    supervisor = MagicMock()
    supervisor.send_llm.return_value = True
    rt = VoiceDialogRuntime(
        emit=emitted.append,
        event_queue=queue.Queue(),
        supervisor=supervisor,
        mmd=MagicMock(),
        audio_input_server=MagicMock(),
        asr_audio_server=MagicMock(),
        asr_compare_audio_server=MagicMock(),
        maai_audio_server=MagicMock(),
    )
    rt._llm_settings.update(
        {
            "llm_watchdog_sec": watchdog_sec,
            "llm_backend": "LM Studio",
            "llm_model": "fake-model",
            "tts_engine": "",
        }
    )
    rt._orchestrator = MagicMock()
    rt._orchestrator.history_snapshot.return_value = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
    }
    return rt, emitted


def _dispatch(rt: VoiceDialogRuntime, *, prefetch: bool = False) -> str:
    rt.dispatch_llm_request(prefetch=prefetch)
    starts = [
        call.args[0]
        for call in rt._supervisor.send_llm.call_args_list
        if call.args and call.args[0].get("type") == "start"
    ]
    if not starts:
        raise AssertionError("LLM start payload was not sent")
    return str(starts[-1]["request_id"])


def _watch_cancels(rt: VoiceDialogRuntime) -> threading.Event:
    cancelled = threading.Event()

    def _send(payload: dict) -> bool:
        if payload.get("type") == "cancel":
            cancelled.set()
        return True

    rt._supervisor.send_llm.side_effect = _send
    return cancelled


def _cancel_payloads(rt: VoiceDialogRuntime) -> list[dict]:
    return [
        call.args[0]
        for call in rt._supervisor.send_llm.call_args_list
        if call.args and call.args[0].get("type") == "cancel"
    ]


def _cleanup_runtime(rt: VoiceDialogRuntime) -> None:
    with rt._llm_pending_lock:
        for request_id in list(rt._llm_pending):
            rt._pop_llm_pending_locked(request_id)


class LlmWatchdogTests(unittest.TestCase):
    def test_normal_request_timeout_cancels_and_finishes_turn(self) -> None:
        rt, emitted = _make_runtime(0.05)
        self.addCleanup(_cleanup_runtime, rt)
        cancelled = _watch_cancels(rt)
        finished = threading.Event()
        rt._orchestrator.on_llm_finished.side_effect = lambda **_kwargs: finished.set()

        request_id = _dispatch(rt)

        self.assertTrue(cancelled.wait(1.0), "watchdog did not cancel the request")
        self.assertTrue(finished.wait(1.0), "watchdog did not finish the turn")
        rt._orchestrator.on_llm_finished.assert_called_once_with(
            ok=False,
            assistant_text="",
            error="llm_timeout",
        )
        self.assertEqual(_cancel_payloads(rt), [{"type": "cancel", "request_id": request_id}])
        self.assertTrue(
            any(
                event.get("type") == "log"
                and "[LLM][WATCHDOG]" in str(event.get("message", ""))
                and request_id in str(event.get("message", ""))
                for event in emitted
            )
        )

    def test_chunk_and_tts_segment_activity_postpone_timeout(self) -> None:
        rt, _emitted = _make_runtime(0.15)
        self.addCleanup(_cleanup_runtime, rt)
        cancelled = _watch_cancels(rt)
        request_id = _dispatch(rt)

        time.sleep(0.06)
        rt.handle_llm_worker_event(
            {"type": "chunk", "request_id": request_id, "text": "partial"}
        )
        time.sleep(0.10)
        rt.handle_llm_worker_event(
            {"type": "tts_segment", "request_id": request_id, "segment_text": ""}
        )

        self.assertFalse(
            cancelled.wait(0.08),
            "watchdog ignored recent chunk/tts_segment activity",
        )
        self.assertTrue(cancelled.wait(0.5), "watchdog did not fire after activity stopped")

    def test_done_cancels_watchdog(self) -> None:
        rt, _emitted = _make_runtime(0.05)
        self.addCleanup(_cleanup_runtime, rt)
        cancelled = _watch_cancels(rt)
        request_id = _dispatch(rt)

        rt.handle_llm_worker_event(
            {
                "type": "done",
                "request_id": request_id,
                "assistant_text": "answer",
                "ttft_sec": 0.01,
                "total_sec": 0.02,
            }
        )

        self.assertFalse(cancelled.wait(0.15))
        rt._orchestrator.on_llm_finished.assert_called_once_with(
            ok=True,
            assistant_text="answer",
            error="",
        )

    def test_uncommitted_prefetch_timeout_marks_prefetch_failed(self) -> None:
        rt, _emitted = _make_runtime(0.05)
        self.addCleanup(_cleanup_runtime, rt)
        cancelled = _watch_cancels(rt)
        failed = threading.Event()
        rt._orchestrator.mark_prefetch_failed.side_effect = failed.set

        _dispatch(rt, prefetch=True)

        self.assertTrue(cancelled.wait(1.0))
        self.assertTrue(failed.wait(1.0))
        rt._orchestrator.mark_prefetch_failed.assert_called_once_with()
        rt._orchestrator.on_llm_finished.assert_not_called()

    def test_watchdog_disabled_for_non_positive_values(self) -> None:
        for watchdog_sec in (0.0, -1.0):
            with self.subTest(watchdog_sec=watchdog_sec):
                rt, _emitted = _make_runtime(watchdog_sec)
                self.addCleanup(_cleanup_runtime, rt)
                cancelled = _watch_cancels(rt)
                request_id = _dispatch(rt)

                self.assertFalse(cancelled.wait(0.10))
                with rt._llm_pending_lock:
                    state = rt._llm_pending.get(request_id)
                    self.assertIsInstance(state, dict)
                    self.assertNotIn("_llm_watchdog_timer", state)
                rt.handle_llm_worker_event(
                    {"type": "done", "request_id": request_id, "assistant_text": "ok"}
                )

    def test_done_racing_watchdog_is_a_noop(self) -> None:
        rt, _emitted = _make_runtime(0.05)
        self.addCleanup(_cleanup_runtime, rt)
        cancel_entered = threading.Event()
        let_cancel_return = threading.Event()
        failed = threading.Event()

        def _send(payload: dict) -> bool:
            if payload.get("type") == "cancel":
                cancel_entered.set()
                let_cancel_return.wait(1.0)
            return True

        rt._supervisor.send_llm.side_effect = _send
        rt._orchestrator.on_llm_finished.side_effect = lambda **_kwargs: failed.set()
        request_id = _dispatch(rt)

        try:
            self.assertTrue(cancel_entered.wait(1.0))
            rt.handle_llm_worker_event(
                {"type": "done", "request_id": request_id, "assistant_text": "late answer"}
            )
            rt._orchestrator.on_llm_finished.assert_not_called()
        finally:
            let_cancel_return.set()

        self.assertTrue(failed.wait(1.0))
        rt._orchestrator.on_llm_finished.assert_called_once_with(
            ok=False,
            assistant_text="",
            error="llm_timeout",
        )


if __name__ == "__main__":
    unittest.main()
