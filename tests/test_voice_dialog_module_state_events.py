"""音声対話開始時の worker ensure と module-state UI 同期を検証する。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from maai_sds.dialog.runtime import VoiceDialogRuntime  # noqa: E402


class _FakeProc:
    @staticmethod
    def is_running() -> bool:
        return True


class _FakeSupervisor:
    def __init__(self, fail_modules: set[str] | None = None) -> None:
        self.fail_modules = fail_modules or set()
        self.calls: list[str] = []
        self.maai_fatal = ""
        self.audio_input = _FakeProc()
        self.audio_output = _FakeProc()

    def _maybe_fail(self, module: str) -> None:
        self.calls.append(module)
        if module in self.fail_modules:
            raise RuntimeError(f"{module} boom")

    def ensure_asr(self, **_kw) -> None:
        self._maybe_fail("asr")

    def ensure_tts(self, **_kw) -> None:
        self._maybe_fail("tts")

    def ensure_llm(self) -> None:
        self._maybe_fail("llm")

    def ensure_maai(self, **_kw) -> None:
        self._maybe_fail("maai")

    def mark_maai_fatal(self, message: str) -> None:
        self.maai_fatal = message


def _make_runtime(supervisor: _FakeSupervisor, events: list[dict]) -> VoiceDialogRuntime:
    runtime = VoiceDialogRuntime.__new__(VoiceDialogRuntime)
    runtime._emit = events.append
    runtime._supervisor = supervisor
    runtime._asr_audio_server = SimpleNamespace(host="127.0.0.1", port=1)
    runtime._maai_audio_server = SimpleNamespace(host="127.0.0.1", port=2)
    runtime._llm_settings = {}
    return runtime


def _settings() -> dict:
    return {
        "tts_engine": "VOICEVOX-Core",
        "worker_configs": {
            "asr": {"asr_engine": "sherpa-onnx"},
            "tts": {"voice": "8"},
            "llm": {"connection": {}},
            "maai": {"mode": "both"},
            "audio_input": {"input_mode": "mic"},
            "audio_output": {"output_mode": "local"},
        },
    }


def _module_states(events: list[dict]) -> list[tuple[str, str]]:
    return [
        (str(event.get("module")), str(event.get("state")))
        for event in events
        if event.get("type") == "module_state"
    ]


class TestVoiceDialogModuleStateEvents(unittest.TestCase):
    def test_all_modules_emit_starting_then_running(self) -> None:
        events: list[dict] = []
        supervisor = _FakeSupervisor()
        runtime = _make_runtime(supervisor, events)

        runtime.ensure_voice_dialog_workers(_settings())

        states = _module_states(events)
        for module in ("asr", "tts", "llm", "maai"):
            self.assertIn((module, "starting"), states)
            self.assertIn((module, "running"), states)
            self.assertLess(
                states.index((module, "starting")), states.index((module, "running"))
            )
        self.assertEqual(supervisor.calls, ["asr", "tts", "llm", "maai"])

    def test_failed_module_emits_error_and_source_tag(self) -> None:
        events: list[dict] = []
        supervisor = _FakeSupervisor(fail_modules={"tts"})
        runtime = _make_runtime(supervisor, events)

        runtime.ensure_voice_dialog_workers(_settings())

        states = _module_states(events)
        self.assertIn(("tts", "starting"), states)
        self.assertIn(("tts", "error"), states)
        self.assertNotIn(("tts", "running"), states)
        # TTS 失敗でも後続（LLM / MaAI）の起動通知は継続する。
        self.assertIn(("llm", "running"), states)
        self.assertIn(("maai", "running"), states)
        error_event = next(
            event
            for event in events
            if event.get("type") == "module_state" and event.get("state") == "error"
        )
        self.assertEqual(error_event.get("source"), "voice_dialog")
        self.assertIn("boom", str(error_event.get("error")))

    def test_late_maai_fatal_replaces_running_state_with_error_reason(self) -> None:
        events: list[dict] = []
        supervisor = _FakeSupervisor()
        runtime = _make_runtime(supervisor, events)

        runtime.on_maai_ingress_event({"type": "fatal", "message": "model load exploded"})

        error_event = next(
            event
            for event in events
            if event.get("type") == "module_state" and event.get("module") == "maai"
        )
        self.assertEqual(error_event.get("state"), "error")
        self.assertEqual(error_event.get("source"), "worker_fatal")
        self.assertEqual(error_event.get("error"), "model load exploded")
        self.assertEqual(supervisor.maai_fatal, "model load exploded")


    def test_maai_ready_forwards_nod_threshold_defaults_to_the_ui(self) -> None:
        events: list[dict] = []
        supervisor = _FakeSupervisor()
        supervisor.mark_maai_ready = lambda: None
        runtime = _make_runtime(supervisor, events)
        defaults = {"count": {"t0": 0.4, "t1": 0.5, "t2": 0.6}, "swing": 0.3}

        runtime.on_maai_ingress_event({"type": "ready", "nod_threshold_defaults": defaults})
        runtime.on_maai_ingress_event({"type": "ready"})

        sent = [e for e in events if e.get("type") == "maai_nod_threshold_defaults"]
        self.assertEqual([e["defaults"] for e in sent], [defaults, None])


class InterruptPlotAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_interrupt_and_prediction_share_origin_on_arrival_and_scroll(self) -> None:
        from maai_sds.maai.plot_window import MaaiPlotWindow, pg

        for delay in (0.0, 0.4):
            with self.subTest(audio_clock_delay=delay):
                window = MaaiPlotWindow()
                try:
                    with patch("maai_sds.maai.plot_window.time.time", return_value=1000.6):
                        window.init_full_layout()
                        window._note_audio_now(1000.6 - delay)
                        window.ingest_result({"p_now": 0.1, "t": 999.8})
                        window.update_voice_dialog_latency(
                            kind="interrupt", latency_sec=None, end_ts=999.8
                        )

                        def interrupt_x() -> float:
                            lines = [
                                item
                                for item in window._voice_latency_plot.items
                                if isinstance(item, pg.InfiniteLine)
                            ]
                            self.assertEqual(len(lines), 1)
                            return lines[0].value()

                        self.assertAlmostEqual(
                            interrupt_x(), 999.8 - window._audio_now_ts()
                        )
                        window.flush_maai_draw(now_ts=1000.6)
                        xs, _ = window.curves["p_now"]["high"].getData()
                        self.assertAlmostEqual(interrupt_x(), xs[-1])
                    with patch("maai_sds.maai.plot_window.time.time", return_value=1001.1):
                        window.tick_wall_clock_plot_scroll()
                        xs, _ = window.curves["p_now"]["high"].getData()
                        self.assertAlmostEqual(interrupt_x(), xs[-1])
                        self.assertAlmostEqual(
                            interrupt_x(), 999.8 - window._audio_now_ts()
                        )
                    self.assertEqual(window._voice_interrupt_action_ts, [999.8])
                finally:
                    window.close()
                    window.deleteLater()


class ModuleBatchStartTests(unittest.TestCase):
    @staticmethod
    def make_panel(states: dict[str, str]) -> SimpleNamespace:
        return SimpleNamespace(
            _MODULE_PANEL_KEYS=("asr", "llm", "tts", "maai"),
            _module_panel_states={key: {"state": value} for key, value in states.items()},
            _module_panel_start=Mock(),
            _set_module_panel_state=Mock(),
            _refresh_module_status_panel=Mock(),
            _log_stream=Mock(),
        )

    def test_skips_running_and_starting_modules(self) -> None:
        from maai_sds.gui.module_status_mixin import ModuleStatusPanelMixin

        panel = self.make_panel(
            {"asr": "running", "llm": "starting", "tts": "stopped", "maai": "error"}
        )
        ModuleStatusPanelMixin._module_panel_start_all(panel)
        self.assertEqual(
            [call.args[0] for call in panel._module_panel_start.call_args_list],
            ["tts", "maai"],
        )
        self.assertFalse(panel._module_start_all_busy)

    def test_reentrant_click_and_one_failure_do_not_stop_other_modules(self) -> None:
        from maai_sds.gui.module_status_mixin import ModuleStatusPanelMixin

        panel = self.make_panel({})

        def launch(module: str) -> None:
            ModuleStatusPanelMixin._module_panel_start_all(panel)
            if module == "asr":
                raise RuntimeError("ASR load failed")

        panel._module_panel_start.side_effect = launch
        ModuleStatusPanelMixin._module_panel_start_all(panel)
        self.assertEqual(
            [call.args[0] for call in panel._module_panel_start.call_args_list],
            ["asr", "llm", "tts", "maai"],
        )
        panel._set_module_panel_state.assert_called_once_with(
            "asr", "error", "ASR load failed"
        )
        self.assertFalse(panel._module_start_all_busy)


if __name__ == "__main__":
    unittest.main()
