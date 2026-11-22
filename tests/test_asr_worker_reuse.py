import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from maai_sds.ipc.orchestrator_supervisor import OrchestratorSupervisor


class AsrWorkerReuseTests(unittest.TestCase):
    def make_supervisor(self):
        supervisor = OrchestratorSupervisor.__new__(OrchestratorSupervisor)
        supervisor._asr_ensure_lock = threading.RLock()
        supervisor._asr_ready = threading.Event()
        supervisor._asr_config_json = ""
        supervisor._asr_route = None
        supervisor._asr_fatal = ""
        supervisor._python_exe = "python"
        supervisor._log = Mock()
        running = [False]

        def start(**kwargs):
            running[0] = True
            supervisor.mark_asr_ready()
            return SimpleNamespace(pid=123)

        def stop():
            running[0] = False

        supervisor.asr = SimpleNamespace(
            is_running=lambda: running[0],
            start=Mock(side_effect=start),
            stop=Mock(side_effect=stop),
        )
        return supervisor

    def test_same_config_and_route_keep_process(self):
        supervisor = self.make_supervisor()
        supervisor.ensure_asr(
            {"engine": "sensevoice", "vad": "silero"},
            "127.0.0.1",
            100,
            timeout_sec=1,
        )
        supervisor.ensure_asr(
            {"vad": "silero", "engine": "sensevoice"},
            "127.0.0.1",
            100,
            timeout_sec=1,
        )
        self.assertEqual(supervisor.asr.start.call_count, 1)
        self.assertEqual(supervisor.asr.stop.call_count, 1)  # Initial cleanup only.

    def test_changed_config_or_route_restarts(self):
        supervisor = self.make_supervisor()
        supervisor.ensure_asr({"engine": "one"}, "127.0.0.1", 100, timeout_sec=1)
        supervisor.ensure_asr({"engine": "two"}, "127.0.0.1", 100, timeout_sec=1)
        supervisor.ensure_asr({"engine": "two"}, "127.0.0.1", 101, timeout_sec=1)
        self.assertEqual(supervisor.asr.start.call_count, 3)
        supervisor._teardown_asr()
        self.assertEqual(supervisor._asr_config_json, "")
        self.assertIsNone(supervisor._asr_route)

    def test_not_ready_restarts(self):
        supervisor = self.make_supervisor()
        supervisor.ensure_asr({}, "127.0.0.1", 100, timeout_sec=1)
        supervisor._asr_ready.clear()
        supervisor.ensure_asr({}, "127.0.0.1", 100, timeout_sec=1)
        self.assertEqual(supervisor.asr.start.call_count, 2)


if __name__ == "__main__":
    unittest.main()
