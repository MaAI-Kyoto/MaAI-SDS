"""ワーカーが ready を返す前に落ちたら、時間切れまで待たずにすぐ失敗にする（orchestrator_supervisor）。"""

from __future__ import annotations

import threading
import time
import unittest

from maai_sds.ipc.orchestrator_supervisor import OrchestratorSupervisor


class _DeadProc:
    pid = 4242
    returncode = 1

    def poll(self) -> int:
        return 1


class _FakeService:
    def __init__(self) -> None:
        self.started = 0

    def start(self, **_kwargs) -> _DeadProc:
        self.started += 1
        return _DeadProc()

    def is_running(self) -> bool:
        return False

    def stop(self) -> None:
        pass


class WorkerExitBeforeReadyTest(unittest.TestCase):
    def test_maai_crash_fails_at_once_instead_of_after_the_timeout(self) -> None:
        sup = OrchestratorSupervisor()
        sup.maai = _FakeService()
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            sup.ensure_maai({}, "127.0.0.1", 0, timeout_sec=60.0)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIn("終了コード 1", str(ctx.exception))
        self.assertIn("[MaAI]", str(ctx.exception))

    def test_fatal_sent_just_before_exit_is_preferred(self) -> None:
        sup = OrchestratorSupervisor()
        sup.asr = _FakeService()
        # 落ちる直前に送られた理由つきの fatal が、少し遅れて届く
        threading.Timer(0.1, lambda: sup.mark_asr_fatal("モデルが見つかりません")).start()
        with self.assertRaises(RuntimeError) as ctx:
            sup.ensure_asr({}, "127.0.0.1", 0, timeout_sec=60.0)
        self.assertEqual(str(ctx.exception), "モデルが見つかりません")


if __name__ == "__main__":
    unittest.main()
