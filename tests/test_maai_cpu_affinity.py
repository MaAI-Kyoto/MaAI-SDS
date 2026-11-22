from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maai_sds.maai.cpu_affinity import _performance_core_mask, linux_cpus_are_virtual


def cpu_set(logical: int, efficiency: int, *, group: int = 0, size: int = 32) -> bytes:
    data = bytearray(size)
    struct.pack_into("<II", data, 0, size, 0)
    struct.pack_into("<H", data, 12, group)
    data[14], data[18] = logical, efficiency
    return bytes(data)


class PerformanceCoreMaskTests(unittest.TestCase):
    def test_uses_real_topology_not_first_eight_processors(self) -> None:
        # 実機 Ultra 9 275HX: P コアの番号は連続していない。
        fast = {0, 1, 10, 11, 12, 13, 22, 23}
        data = b"".join(cpu_set(i, int(i in fast)) for i in range(24))
        self.assertEqual(_performance_core_mask(data, (1 << 24) - 1), 12598275)

    def test_respects_existing_process_mask(self) -> None:
        data = cpu_set(0, 0) + cpu_set(1, 1) + cpu_set(2, 1)
        self.assertEqual(_performance_core_mask(data, 0b011), 0b010)
        self.assertEqual(_performance_core_mask(data, 0b001), 0)

    def test_homogeneous_cpu_is_unchanged(self) -> None:
        self.assertEqual(_performance_core_mask(cpu_set(0, 0) + cpu_set(1, 0), 3), 0)

    def test_skips_unknown_records_and_uses_record_size(self) -> None:
        unknown = struct.pack("<II", 8, 99)
        data = unknown + cpu_set(0, 0, size=40) + cpu_set(1, 1)
        self.assertEqual(_performance_core_mask(data, 3), 2)

    def test_does_not_mix_processor_groups(self) -> None:
        self.assertEqual(_performance_core_mask(cpu_set(0, 0) + cpu_set(1, 1, group=1), 3), 0)

    def test_malformed_records_are_ignored_without_changing_affinity(self) -> None:
        good = cpu_set(0, 0) + cpu_set(1, 1)
        for data in (good + b"\0", good[:-1], struct.pack("<II", 0, 0)):
            with self.subTest(data=data):
                self.assertEqual(_performance_core_mask(data, 3), 0)


class VirtualCpuTests(unittest.TestCase):
    def test_detects_hypervisor_flag(self) -> None:
        guest = "processor\t: 0\nflags\t\t: fpu vme hypervisor lahf_lm\n"
        bare = "processor\t: 0\nflags\t\t: fpu vme lahf_lm\n"
        self.assertTrue(linux_cpus_are_virtual(guest))
        self.assertFalse(linux_cpus_are_virtual(bare))
        self.assertFalse(linux_cpus_are_virtual("flags\t: hypervisorx\n"))


class SerialMultiSubsTests(unittest.TestCase):
    def test_inline_executor_runs_in_caller_thread_and_keeps_errors(self) -> None:
        import threading

        from maai_sds.maai.worker import _InlineExecutor

        ex = _InlineExecutor()
        self.assertEqual(ex.submit(threading.get_ident).result(), threading.get_ident())
        with self.assertRaises(ZeroDivisionError):
            ex.submit(lambda: 1 / 0).result()

    def test_swaps_pool_only_on_virtual_linux(self) -> None:
        from unittest import mock

        from maai_sds.maai import worker

        class Pool:
            closed = False

            def shutdown(self, wait=True):
                self.closed = True

        class Multi:
            def __init__(self):
                self._sub_executor = Pool()

        for platform, virtual, swapped in (("linux", True, True), ("linux", False, False), ("win32", True, False)):
            with self.subTest(platform=platform, virtual=virtual):
                m = Multi()
                pool = m._sub_executor
                with mock.patch.object(worker.sys, "platform", platform), mock.patch(
                    "maai_sds.maai.cpu_affinity.linux_cpus_are_virtual", return_value=virtual
                ):
                    self.assertEqual(worker._serialize_multi_subs_in_vm(m), swapped)
                self.assertEqual(isinstance(m._sub_executor, worker._InlineExecutor), swapped)
                self.assertEqual(pool.closed, swapped)


if __name__ == "__main__":
    unittest.main()
