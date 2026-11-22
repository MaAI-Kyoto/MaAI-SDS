"""Windows hybrid CPU の実際のコア構成から MaAI の実行先を選ぶ。"""

from __future__ import annotations

import ctypes
import struct
import sys


def _performance_core_mask(data: bytes, allowed: int) -> int:
    # SYSTEM_CPU_SET_INFORMATION は可変長。EfficiencyClass は大きいほど高速。
    # https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-system_cpu_set_information
    cores: dict[int, int] = {}
    offset = 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            return 0
        if kind == 0 and size >= 32:
            group = struct.unpack_from("<H", data, offset + 12)[0]
            logical, efficiency = data[offset + 14], data[offset + 18]
            if group == 0 and allowed & (1 << logical):
                cores[logical] = efficiency
        offset += size
    if offset != len(data) or len(set(cores.values())) < 2:
        return 0  # 均一コアの CPU や既存の制限には手を加えない。
    fastest = max(cores.values())
    return sum(1 << logical for logical, efficiency in cores.items() if efficiency == fastest)


def pin_windows_performance_cores() -> tuple[int, ...]:
    """現在のプロセスを高速コアへ固定。非対応環境・API 失敗時は変更しない。"""
    if sys.platform != "win32":
        return ()
    try:
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetActiveProcessorGroupCount.restype = wintypes.WORD
        if kernel.GetActiveProcessorGroupCount() != 1:
            return ()
        kernel.GetProcessAffinityMask.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        )
        kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        kernel.GetSystemCpuSetInformation.argtypes = (
            ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
            wintypes.HANDLE, wintypes.ULONG,
        )
        process = kernel.GetCurrentProcess()
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            return ()
        size = wintypes.ULONG()
        kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), process, 0)
        if not size.value:
            return ()
        data = ctypes.create_string_buffer(size.value)
        if not kernel.GetSystemCpuSetInformation(data, size.value, ctypes.byref(size), process, 0):
            return ()
        mask = _performance_core_mask(data.raw[:size.value], allowed.value)
        if mask and kernel.SetProcessAffinityMask(process, mask):
            return tuple(i for i in range(mask.bit_length()) if mask & (1 << i))
    except (AttributeError, OSError):
        pass
    return ()


def linux_cpus_are_virtual(cpuinfo_text: str | None = None) -> bool:
    """Linux が仮想マシン上（WSL2 / Docker Desktop など）で動いているか。

    仮想 CPU の番号はホストの P/E コアと対応しない（275HX の P コアは Windows では
    0,1,10-13,22,23 なのに、WSL2 では 24 個とも同じコアに見える）。そこで「先頭 8 個」に
    固定しても P コアは選べず、他の処理と同じ 8 個を奪い合うだけになる。
    実測（Docker on Windows・CPU 負荷あり）: 固定あり RTF 0.82、固定なし 0.63。
    """
    if cpuinfo_text is None:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                cpuinfo_text = f.read()
        except OSError:
            return False
    for line in cpuinfo_text.splitlines():
        if line.startswith("flags"):
            return "hypervisor" in line.partition(":")[2].split()
    return False
