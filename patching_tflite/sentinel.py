from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hook_lib import NativeHookLibrary


@dataclass
class MemoryHealthReport:
    rss_bytes: int
    rss_mb: float
    vsize_bytes: int
    vsize_mb: float
    heap_bytes: int
    canary_corruptions: int
    leak_detected: bool
    leak_rate_kb: float

    def to_dict(self) -> dict:
        return {
            "rss_bytes": self.rss_bytes,
            "rss_mb": round(self.rss_mb, 2),
            "vsize_bytes": self.vsize_bytes,
            "vsize_mb": round(self.vsize_mb, 2),
            "heap_bytes": self.heap_bytes,
            "canary_corruptions": self.canary_corruptions,
            "leak_detected": self.leak_detected,
            "leak_rate_kb": round(self.leak_rate_kb, 2),
        }


class MemorySentinel:
    """Monitors memory leaks and buffer integrity canaries."""

    def __init__(self, hook_lib: Optional[NativeHookLibrary] = None):
        self._hook_lib = hook_lib

    def get_health(self) -> MemoryHealthReport:
        """Inspects current resident set size and canary integrity."""
        if not self._hook_lib:
            return MemoryHealthReport(
                rss_bytes=0,
                rss_mb=0.0,
                vsize_bytes=0,
                vsize_mb=0.0,
                heap_bytes=0,
                canary_corruptions=0,
                leak_detected=False,
                leak_rate_kb=0.0,
            )

        from .hook_lib import CMemoryHealthInfo

        c_info = CMemoryHealthInfo()
        self._hook_lib.cdll.sentinel_get_memory_info(ctypes.byref(c_info))

        rss = int(c_info.rss_bytes)
        vsize = int(c_info.vsize_bytes)
        heap = int(c_info.heap_bytes)
        corruptions = int(c_info.canary_corruptions)
        leak = bool(c_info.leak_flag)
        rate = float(c_info.leak_rate_kb)

        return MemoryHealthReport(
            rss_bytes=rss,
            rss_mb=rss / (1024.0 * 1024.0),
            vsize_bytes=vsize,
            vsize_mb=vsize / (1024.0 * 1024.0),
            heap_bytes=heap,
            canary_corruptions=corruptions,
            leak_detected=leak,
            leak_rate_kb=rate,
        )

    def register_canary(self, buffer_ptr: int | ctypes.c_void_p, size: int) -> int:
        """Places a guard canary magic at buffer_ptr + size - 8."""
        if not self._hook_lib:
            return -1
        ptr_val = buffer_ptr if isinstance(buffer_ptr, int) else buffer_ptr.value
        return self._hook_lib.cdll.sentinel_register_canary(
            ctypes.c_void_p(ptr_val), ctypes.c_size_t(size)
        )

    def verify_canaries(self) -> int:
        """Returns the number of corrupted buffer canaries."""
        if not self._hook_lib:
            return 0
        return self._hook_lib.cdll.sentinel_verify_canaries()

    def reset(self) -> None:
        """Resets the sentinel counters and history."""
        if self._hook_lib:
            self._hook_lib.cdll.sentinel_reset()

