from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hook_lib import NativeHookLibrary


@dataclass
class OpProfile:
    op_name: str
    execution_count: int
    total_time_ms: float
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    percent_total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "op_name": self.op_name,
            "execution_count": self.execution_count,
            "total_time_ms": round(self.total_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "percent_total": round(self.percent_total, 2),
        }


class OpProfiler:
    """Manages op-level microsecond profiling breakdown."""

    def __init__(self, hook_lib: Optional[NativeHookLibrary] = None):
        self._hook_lib = hook_lib

    def record_op(self, op_name: str, duration_ms: float) -> None:
        """Records an op execution time in the native engine."""
        if self._hook_lib:
            self._hook_lib.cdll.record_op_execution(
                op_name.encode("utf-8"), ctypes.c_double(duration_ms)
            )

    def get_breakdown(self) -> list[OpProfile]:
        """Returns the list of all profiled ops sorted by total time descending."""
        if not self._hook_lib:
            return []

        from .hook_lib import COpProfileEntry

        count = self._hook_lib.cdll.get_op_profile_count()
        if count <= 0:
            return []

        entries = (COpProfileEntry * count)()
        actual = self._hook_lib.cdll.get_op_profiles(entries, count)

        raw_profiles = []
        grand_total = 0.0
        for i in range(actual):
            e = entries[i]
            op_name = e.op_name.decode("utf-8", errors="replace")
            p = OpProfile(
                op_name=op_name,
                execution_count=int(e.execution_count),
                total_time_ms=float(e.total_time_ms),
                min_time_ms=float(e.min_time_ms),
                max_time_ms=float(e.max_time_ms),
                avg_time_ms=float(e.avg_time_ms),
            )
            grand_total += p.total_time_ms
            raw_profiles.append(p)

        # Compute percentages
        for p in raw_profiles:
            p.percent_total = (p.total_time_ms / grand_total * 100.0) if grand_total > 0 else 0.0

        raw_profiles.sort(key=lambda x: x.total_time_ms, reverse=True)
        return raw_profiles

    def find_bottlenecks(self, top_n: int = 3, threshold_pct: float = 10.0) -> list[OpProfile]:
        """Identifies operators that exceed threshold_pct of total inference latency."""
        breakdown = self.get_breakdown()
        bottlenecks = [p for p in breakdown if p.percent_total >= threshold_pct]
        return bottlenecks[:top_n]

    def reset(self) -> None:
        """Resets all op profile statistics."""
        if self._hook_lib:
            self._hook_lib.cdll.reset_op_profiles()

