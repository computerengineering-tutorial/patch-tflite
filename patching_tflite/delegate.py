from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hook_lib import NativeHookLibrary


@dataclass
class DelegateReport:
    delegate_attached: bool
    delegate_name: str
    delegated_nodes: int
    cpu_fallback_nodes: int
    total_nodes: int
    delegation_ratio: float
    is_gpu_accelerated: bool
    is_npu_accelerated: bool

    def to_dict(self) -> dict:
        return {
            "delegate_attached": self.delegate_attached,
            "delegate_name": self.delegate_name,
            "delegated_nodes": self.delegated_nodes,
            "cpu_fallback_nodes": self.cpu_fallback_nodes,
            "total_nodes": self.total_nodes,
            "delegation_ratio": round(self.delegation_ratio, 4),
            "delegation_percent": round(self.delegation_ratio * 100.0, 2),
            "is_gpu_accelerated": self.is_gpu_accelerated,
            "is_npu_accelerated": self.is_npu_accelerated,
        }


class DelegateTracker:
    """Tracks TFLite delegate attachments and hardware acceleration offload ratios."""

    def __init__(self, hook_lib: Optional[NativeHookLibrary] = None):
        self._hook_lib = hook_lib

    def record_delegate(
        self, delegate_name: str, delegated_nodes: int, cpu_fallback_nodes: int
    ) -> None:
        """Manually records delegate attachment or telemetry."""
        if self._hook_lib:
            self._hook_lib.cdll.record_delegate_attachment(
                delegate_name.encode("utf-8"),
                ctypes.c_uint32(delegated_nodes),
                ctypes.c_uint32(cpu_fallback_nodes),
            )

    def hook_delegate_symbol(
        self, lib_name: Optional[str] = None, symbol_name: Optional[str] = None
    ) -> bool:
        """Installs interception hook on TfLiteInterpreterModifyGraphWithDelegate."""
        if not self._hook_lib:
            return False
        res = self._hook_lib.cdll.install_delegate_hook_symbol(
            lib_name.encode("utf-8") if lib_name else None,
            symbol_name.encode("utf-8") if symbol_name else None,
        )
        return res == 0

    def get_report(self) -> DelegateReport:
        """Retrieves the current delegate status and offload metrics."""
        if not self._hook_lib:
            return DelegateReport(
                delegate_attached=False,
                delegate_name="None",
                delegated_nodes=0,
                cpu_fallback_nodes=0,
                total_nodes=0,
                delegation_ratio=0.0,
                is_gpu_accelerated=False,
                is_npu_accelerated=False,
            )

        from .hook_lib import CDelegateTrackingInfo

        c_info = CDelegateTrackingInfo()
        self._hook_lib.cdll.get_delegate_info(ctypes.byref(c_info))

        name = c_info.delegate_name.decode("utf-8", errors="replace")
        attached = bool(c_info.delegate_attached)
        d_nodes = int(c_info.delegated_nodes)
        fb_nodes = int(c_info.cpu_fallback_nodes)
        total = d_nodes + fb_nodes
        ratio = float(c_info.delegation_ratio)

        name_lower = name.lower()
        is_gpu = any(x in name_lower for x in ["gpu", "opencl", "vulkan", "gl"])
        is_npu = any(x in name_lower for x in ["npu", "nnapi", "dsp", "tpu", "hexagon"])

        return DelegateReport(
            delegate_attached=attached,
            delegate_name=name,
            delegated_nodes=d_nodes,
            cpu_fallback_nodes=fb_nodes,
            total_nodes=total,
            delegation_ratio=ratio,
            is_gpu_accelerated=is_gpu,
            is_npu_accelerated=is_npu,
        )

    def get_warnings(self) -> list[str]:
        """Returns any diagnostic warnings regarding delegate performance."""
        report = self.get_report()
        warnings = []
        if not report.delegate_attached:
            warnings.append("No hardware acceleration delegate attached (pure CPU execution).")
        elif report.delegation_ratio < 0.5 and report.total_nodes > 0:
            warnings.append(
                f"Low hardware delegation ratio ({report.delegation_ratio * 100:.1f}%): "
                f"{report.cpu_fallback_nodes} nodes fell back to CPU."
            )
        return warnings

    def reset(self) -> None:
        """Resets recorded delegate metrics."""
        if self._hook_lib:
            self._hook_lib.cdll.reset_delegate_info()

