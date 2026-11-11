from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional


# C Callback type:
# (call_id, stage, model_name, model_id, subgraph_idx, tensor_idx, tensor_name, dtype, shape, data_ptr, byte_size, elapsed_ms)
TensorCallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_double,
)


class CLatencyProfile(ctypes.Structure):
    _fields_ = [
        ("total_calls", ctypes.c_uint64),
        ("avg_ms", ctypes.c_double),
        ("min_ms", ctypes.c_double),
        ("max_ms", ctypes.c_double),
        ("p50_ms", ctypes.c_double),
        ("p90_ms", ctypes.c_double),
        ("p99_ms", ctypes.c_double),
        ("nan_inf_count", ctypes.c_uint64),
    ]


class COpProfileEntry(ctypes.Structure):
    _fields_ = [
        ("op_name", ctypes.c_char * 64),
        ("execution_count", ctypes.c_uint32),
        ("total_time_ms", ctypes.c_double),
        ("min_time_ms", ctypes.c_double),
        ("max_time_ms", ctypes.c_double),
        ("avg_time_ms", ctypes.c_double),
    ]


class CDelegateTrackingInfo(ctypes.Structure):
    _fields_ = [
        ("delegate_attached", ctypes.c_int),
        ("delegate_name", ctypes.c_char * 64),
        ("delegated_nodes", ctypes.c_uint32),
        ("cpu_fallback_nodes", ctypes.c_uint32),
        ("delegation_ratio", ctypes.c_double),
    ]


class CMemoryHealthInfo(ctypes.Structure):
    _fields_ = [
        ("rss_bytes", ctypes.c_uint64),
        ("vsize_bytes", ctypes.c_uint64),
        ("heap_bytes", ctypes.c_uint64),
        ("canary_corruptions", ctypes.c_int),
        ("leak_flag", ctypes.c_int),
        ("leak_rate_kb", ctypes.c_double),
    ]


def find_hook_library(custom_path: str | Path | None = None) -> Path:
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Custom hook library not found: {p}")

    candidates = [
        os.environ.get("TFLITE_PATCH_LIB"),
        Path(__file__).resolve().parent.parent / "libtflite_patch.so",
        Path(__file__).resolve().parent.parent / "build" / "libtflite_patch.so",
        Path("/usr/local/lib/libtflite_patch.so"),
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.exists():
                return p

    raise FileNotFoundError(
        "Could not find libtflite_patch.so. Please compile it with 'make' or specify the path."
    )


class NativeHookLibrary:
    def __init__(self, lib_path: str | Path | None = None):
        self.path = find_hook_library(lib_path)
        self.cdll = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_GLOBAL)
        self._bind_functions()

    def _bind_functions(self):
        c = self.cdll

        # Config API
        c.set_trace_output_path.argtypes = [ctypes.c_char_p]
        c.set_trace_output_path.restype = None

        c.set_tensor_trace_limit.argtypes = [ctypes.c_int]
        c.set_tensor_trace_limit.restype = None

        c.set_tensor_max_bytes.argtypes = [ctypes.c_size_t]
        c.set_tensor_max_bytes.restype = None

        c.set_tensor_preview_values.argtypes = [ctypes.c_int]
        c.set_tensor_preview_values.restype = None

        c.set_tensor_name_logging.argtypes = [ctypes.c_int]
        c.set_tensor_name_logging.restype = None

        c.set_trace_all_tensors.argtypes = [ctypes.c_int]
        c.set_trace_all_tensors.restype = None

        c.set_target_subgraph.argtypes = [ctypes.c_int]
        c.set_target_subgraph.restype = None

        c.set_custom_tensor_indices.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
        c.set_custom_tensor_indices.restype = None

        c.clear_custom_tensor_indices.argtypes = []
        c.clear_custom_tensor_indices.restype = None

        c.set_interpreter_layout.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        c.set_interpreter_layout.restype = None

        c.get_invoke_call_count.argtypes = []
        c.get_invoke_call_count.restype = ctypes.c_int

        # Callbacks API
        c.set_tensor_callback.argtypes = [TensorCallbackType]
        c.set_tensor_callback.restype = None

        c.clear_tensor_callback.argtypes = []
        c.clear_tensor_callback.restype = None

        # Live tensor override API
        c.set_tensor_override.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        c.set_tensor_override.restype = ctypes.c_int

        c.clear_tensor_overrides.argtypes = []
        c.clear_tensor_overrides.restype = None

        # Fault injection API
        c.set_fault_injection.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int]
        c.set_fault_injection.restype = None

        c.clear_fault_injection.argtypes = []
        c.clear_fault_injection.restype = None

        # Profiling API
        c.get_latency_profile.argtypes = [ctypes.c_char_p, ctypes.POINTER(CLatencyProfile)]
        c.get_latency_profile.restype = None

        c.reset_latency_profile.argtypes = []
        c.reset_latency_profile.restype = None

        # Op-level micro-profiling API
        c.record_op_execution.argtypes = [ctypes.c_char_p, ctypes.c_double]
        c.record_op_execution.restype = None

        c.get_op_profiles.argtypes = [ctypes.POINTER(COpProfileEntry), ctypes.c_int]
        c.get_op_profiles.restype = ctypes.c_int

        c.get_op_profile_count.argtypes = []
        c.get_op_profile_count.restype = ctypes.c_int

        c.reset_op_profiles.argtypes = []
        c.reset_op_profiles.restype = None

        # Delegate tracking API
        c.record_delegate_attachment.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        c.record_delegate_attachment.restype = None

        c.get_delegate_info.argtypes = [ctypes.POINTER(CDelegateTrackingInfo)]
        c.get_delegate_info.restype = None

        c.reset_delegate_info.argtypes = []
        c.reset_delegate_info.restype = None

        c.install_delegate_hook_symbol.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        c.install_delegate_hook_symbol.restype = ctypes.c_int

        # Memory Safety Sentinel API
        c.sentinel_get_memory_info.argtypes = [ctypes.POINTER(CMemoryHealthInfo)]
        c.sentinel_get_memory_info.restype = None

        c.sentinel_register_canary.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        c.sentinel_register_canary.restype = ctypes.c_int

        c.sentinel_verify_canaries.argtypes = []
        c.sentinel_verify_canaries.restype = ctypes.c_int

        c.sentinel_reset.argtypes = []
        c.sentinel_reset.restype = None

        # Model metadata API
        c.register_model_metadata.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
        ]
        c.register_model_metadata.restype = ctypes.c_int

        c.register_model_metadata_with_names.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        c.register_model_metadata_with_names.restype = ctypes.c_int

        c.clear_model_metadata.argtypes = []
        c.clear_model_metadata.restype = None

        # Dump API
        c.enable_tensor_dump.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        c.enable_tensor_dump.restype = ctypes.c_int

        c.disable_tensor_dump.argtypes = []
        c.disable_tensor_dump.restype = None

        # Hook installation API
        c.install_invoke_hook_symbol.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        c.install_invoke_hook_symbol.restype = ctypes.c_int

        c.install_invoke_hook_address.argtypes = [ctypes.c_void_p]
        c.install_invoke_hook_address.restype = ctypes.c_int

        c.install_invoke_hook_offset.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        c.install_invoke_hook_offset.restype = ctypes.c_int

        c.install_invoke_hook_relative.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ssize_t,
            ctypes.c_size_t,
        ]
        c.install_invoke_hook_relative.restype = ctypes.c_int

        c.remove_all_hooks.argtypes = []
        c.remove_all_hooks.restype = ctypes.c_int

        c.is_hook_installed.argtypes = []
        c.is_hook_installed.restype = ctypes.c_int
