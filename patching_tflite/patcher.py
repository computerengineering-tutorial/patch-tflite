from __future__ import annotations

import ctypes
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from patching_tflite.config import PatcherConfig
from patching_tflite.delegate import DelegateReport, DelegateTracker
from patching_tflite.extractor import extract_models_from_library, extract_models_from_process_memory
from patching_tflite.hook_lib import CLatencyProfile, NativeHookLibrary, TensorCallbackType
from patching_tflite.metadata import ModelMetadata, parse_tflite_metadata
from patching_tflite.perfetto import export_perfetto_trace
from patching_tflite.profiler import OpProfile, OpProfiler
from patching_tflite.sentinel import MemoryHealthReport, MemorySentinel
from patching_tflite.session import SessionRecorder
from patching_tflite.shadow import ShadowComparisonResult, ShadowEngine

LOG = logging.getLogger("tflite_patch")


@dataclass
class ProfilingStats:
    total_calls: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    anomalies_detected: int


DTYPE_INT_TO_NAME = {
    1: "f32",
    2: "i32",
    3: "u8",
    4: "i64",
    5: "str",
    6: "bool",
    7: "i16",
    8: "complex64",
    9: "i8",
    10: "f16",
    11: "f64",
    16: "u32",
    17: "u16",
    18: "u64",
    19: "bf16",
}


class TFLitePatcher:
    """
    Generic, standalone framework to intercept, trace, profile, mutate, and dump
    TFLite model inferences from any shared library (.so) that uses TensorFlow Lite.
    """

    def __init__(
        self,
        config: Optional[PatcherConfig] = None,
        *,
        hook_lib_path: Optional[str | Path] = None,
    ):
        self.config = config or PatcherConfig()
        self.hook_lib = NativeHookLibrary(hook_lib_path)
        self._trace_file = Path(
            self.config.trace_path
            or os.environ.get("TFLITE_PATCH_TRACE_FILE")
            or f"/tmp/tflite_patch_trace_{os.getpid()}.log"
        )
        self._last_read_offset = 0
        self._c_callback_ref: Optional[Any] = None
        self._user_callback: Optional[Callable] = None
        self._recorder: Optional[SessionRecorder] = None
        self.profiler = OpProfiler(self.hook_lib)
        self.delegate_tracker = DelegateTracker(self.hook_lib)
        self.sentinel = MemorySentinel(self.hook_lib)
        self.shadow_engine: Optional[ShadowEngine] = None
        self._last_tensors: list[dict] = []
        self._apply_config()

    @property
    def trace_file(self) -> Path:
        return self._trace_file

    def _apply_config(self) -> None:
        c = self.hook_lib.cdll

        # Trace file
        self._trace_file.parent.mkdir(parents=True, exist_ok=True)
        c.set_trace_output_path(str(self._trace_file).encode("utf-8"))

        # Tensor limit & size
        c.set_tensor_trace_limit(int(self.config.tensor_limit))
        c.set_tensor_max_bytes(int(self.config.max_tensor_bytes))
        c.set_tensor_preview_values(int(self.config.preview_values))
        c.set_tensor_name_logging(1 if self.config.tensor_names else 0)

        # Full intermediate op tracing & custom indices
        c.set_trace_all_tensors(1 if self.config.trace_all_tensors else 0)
        c.set_target_subgraph(int(self.config.target_subgraph))
        if self.config.custom_tensor_indices:
            self.set_custom_tensor_indices(self.config.custom_tensor_indices)

        # Memory layout
        c.set_interpreter_layout(
            self.config.subgraph_offset,
            self.config.tensor_array_offset,
            self.config.tensor_count_offset,
            self.config.tensor_size,
        )

        # Dump dir if configured
        if self.config.dump_dir:
            self.enable_dump(self.config.dump_dir, self.config.dump_models)

    # -------------------------------------------------------------------------
    # In-Memory Callbacks (Zero-Copy)
    # -------------------------------------------------------------------------
    def set_tensor_callback(
        self,
        callback_fn: Callable[
            [int, str, str, int, int, str, str, tuple, Any, float], None
        ],
    ) -> None:
        """
        Register a Python callback to receive tensors directly in memory without disk I/O.

        Signature: callback_fn(call_id, stage, model_name, subgraph_idx, tensor_idx,
                               tensor_name, dtype, shape, data, elapsed_ms)
        """
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False

        dtype_map = {
            1: np.float32 if has_numpy else None,
            2: np.int32 if has_numpy else None,
            3: np.uint8 if has_numpy else None,
            4: np.int64 if has_numpy else None,
            6: np.bool_ if has_numpy else None,
            7: np.int16 if has_numpy else None,
            9: np.int8 if has_numpy else None,
            10: np.float16 if has_numpy else None,
            11: np.float64 if has_numpy else None,
        }

        def _c_handler(
            call_id,
            stage,
            model_name,
            model_id,
            subgraph_idx,
            tensor_idx,
            tensor_name,
            dtype_val,
            shape_str,
            data_ptr,
            byte_size,
            elapsed_ms,
        ):
            stage_s = stage.decode("utf-8") if stage else ""
            model_s = model_name.decode("utf-8") if model_name else ""
            tensor_s = tensor_name.decode("utf-8") if tensor_name else ""
            dtype_s = DTYPE_INT_TO_NAME.get(dtype_val, "?")
            shape_s = shape_str.decode("utf-8") if shape_str else "[]"

            shape_tuple: tuple[int, ...] = ()
            if shape_s.startswith("[") and shape_s.endswith("]") and shape_s != "[]":
                shape_tuple = tuple(
                    int(p) for p in shape_s[1:-1].split("x") if p
                )

            payload: Any = None
            if data_ptr and byte_size > 0:
                raw_bytes = ctypes.string_at(data_ptr, byte_size)
                if has_numpy and dtype_val in dtype_map and dtype_map[dtype_val] is not None:
                    arr = np.frombuffer(raw_bytes, dtype=dtype_map[dtype_val])
                    if shape_tuple and arr.size == int(np.prod(shape_tuple)):
                        payload = arr.reshape(shape_tuple)
                    else:
                        payload = arr
                else:
                    payload = raw_bytes

            callback_fn(
                call_id,
                stage_s,
                model_s,
                subgraph_idx,
                tensor_idx,
                tensor_s,
                dtype_s,
                shape_tuple,
                payload,
                elapsed_ms,
            )

        self._c_callback_ref = TensorCallbackType(_c_handler)
        self.hook_lib.cdll.set_tensor_callback(self._c_callback_ref)
        self._user_callback = callback_fn

    def clear_tensor_callback(self) -> None:
        """Clear active in-memory tensor callback."""
        self.hook_lib.cdll.clear_tensor_callback()
        self._c_callback_ref = None
        self._user_callback = None

    # -------------------------------------------------------------------------
    # Live Tensor Overriding / Mutation
    # -------------------------------------------------------------------------
    def set_tensor_override(
        self,
        tensor_idx: int,
        data: Any,
        *,
        is_post: bool = False,
        model_name: Optional[str] = None,
    ) -> bool:
        """
        Override a tensor's data buffer live in memory.

        Args:
            tensor_idx: Index of tensor to mutate.
            data: Raw bytes or NumPy ndarray.
            is_post: If False, mutates inputs before inference (PRE).
                     If True, overrides outputs after inference (POST).
            model_name: Optional filter for a specific model name.
        """
        raw_bytes: bytes
        if hasattr(data, "tobytes"):
            raw_bytes = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            raw_bytes = bytes(data)
        else:
            raise TypeError("Override data must be bytes or a NumPy ndarray.")

        model_arg = model_name.encode("utf-8") if model_name else None
        res = self.hook_lib.cdll.set_tensor_override(
            model_arg,
            1 if is_post else 0,
            ctypes.c_uint32(tensor_idx),
            raw_bytes,
            len(raw_bytes),
        )
        return res == 0

    def clear_tensor_overrides(self) -> None:
        """Remove all active tensor overrides."""
        self.hook_lib.cdll.clear_tensor_overrides()

    # -------------------------------------------------------------------------
    # Performance Profiling & Latency Analytics
    # -------------------------------------------------------------------------
    def get_profiling_stats(self, model_name: Optional[str] = None) -> ProfilingStats:
        """Get aggregate latency percentiles and anomaly metrics."""
        c_profile = CLatencyProfile()
        model_arg = model_name.encode("utf-8") if model_name else None
        self.hook_lib.cdll.get_latency_profile(model_arg, ctypes.byref(c_profile))

        return ProfilingStats(
            total_calls=c_profile.total_calls,
            avg_latency_ms=c_profile.avg_ms,
            min_latency_ms=c_profile.min_ms,
            max_latency_ms=c_profile.max_ms,
            p50_ms=c_profile.p50_ms,
            p90_ms=c_profile.p90_ms,
            p99_ms=c_profile.p99_ms,
            anomalies_detected=c_profile.nan_inf_count,
        )

    def reset_profiling_stats(self) -> None:
        """Reset profiling metrics history."""
        self.hook_lib.cdll.reset_latency_profile()

    def print_profiling_summary(self) -> None:
        """Print a human-readable latency and anomaly summary."""
        stats = self.get_profiling_stats()
        print("\n=== TFLite Inference Profiling Summary ===")
        print(f"Total Invocations:   {stats.total_calls}")
        print(f"Average Latency:     {stats.avg_latency_ms:.3f} ms")
        print(f"Min / Max Latency:   {stats.min_latency_ms:.3f} ms / {stats.max_latency_ms:.3f} ms")
        print(f"P50 (Median):        {stats.p50_ms:.3f} ms")
        print(f"P90:                 {stats.p90_ms:.3f} ms")
        print(f"P99:                 {stats.p99_ms:.3f} ms")
        print(f"NaN/Inf Anomalies:   {stats.anomalies_detected}")
        print("=========================================\n")

    # -------------------------------------------------------------------------
    # Intermediate Op Tracing & Subgraphs
    # -------------------------------------------------------------------------
    def set_trace_all_tensors(self, enabled: bool = True) -> None:
        """Enable or disable full internal graph tensor tracing."""
        self.hook_lib.cdll.set_trace_all_tensors(1 if enabled else 0)

    def set_target_subgraph(self, subgraph_idx: int) -> None:
        """Set specific subgraph index to inspect (-1 for default/first)."""
        self.hook_lib.cdll.set_target_subgraph(subgraph_idx)

    def set_custom_tensor_indices(self, indices: Sequence[int]) -> None:
        """Specify explicit tensor indices across intermediate layers to trace."""
        arr = (ctypes.c_uint32 * len(indices))(*indices)
        self.hook_lib.cdll.set_custom_tensor_indices(arr, len(indices))

    def clear_custom_tensor_indices(self) -> None:
        """Clear custom tensor indices filter."""
        self.hook_lib.cdll.clear_custom_tensor_indices()

    # -------------------------------------------------------------------------
    # Model Metadata Registration
    # -------------------------------------------------------------------------
    def register_model(
        self,
        name: str,
        tensor_count: int,
        inputs: Sequence[int],
        outputs: Sequence[int],
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
    ) -> bool:
        """Explicitly register metadata for a TFLite model."""
        c = self.hook_lib.cdll

        input_arr = (ctypes.c_uint32 * len(inputs))(*inputs) if inputs else None
        output_arr = (ctypes.c_uint32 * len(outputs))(*outputs) if outputs else None

        input_names_arr = (
            (ctypes.c_char_p * len(input_names))(
                *[str(n).encode("utf-8") for n in input_names]
            )
            if input_names
            else None
        )
        output_names_arr = (
            (ctypes.c_char_p * len(output_names))(
                *[str(n).encode("utf-8") for n in output_names]
            )
            if output_names
            else None
        )

        res = c.register_model_metadata_with_names(
            name.encode("utf-8"),
            ctypes.c_uint64(tensor_count),
            input_arr,
            len(inputs),
            input_names_arr,
            output_arr,
            len(outputs),
            output_names_arr,
        )
        return res == 0

    def register_tflite_model(
        self,
        model_source: str | Path | bytes,
        name: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Parse a .tflite model and register its metadata automatically."""
        meta = parse_tflite_metadata(model_source, name=name)
        if meta:
            self.register_model(
                name=meta.name,
                tensor_count=meta.tensor_count,
                inputs=meta.inputs,
                outputs=meta.outputs,
                input_names=meta.input_names,
                output_names=meta.output_names,
            )
        return meta

    def clear_models(self) -> None:
        """Clear all registered model metadata."""
        self.hook_lib.cdll.clear_model_metadata()

    # -------------------------------------------------------------------------
    # Hook Installation & Lifecycle
    # -------------------------------------------------------------------------
    def install_hook_symbol(
        self,
        lib_name: Optional[str | Path] = None,
        symbol: str = "TfLiteInterpreterInvoke"
    ) -> bool:
        lib_arg = str(lib_name).encode("utf-8") if lib_name else None
        sym_arg = symbol.encode("utf-8")
        return self.hook_lib.cdll.install_invoke_hook_symbol(lib_arg, sym_arg) == 0

    def install_hook_address(self, address: int) -> bool:
        return self.hook_lib.cdll.install_invoke_hook_address(ctypes.c_void_p(address)) == 0

    def install_hook_offset(self, base_address: int, offset: int) -> bool:
        return (
            self.hook_lib.cdll.install_invoke_hook_offset(
                ctypes.c_void_p(base_address), ctypes.c_size_t(offset)
            )
            == 0
        )

    def install_hook_relative(
        self,
        anchor_symbol: str,
        symbol_offset: int,
        target_offset: int
    ) -> bool:
        return (
            self.hook_lib.cdll.install_invoke_hook_relative(
                anchor_symbol.encode("utf-8"),
                ctypes.c_ssize_t(symbol_offset),
                ctypes.c_size_t(target_offset),
            )
            == 0
        )

    def is_installed(self) -> bool:
        return self.hook_lib.cdll.is_hook_installed() != 0

    def uninstall_hook(self) -> bool:
        return self.hook_lib.cdll.remove_all_hooks() == 0

    def get_invoke_count(self) -> int:
        return self.hook_lib.cdll.get_invoke_call_count()

    def enable_dump(
        self,
        dump_dir: str | Path,
        models: Optional[Sequence[str] | str] = None
    ) -> bool:
        path_str = str(dump_dir)
        Path(path_str).mkdir(parents=True, exist_ok=True)
        if isinstance(models, (list, tuple)):
            models_csv = ",".join(str(m) for m in models)
        elif models:
            models_csv = str(models)
        else:
            models_csv = None

        models_arg = models_csv.encode("utf-8") if models_csv else None
        return (
            self.hook_lib.cdll.enable_tensor_dump(
                path_str.encode("utf-8"), models_arg
            )
            == 0
        )

    def disable_dump(self) -> None:
        self.hook_lib.cdll.disable_tensor_dump()

    def read_trace_lines(self, since_last: bool = False) -> list[str]:
        if not self._trace_file.exists():
            return []

        try:
            with self._trace_file.open("r", encoding="utf-8", errors="replace") as f:
                if since_last:
                    f.seek(self._last_read_offset)
                    lines = [line.rstrip("\r\n") for line in f]
                    self._last_read_offset = f.tell()
                    return lines
                else:
                    return [line.rstrip("\r\n") for line in f]
        except OSError:
            return []

    def clear_trace(self) -> None:
        if self._trace_file.exists():
            self._trace_file.write_text("")
        self._last_read_offset = 0

    # -------------------------------------------------------------------------
    # Fault Injection & Chaos Testing
    # -------------------------------------------------------------------------
    def set_fault_injection(
        self,
        mode: str,
        param: float = 0.0,
        tensor_idx: int = -1,
    ) -> None:
        """
        Inject live hardware/sensor faults into tensors before inference.

        Modes: 'bit_flip', 'noise', 'zero_out', 'nan_inject'.
        """
        mode_map = {
            "none": 0,
            "bit_flip": 1,
            "noise": 2,
            "zero_out": 3,
            "nan_inject": 4,
        }
        val = mode_map.get(mode.lower())
        if val is None:
            raise ValueError(f"Unknown fault mode: {mode}. Choose from {list(mode_map.keys())}")

        self.hook_lib.cdll.set_fault_injection(val, float(param), int(tensor_idx))

    def clear_fault_injection(self) -> None:
        """Disable active fault injection."""
        self.hook_lib.cdll.clear_fault_injection()

    # -------------------------------------------------------------------------
    # Embedded Model Extraction (Netron Integration)
    # -------------------------------------------------------------------------
    def extract_embedded_models(
        self,
        output_dir: str | Path,
        library_path: Optional[str | Path] = None,
    ) -> list[Path]:
        """
        Extract embedded .tflite models from a library on disk or live process memory.
        """
        if library_path:
            return extract_models_from_library(library_path, output_dir)
        return extract_models_from_process_memory(output_dir)

    # -------------------------------------------------------------------------
    # Session Recording & Perfetto Export
    # -------------------------------------------------------------------------
    def start_recording(self) -> None:
        """Start recording all inference invocations to a session."""
        self._recorder = SessionRecorder()
        prev_user_cb = self._user_callback

        def _recording_cb(
            call_id, stage, model_name, sg_idx, tensor_idx, tensor_name, dtype, shape, data, elapsed_ms
        ):
            if self._recorder:
                self._recorder.record_tensor(
                    call_id, stage, model_name, sg_idx, tensor_idx, tensor_name, dtype, shape, data, elapsed_ms
                )
            if prev_user_cb:
                prev_user_cb(
                    call_id, stage, model_name, sg_idx, tensor_idx, tensor_name, dtype, shape, data, elapsed_ms
                )

        self.set_tensor_callback(_recording_cb)

    def stop_recording(self, output_filepath: str | Path) -> Path:
        """Stop session recording and save to a .tflite_session file."""
        if not self._recorder:
            raise RuntimeError("Recording was not started.")

        dest = Path(output_filepath)
        self._recorder.save(dest)
        self._recorder = None
        self.clear_tensor_callback()
        return dest

    def export_chrome_trace(self, output_json_path: str | Path) -> Path:
        """Export native trace logs to standard Chrome Tracing / Perfetto JSON."""
        return export_perfetto_trace(self._trace_file, output_json_path)

    # -------------------------------------------------------------------------
    # Op-Level Micro-Profiling
    # -------------------------------------------------------------------------
    def record_op(self, op_name: str, duration_ms: float) -> None:
        """Record an individual operator execution duration in milliseconds."""
        self.profiler.record_op(op_name, duration_ms)

    def get_op_breakdown(self) -> list[OpProfile]:
        """Get per-op latency breakdown sorted by total runtime descending."""
        return self.profiler.get_breakdown()

    def find_bottlenecks(self, top_n: int = 3, threshold_pct: float = 10.0) -> list[OpProfile]:
        """Find operator types exceeding threshold_pct of total inference latency."""
        return self.profiler.find_bottlenecks(top_n=top_n, threshold_pct=threshold_pct)

    def reset_op_profiles(self) -> None:
        """Reset op profiling counters."""
        self.profiler.reset()

    # -------------------------------------------------------------------------
    # Delegate & Hardware Acceleration Tracking
    # -------------------------------------------------------------------------
    def record_delegate(
        self, delegate_name: str, delegated_nodes: int, cpu_fallback_nodes: int
    ) -> None:
        """Record hardware accelerator delegate attachment and node counts."""
        self.delegate_tracker.record_delegate(delegate_name, delegated_nodes, cpu_fallback_nodes)

    def get_delegate_report(self) -> DelegateReport:
        """Retrieve hardware accelerator offload ratio and delegate statistics."""
        return self.delegate_tracker.get_report()

    def hook_delegate_symbol(
        self, lib_name: Optional[str] = None, symbol_name: Optional[str] = None
    ) -> bool:
        """Hook delegate attachment symbol."""
        return self.delegate_tracker.hook_delegate_symbol(lib_name, symbol_name)

    def reset_delegate(self) -> None:
        """Reset delegate metrics."""
        self.delegate_tracker.reset()

    # -------------------------------------------------------------------------
    # Memory Safety Sentinel
    # -------------------------------------------------------------------------
    def get_memory_health(self) -> MemoryHealthReport:
        """Inspect resident set size (RSS), virtual memory, leak status and canary integrity."""
        return self.sentinel.get_health()

    def register_memory_canary(self, buffer_ptr: int | ctypes.c_void_p, size: int) -> int:
        """Register a buffer boundary canary guard."""
        return self.sentinel.register_canary(buffer_ptr, size)

    def verify_memory_canaries(self) -> int:
        """Check all registered buffer canaries and return corruption count."""
        return self.sentinel.verify_canaries()

    def reset_memory_sentinel(self) -> None:
        """Reset memory sentinel history and canaries."""
        self.sentinel.reset()

    # -------------------------------------------------------------------------
    # Shadow Execution Engine
    # -------------------------------------------------------------------------
    def attach_shadow_engine(self, shadow_engine: ShadowEngine) -> None:
        """Attach a ShadowEngine for real-time output verification against a reference model."""
        self.shadow_engine = shadow_engine

    def get_shadow_results(self) -> list[ShadowComparisonResult]:
        """Return history of shadow verification results."""
        if not self.shadow_engine:
            return []
        return self.shadow_engine.get_history()

    # -------------------------------------------------------------------------
    # Dynamic Shape Resizing & Dynamic Batching Hook
    # -------------------------------------------------------------------------
    def resize_input_tensor(
        self,
        interpreter_ptr: int | ctypes.c_void_p,
        input_index: int,
        new_dims: list[int],
        library_path: Optional[str | Path] = None,
    ) -> bool:
        """Dynamically reconfigures input tensor dimensions and reallocates tensors."""
        handle = None
        if library_path:
            handle = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        else:
            handle = ctypes.CDLL(None)

        fn_name = "TfLiteInterpreterResizeInputTensor"
        resize_fn = getattr(handle, fn_name, None)
        if not resize_fn:
            LOG.warning(f"Symbol {fn_name} not found in library.")
            return False

        resize_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int32,
        ]
        resize_fn.restype = ctypes.c_int

        dims_arr = (ctypes.c_int * len(new_dims))(*new_dims)
        ptr_val = interpreter_ptr if isinstance(interpreter_ptr, int) else interpreter_ptr.value
        status = resize_fn(ctypes.c_void_p(ptr_val), input_index, dims_arr, len(new_dims))
        if status != 0:
            LOG.error(f"TfLiteInterpreterResizeInputTensor failed with status {status}")
            return False

        alloc_fn = getattr(handle, "TfLiteInterpreterAllocateTensors", None)
        if alloc_fn:
            alloc_fn.argtypes = [ctypes.c_void_p]
            alloc_fn.restype = ctypes.c_int
            alloc_fn(ctypes.c_void_p(ptr_val))

        LOG.info(f"Dynamically resized input tensor #{input_index} to shape {new_dims}")
        return True

    def set_dynamic_batch_size(
        self,
        interpreter_ptr: int | ctypes.c_void_p,
        batch_size: int,
        current_dims: list[int],
        input_index: int = 0,
        library_path: Optional[str | Path] = None,
    ) -> bool:
        """Utility to reconfigure batch size (dimension 0) dynamically."""
        if not current_dims:
            raise ValueError("current_dims must have at least 1 dimension.")
        new_dims = list(current_dims)
        new_dims[0] = batch_size
        return self.resize_input_tensor(
            interpreter_ptr, input_index, new_dims, library_path=library_path
        )

    # -------------------------------------------------------------------------
    # Telemetry Snapshot (for Web UI Dashboard)
    # -------------------------------------------------------------------------
    def get_telemetry_snapshot(self) -> dict:
        """Returns comprehensive diagnostic telemetry snapshot."""
        profile = self.get_profiling_stats()
        op_breakdown = [p.to_dict() for p in self.get_op_breakdown()]
        delegate_report = self.get_delegate_report().to_dict()
        mem_health = self.get_memory_health().to_dict()

        return {
            "total_calls": profile.total_calls if profile else 0,
            "avg_latency_ms": round(profile.avg_latency_ms, 3) if profile else 0.0,
            "min_latency_ms": round(profile.min_latency_ms, 3) if profile else 0.0,
            "max_latency_ms": round(profile.max_latency_ms, 3) if profile else 0.0,
            "p50_ms": round(profile.p50_ms, 3) if profile else 0.0,
            "p90_ms": round(profile.p90_ms, 3) if profile else 0.0,
            "p99_ms": round(profile.p99_ms, 3) if profile else 0.0,
            "anomalies_detected": profile.anomalies_detected if profile else 0,
            "op_breakdown": op_breakdown,
            "delegate": delegate_report,
            "memory": mem_health,
            "shadow_divergences": self.shadow_engine.get_divergence_count()
            if self.shadow_engine
            else 0,
        }

    def __enter__(self) -> "TFLitePatcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.uninstall_hook()
        self.disable_dump()
        self.clear_tensor_callback()
        self.clear_tensor_overrides()
        self.clear_custom_tensor_indices()
        self.clear_fault_injection()
        self.reset_op_profiles()
        self.reset_delegate()
        self.reset_memory_sentinel()
