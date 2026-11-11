"""
patching_tflite - Generic TFLite Runtime Instrumentation, Patching & Mutation Framework.

Works with any shared library (.so) that runs TensorFlow Lite models.
"""

from patching_tflite.config import PatcherConfig
from patching_tflite.delegate import DelegateReport, DelegateTracker
from patching_tflite.diff import TensorDiffReport, compare_dumps, compute_tensor_diff
from patching_tflite.dump import load_tensor_dump
from patching_tflite.extractor import (
    extract_models_from_library,
    extract_models_from_process_memory,
    find_tflite_models_in_bytes,
)
from patching_tflite.metadata import ModelMetadata, parse_tflite_metadata
from patching_tflite.patcher import ProfilingStats, TFLitePatcher
from patching_tflite.perfetto import export_perfetto_trace
from patching_tflite.profiler import OpProfile, OpProfiler
from patching_tflite.quantization import QuantizationParams, dequantize_tensor, quantize_tensor
from patching_tflite.scanner import ScanResult, scan_library
from patching_tflite.sentinel import MemoryHealthReport, MemorySentinel
from patching_tflite.session import RecordedCall, RecordedSession, RecordedTensor, SessionPlayer, SessionRecorder
from patching_tflite.shadow import ShadowComparisonResult, ShadowEngine

__version__ = "0.4.0"

__all__ = [
    "TFLitePatcher",
    "PatcherConfig",
    "ModelMetadata",
    "ProfilingStats",
    "ScanResult",
    "TensorDiffReport",
    "QuantizationParams",
    "RecordedSession",
    "RecordedCall",
    "RecordedTensor",
    "SessionRecorder",
    "SessionPlayer",
    "OpProfile",
    "OpProfiler",
    "DelegateReport",
    "DelegateTracker",
    "MemoryHealthReport",
    "MemorySentinel",
    "ShadowComparisonResult",
    "ShadowEngine",
    "scan_library",
    "load_tensor_dump",
    "parse_tflite_metadata",
    "compare_dumps",
    "compute_tensor_diff",
    "dequantize_tensor",
    "quantize_tensor",
    "extract_models_from_library",
    "extract_models_from_process_memory",
    "find_tflite_models_in_bytes",
    "export_perfetto_trace",
]
