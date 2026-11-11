from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PatcherConfig:
    """Configuration options for TFLite runtime patching, tracing, and profiling."""

    trace_path: str | Path | None = None
    tensor_limit: int = 128
    max_tensor_bytes: int = 16 * 1024 * 1024
    preview_values: int = 0
    tensor_names: bool = True
    dump_dir: str | Path | None = None
    dump_models: list[str] | str | None = None

    # Full intermediate op tracing / custom tensor indices
    trace_all_tensors: bool = False
    target_subgraph: int = -1  # -1 = default / all
    custom_tensor_indices: list[int] | None = None

    # Memory layout offsets (x86_64 defaults)
    subgraph_offset: int = 0x68
    tensor_array_offset: int = 0x08
    tensor_count_offset: int = 0x10
    tensor_size: int = 0x70
