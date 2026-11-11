from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple


def load_tensor_dump(
    meta_path: str | Path,
    *,
    as_numpy: bool = True
) -> Tuple[dict[str, Any], Any]:
    """
    Load a tensor dump (.json metadata + .bin payload) produced by the native patcher.

    Args:
        meta_path: Path to the generated .json metadata file.
        as_numpy: If True and NumPy is available, return the payload as a shaped np.ndarray.

    Returns:
        A tuple of (metadata_dict, payload).
    """
    meta_file = Path(meta_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    bin_rel = metadata.get("bin", "")
    bin_path = Path(bin_rel)
    if not bin_path.is_absolute():
        bin_path = meta_file.parent / bin_path

    if not bin_path.exists():
        raise FileNotFoundError(f"Binary payload file not found: {bin_path}")

    raw_bytes = bin_path.read_bytes()
    if not as_numpy:
        return metadata, raw_bytes

    try:
        import numpy as np
    except ImportError:
        return metadata, raw_bytes

    dtype_map = {
        "f32": np.float32,
        "i32": np.int32,
        "u8": np.uint8,
        "i64": np.int64,
        "bool": np.bool_,
        "i16": np.int16,
        "i8": np.int8,
        "f16": np.float16,
        "f64": np.float64,
        "u32": np.uint32,
        "u16": np.uint16,
        "u64": np.uint64,
    }
    np_dtype = dtype_map.get(metadata.get("dtype"))
    if np_dtype is None:
        return metadata, raw_bytes

    arr = np.frombuffer(raw_bytes, dtype=np_dtype)
    shape_text = metadata.get("shape") or ""
    if shape_text.startswith("[") and shape_text.endswith("]") and shape_text != "[]":
        shape = tuple(int(part) for part in shape_text[1:-1].split("x") if part)
        if shape and arr.size == int(np.prod(shape)):
            arr = arr.reshape(shape)

    return metadata, arr

