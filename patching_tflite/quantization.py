from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass
class QuantizationParams:
    scale: float
    zero_point: int


def dequantize_tensor(
    tensor_data: Any,
    scale: float,
    zero_point: int
) -> Any:
    """
    Dequantize an INT8/UINT8/INT16 tensor array to standard FLOAT32:
        real_val = (q - zero_point) * scale
    """
    try:
        import numpy as np
        arr = np.asarray(tensor_data, dtype=np.float32)
        return (arr - float(zero_point)) * float(scale)
    except ImportError:
        # Pure python fallback for 1D lists
        if isinstance(tensor_data, (bytes, bytearray)):
            int_vals = list(tensor_data)
        elif isinstance(tensor_data, (list, tuple)):
            int_vals = tensor_data
        else:
            raise TypeError("Unsupported tensor data type for dequantization.")

        return [(float(v) - zero_point) * scale for v in int_vals]


def quantize_tensor(
    float_data: Any,
    scale: float,
    zero_point: int,
    dtype: str = "int8"
) -> Any:
    """
    Quantize a FLOAT32 tensor into INT8 or UINT8:
        q = clip(round(real_val / scale) + zero_point, min_val, max_val)
    """
    import numpy as np

    arr = np.asarray(float_data, dtype=np.float32)
    q = np.round(arr / float(scale)) + int(zero_point)

    if dtype == "int8":
        return np.clip(q, -128, 127).astype(np.int8)
    elif dtype == "uint8":
        return np.clip(q, 0, 255).astype(np.uint8)
    elif dtype == "int16":
        return np.clip(q, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported quantized dtype: {dtype}")

