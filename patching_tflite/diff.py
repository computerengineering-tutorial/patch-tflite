from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from patching_tflite.dump import load_tensor_dump


@dataclass
class SingleTensorDiff:
    tensor_name: str
    shape: tuple
    mae: float
    rmse: float
    l1_distance: float
    l2_distance: float
    cosine_similarity: float
    max_absolute_error: float
    top1_match: bool


@dataclass
class TensorDiffReport:
    dir_a: str
    dir_b: str
    tensors_compared: int = 0
    diffs: list[SingleTensorDiff] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Tensor Numerical Drift Report ===",
            f"Set A: {self.dir_a}",
            f"Set B: {self.dir_b}",
            f"Tensors Compared: {self.tensors_compared}",
            "-" * 40,
        ]
        for d in self.diffs:
            lines.append(
                f"[{d.tensor_name}] shape={d.shape} | MAE: {d.mae:.6f} | RMSE: {d.rmse:.6f} | Cosine: {d.cosine_similarity:.6f} | MaxErr: {d.max_absolute_error:.6f} | Top1 Match: {d.top1_match}"
            )
        lines.append("=====================================")
        return "\n".join(lines)

    def print_report(self) -> None:
        print(self.summary())


def compute_tensor_diff(arr_a: Any, arr_b: Any, tensor_name: str = "tensor") -> SingleTensorDiff:
    """
    Compute numerical drift metrics between two arrays.
    """
    import numpy as np

    a = np.asarray(arr_a, dtype=np.float64).flatten()
    b = np.asarray(arr_b, dtype=np.float64).flatten()

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    l1 = float(np.sum(diff))
    l2 = float(np.linalg.norm(diff))
    max_err = float(np.max(diff)) if len(diff) > 0 else 0.0

    # Cosine similarity
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 1e-12 and norm_b > 1e-12:
        cosine = float(np.dot(a, b) / (norm_a * norm_b))
    else:
        cosine = 1.0 if np.allclose(a, b) else 0.0

    # Top-1 match
    top1 = bool(np.argmax(a) == np.argmax(b)) if len(a) > 0 else True

    return SingleTensorDiff(
        tensor_name=tensor_name,
        shape=getattr(arr_a, "shape", (len(a),)),
        mae=mae,
        rmse=rmse,
        l1_distance=l1,
        l2_distance=l2,
        cosine_similarity=cosine,
        max_absolute_error=max_err,
        top1_match=top1,
    )


def compare_dumps(dir_a: str | Path, dir_b: str | Path) -> TensorDiffReport:
    """
    Compare two directories of dumped tensors and generate a numerical drift report.
    """
    pa = Path(dir_a).resolve()
    pb = Path(dir_b).resolve()

    report = TensorDiffReport(dir_a=str(pa), dir_b=str(pb))

    files_a = {f.name: f for f in pa.glob("**/*.json")}
    files_b = {f.name: f for f in pb.glob("**/*.json")}

    common_names = sorted(list(set(files_a.keys()) & set(files_b.keys())))
    for name in common_names:
        meta_a, arr_a = load_tensor_dump(files_a[name], as_numpy=True)
        meta_b, arr_b = load_tensor_dump(files_b[name], as_numpy=True)

        try:
            d = compute_tensor_diff(arr_a, arr_b, tensor_name=meta_a.get("tensor_name") or name)
            report.diffs.append(d)
            report.tensors_compared += 1
        except Exception:
            continue

    return report

