from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
import numpy as np

logger = logging.getLogger("patching_tflite.shadow")


@dataclass
class ShadowComparisonResult:
    call_id: int
    model_name: str
    tensor_idx: int
    mae: float
    rmse: float
    max_abs_diff: float
    cosine_sim: float
    top1_match: bool
    has_divergence: bool

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "model_name": self.model_name,
            "tensor_idx": self.tensor_idx,
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
            "max_abs_diff": round(self.max_abs_diff, 6),
            "cosine_sim": round(self.cosine_sim, 6),
            "top1_match": self.top1_match,
            "has_divergence": self.has_divergence,
        }


class ShadowEngine:
    """Verifies live .so inference outputs against a shadow reference model in lockstep."""

    def __init__(
        self,
        reference_model: Optional[str | Callable[[List[np.ndarray]], List[np.ndarray]]] = None,
        divergence_cosine_threshold: float = 0.999,
        divergence_mae_threshold: float = 1e-3,
        model_name: str = "shadow_model",
    ):
        self.model_name = model_name
        self.divergence_cosine_threshold = divergence_cosine_threshold
        self.divergence_mae_threshold = divergence_mae_threshold
        self._history: list[ShadowComparisonResult] = []
        self._divergence_count: int = 0
        self._ref_interpreter = None
        self._ref_fn = None

        if callable(reference_model):
            self._ref_fn = reference_model
        elif isinstance(reference_model, str):
            self._init_tflite_reference(reference_model)

    def _init_tflite_reference(self, model_path: str):
        try:
            from ai_edge_litert.interpreter import Interpreter
            self._ref_interpreter = Interpreter(model_path=model_path)
            self._ref_interpreter.allocate_tensors()
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                self._ref_interpreter = tflite.Interpreter(model_path=model_path)
                self._ref_interpreter.allocate_tensors()
            except ImportError:
                raise RuntimeError(
                    "Neither ai_edge_litert nor tflite_runtime is available to load .tflite reference model."
                )

    def run_reference(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Runs inputs through the reference model and returns outputs."""
        if self._ref_fn:
            return self._ref_fn(inputs)

        if self._ref_interpreter:
            input_details = self._ref_interpreter.get_input_details()
            output_details = self._ref_interpreter.get_output_details()

            for i, in_arr in enumerate(inputs):
                if i < len(input_details):
                    self._ref_interpreter.set_tensor(input_details[i]["index"], in_arr)

            self._ref_interpreter.invoke()
            outputs = [self._ref_interpreter.get_tensor(od["index"]) for od in output_details]
            return outputs

        raise RuntimeError("No reference model or callable configured in ShadowEngine.")

    def compare_tensors(
        self,
        so_arr: np.ndarray,
        ref_arr: np.ndarray,
        call_id: int = 0,
        tensor_idx: int = 0,
    ) -> ShadowComparisonResult:
        """Computes statistical divergence metrics between live .so output and reference output."""
        a = np.asarray(so_arr, dtype=np.float64).flatten()
        b = np.asarray(ref_arr, dtype=np.float64).flatten()

        if a.shape != b.shape:
            # Shape mismatch
            res = ShadowComparisonResult(
                call_id=call_id,
                model_name=self.model_name,
                tensor_idx=tensor_idx,
                mae=float("inf"),
                rmse=float("inf"),
                max_abs_diff=float("inf"),
                cosine_sim=0.0,
                top1_match=False,
                has_divergence=True,
            )
            self._history.append(res)
            self._divergence_count += 1
            return res

        diff = np.abs(a - b)
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(diff))

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 1e-12 and norm_b > 1e-12:
            cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
        elif norm_a <= 1e-12 and norm_b <= 1e-12:
            cosine_sim = 1.0
        else:
            cosine_sim = 0.0

        top1 = bool(np.argmax(a) == np.argmax(b)) if len(a) > 0 else True
        diverged = (cosine_sim < self.divergence_cosine_threshold) or (
            mae > self.divergence_mae_threshold
        )

        res = ShadowComparisonResult(
            call_id=call_id,
            model_name=self.model_name,
            tensor_idx=tensor_idx,
            mae=mae,
            rmse=rmse,
            max_abs_diff=max_abs,
            cosine_sim=cosine_sim,
            top1_match=top1,
            has_divergence=diverged,
        )
        self._history.append(res)
        if diverged:
            self._divergence_count += 1
            logger.warning(
                f"[SHADOW DIVERGENCE] call={call_id} tensor={tensor_idx} "
                f"cosine={cosine_sim:.5f} mae={mae:.5f}"
            )

        return res

    def verify_outputs(
        self,
        so_outputs: list[np.ndarray],
        inputs: list[np.ndarray],
        call_id: int = 0,
    ) -> list[ShadowComparisonResult]:
        """Runs the reference model and compares with actual .so outputs."""
        ref_outputs = self.run_reference(inputs)
        results = []
        for idx, (so_out, ref_out) in enumerate(zip(so_outputs, ref_outputs)):
            r = self.compare_tensors(so_out, ref_out, call_id=call_id, tensor_idx=idx)
            results.append(r)
        return results

    def get_divergence_count(self) -> int:
        return self._divergence_count

    def get_history(self) -> list[ShadowComparisonResult]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._divergence_count = 0

