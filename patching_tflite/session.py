from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class RecordedTensor:
    tensor_idx: int
    name: str
    stage: str  # "PRE" or "POST"
    dtype: str
    shape: list[int]
    data_b64: str


@dataclass
class RecordedCall:
    call_id: int
    model_name: str
    subgraph_idx: int
    elapsed_ms: float
    inputs: list[RecordedTensor] = field(default_factory=list)
    outputs: list[RecordedTensor] = field(default_factory=list)


@dataclass
class RecordedSession:
    version: str = "1.0"
    total_calls: int = 0
    calls: list[RecordedCall] = field(default_factory=list)


class SessionRecorder:
    """Records complete inference sessions into portable, self-contained files."""

    def __init__(self):
        self.session = RecordedSession()
        self._current_calls: dict[int, RecordedCall] = {}

    def record_tensor(
        self,
        call_id: int,
        stage: str,
        model_name: str,
        subgraph_idx: int,
        tensor_idx: int,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        data: Any,
        elapsed_ms: float,
    ) -> None:
        if call_id not in self._current_calls:
            self._current_calls[call_id] = RecordedCall(
                call_id=call_id,
                model_name=model_name,
                subgraph_idx=subgraph_idx,
                elapsed_ms=elapsed_ms,
            )

        call_obj = self._current_calls[call_id]
        if elapsed_ms > 0.0:
            call_obj.elapsed_ms = elapsed_ms

        # Encode data to base64
        raw_bytes: bytes
        if hasattr(data, "tobytes"):
            raw_bytes = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            raw_bytes = bytes(data)
        else:
            raw_bytes = b""

        b64 = base64.b64encode(raw_bytes).decode("ascii")
        t_rec = RecordedTensor(
            tensor_idx=tensor_idx,
            name=name,
            stage=stage,
            dtype=dtype,
            shape=list(shape) if shape else [],
            data_b64=b64,
        )

        if stage == "PRE":
            call_obj.inputs.append(t_rec)
        else:
            call_obj.outputs.append(t_rec)

    def save(self, filepath: str | Path) -> None:
        p = Path(filepath).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        self.session.calls = list(self._current_calls.values())
        self.session.total_calls = len(self.session.calls)

        data = asdict(self.session)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


class SessionPlayer:
    """Loads and replays previously recorded TFLite inference sessions."""

    def __init__(self, filepath: str | Path):
        self.path = Path(filepath).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Session file not found: {self.path}")

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.session = RecordedSession(
            version=raw.get("version", "1.0"),
            total_calls=raw.get("total_calls", 0),
        )

        for c_data in raw.get("calls", []):
            call = RecordedCall(
                call_id=c_data["call_id"],
                model_name=c_data["model_name"],
                subgraph_idx=c_data["subgraph_idx"],
                elapsed_ms=c_data["elapsed_ms"],
            )
            for inp in c_data.get("inputs", []):
                call.inputs.append(RecordedTensor(**inp))
            for out in c_data.get("outputs", []):
                call.outputs.append(RecordedTensor(**out))
            self.session.calls.append(call)

    def get_calls(self) -> list[RecordedCall]:
        return self.session.calls

    @staticmethod
    def decode_tensor_data(tensor_rec: RecordedTensor, as_numpy: bool = True) -> Any:
        raw_bytes = base64.b64decode(tensor_rec.data_b64)
        if not as_numpy:
            return raw_bytes

        try:
            import numpy as np
            dtype_map = {
                "f32": np.float32,
                "i32": np.int32,
                "u8": np.uint8,
                "i64": np.int64,
                "bool": np.bool_,
                "i16": np.int16,
                "i8": np.int8,
                "f16": np.float16,
            }
            np_dt = dtype_map.get(tensor_rec.dtype)
            if np_dt is not None:
                arr = np.frombuffer(raw_bytes, dtype=np_dt)
                if tensor_rec.shape and arr.size == int(np.prod(tensor_rec.shape)):
                    return arr.reshape(tensor_rec.shape)
                return arr
        except ImportError:
            pass

        return raw_bytes

