from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelMetadata:
    name: str
    tensor_count: int
    inputs: list[int] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


def parse_tflite_metadata(
    model_source: str | Path | bytes,
    name: Optional[str] = None
) -> Optional[ModelMetadata]:
    """
    Parses TFLite boundary metadata (input/output tensor indices, names, and total count).

    Args:
        model_source: Path to .tflite file or raw model bytes.
        name: Optional label for the model (defaults to file stem or 'unnamed_model').

    Returns:
        A ModelMetadata instance or None if parsing fails.
    """
    data: bytes
    if isinstance(model_source, (str, Path)):
        p = Path(model_source)
        if not p.exists():
            return None
        data = p.read_bytes()
        if name is None:
            name = p.name
    elif isinstance(model_source, (bytes, bytearray)):
        data = bytes(model_source)
        if name is None:
            name = "in_memory_model"
    else:
        return None

    # Strategy 1: ai_edge_litert
    try:
        import ai_edge_litert.interpreter as litert
        interp = litert.Interpreter(model_content=data)
        tensor_details = interp.get_tensor_details()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()

        inputs = [int(d["index"]) for d in input_details]
        input_names = [str(d.get("name", "")) for d in input_details]
        outputs = [int(d["index"]) for d in output_details]
        output_names = [str(d.get("name", "")) for d in output_details]

        return ModelMetadata(
            name=name,
            tensor_count=len(tensor_details),
            inputs=inputs,
            input_names=input_names,
            outputs=outputs,
            output_names=output_names,
        )
    except Exception:
        pass

    # Strategy 2: tflite package if installed
    try:
        import tflite
        model = tflite.Model.GetRootAsModel(data, 0)
        subgraph = model.Subgraphs(0)
        inputs = [int(subgraph.Inputs(i)) for i in range(subgraph.InputsLength())]
        outputs = [int(subgraph.Outputs(i)) for i in range(subgraph.OutputsLength())]

        def get_tensor_name(idx: int) -> str:
            t = subgraph.Tensors(idx)
            raw = t.Name()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw or "")

        input_names = [get_tensor_name(i) for i in inputs]
        output_names = [get_tensor_name(i) for i in outputs]

        return ModelMetadata(
            name=name,
            tensor_count=int(subgraph.TensorsLength()),
            inputs=inputs,
            input_names=input_names,
            outputs=outputs,
            output_names=output_names,
        )
    except Exception:
        pass

    return None

