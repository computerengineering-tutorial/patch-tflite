from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List


def export_perfetto_trace(trace_log_path: str | Path, output_json_path: str | Path) -> Path:
    """
    Convert a TFLite native trace log file into a standard Chrome Tracing / Perfetto JSON timeline.
    Compatible with https://ui.perfetto.dev and chrome://tracing.
    """
    src = Path(trace_log_path).resolve()
    dst = Path(output_json_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Trace log not found: {src}")

    events = []

    # Regex patterns for trace lines:
    # [1792928104.861035] [INVOKE] call=1 start model=m001 name=...
    # [1792928104.861131] [INVOKE] call=1 end model=m001 name=... result=0 elapsed_ms=0.096
    start_pattern = re.compile(
        r"\[(?P<ts>\d+\.\d+)\]\s+\[INVOKE\]\s+call=(?P<call_id>\d+)\s+start\s+model=(?P<model>\S+)\s+name=(?P<name>\S+)"
    )
    end_pattern = re.compile(
        r"\[(?P<ts>\d+\.\d+)\]\s+\[INVOKE\]\s+call=(?P<call_id>\d+)\s+end\s+model=(?P<model>\S+)\s+name=(?P<name>\S+).*?elapsed_ms=(?P<dur>\d+\.\d+)"
    )
    tensor_pattern = re.compile(
        r"\[(?P<ts>\d+\.\d+)\]\s+\[TENSOR\]\s+call=(?P<call_id>\d+)\s+(?P<stage>\w+)\s+(?P<kind>\w+)\s+t(?P<idx>\d+)\s+type=(?P<type>\w+)\s+shape=(?P<shape>\[.*?\])\s+bytes=(?P<bytes>\d+)"
    )

    call_starts: dict[int, dict[str, Any]] = {}

    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        # Match invoke start
        m_start = start_pattern.search(line)
        if m_start:
            cid = int(m_start.group("call_id"))
            ts_us = float(m_start.group("ts")) * 1_000_000.0  # microseconds
            call_starts[cid] = {
                "ts": ts_us,
                "model": m_start.group("model"),
                "name": m_start.group("name"),
            }
            continue

        # Match invoke end
        m_end = end_pattern.search(line)
        if m_end:
            cid = int(m_end.group("call_id"))
            end_ts_us = float(m_end.group("ts")) * 1_000_000.0
            dur_us = float(m_end.group("dur")) * 1_000.0

            start_info = call_starts.pop(cid, None)
            start_ts_us = start_info["ts"] if start_info else (end_ts_us - dur_us)
            model_name = m_end.group("name")

            events.append({
                "name": f"Invoke: {model_name} (call {cid})",
                "cat": "tflite_inference",
                "ph": "X",  # Complete event
                "ts": start_ts_us,
                "dur": dur_us,
                "pid": 1,
                "tid": 1,
                "args": {
                    "call_id": cid,
                    "model": m_end.group("model"),
                    "name": model_name,
                    "elapsed_ms": float(m_end.group("dur")),
                },
            })
            continue

        # Match tensor event
        m_tensor = tensor_pattern.search(line)
        if m_tensor:
            cid = int(m_tensor.group("call_id"))
            ts_us = float(m_tensor.group("ts")) * 1_000_000.0
            events.append({
                "name": f"Tensor {m_tensor.group('stage')}: t{m_tensor.group('idx')}",
                "cat": "tensor",
                "ph": "i",  # Instant event
                "ts": ts_us,
                "pid": 1,
                "tid": 1,
                "s": "t",
                "args": {
                    "call_id": cid,
                    "stage": m_tensor.group("stage"),
                    "kind": m_tensor.group("kind"),
                    "shape": m_tensor.group("shape"),
                    "bytes": int(m_tensor.group("bytes")),
                    "dtype": m_tensor.group("type"),
                },
            })

    output_data = {"traceEvents": events, "displayTimeUnit": "ms"}
    dst.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    return dst

