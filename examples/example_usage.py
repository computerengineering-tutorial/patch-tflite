#!/usr/bin/env python3
"""
Example demonstrating the complete capabilities of the generic TFLite patching framework:
- Scanning an unknown .so for TFLite symbols
- In-memory zero-copy tensor callbacks
- Live tensor mutation / overriding (PRE and POST)
- Tracing all intermediate layer tensors
- Performance analytics (P50/P90/P99 latency & anomaly detection)
- Tensor dumping to .bin / .json
"""

import numpy as np
from pathlib import Path
from patching_tflite import (
    TFLitePatcher,
    PatcherConfig,
    load_tensor_dump,
    scan_library,
)


def main():
    target_so = "/path/to/any_tflite_runtime.so"

    # 1. Automatic ELF Scanning
    print("--- 1. Library Inspection ---")
    if Path(target_so).exists():
        scan_res = scan_library(target_so)
        print(scan_res.summary())
        symbol_to_hook = scan_res.recommended_symbol or "TfLiteInterpreterInvoke"
    else:
        print(f"Target {target_so} not found; using default symbol.")
        symbol_to_hook = "TfLiteInterpreterInvoke"

    # 2. Configure Patcher with advanced features
    config = PatcherConfig(
        trace_path="/tmp/tflite_trace.log",
        tensor_limit=128,
        preview_values=8,
        dump_dir="/tmp/tflite_dumps",
        dump_models="my_model",
        trace_all_tensors=True,  # Trace all intermediate layers, not just boundaries
    )

    with TFLitePatcher(config) as patcher:
        # 3. Register Model Metadata
        patcher.register_model(
            name="my_model",
            tensor_count=4,
            inputs=[0],
            outputs=[3],
            input_names=["input_tensor"],
            output_names=["output_tensor"],
        )

        # 4. In-Memory Zero-Copy Streaming Callback
        def on_tensor_stream(call_id, stage, model, sg_idx, tidx, name, dtype, shape, data, elapsed):
            if isinstance(data, np.ndarray):
                print(f"[STREAM] Call {call_id} | {stage} | Tensor #{tidx} ({name}) | Shape {shape} | Min: {data.min():.2f}")
            else:
                print(f"[STREAM] Call {call_id} | {stage} | Tensor #{tidx} ({name}) | {len(data)} bytes")

        patcher.set_tensor_callback(on_tensor_stream)

        # 5. Live Tensor Injection / Mutation
        # Mutate input 0 before inference:
        custom_input = np.ones((1, 4), dtype=np.float32) * 5.0
        patcher.set_tensor_override(0, custom_input, is_post=False)

        # 6. Install Hook on target library
        print(f"\nAttaching hook to {symbol_to_hook}...")
        # patcher.install_hook_symbol(target_so, symbol_to_hook)

        # 7. Run your normal model inference here ...
        # (your library calls Invoke)

        # 8. Performance Analytics & Profiling
        stats = patcher.get_profiling_stats()
        patcher.print_profiling_summary()

    print("\nHooks uninstalled and environment cleaned up successfully.")


if __name__ == "__main__":
    main()
