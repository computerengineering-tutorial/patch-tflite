#!/usr/bin/env python3
"""
Advanced Features Demo for patching_tflite:
1. Op-Level Micro-Profiling & Bottleneck Discovery
2. Hardware Accelerator Delegate Tracking (GPU/NPU/DSP)
3. Dynamic Input Tensor Resizing & Dynamic Batching
4. Side-by-Side Shadow Execution Verification
5. Memory Safety Sentinel & Buffer Canary Verification
6. Live Web UI Dashboard integration
"""

import ctypes
import numpy as np
from pathlib import Path
from patching_tflite import (
    TFLitePatcher,
    PatcherConfig,
    ShadowEngine,
)


def main():
    print("=== TFLite Patch Advanced Diagnostic Suite ===\n")

    # Load mock TFLite library for demonstration
    mock_so = Path(__file__).resolve().parent.parent / "tests" / "libmock_tflite.so"
    mock_lib = ctypes.CDLL(str(mock_so), mode=ctypes.RTLD_GLOBAL)
    mock_lib.mock_tflite_create.argtypes = [ctypes.c_int]
    mock_lib.mock_tflite_create.restype = ctypes.c_void_p
    mock_lib.mock_tflite_free.argtypes = [ctypes.c_void_p]
    mock_lib.TfLiteInterpreterInvoke.argtypes = [ctypes.c_void_p]
    mock_lib.TfLiteInterpreterInvoke.restype = ctypes.c_int

    with TFLitePatcher() as patcher:
        patcher.register_model("demo_model", 2, inputs=[0], outputs=[1])
        patcher.install_hook_symbol(mock_so, "TfLiteInterpreterInvoke")

        # -------------------------------------------------------------
        # 1. Op-Level Micro-Profiling
        # -------------------------------------------------------------
        print("1. Profiling Operators...")
        patcher.record_op("CONV_2D", 14.5)
        patcher.record_op("CONV_2D", 16.2)
        patcher.record_op("DEPTHWISE_CONV", 5.8)
        patcher.record_op("FULLY_CONNECTED", 4.1)
        patcher.record_op("SOFTMAX", 1.2)

        breakdown = patcher.get_op_breakdown()
        print(f"  Profiled {len(breakdown)} operators:")
        for op in breakdown:
            print(f"    - {op.op_name:18s}: {op.total_time_ms:.2f} ms ({op.percent_total:.1f}%) [avg {op.avg_time_ms:.2f} ms]")

        bottlenecks = patcher.find_bottlenecks(top_n=1, threshold_pct=25.0)
        if bottlenecks:
            print(f"  [!] Primary bottleneck identified: {bottlenecks[0].op_name} ({bottlenecks[0].percent_total:.1f}% of total latency)")

        # -------------------------------------------------------------
        # 2. Hardware Acceleration Delegate Tracking
        # -------------------------------------------------------------
        print("\n2. Hardware Acceleration & Delegate Tracking...")
        patcher.record_delegate("GPU_OpenCL_Delegate", delegated_nodes=38, cpu_fallback_nodes=6)
        rep = patcher.get_delegate_report()
        print(f"  Delegate: {rep.delegate_name}")
        print(f"  Offload Ratio: {rep.delegation_ratio * 100:.1f}% ({rep.delegated_nodes}/{rep.total_nodes} nodes)")
        print(f"  GPU Accelerated: {rep.is_gpu_accelerated} | NPU Accelerated: {rep.is_npu_accelerated}")

        # -------------------------------------------------------------
        # 3. Dynamic Shape Resizing & Dynamic Batching
        # -------------------------------------------------------------
        print("\n3. Dynamic Shape Resizing...")
        interp = mock_lib.mock_tflite_create(4)
        try:
            print("  Initial shape: [1, 4]")
            # Dynamically resize input #0 to [1, 8]
            success = patcher.resize_input_tensor(interp, input_index=0, new_dims=[1, 8], library_path=mock_so)
            print(f"  Resized input #0 to [1, 8]: success={success}")

            # Reconfigure dynamic batch size: [1, 8] -> [4, 8]
            batch_success = patcher.set_dynamic_batch_size(interp, batch_size=4, current_dims=[1, 8], library_path=mock_so)
            print(f"  Reconfigured dynamic batch size to 4: success={batch_success}")
        finally:
            mock_lib.mock_tflite_free(interp)

        # -------------------------------------------------------------
        # 4. Shadow Execution Engine (Side-by-Side Validation)
        # -------------------------------------------------------------
        print("\n4. Shadow Execution Engine...")
        # Define shadow reference model
        shadow = ShadowEngine(
            reference_model=lambda inputs: [x * 2.0 for x in inputs],
            divergence_cosine_threshold=0.999
        )
        patcher.attach_shadow_engine(shadow)

        input_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        simulated_output = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

        results = shadow.verify_outputs([simulated_output], [input_arr], call_id=1)
        r = results[0]
        print(f"  Shadow Verification: Cosine Sim = {r.cosine_sim:.5f} | MAE = {r.mae:.6f} | Divergence = {r.has_divergence}")

        # -------------------------------------------------------------
        # 5. Memory Safety Sentinel & Canary Guard
        # -------------------------------------------------------------
        print("\n5. Memory Safety Sentinel...")
        mem = patcher.get_memory_health()
        print(f"  Current RSS Memory: {mem.rss_mb:.2f} MB | VSize: {mem.vsize_mb:.2f} MB")
        print(f"  Leak Status: {'LEAK DETECTED' if mem.leak_detected else 'HEALTHY'}")

        # Guard canary check
        buf = (ctypes.c_char * 64)()
        buf_addr = ctypes.addressof(buf)
        patcher.register_memory_canary(buf_addr, 64)
        print(f"  Buffer canary status: {patcher.verify_memory_canaries()} corruptions (clean)")

        # -------------------------------------------------------------
        # 6. Telemetry Snapshot for Web UI Dashboard
        # -------------------------------------------------------------
        print("\n6. Telemetry Snapshot for Live Web UI Dashboard:")
        snapshot = patcher.get_telemetry_snapshot()
        print(f"  Status Payload Keys: {list(snapshot.keys())}")
        print("\n  Launch the interactive React + TypeScript UI Dashboard with:")
        print("    $ tflite-patch ui --port 8080 --demo")
        print("    Open: http://localhost:8080 in your browser.")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

