import ctypes
import unittest
from pathlib import Path
import numpy as np

from patching_tflite import TFLitePatcher, ProfilingStats


class TestProfilingAndAnomaly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests_dir = Path(__file__).resolve().parent
        cls.mock_so = cls.tests_dir / "libmock_tflite.so"
        cls.mock_lib = ctypes.CDLL(str(cls.mock_so), mode=ctypes.RTLD_GLOBAL)

        cls.mock_lib.mock_tflite_create.argtypes = [ctypes.c_int]
        cls.mock_lib.mock_tflite_create.restype = ctypes.c_void_p

        cls.mock_lib.mock_tflite_free.argtypes = [ctypes.c_void_p]
        cls.mock_lib.mock_tflite_free.restype = None

        cls.mock_lib.TfLiteInterpreterInvoke.argtypes = [ctypes.c_void_p]
        cls.mock_lib.TfLiteInterpreterInvoke.restype = ctypes.c_int

    def test_profiling_and_nan_detection(self):
        patcher = TFLitePatcher()
        patcher.reset_profiling_stats()

        patcher.register_model(
            name="profiling_model",
            tensor_count=2,
            inputs=[0],
            outputs=[1],
        )
        patcher.install_hook_symbol(self.mock_so, "TfLiteInterpreterInvoke")

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            # 1. Run 3 regular inferences
            for _ in range(3):
                status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
                self.assertEqual(status, 0)

            stats = patcher.get_profiling_stats()
            self.assertIsInstance(stats, ProfilingStats)
            self.assertEqual(stats.total_calls, 3)
            self.assertGreaterEqual(stats.avg_latency_ms, 0.0)
            self.assertLessEqual(stats.min_latency_ms, stats.max_latency_ms)
            self.assertEqual(stats.anomalies_detected, 0)

            # 2. Inject NaN into input tensor to trigger anomaly detection
            nan_arr = np.array([float("nan"), 1.0, 2.0, 3.0], dtype=np.float32)
            patcher.set_tensor_override(0, nan_arr, is_post=False)

            self.mock_lib.TfLiteInterpreterInvoke(wrapper)

            # Check that anomaly was detected and logged
            stats_after = patcher.get_profiling_stats()
            self.assertGreater(stats_after.anomalies_detected, 0)

            trace_lines = patcher.read_trace_lines()
            trace_text = "\n".join(trace_lines)
            self.assertIn("ANOMALY", trace_text)
            self.assertIn("has_nan=1", trace_text)

        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.clear_tensor_overrides()
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

