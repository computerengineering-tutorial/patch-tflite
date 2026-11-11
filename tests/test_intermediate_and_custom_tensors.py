import ctypes
import unittest
from pathlib import Path

from patching_tflite import TFLitePatcher, PatcherConfig


class TestIntermediateAndCustomTensors(unittest.TestCase):
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

    def test_trace_all_tensors(self):
        config = PatcherConfig(trace_all_tensors=True)
        patcher = TFLitePatcher(config)
        patcher.register_model(
            name="all_tensors_model",
            tensor_count=2,
            inputs=[0],
            outputs=[1],
        )
        patcher.install_hook_symbol(self.mock_so, "TfLiteInterpreterInvoke")

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            trace_lines = patcher.read_trace_lines()
            trace_text = "\n".join(trace_lines)
            self.assertIn("PRE all t0", trace_text)
            self.assertIn("POST all t1", trace_text)
        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.uninstall_hook()

    def test_custom_tensor_indices(self):
        patcher = TFLitePatcher()
        patcher.register_model(
            name="custom_tensors_model",
            tensor_count=2,
            inputs=[],
            outputs=[],
        )
        patcher.set_custom_tensor_indices([1])
        patcher.install_hook_symbol(self.mock_so, "TfLiteInterpreterInvoke")

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            trace_lines = patcher.read_trace_lines()
            trace_text = "\n".join(trace_lines)
            self.assertIn("PRE custom t1", trace_text)
            self.assertIn("POST custom t1", trace_text)
        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.clear_custom_tensor_indices()
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

