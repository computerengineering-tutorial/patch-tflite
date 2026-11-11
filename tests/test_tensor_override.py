import ctypes
import unittest
from pathlib import Path
import numpy as np

from patching_tflite import TFLitePatcher, PatcherConfig


class TestTensorOverride(unittest.TestCase):
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

        cls.mock_lib.mock_tflite_get_output.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        cls.mock_lib.mock_tflite_get_output.restype = None

    def test_pre_input_override(self):
        patcher = TFLitePatcher()
        patcher.register_model(
            name="mock_override_model",
            tensor_count=2,
            inputs=[0],
            outputs=[1],
        )
        patcher.install_hook_symbol(self.mock_so, "TfLiteInterpreterInvoke")

        # Mutate input tensor (idx 0) to [10.0, 20.0, 30.0, 40.0] before inference
        new_inputs = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        ok = patcher.set_tensor_override(0, new_inputs, is_post=False)
        self.assertTrue(ok)

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            # Model computes output = input * 2.0 = [20, 40, 60, 80]
            out = (ctypes.c_float * 4)()
            self.mock_lib.mock_tflite_get_output(wrapper, out, 4)
            self.assertEqual(list(out), [20.0, 40.0, 60.0, 80.0])
        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.clear_tensor_overrides()
            patcher.uninstall_hook()

    def test_post_output_override(self):
        patcher = TFLitePatcher()
        patcher.register_model(
            name="mock_override_model2",
            tensor_count=2,
            inputs=[0],
            outputs=[1],
        )
        patcher.install_hook_symbol(self.mock_so, "TfLiteInterpreterInvoke")

        # Override output tensor (idx 1) after inference to [99.0, 99.0, 99.0, 99.0]
        mock_output = np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32)
        patcher.set_tensor_override(1, mock_output, is_post=True)

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            out = (ctypes.c_float * 4)()
            self.mock_lib.mock_tflite_get_output(wrapper, out, 4)
            self.assertEqual(list(out), [99.0, 99.0, 99.0, 99.0])
        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.clear_tensor_overrides()
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

