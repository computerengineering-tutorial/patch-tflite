import ctypes
import unittest
from pathlib import Path

from patching_tflite import TFLitePatcher


class TestFaultInjection(unittest.TestCase):
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

    def test_zero_out_fault(self):
        patcher = TFLitePatcher()
        patcher.register_model("fault_model", 2, inputs=[0], outputs=[1])
        patcher.install_hook_symbol(self.mock_so, "TfLiteInterpreterInvoke")

        # Configure zero_out fault on input tensor #0
        patcher.set_fault_injection("zero_out", tensor_idx=0)

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            # Since input was zeroed out, output = 0.0 * 2 = 0.0
            out = (ctypes.c_float * 4)()
            self.mock_lib.mock_tflite_get_output(wrapper, out, 4)
            self.assertEqual(list(out), [0.0, 0.0, 0.0, 0.0])
        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.clear_fault_injection()
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

