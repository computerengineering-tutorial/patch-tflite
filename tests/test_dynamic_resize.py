import ctypes
from pathlib import Path
import unittest

from patching_tflite import TFLitePatcher


class TestDynamicResize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests_dir = Path(__file__).resolve().parent
        cls.mock_so = cls.tests_dir / "libmock_tflite.so"
        cls.mock_lib = ctypes.CDLL(str(cls.mock_so), mode=ctypes.RTLD_GLOBAL)

        cls.mock_lib.mock_tflite_create.argtypes = [ctypes.c_int]
        cls.mock_lib.mock_tflite_create.restype = ctypes.c_void_p

        cls.mock_lib.mock_tflite_free.argtypes = [ctypes.c_void_p]
        cls.mock_lib.mock_tflite_free.restype = None

        cls.mock_lib.mock_tflite_get_tensor_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cls.mock_lib.mock_tflite_get_tensor_bytes.restype = ctypes.c_size_t

    def test_dynamic_shape_resizing(self):
        patcher = TFLitePatcher()
        wrapper = self.mock_lib.mock_tflite_create(4)  # 4 floats = 16 bytes
        try:
            initial_bytes = self.mock_lib.mock_tflite_get_tensor_bytes(wrapper, 0)
            self.assertEqual(initial_bytes, 4 * ctypes.sizeof(ctypes.c_float))

            # Dynamically resize input #0 to shape [1, 8] (8 floats = 32 bytes)
            success = patcher.resize_input_tensor(
                wrapper, input_index=0, new_dims=[1, 8], library_path=self.mock_so
            )
            self.assertTrue(success)

            new_bytes = self.mock_lib.mock_tflite_get_tensor_bytes(wrapper, 0)
            self.assertEqual(new_bytes, 8 * ctypes.sizeof(ctypes.c_float))

            # Test dynamic batch sizing
            success_batch = patcher.set_dynamic_batch_size(
                wrapper, batch_size=4, current_dims=[1, 8], input_index=0, library_path=self.mock_so
            )
            self.assertTrue(success_batch)
            batch_bytes = self.mock_lib.mock_tflite_get_tensor_bytes(wrapper, 0)
            self.assertEqual(batch_bytes, 4 * 8 * ctypes.sizeof(ctypes.c_float))

        finally:
            self.mock_lib.mock_tflite_free(wrapper)


if __name__ == "__main__":
    unittest.main()

