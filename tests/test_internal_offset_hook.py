import ctypes
import tempfile
import unittest
from pathlib import Path

from patching_tflite import TFLitePatcher, PatcherConfig


class TestInternalOffsetHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests_dir = Path(__file__).resolve().parent
        cls.mock_so = cls.tests_dir / "libmock_tflite.so"
        cls.mock_lib = ctypes.CDLL(str(cls.mock_so), mode=ctypes.RTLD_GLOBAL)

        cls.mock_lib.mock_tflite_create.argtypes = [ctypes.c_int]
        cls.mock_lib.mock_tflite_create.restype = ctypes.c_void_p

        cls.mock_lib.mock_tflite_free.argtypes = [ctypes.c_void_p]
        cls.mock_lib.mock_tflite_free.restype = None

        cls.mock_lib.mock_internal_invoke.argtypes = [ctypes.c_void_p]
        cls.mock_lib.mock_internal_invoke.restype = ctypes.c_int

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = Path(self.temp_dir.name) / "trace.log"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hook_by_address(self):
        patcher = TFLitePatcher(PatcherConfig(trace_path=self.trace_file))

        # Get virtual address of mock_internal_invoke
        internal_func_addr = ctypes.cast(
            self.mock_lib.mock_internal_invoke, ctypes.c_void_p
        ).value
        self.assertIsNotNone(internal_func_addr)

        # Hook directly at address
        ok = patcher.install_hook_address(internal_func_addr)
        self.assertTrue(ok)
        self.assertTrue(patcher.is_installed())

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            # Call mock_internal_invoke directly passing the C++ impl pointer
            # Note: wrapper->impl is at offset 0
            impl_ptr = ctypes.cast(wrapper, ctypes.POINTER(ctypes.c_void_p)).contents.value
            status = self.mock_lib.mock_internal_invoke(impl_ptr)
            self.assertEqual(status, 0)

            # Check trace
            trace_lines = patcher.read_trace_lines()
            trace_text = "\n".join(trace_lines)
            self.assertIn("[INVOKE] call=", trace_text)
            self.assertIn("result=0", trace_text)
            self.assertIn("end", trace_text)
        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

