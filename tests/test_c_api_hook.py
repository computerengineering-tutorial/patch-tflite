import ctypes
import os
import tempfile
import unittest
from pathlib import Path

from patching_tflite import TFLitePatcher, PatcherConfig


class TestCAPITfliteHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests_dir = Path(__file__).resolve().parent
        cls.mock_so = cls.tests_dir / "libmock_tflite.so"
        if not cls.mock_so.exists():
            # Compile mock library if not already compiled
            c_src = cls.tests_dir / "mock_tflite_so.c"
            import subprocess
            subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O2", "-o", str(cls.mock_so), str(c_src)],
                check=True
            )

        # Load mock C library
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

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = Path(self.temp_dir.name) / "trace.log"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hook_and_invoke_lifecycle(self):
        config = PatcherConfig(
            trace_path=self.trace_file,
            tensor_limit=16,
            preview_values=4,
            tensor_names=True,
        )
        patcher = TFLitePatcher(config)

        # Register mock model metadata
        registered = patcher.register_model(
            name="mock_add_model",
            tensor_count=2,
            inputs=[0],
            outputs=[1],
            input_names=["input_tensor"],
            output_names=["output_tensor"],
        )
        self.assertTrue(registered)

        # Install hook on standard TfLiteInterpreterInvoke in mock .so
        installed = patcher.install_hook_symbol(
            lib_name=self.mock_so,
            symbol="TfLiteInterpreterInvoke"
        )
        self.assertTrue(installed)
        self.assertTrue(patcher.is_installed())

        # Create mock interpreter instance
        wrapper = self.mock_lib.mock_tflite_create(4)
        self.assertIsNotNone(wrapper)

        try:
            # First invoke
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            # Check that computation succeeded (input=[1,2,3,4] * 2 = [2,4,6,8])
            out_buf = (ctypes.c_float * 4)()
            self.mock_lib.mock_tflite_get_output(wrapper, out_buf, 4)
            self.assertEqual(list(out_buf), [2.0, 4.0, 6.0, 8.0])

            # Verify hook call counter
            self.assertEqual(patcher.get_invoke_count(), 1)

            # Verify trace content
            trace_lines = patcher.read_trace_lines()
            trace_text = "\n".join(trace_lines)
            self.assertIn("[INVOKE] call=1 start", trace_text)
            self.assertIn("name=mock_add_model", trace_text)
            self.assertIn("PRE input t0", trace_text)
            self.assertIn("name=input_tensor", trace_text)
            self.assertIn("POST output t1", trace_text)
            self.assertIn("name=output_tensor", trace_text)
            self.assertIn("[INVOKE] call=1 end", trace_text)

            # Uninstall hook
            uninstalled = patcher.uninstall_hook()
            self.assertTrue(uninstalled)
            self.assertFalse(patcher.is_installed())

            # Second invoke (unhooked)
            status2 = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status2, 0)
            # Invoke count must remain 1
            self.assertEqual(patcher.get_invoke_count(), 1)

        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

