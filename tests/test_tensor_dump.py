import ctypes
import tempfile
import unittest
from pathlib import Path

from patching_tflite import TFLitePatcher, PatcherConfig, load_tensor_dump


class TestTensorDump(unittest.TestCase):
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

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dump_dir = Path(self.temp_dir.name) / "dumps"
        self.trace_file = Path(self.temp_dir.name) / "trace.log"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tensor_dump_and_load(self):
        config = PatcherConfig(
            trace_path=self.trace_file,
            dump_dir=self.dump_dir,
            dump_models="mock_model",
        )
        patcher = TFLitePatcher(config)

        patcher.register_model(
            name="mock_model",
            tensor_count=2,
            inputs=[0],
            outputs=[1],
            input_names=["input_0"],
            output_names=["output_0"],
        )

        patcher.install_hook_symbol(
            lib_name=self.mock_so,
            symbol="TfLiteInterpreterInvoke"
        )

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            # Check that dump directory and files were created
            json_files = list(self.dump_dir.glob("*/*.json"))
            self.assertEqual(len(json_files), 2)  # 1 PRE input + 1 POST output

            # Load the input dump
            pre_json = [f for f in json_files if "_PRE_input_" in f.name][0]
            pre_meta, pre_arr = load_tensor_dump(pre_json, as_numpy=True)

            self.assertEqual(pre_meta["stage"], "PRE")
            self.assertEqual(pre_meta["kind"], "input")
            self.assertEqual(pre_meta["tensor_name"], "input_0")
            self.assertEqual(pre_meta["dtype"], "f32")
            self.assertEqual(pre_meta["shape"], "[1x4]")
            # Input was [1.0, 2.0, 3.0, 4.0]
            self.assertAlmostEqual(float(pre_arr[0][0]), 1.0)
            self.assertAlmostEqual(float(pre_arr[0][3]), 4.0)

            # Load the output dump
            post_json = [f for f in json_files if "_POST_output_" in f.name][0]
            post_meta, post_arr = load_tensor_dump(post_json, as_numpy=True)

            self.assertEqual(post_meta["stage"], "POST")
            self.assertEqual(post_meta["kind"], "output")
            self.assertEqual(post_meta["tensor_name"], "output_0")
            self.assertEqual(post_meta["dtype"], "f32")
            self.assertEqual(post_meta["shape"], "[1x4]")
            # Output was [2.0, 4.0, 6.0, 8.0]
            self.assertAlmostEqual(float(post_arr[0][0]), 2.0)
            self.assertAlmostEqual(float(post_arr[0][3]), 8.0)

        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

