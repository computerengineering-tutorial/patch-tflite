import ctypes
import unittest
from pathlib import Path
import numpy as np

from patching_tflite import TFLitePatcher


class TestTensorCallback(unittest.TestCase):
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

    def test_in_memory_callback(self):
        patcher = TFLitePatcher()
        patcher.register_model(
            name="callback_test_model",
            tensor_count=2,
            inputs=[0],
            outputs=[1],
            input_names=["in_t"],
            output_names=["out_t"],
        )
        patcher.install_hook_symbol(self.mock_so, "TfLiteInterpreterInvoke")

        received_events = []

        def my_callback(
            call_id,
            stage,
            model_name,
            subgraph_idx,
            tensor_idx,
            tensor_name,
            dtype,
            shape,
            data,
            elapsed_ms,
        ):
            received_events.append({
                "call_id": call_id,
                "stage": stage,
                "model_name": model_name,
                "tensor_idx": tensor_idx,
                "tensor_name": tensor_name,
                "dtype": dtype,
                "shape": shape,
                "data": np.copy(data) if isinstance(data, np.ndarray) else data,
                "elapsed_ms": elapsed_ms,
            })

        patcher.set_tensor_callback(my_callback)

        wrapper = self.mock_lib.mock_tflite_create(4)
        try:
            status = self.mock_lib.TfLiteInterpreterInvoke(wrapper)
            self.assertEqual(status, 0)

            # We expect 2 callback events: 1 PRE and 1 POST
            self.assertEqual(len(received_events), 2)

            pre = received_events[0]
            self.assertEqual(pre["stage"], "PRE")
            self.assertEqual(pre["tensor_idx"], 0)
            self.assertEqual(pre["tensor_name"], "in_t")
            self.assertEqual(pre["dtype"], "f32")
            self.assertEqual(pre["shape"], (1, 4))
            self.assertEqual(list(pre["data"].flatten()), [1.0, 2.0, 3.0, 4.0])

            post = received_events[1]
            self.assertEqual(post["stage"], "POST")
            self.assertEqual(post["tensor_idx"], 1)
            self.assertEqual(post["tensor_name"], "out_t")
            self.assertEqual(post["dtype"], "f32")
            self.assertEqual(post["shape"], (1, 4))
            self.assertEqual(list(post["data"].flatten()), [2.0, 4.0, 6.0, 8.0])

        finally:
            self.mock_lib.mock_tflite_free(wrapper)
            patcher.clear_tensor_callback()
            patcher.uninstall_hook()


if __name__ == "__main__":
    unittest.main()

