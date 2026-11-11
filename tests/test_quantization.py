import unittest
import numpy as np

from patching_tflite import quantize_tensor, dequantize_tensor


class TestQuantization(unittest.TestCase):
    def test_quantize_and_dequantize_roundtrip(self):
        original = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        scale = 0.01
        zero_point = 0

        # Quantize to int8
        q = quantize_tensor(original, scale=scale, zero_point=zero_point, dtype="int8")
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(list(q), [-100, -50, 0, 50, 100])

        # Dequantize back to float32
        recovered = dequantize_tensor(q, scale=scale, zero_point=zero_point)
        self.assertTrue(np.allclose(original, recovered, atol=1e-3))

    def test_uint8_quantization(self):
        original = np.array([0.0, 1.28, 2.55], dtype=np.float32)
        scale = 0.01
        zero_point = 0

        q = quantize_tensor(original, scale=scale, zero_point=zero_point, dtype="uint8")
        self.assertEqual(q.dtype, np.uint8)
        self.assertEqual(list(q), [0, 128, 255])

        recovered = dequantize_tensor(q, scale=scale, zero_point=zero_point)
        self.assertTrue(np.allclose(original, recovered, atol=1e-2))


if __name__ == "__main__":
    unittest.main()

