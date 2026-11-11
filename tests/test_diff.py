import tempfile
import unittest
from pathlib import Path
import numpy as np

from patching_tflite import compute_tensor_diff, compare_dumps


class TestDiff(unittest.TestCase):
    def test_identical_tensors(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.copy(a)

        d = compute_tensor_diff(a, b, "test_tensor")
        self.assertAlmostEqual(d.mae, 0.0)
        self.assertAlmostEqual(d.rmse, 0.0)
        self.assertAlmostEqual(d.l1_distance, 0.0)
        self.assertAlmostEqual(d.cosine_similarity, 1.0)
        self.assertTrue(d.top1_match)

    def test_divergent_tensors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        d = compute_tensor_diff(a, b, "logits")
        self.assertAlmostEqual(d.mae, 2.0 / 3.0)
        self.assertAlmostEqual(d.cosine_similarity, 0.0)
        self.assertFalse(d.top1_match)

    def test_compare_dumps_empty(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            report = compare_dumps(tmp1, tmp2)
            self.assertEqual(report.tensors_compared, 0)
            self.assertIn("=== Tensor Numerical Drift Report ===", report.summary())


if __name__ == "__main__":
    unittest.main()

