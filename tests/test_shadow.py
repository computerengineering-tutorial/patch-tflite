import numpy as np
import unittest

from patching_tflite import ShadowComparisonResult, ShadowEngine, TFLitePatcher


class TestShadowEngine(unittest.TestCase):
    def test_identical_shadow_comparison(self):
        engine = ShadowEngine(
            reference_model=lambda inputs: [x * 2.0 for x in inputs],
            divergence_cosine_threshold=0.999,
        )

        in_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        so_out = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

        results = engine.verify_outputs([so_out], [in_arr], call_id=1)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertAlmostEqual(r.cosine_sim, 1.0, places=4)
        self.assertAlmostEqual(r.mae, 0.0, places=5)
        self.assertTrue(r.top1_match)
        self.assertFalse(r.has_divergence)
        self.assertEqual(engine.get_divergence_count(), 0)

    def test_divergent_shadow_comparison(self):
        engine = ShadowEngine(
            reference_model=lambda inputs: [x * 2.0 for x in inputs],
            divergence_cosine_threshold=0.999,
            divergence_mae_threshold=0.01,
        )

        in_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # Drastically divergent output from corrupted / mutant .so
        so_out = np.array([2.0, 4.0, 0.0, -8.0], dtype=np.float32)

        results = engine.verify_outputs([so_out], [in_arr], call_id=2)
        r = results[0]
        self.assertTrue(r.has_divergence)
        self.assertLess(r.cosine_sim, 0.999)
        self.assertEqual(engine.get_divergence_count(), 1)


if __name__ == "__main__":
    unittest.main()

