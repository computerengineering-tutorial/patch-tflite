import unittest
from patching_tflite import TFLitePatcher, OpProfile, OpProfiler


class TestOpProfiler(unittest.TestCase):
    def test_op_recording_and_breakdown(self):
        with TFLitePatcher() as patcher:
            patcher.reset_op_profiles()

            # Record sample op executions
            patcher.record_op("CONV_2D", 10.0)
            patcher.record_op("CONV_2D", 20.0)
            patcher.record_op("DEPTHWISE_CONV", 5.0)
            patcher.record_op("SOFTMAX", 1.0)

            breakdown = patcher.get_op_breakdown()
            self.assertEqual(len(breakdown), 3)

            # CONV_2D should be top op
            top_op = breakdown[0]
            self.assertEqual(top_op.op_name, "CONV_2D")
            self.assertEqual(top_op.execution_count, 2)
            self.assertAlmostEqual(top_op.total_time_ms, 30.0, places=2)
            self.assertAlmostEqual(top_op.min_time_ms, 10.0, places=2)
            self.assertAlmostEqual(top_op.max_time_ms, 20.0, places=2)
            self.assertAlmostEqual(top_op.avg_time_ms, 15.0, places=2)

            # Total is 36ms, CONV_2D is 30/36 = 83.33%
            self.assertGreater(top_op.percent_total, 80.0)

            # Bottleneck detection (>15%)
            bottlenecks = patcher.find_bottlenecks(top_n=2, threshold_pct=15.0)
            self.assertIn("CONV_2D", [b.op_name for b in bottlenecks])

            patcher.reset_op_profiles()
            self.assertEqual(len(patcher.get_op_breakdown()), 0)


if __name__ == "__main__":
    unittest.main()

