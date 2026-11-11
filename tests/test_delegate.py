import unittest
from patching_tflite import TFLitePatcher, DelegateReport, DelegateTracker


class TestDelegateTracker(unittest.TestCase):
    def test_delegate_tracking(self):
        with TFLitePatcher() as patcher:
            patcher.reset_delegate()

            initial = patcher.get_delegate_report()
            self.assertFalse(initial.delegate_attached)

            # Record GPU delegate
            patcher.record_delegate("GPU_OpenCL_Delegate", delegated_nodes=80, cpu_fallback_nodes=20)

            report = patcher.get_delegate_report()
            self.assertTrue(report.delegate_attached)
            self.assertEqual(report.delegate_name, "GPU_OpenCL_Delegate")
            self.assertEqual(report.delegated_nodes, 80)
            self.assertEqual(report.cpu_fallback_nodes, 20)
            self.assertEqual(report.total_nodes, 100)
            self.assertAlmostEqual(report.delegation_ratio, 0.8, places=3)
            self.assertTrue(report.is_gpu_accelerated)
            self.assertFalse(report.is_npu_accelerated)

            # Record NPU delegate
            patcher.record_delegate("Hexagon_DSP_Delegate", delegated_nodes=10, cpu_fallback_nodes=40)
            npu_report = patcher.get_delegate_report()
            self.assertTrue(npu_report.is_npu_accelerated)
            self.assertAlmostEqual(npu_report.delegation_ratio, 0.2, places=3)

            # Warnings check
            warnings = patcher.delegate_tracker.get_warnings()
            self.assertTrue(any("Low hardware delegation" in w for w in warnings))

            patcher.reset_delegate()
            self.assertFalse(patcher.get_delegate_report().delegate_attached)


if __name__ == "__main__":
    unittest.main()

