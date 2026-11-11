import unittest
from pathlib import Path

from patching_tflite import scan_library, ScanResult


class TestScanner(unittest.TestCase):
    def test_scan_mock_library(self):
        tests_dir = Path(__file__).resolve().parent
        mock_so = tests_dir / "libmock_tflite.so"

        res = scan_library(mock_so)
        self.assertIsInstance(res, ScanResult)
        self.assertTrue(res.is_valid_elf)
        self.assertTrue(res.has_c_api)
        self.assertEqual(res.c_api_invoke_symbol, "TfLiteInterpreterInvoke")
        self.assertEqual(res.recommended_symbol, "TfLiteInterpreterInvoke")
        self.assertEqual(res.recommended_mode, "symbol")

        # Test summary output
        summary = res.summary()
        self.assertIn("Valid ELF", summary)
        self.assertIn("TfLiteInterpreterInvoke", summary)


if __name__ == "__main__":
    unittest.main()

