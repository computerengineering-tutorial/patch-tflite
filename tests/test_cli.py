import subprocess
import sys
import unittest
from pathlib import Path


class TestCLI(unittest.TestCase):
    def test_cli_scan(self):
        tests_dir = Path(__file__).resolve().parent
        mock_so = tests_dir / "libmock_tflite.so"
        pkg_root = tests_dir.parent

        proc = subprocess.run(
            [sys.executable, "-m", "patching_tflite.cli", "scan", str(mock_so)],
            capture_output=True,
            text=True,
            cwd=str(pkg_root),
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Valid ELF:            True", proc.stdout)
        self.assertIn("TfLiteInterpreterInvoke", proc.stdout)


if __name__ == "__main__":
    unittest.main()

