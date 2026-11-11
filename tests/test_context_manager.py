import unittest

from patching_tflite import TFLitePatcher, PatcherConfig


class TestContextManager(unittest.TestCase):
    def test_context_manager_cleanup(self):
        with TFLitePatcher() as patcher:
            self.assertFalse(patcher.is_installed())

        # Outside the context manager, hooks should be uninstalled
        self.assertFalse(patcher.is_installed())


if __name__ == "__main__":
    unittest.main()

