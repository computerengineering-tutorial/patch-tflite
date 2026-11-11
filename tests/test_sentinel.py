import ctypes
import unittest

from patching_tflite import MemoryHealthReport, MemorySentinel, TFLitePatcher


class TestMemorySentinel(unittest.TestCase):
    def test_memory_health_inspection(self):
        with TFLitePatcher() as patcher:
            health = patcher.get_memory_health()
            self.assertIsInstance(health, MemoryHealthReport)
            self.assertGreater(health.rss_bytes, 0)
            self.assertGreater(health.rss_mb, 0.0)
            self.assertFalse(health.leak_detected)

    def test_buffer_canary_overrun_detection(self):
        with TFLitePatcher() as patcher:
            patcher.reset_memory_sentinel()

            # Allocate 64-byte buffer
            buf = (ctypes.c_char * 64)()
            buf_addr = ctypes.addressof(buf)

            # Register canary at the end of the buffer
            slot = patcher.register_memory_canary(buf_addr, 64)
            self.assertGreaterEqual(slot, 0)

            # Initially no corruptions
            corruptions = patcher.verify_memory_canaries()
            self.assertEqual(corruptions, 0)

            # Simulate buffer overrun corrupting the last 8 bytes
            ctypes.memset(ctypes.c_void_p(buf_addr + 56), 0xFF, 8)

            # Verify detects corruption
            corruptions_after = patcher.verify_memory_canaries()
            self.assertEqual(corruptions_after, 1)

            # Reset
            patcher.reset_memory_sentinel()
            self.assertEqual(patcher.verify_memory_canaries(), 0)


if __name__ == "__main__":
    unittest.main()

