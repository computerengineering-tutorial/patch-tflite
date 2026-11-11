import struct
import tempfile
import unittest
from pathlib import Path

from patching_tflite import find_tflite_models_in_bytes, extract_models_from_library


class TestExtractor(unittest.TestCase):
    def test_find_tflite_in_bytes(self):
        # Create a synthetic binary buffer embedding a fake FlatBuffer with magic TFL3
        fake_binary = bytearray(b"\x90" * 1024)
        
        # Embed FlatBuffer header: [uint32 root_offset = 0x1c][magic = 'TFL3'][payload ...]
        model_payload = struct.pack("<I", 0x1c) + b"TFL3" + (b"\xaa" * 2048)
        offset = 256
        fake_binary[offset : offset + len(model_payload)] = model_payload

        found = find_tflite_models_in_bytes(bytes(fake_binary), min_size=512)
        self.assertGreaterEqual(len(found), 1)
        self.assertEqual(found[0][4:8], b"TFL3")

    def test_extract_from_dummy_library(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_so = Path(tmpdir) / "libdummy.so"
            out_dir = Path(tmpdir) / "out"

            # Create dummy ELF file with embedded TFLite model
            elf_header = b"\x7fELF" + (b"\x00" * 60)
            model_bytes = struct.pack("<I", 0x20) + b"TFL3" + (b"\x55" * 4096)
            dummy_so.write_bytes(elf_header + (b"\x00" * 100) + model_bytes)

            extracted = extract_models_from_library(dummy_so, out_dir)
            self.assertEqual(len(extracted), 1)
            self.assertTrue(extracted[0].exists())
            self.assertEqual(extracted[0].read_bytes()[4:8], b"TFL3")


if __name__ == "__main__":
    unittest.main()

