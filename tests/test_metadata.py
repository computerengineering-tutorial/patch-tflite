import unittest
from pathlib import Path

from patching_tflite import TFLitePatcher, parse_tflite_metadata, ModelMetadata


class TestMetadata(unittest.TestCase):
    def test_manual_metadata_registration(self):
        patcher = TFLitePatcher()

        ok = patcher.register_model(
            name="test_custom_net",
            tensor_count=10,
            inputs=[0, 1],
            outputs=[9],
            input_names=["input_image", "input_scale"],
            output_names=["output_logits"],
        )
        self.assertTrue(ok)
        patcher.clear_models()

    def test_parse_tflite_file_if_available(self):
        # Look for any .tflite file in workspace to verify parser
        repo_root = Path(__file__).resolve().parent.parent.parent
        tflite_files = list(repo_root.glob("**/*.tflite"))

        if tflite_files:
            sample_model = tflite_files[0]
            meta = parse_tflite_metadata(sample_model)
            self.assertIsNotNone(meta)
            self.assertIsInstance(meta, ModelMetadata)
            self.assertGreater(meta.tensor_count, 0)
            self.assertGreater(len(meta.inputs), 0)
            self.assertGreater(len(meta.outputs), 0)


if __name__ == "__main__":
    unittest.main()

