import tempfile
import unittest
from pathlib import Path
import numpy as np

from patching_tflite import SessionRecorder, SessionPlayer


class TestSession(unittest.TestCase):
    def test_record_and_replay_session(self):
        recorder = SessionRecorder()

        # Record Call 1
        dummy_in = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dummy_out = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

        recorder.record_tensor(
            call_id=1,
            stage="PRE",
            model_name="test_model",
            subgraph_idx=0,
            tensor_idx=0,
            name="input_tensor",
            dtype="f32",
            shape=(1, 4),
            data=dummy_in,
            elapsed_ms=0.0,
        )
        recorder.record_tensor(
            call_id=1,
            stage="POST",
            model_name="test_model",
            subgraph_idx=0,
            tensor_idx=1,
            name="output_tensor",
            dtype="f32",
            shape=(1, 4),
            data=dummy_out,
            elapsed_ms=1.234,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "run.tflite_session"
            recorder.save(session_file)
            self.assertTrue(session_file.exists())

            # Load via SessionPlayer
            player = SessionPlayer(session_file)
            calls = player.get_calls()
            self.assertEqual(len(calls), 1)

            c = calls[0]
            self.assertEqual(c.call_id, 1)
            self.assertEqual(c.model_name, "test_model")
            self.assertEqual(len(c.inputs), 1)
            self.assertEqual(len(c.outputs), 1)

            # Decode tensor arrays
            in_arr = SessionPlayer.decode_tensor_data(c.inputs[0], as_numpy=True)
            self.assertTrue(np.allclose(in_arr, dummy_in))

            out_arr = SessionPlayer.decode_tensor_data(c.outputs[0], as_numpy=True)
            self.assertTrue(np.allclose(out_arr, dummy_out))


if __name__ == "__main__":
    unittest.main()

