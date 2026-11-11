import json
import tempfile
import unittest
from pathlib import Path

from patching_tflite import export_perfetto_trace


class TestPerfetto(unittest.TestCase):
    def test_export_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_log = Path(tmpdir) / "trace.log"
            output_json = Path(tmpdir) / "trace.perfetto.json"

            # Write mock trace log lines
            trace_log.write_text(
                "[1792928104.100000] [INVOKE] call=1 start model=m001 name=mock_model\n"
                "[1792928104.100100] [TENSOR] call=1 PRE input t0 type=f32 shape=[1x4] bytes=16\n"
                "[1792928104.100500] [INVOKE] call=1 end model=m001 name=mock_model result=0 elapsed_ms=0.400\n"
            )

            res = export_perfetto_trace(trace_log, output_json)
            self.assertTrue(res.exists())

            data = json.loads(res.read_text())
            self.assertIn("traceEvents", data)
            events = data["traceEvents"]
            self.assertGreaterEqual(len(events), 1)

            invoke_events = [e for e in events if "Invoke:" in e["name"]]
            self.assertEqual(len(invoke_events), 1)
            self.assertEqual(invoke_events[0]["ph"], "X")
            self.assertEqual(invoke_events[0]["args"]["name"], "mock_model")


if __name__ == "__main__":
    unittest.main()

