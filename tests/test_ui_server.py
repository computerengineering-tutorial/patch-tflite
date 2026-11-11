import json
import socket
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
import urllib.request

from patching_tflite.server import DashboardHTTPHandler, broadcast_event


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestUIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        cls.server = ThreadingHTTPServer(("127.0.0.1", cls.port), DashboardHTTPHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_status_api(self):
        url = f"http://127.0.0.1:{self.port}/api/status"
        with urllib.request.urlopen(url) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode())
            self.assertEqual(data.get("status"), "online")
            self.assertEqual(data.get("version"), "0.4.0")

    def test_get_telemetry_api(self):
        url = f"http://127.0.0.1:{self.port}/api/telemetry"
        with urllib.request.urlopen(url) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode())
            self.assertIn("op_breakdown", data)
            self.assertIn("delegate", data)
            self.assertIn("memory", data)

    def test_post_event_api(self):
        url = f"http://127.0.0.1:{self.port}/api/event"
        payload = json.dumps({
            "type": "telemetry",
            "data": {"total_calls": 999, "fps": 120.0}
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            resp_data = json.loads(response.read().decode())
            self.assertEqual(resp_data.get("status"), "ok")

        # Verify state updated
        telemetry_url = f"http://127.0.0.1:{self.port}/api/telemetry"
        with urllib.request.urlopen(telemetry_url) as response:
            data = json.loads(response.read().decode())
            self.assertEqual(data.get("total_calls"), 999)
            self.assertEqual(data.get("fps"), 120.0)

    def test_get_root_serves_html(self):
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as response:
            self.assertEqual(response.status, 200)
            html = response.read().decode("utf-8")
            self.assertIn("<html", html.lower())


if __name__ == "__main__":
    unittest.main()

