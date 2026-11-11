from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import random
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("tflite_patch.server")

# Global telemetry state shared across server threads
_TELEMETRY_LOCK = threading.Lock()
_CLIENTS_LOCK = threading.Lock()
_SSE_CLIENTS: List[threading.Event] = []
_CLIENT_QUEUES: Dict[int, List[dict]] = {}

_CURRENT_STATE: Dict[str, Any] = {
    "connected": True,
    "model_name": "tflite_graph_opt.so",
    "total_calls": 0,
    "fps": 0.0,
    "avg_latency_ms": 0.0,
    "min_latency_ms": 0.0,
    "max_latency_ms": 0.0,
    "p50_ms": 0.0,
    "p90_ms": 0.0,
    "p99_ms": 0.0,
    "anomalies_detected": 0,
    "history": [],  # list of {time, latency_ms, fps}
    "op_breakdown": [
        {"op_name": "CONV_2D", "total_time_ms": 14.2, "percent_total": 45.2, "execution_count": 12},
        {"op_name": "DEPTHWISE_CONV_2D", "total_time_ms": 6.8, "percent_total": 21.6, "execution_count": 8},
        {"op_name": "FULLY_CONNECTED", "total_time_ms": 5.1, "percent_total": 16.2, "execution_count": 4},
        {"op_name": "RESHAPE", "total_time_ms": 2.3, "percent_total": 7.3, "execution_count": 14},
        {"op_name": "SOFTMAX", "total_time_ms": 1.8, "percent_total": 5.7, "execution_count": 2},
        {"op_name": "ADD", "total_time_ms": 1.2, "percent_total": 3.8, "execution_count": 6},
    ],
    "delegate": {
        "delegate_attached": True,
        "delegate_name": "GPU_Delegate_Vulkan",
        "delegated_nodes": 42,
        "cpu_fallback_nodes": 6,
        "total_nodes": 48,
        "delegation_percent": 87.5,
        "is_gpu_accelerated": True,
        "is_npu_accelerated": False,
    },
    "memory": {
        "rss_mb": 48.6,
        "vsize_mb": 182.4,
        "heap_bytes": 32180224,
        "canary_corruptions": 0,
        "leak_detected": False,
        "leak_rate_kb": 0.0,
    },
    "shadow": {
        "divergences": 0,
        "cosine_sim": 0.9998,
        "mae": 0.00012,
    },
    "recent_tensors": [],
    "recent_anomalies": [],
}


def broadcast_event(event_type: str, data: dict) -> None:
    """Dispatches a telemetry event to all connected SSE clients."""
    with _TELEMETRY_LOCK:
        if event_type == "telemetry":
            _CURRENT_STATE.update(data)
        elif event_type == "anomaly":
            _CURRENT_STATE["recent_anomalies"].insert(0, data)
            _CURRENT_STATE["recent_anomalies"] = _CURRENT_STATE["recent_anomalies"][:50]
        elif event_type == "tensor":
            _CURRENT_STATE["recent_tensors"].insert(0, data)
            _CURRENT_STATE["recent_tensors"] = _CURRENT_STATE["recent_tensors"][:10]

    payload = json.dumps({"type": event_type, "data": data})

    with _CLIENTS_LOCK:
        for cid, queue in list(_CLIENT_QUEUES.items()):
            queue.append(payload)


def get_current_state() -> dict:
    with _TELEMETRY_LOCK:
        return dict(_CURRENT_STATE)


def get_ui_dist_dir() -> Path:
    # Path relative to this file
    p = Path(__file__).resolve().parent.parent / "ui" / "dist"
    return p


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    server_version = "TFLitePatchDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy access logs
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._handle_api_status()
        elif path == "/api/telemetry":
            self._handle_api_telemetry()
        elif path == "/api/stream":
            self._handle_sse_stream()
        else:
            self._handle_static_file(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/event":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                event_type = payload.get("type", "telemetry")
                data = payload.get("data", {})
                broadcast_event(event_type, data)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.wfile.write(f'{{"error":"{e}"}}'.encode())
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def _handle_api_status(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        resp = {"status": "online", "time": time.time(), "version": "0.4.0"}
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def _handle_api_telemetry(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        state = get_current_state()
        self.wfile.write(json.dumps(state).encode("utf-8"))

    def _handle_sse_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        client_id = id(self)
        queue: List[str] = []

        with _CLIENTS_LOCK:
            _CLIENT_QUEUES[client_id] = queue

        # Send initial snapshot
        init_payload = json.dumps({"type": "init", "data": get_current_state()})
        try:
            self.wfile.write(f"data: {init_payload}\n\n".encode("utf-8"))
            self.wfile.flush()

            while True:
                time.sleep(0.05)
                items_to_send = []
                with _CLIENTS_LOCK:
                    if client_id in _CLIENT_QUEUES and _CLIENT_QUEUES[client_id]:
                        items_to_send = list(_CLIENT_QUEUES[client_id])
                        _CLIENT_QUEUES[client_id].clear()

                for msg in items_to_send:
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _CLIENTS_LOCK:
                _CLIENT_QUEUES.pop(client_id, None)

    def _handle_static_file(self, req_path: str) -> None:
        dist_dir = get_ui_dist_dir()

        if req_path in ("/", ""):
            file_path = dist_dir / "index.html"
        else:
            rel = req_path.lstrip("/")
            file_path = dist_dir / rel

        # Single page application fallback: if file doesn't exist, serve index.html
        if not file_path.exists() or file_path.is_dir():
            file_path = dist_dir / "index.html"

        if not file_path.exists():
            # If UI hasn't been built yet, show helpful fallback HTML
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            fallback_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TFLite Patch Dashboard</title>
    <style>
        body { background: #09090b; color: #f4f4f5; font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .card { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 32px; max-width: 500px; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { color: #10b981; font-size: 24px; margin-top: 0; }
        code { background: #27272a; padding: 4px 8px; border-radius: 4px; font-family: monospace; color: #38bdf8; }
    </style>
</head>
<body>
    <div class="card">
        <h1>TFLite Patch Dashboard</h1>
        <p>The React UI is building or not compiled yet.</p>
        <p>Run: <code>cd patching_tflite/ui && pnpm build</code></p>
        <p>Telemetry API is active at <code>/api/status</code> and <code>/api/telemetry</code>.</p>
    </div>
</body>
</html>"""
            self.wfile.write(fallback_html.encode("utf-8"))
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)


def run_demo_telemetry_loop(stop_event: threading.Event) -> None:
    """Generates lively mock telemetry for demo/preview purposes."""
    call_count = 0
    history = []
    base_time = time.time()

    ops_pool = [
        {"op_name": "CONV_2D", "base_ms": 14.5, "variance": 2.1},
        {"op_name": "DEPTHWISE_CONV_2D", "base_ms": 6.8, "variance": 1.2},
        {"op_name": "FULLY_CONNECTED", "base_ms": 5.2, "variance": 0.8},
        {"op_name": "RESHAPE", "base_ms": 2.1, "variance": 0.3},
        {"op_name": "SOFTMAX", "base_ms": 1.7, "variance": 0.2},
        {"op_name": "ADD", "base_ms": 1.1, "variance": 0.15},
        {"op_name": "AVERAGE_POOL_2D", "base_ms": 0.9, "variance": 0.1},
    ]

    while not stop_event.is_set():
        time.sleep(0.5)
        call_count += int(random.uniform(8, 25))
        now = time.time()

        # Latency with slight oscillation
        cycle = math.sin(now * 0.5) * 4.0
        current_latency = max(8.0, 31.5 + cycle + random.uniform(-2.5, 3.0))
        fps = 1000.0 / current_latency * random.uniform(0.85, 1.15)

        history.append({
            "timestamp": round(now - base_time, 1),
            "latency_ms": round(current_latency, 2),
            "fps": round(fps, 1),
        })
        if len(history) > 30:
            history.pop(0)

        # Per-op latency updates
        op_breakdown = []
        tot_op_time = 0.0
        for op in ops_pool:
            dur = max(0.1, op["base_ms"] + random.uniform(-op["variance"], op["variance"]))
            tot_op_time += dur
            op_breakdown.append({
                "op_name": op["op_name"],
                "total_time_ms": round(dur, 2),
                "execution_count": random.randint(4, 20),
            })
        for op in op_breakdown:
            op["percent_total"] = round((op["total_time_ms"] / tot_op_time) * 100.0, 1)
        op_breakdown.sort(key=lambda x: x["total_time_ms"], reverse=True)

        # Generate mock 2D/3D tensor heatmap
        tensor_data = []
        for r in range(16):
            row = []
            for c in range(16):
                val = math.sin(r * 0.4 + now) * math.cos(c * 0.4 - now)
                row.append(round(val, 3))
            tensor_data.append(row)

        rss_mb = 48.0 + (call_count * 0.001) % 15.0

        telemetry_update = {
            "total_calls": call_count,
            "fps": round(fps, 1),
            "avg_latency_ms": round(current_latency, 2),
            "min_latency_ms": round(current_latency * 0.8, 2),
            "max_latency_ms": round(current_latency * 1.5, 2),
            "p50_ms": round(current_latency * 0.96, 2),
            "p90_ms": round(current_latency * 1.18, 2),
            "p99_ms": round(current_latency * 1.42, 2),
            "anomalies_detected": int(call_count * 0.02),
            "history": history,
            "op_breakdown": op_breakdown,
            "memory": {
                "rss_mb": round(rss_mb, 2),
                "vsize_mb": round(rss_mb * 3.5, 2),
                "heap_bytes": int(rss_mb * 1024 * 700),
                "canary_corruptions": 0,
                "leak_detected": False,
                "leak_rate_kb": 0.0,
            },
            "delegate": {
                "delegate_attached": True,
                "delegate_name": "GPU_Vulkan_Delegate",
                "delegated_nodes": 42,
                "cpu_fallback_nodes": 6,
                "total_nodes": 48,
                "delegation_percent": 87.5,
                "is_gpu_accelerated": True,
                "is_npu_accelerated": False,
            },
            "shadow": {
                "divergences": 0,
                "cosine_sim": round(0.9995 + random.uniform(-0.0002, 0.0003), 5),
                "mae": round(0.00012 + random.uniform(-0.00003, 0.00005), 6),
            },
            "tensor_preview": {
                "tensor_idx": 1,
                "name": "Identity:0",
                "shape": [1, 16, 16, 1],
                "dtype": "f32",
                "min": -1.0,
                "max": 1.0,
                "data": tensor_data,
            },
        }

        broadcast_event("telemetry", telemetry_update)


def start_server(host: str = "0.0.0.0", port: int = 8080, demo: bool = False) -> None:
    """Launches the Web UI Dashboard HTTP and SSE Telemetry Server."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DashboardHTTPHandler)
    logger.info(f"Dashboard server started at http://{host}:{port}")
    print(f"\n=======================================================")
    print(f"  TFLite Patch Live Web UI Dashboard")
    print(f"  URL: http://localhost:{port} (or http://{host}:{port})")
    print(f"  SSE Stream: /api/stream")
    print(f"  Demo Simulation: {'ENABLED' if demo else 'DISABLED'}")
    print(f"=======================================================\n")

    stop_event = threading.Event()
    if demo:
        sim_thread = threading.Thread(target=run_demo_telemetry_loop, args=(stop_event,), daemon=True)
        sim_thread.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        stop_event.set()
        httpd.shutdown()
        httpd.server_close()

