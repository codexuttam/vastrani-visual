"""
intent/api.py

Phase 10: Optional local HTTP API (stdlib only, disabled by default).
Binds to 127.0.0.1 by default. Enable with INTENT_API_ENABLED=true.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from intent.service import IntentService

MAX_BODY_BYTES = 8 * 1024


def make_handler(service: IntentService):
    post_routes = {
        "/api/intent/parse": service.parse,
        "/api/intent/execute": service.execute,
        "/api/intent/confirm": service.confirm,
        "/api/intent/cancel": service.cancel,
    }

    class IntentHandler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/intent/schema":
                return self._send(200, service.schema())
            self._send(404, {"success": False, "message": "Not found."})

        def do_POST(self):
            route = post_routes.get(self.path)
            if route is None:
                return self._send(404, {"success": False, "message": "Not found."})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                return self._send(413, {"success": False, "message": "Request too large."})
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._send(400, {"success": False, "status": "BAD_REQUEST", "message": "Malformed JSON."})
            result = route(payload)
            self._send(400 if result.get("status") == "BAD_REQUEST" else 200, result)

        def log_message(self, *args):  # keep stdout clean; engine logs structurally
            pass

    return IntentHandler


class IntentAPIServer:
    def __init__(self, service: IntentService, host: str = "127.0.0.1", port: int = 8765):
        self.httpd = ThreadingHTTPServer((host, port), make_handler(service))
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self):
        return self.httpd.server_address

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
