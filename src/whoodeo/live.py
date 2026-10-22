"""A local web page with the latest frames.

The page is served from this process. ``show`` copies the newest images and
returns; a background thread encodes PNG with Pillow. Closing the browser
does not stop the caller. Boxes passed to ``show`` (face boxes in training)
are drawn into the PNG on that thread and listed in state.json.
"""

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from whoodeo.video import png_bytes

_PAGE = (Path(__file__).with_name("live.html")).read_bytes()
_COPY_INTERVAL = 0.2
_NAME = re.compile(r"[A-Za-z0-9_-]+")
_BOX_COLOR = (255, 214, 0)
_BOX_LINE = 2


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.server.live.handle_get(self)


def bind_address(text):
    """Parse ``host:port``. Port 0 asks the OS for a free port."""
    raw = str(text)
    if raw.startswith("["):
        end = raw.find("]")
        if end < 0 or not raw[end:].startswith("]:"):
            raise SystemExit(f"live bind must be host:port, not {raw}")
        host = raw[1:end]
        port_text = raw[end + 2:]
    else:
        host, sep, port_text = raw.rpartition(":")
        if not sep:
            raise SystemExit(f"live bind must be host:port, not {raw}")
    if not host or not port_text.isdecimal():
        raise SystemExit(f"live bind must be host:port, not {raw}")
    port = int(port_text)
    if port > 65535:
        raise SystemExit(f"live bind port {port} is out of range")
    return host, port


def add_live_args(parser):
    parser.add_argument("--no-preview", action="store_true", help="do not serve the live page")
    parser.add_argument(
        "--live-bind",
        default="127.0.0.1:8765",
        help="host:port for the live page (default 127.0.0.1:8765; port 0 picks a free port)",
    )


def open_live(bind, enabled=True):
    if not enabled:
        return None
    host, port = bind_address(bind)
    return Live(host, port)


class Live:
    def __init__(self, host="127.0.0.1", port=8765):
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending = None
        self._pngs = {}
        self._boxes = {}
        self._status = ""
        self._generation = 0
        self._next_copy = 0.0
        self._reported_encode_error = False
        self._closed = False
        try:
            self._httpd = _Server((host, port), _Handler)
        except OSError as exc:
            raise SystemExit(f"cannot bind live page at {host}:{port}: {exc}") from exc
        self._httpd.live = self
        actual_port = self._httpd.server_address[1]
        self.url = f"http://{_url_host(host)}:{actual_port}/"
        self._encoder = threading.Thread(target=self._encode_loop, name="whoodeo-live-png", daemon=True)
        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="whoodeo-live",
            daemon=True,
        )
        self._encoder.start()
        self._server_thread.start()
        print(f"live {self.url}", flush=True)

    def show(self, frames, boxes=None):
        """Replace the images on the page. ``frames`` maps a name to HWC uint8 RGB.

        ``boxes`` maps a frame name to rows of x, y, w, h (and a score) in
        that frame's pixels; they are drawn as outlines. Calls closer than
        five per second are dropped, and this never waits for a browser or
        for the PNG encoder.
        """
        if not isinstance(frames, dict) or not frames:
            raise ValueError("show() needs at least one named frame")
        for name, image in frames.items():
            if not isinstance(name, str) or _NAME.fullmatch(name) is None:
                raise ValueError(f"frame name {name!r} must be letters, digits, '_' or '-'")
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or image.shape[0] < 1
                or image.shape[1] < 1
            ):
                raise ValueError(f"frame {name} must be HWC uint8 RGB")
        now = time.monotonic()
        with self._lock:
            if now < self._next_copy:
                return
            self._next_copy = now + _COPY_INTERVAL
        copied = {
            name: np.array(image, dtype=np.uint8, copy=True, order="C")
            for name, image in frames.items()
        }
        rows = {}
        for name, found in (boxes or {}).items():
            if name not in copied or found is None:
                continue
            array = np.asarray(found, dtype=np.float32)
            array = array.reshape(len(array), -1) if array.size else np.zeros((0, 4), np.float32)
            rows[name] = [[round(float(value), 1) for value in row[:5]] for row in array]
        with self._lock:
            self._pending = (copied, rows)
        self._wake.set()

    def status(self, text):
        """One line of text above the image."""
        with self._lock:
            self._status = str(text)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._server_thread.join(timeout=2)
        self._encoder.join(timeout=2)

    def handle_get(self, handler):
        path = urlparse(handler.path).path
        if path == "/":
            _send(handler, 200, _PAGE, "text/html; charset=utf-8")
            return
        if path == "/state.json":
            _send(handler, 200, json.dumps(self._state()).encode(), "application/json")
            return
        if path.startswith("/frames/") and path.endswith(".png"):
            name = unquote(path[len("/frames/"):-len(".png")])
            with self._lock:
                blob = self._pngs.get(name)
            if blob is None:
                _send(handler, 404, b"", "text/plain; charset=utf-8")
                return
            _send(handler, 200, blob, "image/png")
            return
        _send(handler, 404, b"", "text/plain; charset=utf-8")

    def _state(self):
        with self._lock:
            return {
                "status": self._status,
                "generation": self._generation,
                "frames": list(self._pngs),
                "boxes": self._boxes,
            }

    def _encode_loop(self):
        while not self._stop.is_set():
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            images, rows = pending
            try:
                for name, found in rows.items():
                    _draw_boxes(images[name], found)
                encoded = {name: png_bytes(image) for name, image in images.items()}
            except Exception as exc:
                with self._lock:
                    if not self._reported_encode_error:
                        self._reported_encode_error = True
                        print(f"live frame encode failed: {exc}", flush=True)
                continue
            with self._lock:
                self._reported_encode_error = False
                self._pngs = encoded
                self._boxes = rows
                self._generation += 1


def _draw_boxes(image, rows):
    """Outline each x, y, w, h row in place, clipped to the image."""
    height, width = image.shape[:2]
    line = _BOX_LINE
    for row in rows:
        x0 = int(np.clip(round(row[0]), 0, width - 1))
        y0 = int(np.clip(round(row[1]), 0, height - 1))
        x1 = int(np.clip(round(row[0] + row[2]), x0 + 1, width))
        y1 = int(np.clip(round(row[1] + row[3]), y0 + 1, height))
        image[y0:min(y0 + line, y1), x0:x1] = _BOX_COLOR
        image[max(y1 - line, y0):y1, x0:x1] = _BOX_COLOR
        image[y0:y1, x0:min(x0 + line, x1)] = _BOX_COLOR
        image[y0:y1, max(x1 - line, x0):x1] = _BOX_COLOR


def _url_host(host):
    if ":" in host:
        return f"[{host}]"
    return host


def _send(handler, code, body, content_type):
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)
