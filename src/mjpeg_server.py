# TEMPORARY: a quick web video server to compare its latency with Lucy's rosbridge STREAM.
# It will be deleted, and it is not meant to move into Lucy, now or later.
# Remove this file and every spot marked TEMPORARY in src/head_master.py.

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

HOST = "127.0.0.1"  # a live camera feed of a person stays on this machine
PORT = 8090
WIDTH = 960
JPEG_QUALITY = 80
FRAME_WAIT_SEC = 2.0


class MjpegServer:
    """Serves the latest coach frame as MJPEG at http://127.0.0.1:8090/; encodes only while someone watches."""

    def __init__(self, host: str = HOST, port: int = PORT):
        self.frame = None
        self.count = 0
        self.changed = threading.Condition()
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"TEMPORARY MJPEG stream at http://{host}:{port}/")

    def push(self, frame) -> None:
        with self.changed:
            self.frame = frame
            self.count += 1
            self.changed.notify_all()

    def next_jpeg(self, seen: int) -> tuple[bytes | None, int]:
        with self.changed:
            self.changed.wait_for(lambda: self.count != seen, timeout=FRAME_WAIT_SEC)
            frame, count = self.frame, self.count
        if frame is None:
            return None, count
        scale = WIDTH / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return (jpeg.tobytes() if ok else None), count

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ("/", "/stream.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        jpeg, seen = server.next_jpeg(seen)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                        self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, *args):
                pass

        return Handler
