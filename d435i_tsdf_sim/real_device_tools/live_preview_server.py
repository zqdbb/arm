#!/usr/bin/env python3
"""Read-only D435i RGB-D preview served as a small MJPEG web page."""

import argparse
import json
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs


class PreviewState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = {"ready": False}
        self.stopping = threading.Event()

    def update(self, image, status):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.status = status

    def snapshot(self):
        with self.lock:
            return self.jpeg, dict(self.status)


def capture_loop(state, width, height, fps):
    reconnects = 0
    while not state.stopping.is_set():
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        started = False
        try:
            profile = pipeline.start(config)
            started = True
            align = rs.align(rs.stream.color)
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            serial = profile.get_device().get_info(rs.camera_info.serial_number)
            usb = profile.get_device().get_info(rs.camera_info.usb_type_descriptor)
            for _ in range(fps):
                pipeline.wait_for_frames(5000)
            while not state.stopping.is_set():
                frames = align.process(pipeline.wait_for_frames(5000))
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color = np.asanyarray(color_frame.get_data()).copy()
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_m = depth_raw.astype(np.float32) * scale
                valid = depth_raw > 0

                # Central guide: the reconstruction target should occupy most of this box.
                x0, x1 = int(width * 0.22), int(width * 0.78)
                y0, y1 = int(height * 0.14), int(height * 0.86)
                cv2.rectangle(color, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.drawMarker(
                    color,
                    (width // 2, height // 2),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    28,
                    2,
                )

                center = depth_m[height // 2 - 5:height // 2 + 6,
                                 width // 2 - 5:width // 2 + 6]
                center = center[center > 0]
                center_m = float(np.median(center)) if center.size else None
                valid_percent = float(valid.mean() * 100.0)

                message = "Place target inside yellow box; clear foreground/background"
                cv2.rectangle(color, (0, 0), (width, 58), (0, 0, 0), -1)
                cv2.putText(color, message, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (0, 255, 255), 1, cv2.LINE_AA)
                depth_text = "center: --" if center_m is None else f"center: {center_m:.3f} m"
                cv2.putText(color, f"{depth_text}  valid: {valid_percent:.1f}%",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 1, cv2.LINE_AA)

                depth_vis = np.clip(depth_m / 2.0 * 255.0, 0, 255).astype(np.uint8)
                depth_vis[~valid] = 0
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                depth_color[~valid] = 0
                cv2.rectangle(depth_color, (x0, y0), (x1, y1), (255, 255, 255), 2)
                cv2.putText(depth_color, "Aligned depth: 0-2 m", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1,
                            cv2.LINE_AA)

                combined = np.hstack([color, depth_color])
                state.update(combined, {
                    "ready": True,
                    "serial": serial,
                    "usb": usb,
                    "resolution": [width, height],
                    "fps": fps,
                    "center_depth_m": center_m,
                    "valid_depth_percent": valid_percent,
                    "reconnect_count": reconnects,
                    "updated_unix": time.time(),
                })
        except Exception as exc:
            reconnects += 1
            _, old = state.snapshot()
            old.update({
                "ready": False,
                "error": repr(exc),
                "reconnect_count": reconnects,
                "reconnecting": True,
            })
            with state.lock:
                state.status = old
        finally:
            if started:
                try:
                    pipeline.stop()
                except Exception:
                    pass
        if not state.stopping.wait(1.0):
            continue


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>D435i RGB-D Live Preview</title>
<style>
body{margin:0;background:#101216;color:#eee;font-family:system-ui,sans-serif;text-align:center}
h2{margin:14px 0 4px}.note{color:#ffd54f;margin:0 0 12px}
img{width:min(98vw,1280px);height:auto;border:1px solid #444;background:#000}
#status{font-family:monospace;margin:10px;color:#8fe388}
</style></head><body>
<h2>D435i 实时取景</h2>
<p class="note">把家具放入黄色框并尽量占满，移走前景遮挡物；此页面不会控制机械臂或转台。</p>
<img src="/stream.mjpg" alt="D435i preview">
<div id="status">连接中...</div>
<script>setInterval(async()=>{try{const r=await fetch('/status.json',{cache:'no-store'});const s=await r.json();document.getElementById('status').textContent=JSON.stringify(s)}catch(e){}},1000)</script>
</body></html>""".encode("utf-8")


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(HTML)))
                self.end_headers()
                self.wfile.write(HTML)
                return
            if self.path == "/status.json":
                _, status = state.snapshot()
                body = json.dumps(status, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/snapshot.jpg":
                jpeg, _ = state.snapshot()
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while not state.stopping.is_set():
                        jpeg, _ = state.snapshot()
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 15.0)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    state = PreviewState()
    worker = threading.Thread(
        target=capture_loop,
        args=(state, args.width, args.height, args.fps),
        daemon=True,
    )
    worker.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

    def stop(*_):
        state.stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    finally:
        state.stopping.set()
        worker.join(timeout=3)
        server.server_close()


if __name__ == "__main__":
    main()
