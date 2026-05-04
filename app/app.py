"""
app.py — Flask Application (Performance-Optimized)
============================
KEY ARCHITECTURE DECISIONS (unchanged):

  SINGLE PROCESSING THREAD
  ─────────────────────────
  All MediaPipe calls happen in _processing_thread(), which is the ONLY
  consumer of the camera queue.  The Flask generator thread only reads
  pre-processed results from _latest_result.

  FRAME QUEUE (maxsize=1)
  ───────────────────────
  WebcamStream puts frames into a Queue(maxsize=1), always keeping the
  latest frame and discarding stale ones.

  FLASK GENERATOR (read-only)
  ────────────────────────────
  generate_frames() reads from _latest_result, never calls process_frame().

PERFORMANCE OPTIMIZATIONS APPLIED:
  [OPT-A] Target FPS raised from 25 → 30 for smoother display; the per-frame
          cost drop from detection.py optimizations creates headroom for this.
  [OPT-B] MJPEG encode quality lowered to 70 (was implicitly 95 in most
          cv2.imencode defaults) — reduces encode time and network payload
          with no visible difference on a local stream.
  [OPT-C] Processing thread skips every other frame (PROC_SKIP=2) when
          the queue is consistently full, indicating the camera produces
          frames faster than we can process them.  Metrics and annotation
          from the previous frame are served instead.
  [OPT-D] WebcamStream warm-up loop reduced from 10 → 5 reads; 10 was
          unnecessarily long for most cameras.
  [OPT-E] Camera buffer size already set to 1; added explicit FourCC hint
          (MJPG) to request hardware-compressed frames from the driver,
          reducing USB bandwidth and V4L2 copy overhead.
  [OPT-F] resize_frame call in the processing thread removed — downscaling
          is now handled inside DrowsinessDetector.process_frame() via the
          PROC_WIDTH/PROC_HEIGHT constants, keeping the full-res frame
          available for annotation without a redundant resize step here.
"""

import cv2
import time
import queue
import threading
import numpy as np
from pathlib import Path
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "drowsy_model.pth"

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Detector ──────────────────────────────────────────────────
from detection import DrowsinessDetector
from utils import mjpeg_response, resize_frame

detector = DrowsinessDetector(model_path=str(MODEL_PATH))

# ── Shared state ──────────────────────────────────────────────
_metrics_lock  = threading.Lock()
latest_metrics = {
    "status":          "FOCUSED",
    "ear":             0.0,
    "yaw":             0.0,
    "eye_state":       "unknown",
    "eye_conf":        0.0,
    "attention_score": 100,
    "face_detected":   False,
    "fps":             0.0,
    "timestamp":       time.time(),
}

_result_lock   = threading.Lock()
_latest_result = {
    "annotated_frame": np.zeros((480, 640, 3), dtype=np.uint8),
}
_frame_ready = threading.Event()

# [OPT-B] JPEG encode quality — 70 is visually indistinguishable at 640×480
# on a local network stream but reduces imencode + I/O cost noticeably.
_JPEG_QUALITY = 70


# ── WebcamStream ──────────────────────────────────────────────
class WebcamStream:
    """
    Captures camera frames in a dedicated thread and exposes them via
    a Queue(maxsize=1).
    """

    def __init__(self, src: int = 0):
        self._cap = cv2.VideoCapture(src)

        # [OPT-E] Request MJPEG from camera driver — many webcams support it
        # and it reduces raw USB bandwidth significantly vs. uncompressed YUY2.
        # Falls back gracefully if the camera doesn't support it.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # [OPT-D] Reduced warm-up from 10 → 5 reads; sufficient for most
        # cameras to stabilise auto-exposure without wasting startup time.
        for _ in range(5):
            self._cap.read()

        if not self._cap.isOpened():
            print("❌ Camera NOT opened")
        else:
            print("✅ Camera opened")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Keep driver buffer at 1 — reduces capture latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # maxsize=1: always latest frame
        self.queue    = queue.Queue(maxsize=1)
        self._running = True

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()

            if not ret or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue

            # Drop old frame if queue is full — always keeps latest
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put_nowait(frame)

    @property
    def is_ready(self) -> bool:
        return not self.queue.empty()

    def release(self):
        self._running = False
        self._cap.release()


cam = WebcamStream(src=0)


# ── Processing thread ─────────────────────────────────────────
# [OPT-A] Raised target FPS from 25 → 30.  The detector's internal
#   downscaling + metric-skipping frees enough CPU budget for this.
_TARGET_FPS     = 30
_FRAME_INTERVAL = 1.0 / _TARGET_FPS

# [OPT-C] Frame-skip ratio for the processing thread: when the camera
#   queue is consistently full (camera outruns processing), skip every
#   other detection pass and re-publish the previous annotated frame.
#   This keeps the video stream fluid without stalling the capture thread.
PROC_SKIP        = 2   # process 1 out of every PROC_SKIP frames
_proc_skip_count = 0


def _processing_thread():
    """
    THE ONLY THREAD THAT CALLS detector.process_frame().
    Architecture unchanged; processing cadence optimised.
    """
    global latest_metrics, _proc_skip_count

    last_process_t = 0.0

    while True:
        # Rate-limit: wait until next processing slot
        now   = time.monotonic()
        sleep = _FRAME_INTERVAL - (now - last_process_t)
        if sleep > 0:
            time.sleep(sleep)

        # Get the latest frame; wait up to 0.1 s before looping
        try:
            frame = cam.queue.get(timeout=0.1)
        except queue.Empty:
            continue

        last_process_t = time.monotonic()

        # [OPT-C] If the camera queue was full (overproducing), skip
        # every other processing cycle.  The generator will re-serve the
        # last annotated frame, keeping video smooth without extra CPU cost.
        _proc_skip_count += 1
        if cam.queue.full() and _proc_skip_count % PROC_SKIP != 0:
            # Queue still has a fresh frame — skip this cycle
            continue

        # [OPT-F] No resize_frame() here — DrowsinessDetector now handles
        # internal downscaling.  Passing the full-res frame lets the detector
        # return a full-res annotated_frame for streaming.
        try:
            result = detector.process_frame(frame)
        except Exception as e:
            print(f"❌ process_frame error: {e}")
            continue

        # Fallback: if annotated_frame is missing/empty, use raw frame
        annotated = result.get("annotated_frame")
        if annotated is None or annotated.size == 0:
            annotated = frame

        # Publish processed frame for the generator
        with _result_lock:
            _latest_result["annotated_frame"] = annotated

        # Signal generator that a new frame is ready
        _frame_ready.set()
        _frame_ready.clear()

        # Publish metrics for /metrics endpoint
        with _metrics_lock:
            latest_metrics = {
                "status":          result["status"],
                "ear":             result["ear"],
                "mar":             result["mar"],
                "yaw":             result["yaw"],
                "head_direction":  result["head_direction"],
                "eye_state":       result["eye_state"],
                "eye_conf":        result["eye_conf"],
                "attention_score": result["attention_score"],
                "face_detected":   result["face_detected"],
                "fps":             round(result["fps"], 1),
                "timestamp":       time.time(),
            }


_proc_thread = threading.Thread(target=_processing_thread, daemon=True)
_proc_thread.start()


# ── Flask generator ───────────────────────────────────────────
def generate_frames():
    """
    Reads annotated frames from _latest_result and yields MJPEG chunks.
    Never calls detector.process_frame() directly.
    """
    # Wait for the first frame to be ready (up to 5 s)
    _frame_ready.wait(timeout=5.0)

    # [OPT-B] Pre-build the encode params list once instead of per-frame
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]

    while True:
        _frame_ready.wait(timeout=0.01)

        with _result_lock:
            frame = _latest_result["annotated_frame"]

        if frame is None or frame.size == 0:
            continue

        # [OPT-B] Use explicit quality param for faster, smaller JPEG encode
        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/metrics")
def metrics():
    with _metrics_lock:
        return jsonify(dict(latest_metrics))

@app.route("/logs")
def logs():
    events = detector.get_events()
    return jsonify({"events": events[-50:], "total": len(events)})

@app.route("/summary")
def summary():
    return jsonify(detector.get_summary())

@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": detector.model_loaded,
        "camera_ok":    cam.is_ready,
        "uptime":       round(time.time() - app.start_time, 1),
    })

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.start_time = time.time()
    print("=" * 60)
    print("🚀 Starting Drowsiness Detection System")
    print("👉 Open: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)