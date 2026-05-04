"""
app.py — Flask Application
============================
KEY ARCHITECTURE DECISIONS:

  SINGLE PROCESSING THREAD
  ─────────────────────────
  All MediaPipe calls happen in _processing_thread(), which is the ONLY
  consumer of the camera queue.  The Flask generator thread only reads
  pre-processed results from _latest_result.  This eliminates the root
  cause of "Packet timestamp mismatch": MediaPipe is called from exactly
  one OS thread, in strict sequential order, with no concurrent access.

  FRAME QUEUE (maxsize=1)
  ───────────────────────
  WebcamStream puts frames into a Queue(maxsize=1).  If the queue is full,
  the old frame is discarded and the new one takes its place.  This means
  the processing thread always gets the LATEST frame, never a stale one,
  and never blocks the camera capture thread.

  FLASK GENERATOR (read-only)
  ────────────────────────────
  generate_frames() reads from _latest_result (a threading.Event + dict
  protected by a RLock).  It never calls detector.process_frame() directly.
  Multiple Flask client connections therefore share the same processed
  result without triggering duplicate MediaPipe calls.
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
# _metrics_lock protects latest_metrics (read by /metrics, written by
# the processing thread).
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

# _result_lock protects _latest_result (read by generator, written by
# the processing thread).
_result_lock   = threading.Lock()
_latest_result = {
    "annotated_frame": np.zeros((480, 640, 3), dtype=np.uint8),
}
# Event lets the generator block cheaply until a new frame is ready
# instead of busy-polling.
_frame_ready = threading.Event()


# ── WebcamStream ──────────────────────────────────────────────
class WebcamStream:
    """
    Captures camera frames in a dedicated thread and exposes them via
    a Queue(maxsize=1).

    Queue semantics:
      - put_nowait() with a full queue raises queue.Full; we catch it,
        discard the stale frame, and put the new one in.  This guarantees
        the processing thread always receives the most recent frame.
      - The processing thread calls get() with a timeout so it never
        blocks forever if the camera stalls.
    """

    def __init__(self, src: int = 0):
        self._cap = cv2.VideoCapture(src)

    # 🔥 ADD THIS BLOCK (camera warm-up)
        for _ in range(10):
            self._cap.read()

        if not self._cap.isOpened():
            print("❌ Camera NOT opened")
        else:
            print("✅ Camera opened")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # keep driver buffer at 1 — reduces latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # maxsize=1: always latest frame
        self.queue = queue.Queue(maxsize=1)
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

        # 🔥 Drop old frame if queue is full
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
# Target processing rate.  25 FPS keeps MediaPipe happy and leaves
# headroom for CNN inference without overloading the CPU.
_TARGET_FPS      = 25
_FRAME_INTERVAL  = 1.0 / _TARGET_FPS


def _processing_thread():
    """
    THE ONLY THREAD THAT CALLS detector.process_frame().

    This is the architectural fix for MediaPipe timestamp errors:
      - One thread, strictly sequential calls, no concurrency.
      - Frames arrive via cam.queue at camera FPS; we rate-limit
        processing to _TARGET_FPS to avoid flooding MediaPipe.
      - The processed result is stored in _latest_result for the
        Flask generator to read asynchronously.
    """
    global latest_metrics

    last_process_t = 0.0

    while True:
        # Rate-limit: wait until the next processing slot
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

        # Resize before processing (keeps MediaPipe input size consistent)
        small = resize_frame(frame, max_width=640)

        try:
            result = detector.process_frame(small)
        except Exception as e:
            print(f"❌ process_frame error: {e}")
            continue

        # Fallback: if annotated_frame is missing/empty, use raw resized frame
        annotated = result.get("annotated_frame")
        if annotated is None or annotated.size == 0:
            annotated = small

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

    This function runs in the Flask request thread.  It NEVER calls
    detector.process_frame() — it only reads pre-processed frames.

    _frame_ready.wait(timeout) provides back-pressure: if the processing
    thread is slow, the generator waits instead of hammering the lock.
    """
    # Wait for the first frame to be ready (up to 5 s)
    _frame_ready.wait(timeout=5.0)

    while True:
        _frame_ready.wait(timeout=0.01)

        with _result_lock:
            frame = _latest_result["annotated_frame"]

        if frame is None or frame.size == 0:
            continue

        yield mjpeg_response(frame)


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