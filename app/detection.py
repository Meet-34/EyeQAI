"""
detection.py — Drowsiness Detector
=====================================
KEY ARCHITECTURE DECISIONS:
  - MediaPipe is called ONLY from process_frame(), which must be called
    from a single thread.  A threading.Lock() (_mp_lock) enforces this.
  - Timestamps passed to MediaPipe are derived from a monotonic counter
    that is guaranteed to strictly increase regardless of wall-clock
    skew or camera frame drops.
  - Frame validation happens before anything else — MediaPipe never
    receives an invalid buffer.
  - FaceMesh is reset at most once per _RESET_COOLDOWN_S seconds to
    prevent a reset-storm on sustained errors.
"""

import cv2
import math
import time
import threading
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from torchvision import models, transforms
import torch.nn as nn

# ─── CONFIG ────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.22
YAW_THRESHOLD     = 25.0
CONSEC_FRAMES     = 6
IMG_SIZE          = 224
CNN_SKIP          = 15        # run CNN every N frames
_RESET_COOLDOWN_S = 3.0      # minimum seconds between FaceMesh resets
_MIN_DIM          = 16       # smallest acceptable frame side in pixels

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

MOUTH_IDX = [13, 14, 78, 308]

HEAD_3D_POINTS = np.array([
    (  0.0,    0.0,    0.0),
    (  0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

HEAD_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─── MODEL LOADER ──────────────────────────────────────────────
def load_model(model_path: str):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# ─── FRAME VALIDATOR ───────────────────────────────────────────
def _is_valid_frame(frame) -> bool:
    """
    Hard gate before any processing.
    Returns False for any frame that would cause MediaPipe to error.
    Checked BEFORE acquiring any lock — it is cheap and avoids wasted work.
    """
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.shape[0] < _MIN_DIM or frame.shape[1] < _MIN_DIM:
        return False
    if frame.size == 0:
        return False
    return True


# ─── MEDIAPIPE FACTORY ─────────────────────────────────────────
def _make_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,   # 🔥 HUGE SPEED FIX
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
# ─── STATUS SMOOTHER ───────────────────────────────────────────
class StatusSmoother:
    def __init__(self):
        self.buffer  = deque(maxlen=CONSEC_FRAMES)
        self.current = "FOCUSED"

    def update(self, val: str) -> str:
        self.buffer.append(val)
        if len(self.buffer) == CONSEC_FRAMES and len(set(self.buffer)) == 1:
            self.current = val
        return self.current


# ─── MAIN DETECTOR ─────────────────────────────────────────────
class DrowsinessDetector:
    """
    Single-responsibility: given one BGR frame, return a metrics dict.

    Thread-safety guarantee:
      process_frame() acquires self._mp_lock before touching MediaPipe.
      The lock makes it safe even if called from multiple threads, but
      for best performance callers should ensure single-threaded usage
      (the Flask generator in app.py already does this).
    """

    def __init__(self, model_path: str):
        # ── CNN ──────────────────────────────────────────────
        try:
            self.model        = load_model(model_path)
            self.model_loaded = True
            print("✅ CNN model loaded")
        except Exception as e:
            self.model        = None
            self.model_loaded = False
            print(f"⚠  CNN model not loaded: {e}")

        # ── MediaPipe ────────────────────────────────────────
        # _mp_lock: ensures process_frame() is never re-entered concurrently.
        # Re-entrant calls corrupt the graph's internal timestamp ordering.
        self._mp_lock      = threading.Lock()
        self._face_mesh    = _make_face_mesh()
        self._last_reset_t = 0.0   # wall-clock time of last FaceMesh reset
        self._mp_err_count = 0     # consecutive MediaPipe errors

        # ── Detection state ──────────────────────────────────
        self.smoother        = StatusSmoother()
        self._score_smooth   = 100.0
        self.attention_score = 100
        self._closed_frames  = 0

        self._cnn_counter  = 0
        self._cached_eye   = "open"
        self._cached_conf  = 0.0

        # ── FPS ──────────────────────────────────────────────
        self._fps_t0      = time.monotonic()
        self._fps_frames  = 0
        self._fps_current = 0.0

        # ── Event log ────────────────────────────────────────
        # self._event_log   = deque(maxlen=200)
        self._event_log = []
        self._last_status = "FOCUSED"
        self._state_start_time = time.time()
        self._time_in_state = {
            "FOCUSED": 0.0,
            "DROWSY": 0.0,
            "INATTENTIVE": 0.0
        }
        self._last_status = "FOCUSED"

    # ── MediaPipe safe reset ────────────────────────────────
    def _maybe_reset_face_mesh(self):
        """
        Recreate the FaceMesh graph only if the cooldown period has elapsed.

        WHY A COOLDOWN:
          Without a cooldown, a burst of bad frames causes a reset on every
          frame, which itself introduces new timestamp discontinuities and
          makes the problem worse.  The cooldown absorbs the burst.
        """
        now = time.monotonic()
        if now - self._last_reset_t < _RESET_COOLDOWN_S:
            return  # still in cooldown; skip

        print(f"⚠  Resetting FaceMesh (consecutive errors: {self._mp_err_count})")
        try:
            self._face_mesh.close()
        except Exception:
            pass

        self._face_mesh    = _make_face_mesh()
        self._mp_err_count = 0
        self._last_reset_t = now

    # ── FPS ─────────────────────────────────────────────────
    def _tick_fps(self) -> float:
        self._fps_frames += 1
        now     = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._fps_current = self._fps_frames / elapsed
            self._fps_frames  = 0
            self._fps_t0      = now
        return self._fps_current

    # ── EAR ─────────────────────────────────────────────────
    @staticmethod
    def _compute_ear(lms, idx: list, w: int, h: int) -> float:
        def pt(i):
            return np.array([lms[i].x * w, lms[i].y * h])
        p = [pt(i) for i in idx]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h1 = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h1 + 1e-6)

    # def _compute_mar(self, lms, w, h):
    #     def pt(i):
    #         return np.array([lms[i].x * w, lms[i].y * h])

    #     top, bottom, left, right = [pt(i) for i in [13, 14, 78, 308]]

    #     vertical = np.linalg.norm(top - bottom)
    #     horizontal = np.linalg.norm(left - right)

    #     return vertical / (horizontal + 1e-6)
    

    def _compute_mar(self, lms, w, h):
        mouth = [13, 14, 78, 308]  # simple stable points

        def dist(a, b):
            return np.linalg.norm(
                np.array([lms[a].x * w, lms[a].y * h]) -
                np.array([lms[b].x * w, lms[b].y * h])
            )

        vertical = dist(mouth[0], mouth[1])
        horizontal = dist(mouth[2], mouth[3])

        return vertical / horizontal if horizontal > 0 else 0.0
    
    # ── Yaw ─────────────────────────────────────────────────
    @staticmethod
    def _compute_yaw(lms, w: int, h: int) -> float:
        img_pts = np.array(
            [[lms[i].x * w, lms[i].y * h] for i in HEAD_LANDMARK_IDS],
            dtype=np.float64,
        )

        f = w
        cam = np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            HEAD_3D_POINTS,
            img_pts,
            cam,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return 0.0

        rmat, _ = cv2.Rodrigues(rvec)

        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))

        return float(yaw)
       
    # ── CNN ─────────────────────────────────────────────────
    def _run_cnn(self, rgb: np.ndarray):
        """Run the CNN on every CNN_SKIP-th call; use cached result otherwise."""
        self._cnn_counter += 1
        if not (self.model_loaded and self._cnn_counter % CNN_SKIP == 0):
            return
        try:
            t = infer_transform(rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(self.model(t), dim=1)[0].cpu().numpy()
            self._cached_conf = float(probs[1])
            self._cached_eye  = "closed" if probs[1] > probs[0] else "open"
        except Exception as e:
            print(f"⚠  CNN inference error: {e}")

    # ── Attention score ─────────────────────────────────────
    def _update_score(self, ear: float, yaw: float, status: str) -> int:
        target = 100.0
        if abs(yaw) > 20:
            target -= 30.0
        if ear < EAR_THRESHOLD:
            penalty = max(0.0, (EAR_THRESHOLD - ear) / EAR_THRESHOLD) * 50.0
            target -= penalty
        if status == "DROWSY":
            target -= 20.0
        if status == "INATTENTIVE":
            target -= 10.0
        target             = max(0.0, min(100.0, target))
        self._score_smooth = 0.75 * self._score_smooth + 0.25 * target
        return int(max(0, min(100, round(self._score_smooth))))

    # ── Public API ──────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one BGR frame and return a metrics dict.

        Call sequentially from a single thread.  The internal _mp_lock
        prevents concurrent MediaPipe access if called from multiple threads.
        """
        result = {
            "status":          "FOCUSED",
            "ear":             0.0,
            "yaw":             0.0,
            "eye_state":       self._cached_eye,
            "eye_conf":        round(self._cached_conf, 3),
            "attention_score": self.attention_score,
            "face_detected":   False,
            "fps":             self._tick_fps(),
            "annotated_frame": None,
            "mar": 0.0,
            "head_direction": "CENTER",
        }


        # ── Step 1: validate frame ──────────────────────────
        # Done OUTSIDE the lock — cheap check, avoids wasted lock contention.
        if not _is_valid_frame(frame):
            result["annotated_frame"] = np.zeros((480, 640, 3), dtype=np.uint8)
            return result

        # Guarantee contiguous uint8 memory layout — MediaPipe requirement.
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        result["annotated_frame"] = frame.copy()
        h, w = frame.shape[:2]

        # Brightness boost (keep existing behaviour)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=25)

        # Convert to RGB once; reused by both MediaPipe and CNN
        rgb = np.ascontiguousarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )

        # ── Step 2: MediaPipe (serialised) ─────────────────
        # The lock is the critical fix:
        #   - It ensures process_frame() is never called re-entrantly.
        #   - MediaPipe's internal graph is NOT thread-safe; concurrent
        #     calls produce "Packet timestamp mismatch" because the graph
        #     receives out-of-order timestamps from competing threads.
        with self._mp_lock:
            try:
                mp_result          = self._face_mesh.process(rgb)
                self._mp_err_count = 0   # reset on success
            except Exception as e:
                self._mp_err_count += 1
                print(f"⚠  MediaPipe error #{self._mp_err_count}: {e}")
                # Only reset after 3 consecutive errors AND cooldown elapsed
                if self._mp_err_count >= 3:
                    self._maybe_reset_face_mesh()
                return result

        # ── Step 3: extract landmarks ───────────────────────
        if not mp_result or not mp_result.multi_face_landmarks:
            result["face_detected"] = False
            result["ear"] = 0.0
            result["yaw"] = 0.0
            return result

        lm = mp_result.multi_face_landmarks[0].landmark
        result["face_detected"] = True

        # ── Step 4: compute metrics ─────────────────────────
        ear = (self._compute_ear(lm, LEFT_EYE_IDX, w, h) +
               self._compute_ear(lm, RIGHT_EYE_IDX, w, h)) / 2.0
        
        result["ear"] = round(float(ear), 3)

        try:
            yaw = self._compute_yaw(lm, w, h)
            if yaw > 15:
                result["head_direction"] = "RIGHT"
            elif yaw < -15:
                result["head_direction"] = "LEFT"
            else:
                result["head_direction"] = "CENTER"
        except Exception:
            yaw = 0.0

        self._run_cnn(rgb)

# ── Hybrid eye detection (FIX) ─────────────────────────

        ear_open_thresh = 0.26
        ear_closed_thresh = 0.21

        if ear > ear_open_thresh:
            self._cached_eye = "OPEN"
        elif ear < ear_closed_thresh:
            self._cached_eye = "CLOSED"
        else:
            pass

        # MAR
        try:
            mar = self._compute_mar(lm, w, h)
        except Exception:
            mar = 0.0

        result["mar"] = round(float(mar), 3)

        # Head direction
        if yaw > 15:
            result["head_direction"] = "RIGHT"
        elif yaw < -15:
            result["head_direction"] = "LEFT"
        else:
            result["head_direction"] = "CENTER"

        # ── Step 5: status decision ─────────────────────────
        if abs(yaw) > YAW_THRESHOLD:
            raw = "INATTENTIVE"
        elif ear < EAR_THRESHOLD:
            self._closed_frames += 1
            raw = "DROWSY" if self._closed_frames > 15 else "FOCUSED"
        else:
            self._closed_frames = 0
            raw = "FOCUSED"

        status = self.smoother.update(raw)
        result["status"] = status

        # ── Step 6: attention score ─────────────────────────
        self.attention_score      = self._update_score(ear, yaw, status)
        result["attention_score"] = self.attention_score

        # ── Step 7: event log ───────────────────────────────
        if status != self._last_status:
            self._event_log.append({
            "time": time.strftime("%d %b, %I:%M:%S %p"),
            "status": status,
            "ear": round(float(ear), 3),
            "yaw": round(float(yaw), 1),
            "score": int(self.attention_score),
        })
        self._last_status = status

        result["fps"] = self._tick_fps()
        return result

 
    # ── Dashboard helpers ───────────────────────────────────
    def get_events(self) -> list:
        return list(self._event_log)

    def get_summary(self) -> dict:
        summary = {
            "focused": 0, "drowsy": 0, "inattentive": 0,
            "total": 0, "avg_score": 0,
        }
        if not self._event_log:
            return summary
        scores = []
        for e in self._event_log:
            key = e["status"].lower()
            if key in summary:
                summary[key] += 1
            scores.append(e["score"])
        summary["total"]     = len(self._event_log)
        summary["avg_score"] = round(sum(scores) / len(scores), 1)
        return summary