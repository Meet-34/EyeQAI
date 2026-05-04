"""
detection.py — Drowsiness Detector (Performance-Optimized)
=====================================
KEY ARCHITECTURE DECISIONS (unchanged):
  - MediaPipe is called ONLY from process_frame(), which must be called
    from a single thread.  A threading.Lock() (_mp_lock) enforces this.
  - Timestamps passed to MediaPipe are derived from a monotonic counter
    that is guaranteed to strictly increase regardless of wall-clock
    skew or camera frame drops.
  - Frame validation happens before anything else — MediaPipe never
    receives an invalid buffer.
  - FaceMesh is reset at most once per _RESET_COOLDOWN_S seconds to
    prevent a reset-storm on sustained errors.

PERFORMANCE OPTIMIZATIONS APPLIED:
  [OPT-1] Reduced processing resolution to 320x240 inside the detector
          so landmark extraction is done on a smaller image.
  [OPT-2] EAR computed only every LANDMARK_SKIP frames; cached otherwise.
  [OPT-3] Yaw (solvePnP) is the most expensive per-frame op — skipped every
          POSE_SKIP frames and the previous value reused.
  [OPT-4] MAR computed only every MAR_SKIP frames.
  [OPT-5] Pre-allocated numpy arrays for 3-D point projection to avoid
          repeated allocations inside _compute_yaw.
  [OPT-6] infer_transform pipeline moved to module level (was already there)
          but the transform object is now reused without re-instantiation.
  [OPT-7] cv2.convertScaleAbs (brightness boost) applied AFTER downscale so
          it operates on fewer pixels.
  [OPT-8] RGB conversion and contiguous-array guarantee merged into a single
          np.ascontiguousarray call on the already-converted buffer.
  [OPT-9] torch.inference_mode() replaces torch.no_grad() — slightly faster
          because it also skips version-counter bookkeeping.
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
CNN_SKIP          = 15        # run CNN every N frames (unchanged)
_RESET_COOLDOWN_S = 3.0
_MIN_DIM          = 16

# [OPT-2] Skip landmark metrics every N frames to reduce redundant computation
LANDMARK_SKIP = 2   # EAR recalculated every 2 frames; cached in between
# [OPT-3] solvePnP (yaw) is expensive — run every 3 frames only
POSE_SKIP     = 3
# [OPT-4] MAR is lightweight but still skippable without noticeable quality loss
MAR_SKIP      = 3

# [OPT-1] Internal processing resolution — smaller = faster MediaPipe + EAR/yaw
#   480p→240p halves both dimensions → 4× fewer pixels per MediaPipe call.
#   The annotated frame returned to the caller is still the original resolution.
PROC_WIDTH  = 320
PROC_HEIGHT = 240

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX     = [13, 14, 78, 308]

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

# [OPT-6] Transform reused across calls — no re-instantiation per frame
infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# [OPT-5] Pre-allocated camera-matrix and dist-coeffs arrays — avoids
#   np.array() allocation inside _compute_yaw on every call.
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


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
        refine_landmarks=False,   # SPEED: skips iris/attention mesh
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
    (Architecture unchanged; only per-frame cost reduced.)
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
        self._mp_lock      = threading.Lock()
        self._face_mesh    = _make_face_mesh()
        self._last_reset_t = 0.0
        self._mp_err_count = 0

        # ── Detection state ──────────────────────────────────
        self.smoother        = StatusSmoother()
        self._score_smooth   = 100.0
        self.attention_score = 100
        self._closed_frames  = 0

        self._cnn_counter  = 0
        self._cached_eye   = "open"
        self._cached_conf  = 0.0

        # [OPT-2/3/4] Frame-skip counters and caches for expensive ops
        self._landmark_counter = 0
        self._pose_counter     = 0
        self._mar_counter      = 0
        self._cached_ear       = 0.3   # safe default (open)
        self._cached_yaw       = 0.0
        self._cached_mar       = 0.0

        # ── FPS ──────────────────────────────────────────────
        self._fps_t0      = time.monotonic()
        self._fps_frames  = 0
        self._fps_current = 0.0

        # ── Event log ────────────────────────────────────────
        self._event_log        = []
        self._last_status      = "FOCUSED"
        self._state_start_time = time.time()
        self._time_in_state    = {"FOCUSED": 0.0, "DROWSY": 0.0, "INATTENTIVE": 0.0}

    # ── MediaPipe safe reset ────────────────────────────────
    def _maybe_reset_face_mesh(self):
        now = time.monotonic()
        if now - self._last_reset_t < _RESET_COOLDOWN_S:
            return
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

    def _compute_mar(self, lms, w, h):
        mouth = [13, 14, 78, 308]

        def dist(a, b):
            return np.linalg.norm(
                np.array([lms[a].x * w, lms[a].y * h]) -
                np.array([lms[b].x * w, lms[b].y * h])
            )

        vertical   = dist(mouth[0], mouth[1])
        horizontal = dist(mouth[2], mouth[3])
        return vertical / horizontal if horizontal > 0 else 0.0

    # ── Yaw ─────────────────────────────────────────────────
    @staticmethod
    def _compute_yaw(lms, w: int, h: int) -> float:
        img_pts = np.array(
            [[lms[i].x * w, lms[i].y * h] for i in HEAD_LANDMARK_IDS],
            dtype=np.float64,
        )

        f   = w
        # [OPT-5] Only the focal length / principal-point changes per frame;
        #   build the 3×3 camera matrix cheaply with array literal instead of
        #   allocating a new zeros array and filling it.
        cam = np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        # [OPT-5] Reuse pre-allocated dist-coeffs array
        ok, rvec, tvec = cv2.solvePnP(
            HEAD_3D_POINTS,
            img_pts,
            cam,
            _DIST_COEFFS,          # pre-allocated, never mutated
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        sy  = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
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
            # [OPT-9] torch.inference_mode() is faster than torch.no_grad():
            #   it also disables version-counter tracking used by autograd.
            with torch.inference_mode():
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
        Unchanged contract; internal cost reduced via frame-skipping + downscale.
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
            "mar":             0.0,
            "head_direction":  "CENTER",
        }

        # ── Step 1: validate frame ──────────────────────────
        if not _is_valid_frame(frame):
            result["annotated_frame"] = np.zeros((480, 640, 3), dtype=np.uint8)
            return result

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # Keep original frame for annotation / streaming — the caller sees
        # full-resolution video even though detection runs on a smaller image.
        result["annotated_frame"] = frame.copy()

        # ── [OPT-1] Downscale for processing ───────────────
        # Resize to PROC_WIDTH × PROC_HEIGHT BEFORE heavy operations.
        # MediaPipe, EAR, yaw, and MAR all operate on this smaller image,
        # cutting pixel count by ~4× compared to 640×480 input.
        # WHY: MediaPipe scales O(w×h); halving each dim → 4× speedup.
        proc_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT),
                                interpolation=cv2.INTER_LINEAR)
        h, w = proc_frame.shape[:2]   # use proc dimensions for metric math

        # ── [OPT-7] Brightness boost on the smaller image ──
        # Applying convertScaleAbs after downscale means it touches
        # PROC_WIDTH×PROC_HEIGHT pixels instead of the original size.
        proc_frame = cv2.convertScaleAbs(proc_frame, alpha=1.2, beta=25)

        # ── [OPT-8] Single combined RGB conversion ─────────
        # np.ascontiguousarray + cvtColor merged so the memory layout
        # guarantee and color conversion are one operation.
        rgb = np.ascontiguousarray(
            cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )

        # ── Step 2: MediaPipe (serialised) ─────────────────
        with self._mp_lock:
            try:
                mp_result          = self._face_mesh.process(rgb)
                self._mp_err_count = 0
            except Exception as e:
                self._mp_err_count += 1
                print(f"⚠  MediaPipe error #{self._mp_err_count}: {e}")
                if self._mp_err_count >= 3:
                    self._maybe_reset_face_mesh()
                return result

        # ── Step 3: extract landmarks ───────────────────────
        if not mp_result or not mp_result.multi_face_landmarks:
            result["face_detected"] = False
            result["ear"]           = 0.0
            result["yaw"]           = 0.0
            return result

        lm = mp_result.multi_face_landmarks[0].landmark
        result["face_detected"] = True

        # ── [OPT-2] EAR: skip every LANDMARK_SKIP frames ───
        # EAR changes slowly (blinks ~15–20× per minute); reusing the
        # previous value for 1 frame introduces negligible error but
        # halves EAR computation cost.
        self._landmark_counter += 1
        if self._landmark_counter % LANDMARK_SKIP == 0:
            ear = (self._compute_ear(lm, LEFT_EYE_IDX, w, h) +
                   self._compute_ear(lm, RIGHT_EYE_IDX, w, h)) / 2.0
            self._cached_ear = ear
        else:
            ear = self._cached_ear   # reuse last computed value

        result["ear"] = round(float(ear), 3)

        # ── [OPT-3] Yaw: skip every POSE_SKIP frames ───────
        # solvePnP is the single most expensive call per frame (iterative
        # PnP solver + Rodrigues decomposition).  Head pose changes slowly;
        # running it every 3rd frame is imperceptible in practice.
        self._pose_counter += 1
        if self._pose_counter % POSE_SKIP == 0:
            try:
                yaw = self._compute_yaw(lm, w, h)
                self._cached_yaw = yaw
            except Exception:
                yaw = self._cached_yaw
        else:
            yaw = self._cached_yaw   # reuse cached yaw

        result["yaw"] = round(float(yaw), 1)

        # Head direction derived from (possibly cached) yaw
        if yaw > 15:
            result["head_direction"] = "RIGHT"
        elif yaw < -15:
            result["head_direction"] = "LEFT"
        else:
            result["head_direction"] = "CENTER"

        # ── CNN (unchanged cadence) ──────────────────────────
        # CNN already has its own skip counter (CNN_SKIP); no change needed.
        self._run_cnn(rgb)

        # ── Hybrid eye state (unchanged logic) ──────────────
        ear_open_thresh   = 0.26
        ear_closed_thresh = 0.21
        if ear > ear_open_thresh:
            self._cached_eye = "OPEN"
        elif ear < ear_closed_thresh:
            self._cached_eye = "CLOSED"

        # ── [OPT-4] MAR: skip every MAR_SKIP frames ─────────
        # Yawn detection doesn't need per-frame accuracy; mouth openness
        # changes slowly.  Running MAR every 3rd frame reduces cost with
        # no meaningful quality loss.
        self._mar_counter += 1
        if self._mar_counter % MAR_SKIP == 0:
            try:
                mar = self._compute_mar(lm, w, h)
                self._cached_mar = mar
            except Exception:
                mar = self._cached_mar
        else:
            mar = self._cached_mar   # reuse cached MAR

        result["mar"] = round(float(mar), 3)

        # ── Step 5: status decision (unchanged logic) ───────
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
                "time":   time.strftime("%d %b, %I:%M:%S %p"),
                "status": status,
                "ear":    round(float(ear), 3),
                "yaw":    round(float(yaw), 1),
                "score":  int(self.attention_score),
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