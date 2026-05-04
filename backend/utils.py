"""
utils.py — Shared Utilities
============================
"""
import cv2
import numpy as np
import base64
from typing import Optional


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame as JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Encode frame as base64 JPEG string (for JSON responses)."""
    jpg = frame_to_jpeg_bytes(frame, quality)
    return base64.b64encode(jpg).decode("utf-8")


def resize_frame(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Downscale frame for faster processing while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def mjpeg_response(frame: np.ndarray) -> bytes:
    """Wrap a JPEG frame in multipart/x-mixed-replace boundary."""
    jpg = frame_to_jpeg_bytes(frame)
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    )


class RollingBuffer:
    """Thread-safe rolling metric buffer for smoothing."""
    def __init__(self, maxlen: int = 30):
        from collections import deque
        self._buf = deque(maxlen=maxlen)

    def push(self, value: float):
        self._buf.append(value)

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.mean(self._buf))

    def latest(self) -> Optional[float]:
        return self._buf[-1] if self._buf else None
