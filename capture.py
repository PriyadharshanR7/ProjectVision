"""
Video capture with automatic fault detection.

Wraps cv2.VideoCapture and tracks the last time a valid frame was received.
If no good frame arrives for longer than FAULT_TIMEOUT_SEC the `fault` flag
is raised so other modules (overlay, speech) can react.
"""

import time
import cv2

from config import FRAME_WIDTH, FRAME_HEIGHT, FAULT_TIMEOUT_SEC


class Capture:
    """Thin wrapper around cv2.VideoCapture with a built-in fault timer."""

    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        # Request resolution (best-effort; driver may ignore)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._last_good: float = time.monotonic()
        self.fault: bool = False

    # ── public API ──────────────────────────────────────────────
    def read(self):
        """Return a BGR frame or *None*.  Updates `self.fault`."""
        ret, frame = self.cap.read()
        now = time.monotonic()
        if ret and frame is not None:
            self._last_good = now
            self.fault = False
            return frame
        # No frame
        if now - self._last_good > FAULT_TIMEOUT_SEC:
            self.fault = True
        return None

    def release(self):
        """Release the underlying capture device."""
        self.cap.release()
