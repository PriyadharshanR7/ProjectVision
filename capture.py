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
        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        # Request resolution (best-effort; driver may ignore)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._last_good: float = time.monotonic()
        self.fault: bool = False
        self.eof: bool = False

    # ── public API ──────────────────────────────────────────────
    def read(self):
        """Return a BGR frame or *None*.  Updates `self.fault` / `self.eof`."""
        ret, frame = self.cap.read()
        now = time.monotonic()
        if ret and frame is not None:
            self._last_good = now
            self.fault = False
            self.eof = False
            return frame
        # Video files: treat a failed read as end-of-file, not a camera fault.
        if self.is_file:
            self.eof = True
            return None
        if now - self._last_good > FAULT_TIMEOUT_SEC:
            self.fault = True
        return None

    def progress(self) -> tuple[int, int]:
        """Return (current_frame, total_frames).  total may be 0 for live cams."""
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return pos, total

    def release(self):
        """Release the underlying capture device."""
        self.cap.release()
