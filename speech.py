"""
Offline TTS (pyttsx3) on a background thread with per-object cooldown.

Each tracked object has a cooldown timer (default 3 s).  The cooldown
can be interrupted early *only* if the object's distance bucket
**escalates** to a nearer bucket (e.g. mid → near).
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import pyttsx3

from tracker import Track
from config import COOLDOWN_SEC, TTS_RATE

_BUCKET_ORDER = {"far": 0, "mid": 1, "near": 2}


class Speaker:
    """Thread-safe TTS announcer with cooldown management."""

    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        self.enabled = enabled
        self._engine = None
        if enabled:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", TTS_RATE)
            except Exception as exc:
                print(f"[speech] TTS init failed: {exc}")
                self.enabled = False

        self._lock = threading.Lock()
        # track_id → (expiry_monotonic, last_distance_bucket)
        self._cooldowns: dict[int, tuple[float, str]] = {}
        self._log_fh = None
        if log_file:
            self._log_fh = open(log_file, "a", encoding="utf-8")
        self._fault_cd: float = 0.0  # next allowed fault announcement

    # ── low-level ───────────────────────────────────────────────
    def _say(self, text: str):
        """Enqueue text on a daemon thread so the pipeline never blocks."""
        if not self.enabled or self._engine is None:
            return

        def _run():
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass  # swallow rare COM / driver errors

        threading.Thread(target=_run, daemon=True).start()

    def _log(self, text: str):
        if self._log_fh:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self._log_fh.write(f"[{ts}] {text}\n")
            self._log_fh.flush()

    # ── public API ──────────────────────────────────────────────
    def announce(self, top_tracks: List[Track]):
        """Speak the top-priority detections, respecting cooldowns."""
        now = time.monotonic()
        for trk in top_tracks:
            det = trk.det
            bucket = det["distance"]
            tid = trk.id
            phrase = f"{det['label']}, {det['direction']}, {bucket}"

            if tid in self._cooldowns:
                expiry, last_bucket = self._cooldowns[tid]
                escalated = _BUCKET_ORDER.get(bucket, 0) > _BUCKET_ORDER.get(
                    last_bucket, 0
                )
                if now < expiry and not escalated:
                    continue  # still cooling down and didn't escalate

            self._cooldowns[tid] = (now + COOLDOWN_SEC, bucket)
            self._say(phrase)
            self._log(phrase)

    def fault(self, msg: str = "Camera fault"):
        """Announce a system fault (rate-limited to avoid spam)."""
        now = time.monotonic()
        if now >= self._fault_cd:
            self._say(msg)
            self._log(msg)
            self._fault_cd = now + 5.0

    def shutdown(self):
        """Clean up resources."""
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
