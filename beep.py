"""
Optional proximity beep using *sounddevice* to emit sine-wave tones.

The frequency and repeat rate increase as the single nearest object
gets closer:
    near → 1 000 Hz, fast repeat
    mid  → 600 Hz, medium repeat
    far  → 350 Hz, slow repeat
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from config import BEEP_SAMPLE_RATE, BEEP_FREQS, BEEP_DURATIONS, BEEP_INTERVALS


class Beeper:
    """Background thread that plays proximity beeps via sounddevice."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._sd = None
        if enabled:
            try:
                import sounddevice as sd  # noqa: F811

                self._sd = sd
            except ImportError:
                print("[beep] sounddevice not available – beep disabled")
                self.enabled = False
        self._thread: threading.Thread | None = None
        self._running = False
        self._bucket = "far"
        self._lock = threading.Lock()

    # ── public ──────────────────────────────────────────────────
    def update(self, bucket: str):
        """Set the current distance bucket for the nearest object."""
        with self._lock:
            self._bucket = bucket

    def start(self):
        """Begin the beeping loop in a daemon thread."""
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the loop to end and wait briefly."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── internals ───────────────────────────────────────────────
    def _loop(self):
        while self._running:
            with self._lock:
                bucket = self._bucket

            freq = BEEP_FREQS.get(bucket, 350)
            dur = BEEP_DURATIONS.get(bucket, 0.18)
            interval = BEEP_INTERVALS.get(bucket, 1.2)

            n_samples = int(BEEP_SAMPLE_RATE * dur)
            t = np.linspace(0, dur, n_samples, dtype=np.float32)
            # Smooth envelope (half-sine) to avoid audible clicks
            envelope = np.sin(
                np.pi * np.linspace(0, 1, n_samples, dtype=np.float32)
            )
            wave = (0.3 * np.sin(2 * math.pi * freq * t) * envelope).astype(
                np.float32
            )

            try:
                self._sd.play(wave, BEEP_SAMPLE_RATE, blocking=True)
            except Exception:
                pass  # swallow device errors silently

            time.sleep(interval)
