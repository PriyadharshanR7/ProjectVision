"""
Lightweight IOU-based multi-object tracker.

Assigns persistent integer IDs to detections across frames and maintains
a rolling bounding-box–area history so downstream modules can flag
objects that are *approaching* (area growing over time).
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from config import IOU_THRESHOLD, MAX_LOST_FRAMES, AREA_HISTORY_LEN, APPROACH_RATIO


# ── helpers ─────────────────────────────────────────────────────
def _iou(a: list, b: list) -> float:
    """Intersection-over-union for two [x1, y1, x2, y2] boxes."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Track object ────────────────────────────────────────────────
class Track:
    """Single tracked object with a persistent ID."""

    _next_id: int = 1

    def __init__(self, det: dict):
        self.id: int = Track._next_id
        Track._next_id += 1
        self.det: dict = det
        self.lost: int = 0
        self.area_history: deque = deque(maxlen=AREA_HISTORY_LEN)
        self._record_area()
        # Staleness tracking: how many consecutive frames in the same bucket
        self.frames_seen: int = 1
        self.bucket_streak: int = 1
        self.last_bucket: str = det.get("distance", "far")

    # ── internals ───────────────────────────────────────────────
    def _record_area(self):
        x1, y1, x2, y2 = self.det["bbox"]
        self.area_history.append((x2 - x1) * (y2 - y1))

    # ── public ──────────────────────────────────────────────────
    def update(self, det: dict):
        """Re-associate this track with a new detection."""
        new_bucket = det.get("distance", "far")
        if new_bucket == self.last_bucket:
            self.bucket_streak += 1
        else:
            self.bucket_streak = 1
            self.last_bucket = new_bucket
        self.det = det
        self.lost = 0
        self.frames_seen += 1
        self._record_area()

    @property
    def approaching(self) -> bool:
        """True if the object's apparent size is growing over the window."""
        if len(self.area_history) < 3:
            return False
        areas = list(self.area_history)
        mid = len(areas) // 2
        first_half = float(np.mean(areas[:mid]))
        second_half = float(np.mean(areas[mid:]))
        if first_half <= 0:
            return False
        return second_half > first_half * APPROACH_RATIO


# ── Tracker ─────────────────────────────────────────────────────
class Tracker:
    """Greedy IOU matcher that maintains a list of Track objects."""

    def __init__(self):
        self.tracks: List[Track] = []

    def reset(self):
        """Drop all tracks and reset the global ID counter (new video)."""
        self.tracks = []
        Track._next_id = 1

    def update(self, fused_detections: list[dict]) -> List[Track]:
        """Match *fused_detections* to existing tracks; return all live tracks."""

        # First frame — seed tracks
        if not self.tracks:
            for det in fused_detections:
                self.tracks.append(Track(det))
            return self.tracks

        # Build scored (IOU, track_idx, det_idx) pairs
        pairs = []
        for ti, trk in enumerate(self.tracks):
            for di, det in enumerate(fused_detections):
                score = _iou(trk.det["bbox"], det["bbox"])
                if score >= IOU_THRESHOLD:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)  # greedily match highest IOU first

        used_trk: set[int] = set()
        used_det: set[int] = set()
        for score, ti, di in pairs:
            if ti in used_trk or di in used_det:
                continue
            self.tracks[ti].update(fused_detections[di])
            used_trk.add(ti)
            used_det.add(di)

        # Increment lost counter for unmatched tracks
        for ti in range(len(self.tracks)):
            if ti not in used_trk:
                self.tracks[ti].lost += 1

        # Spawn new tracks for unmatched detections
        for di, det in enumerate(fused_detections):
            if di not in used_det:
                self.tracks.append(Track(det))

        # Prune tracks lost too long
        self.tracks = [t for t in self.tracks if t.lost <= MAX_LOST_FRAMES]

        return self.tracks
