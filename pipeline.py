"""
Shared ObstacleGuard processing step used by the CLI and the GUI.

Takes a BGR frame and runs detection → depth → fusion → tracking →
priority → speech / beep → overlay, returning a structured result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import FRAME_WIDTH, FRAME_HEIGHT, TOP_K, MIN_ALERT_SCORE
from detector import Detector
from depth import DepthEstimator
from fusion import fuse
from tracker import Tracker, Track
from priority import scored_tracks, rank
from speech import Speaker
from beep import Beeper
from overlay import draw


@dataclass
class FrameResult:
    vis: np.ndarray
    tracks: list
    top: list
    scores: dict  # track_id → float
    fps: float
    announced: list[str]
    nearest_distance: Optional[str]
    fault: bool
    flags: list[str] = field(default_factory=list)


class ObstaclePipeline:
    """One-shot model holders plus per-session tracker / audio state."""

    def __init__(
        self,
        detector: Detector,
        depth_est: DepthEstimator,
        *,
        speech: bool = True,
        beep: bool = True,
        log_file: Optional[str] = None,
    ):
        self.detector = detector
        self.depth_est = depth_est
        self.tracker = Tracker()
        self.speaker = Speaker(enabled=speech, log_file=log_file)
        self.beeper = Beeper(enabled=beep)
        self.beeper.start()
        self._fps = 0.0

    def set_speech(self, enabled: bool) -> None:
        self.speaker.enabled = bool(enabled) and self.speaker.available

    def set_beep(self, enabled: bool) -> None:
        self.beeper.enabled = enabled and self.beeper._sd is not None
        if self.beeper.enabled:
            self.beeper.start()
        else:
            self.beeper.update(None)

    def reset(self) -> None:
        """Clear tracker IDs and TTS cooldowns between sources."""
        self.tracker.reset()
        self.speaker._cooldowns.clear()
        self.beeper.update(None)
        self._fps = 0.0

    def shutdown(self) -> None:
        self.beeper.stop()
        self.speaker.shutdown()

    def process_fault(self) -> FrameResult:
        blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        vis = draw(blank, [], self._fps, fault=True)
        return FrameResult(
            vis=vis,
            tracks=[],
            top=[],
            scores={},
            fps=self._fps,
            announced=[],
            nearest_distance=None,
            fault=True,
            flags=["CAMERA FAULT"],
        )

    def process_frame(self, frame) -> FrameResult:
        t0 = time.monotonic()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        detections = self.detector.detect(frame)
        depth_map = self.depth_est.estimate(frame)
        fused = fuse(detections, depth_map)
        tracks = self.tracker.update(fused)
        scored = scored_tracks(tracks)
        scores = {trk.id: score for score, trk in scored}

        # Only alert-worthy tracks reach the speaker (may be empty → silence)
        top = rank(tracks)
        announced = self.speaker.announce(top)

        # Prune stale cooldowns for tracks that have been pruned by the tracker
        active_ids = {t.id for t in tracks}
        stale_cds = [tid for tid in self.speaker._cooldowns if tid not in active_ids]
        for tid in stale_cds:
            del self.speaker._cooldowns[tid]

        # Gate the beeper: only activate when the nearest *alert-worthy*
        # object is near or mid.  If nothing is alert-worthy, silence.
        alert_ids = {t.id for t in top}
        active_alert = [
            t for t in tracks
            if t.lost == 0 and t.id in alert_ids
        ]
        nearest_distance: Optional[str] = None
        if active_alert:
            # Higher depth_val = closer in MiDaS inverse-depth
            nearest = max(active_alert, key=lambda t: t.det.get("depth_val", 0))
            nearest_distance = nearest.det.get("distance", "far")
            self.beeper.update(nearest_distance)
        else:
            self.beeper.update(None)

        dt = time.monotonic() - t0
        self._fps = 1.0 / dt if dt > 0 else 0.0

        flags = [_flag_text(score, trk, alert_ids) for score, trk in scored[:TOP_K]]
        vis = draw(frame.copy(), tracks, self._fps, flags=flags)

        return FrameResult(
            vis=vis,
            tracks=tracks,
            top=top,
            scores=scores,
            fps=self._fps,
            announced=announced,
            nearest_distance=nearest_distance,
            fault=False,
            flags=flags,
        )


def _flag_text(score: float, trk: Track, alert_ids: set[int] | None = None) -> str:
    det = trk.det
    bang = " !" if trk.approaching else ""
    prefix = "ALERT" if (alert_ids and trk.id in alert_ids) else "watch"
    return (
        f"{prefix} #{trk.id} {det['label']} {det['direction']} "
        f"{det['distance']}{bang}  ({score:.1f})"
    )


def load_models(status_cb=None) -> tuple[Detector, DepthEstimator]:
    """Load YOLO + MiDaS once (slow; call off the UI thread)."""
    if status_cb:
        status_cb("Loading YOLOv8n…")
    detector = Detector()
    if status_cb:
        status_cb("Loading MiDaS-small…")
    depth_est = DepthEstimator()
    if status_cb:
        status_cb("Models ready.")
    return detector, depth_est
