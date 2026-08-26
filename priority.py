"""
Priority scoring and ranking of tracked objects.

Combines five signals to compute a threat score:
    1. Distance bucket weight  (near > mid > far)
    2. Object class weight     (vehicle > person/cyclist > static)
    3. Direction bias           (center > left/right)
    4. Approaching bonus        (object getting closer)
    5. Staleness penalty        (same bucket for many frames → fade out)

Only tracks whose score meets MIN_ALERT_SCORE are returned for
announcement, so the system stays *silent* when nothing is dangerous.
"""

from __future__ import annotations

from typing import List

from tracker import Track
from config import (
    DISTANCE_WEIGHTS,
    VEHICLE_CLASSES,
    PERSON_CYCLE_CLASSES,
    CLASS_WEIGHTS,
    APPROACH_BONUS,
    TOP_K,
    DIRECTION_WEIGHTS,
    MIN_ALERT_SCORE,
    STALENESS_FRAMES,
    STALENESS_PENALTY,
)


def _class_weight(label: str) -> float:
    if label in VEHICLE_CLASSES:
        return CLASS_WEIGHTS["vehicle"]
    if label in PERSON_CYCLE_CLASSES:
        return CLASS_WEIGHTS["person_cycle"]
    return CLASS_WEIGHTS["static"]


def score_track(trk: Track) -> float:
    """Compute the priority score for a single track.

    Higher = more urgent.  Score is built additively, then dampened by
    detection confidence so flickering low-conf detections are naturally
    suppressed without ever zeroing real threats.
    """
    det = trk.det

    # ── additive components ──────────────────────────────────────
    base = DISTANCE_WEIGHTS.get(det.get("distance", "far"), 1.0)
    base += _class_weight(det.get("label", "object"))
    base += DIRECTION_WEIGHTS.get(det.get("direction", "center"), 1.0)

    if trk.approaching:
        base += APPROACH_BONUS

    # ── staleness penalty ────────────────────────────────────────
    # Object sitting in the same distance bucket for many frames
    if trk.bucket_streak >= STALENESS_FRAMES:
        base -= STALENESS_PENALTY

    # ── confidence dampening ─────────────────────────────────────
    # Only suppress very low-confidence detections; high-conf has
    # negligible effect so the additive score controls alert behaviour.
    conf = det.get("conf", 1.0)
    if conf < 0.5:
        base *= (0.6 + conf * 0.8)     # at conf=0.3 → ×0.84
                                         # at conf=0.4 → ×0.92
                                         # at conf=0.5+ → no penalty

    return max(base, 0.0)


def scored_tracks(tracks: List[Track]) -> list[tuple[float, Track]]:
    """Score every *active* track and return (score, track) pairs, highest first."""
    scored: list[tuple[float, Track]] = []
    for trk in tracks:
        if trk.lost > 0:
            continue
        scored.append((score_track(trk), trk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def rank(tracks: List[Track]) -> List[Track]:
    """Return up to TOP_K tracks that exceed the alert threshold.

    If nothing is dangerous, returns an empty list → silence.
    """
    return [
        trk
        for score, trk in scored_tracks(tracks)[:TOP_K]
        if score >= MIN_ALERT_SCORE
    ]
