"""
Priority scoring and ranking of tracked objects.

Combines three signals:
    1. Distance bucket weight  (near > mid > far)
    2. Object class weight     (vehicle > person/cyclist > static)
    3. Approaching bonus       (if the object is getting closer)

Returns only the top-K tracks for announcement.
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
)


def _class_weight(label: str) -> float:
    if label in VEHICLE_CLASSES:
        return CLASS_WEIGHTS["vehicle"]
    if label in PERSON_CYCLE_CLASSES:
        return CLASS_WEIGHTS["person_cycle"]
    return CLASS_WEIGHTS["static"]


def rank(tracks: List[Track]) -> List[Track]:
    """Score every *active* track and return the top-K by priority."""
    scored: list[tuple[float, Track]] = []
    for trk in tracks:
        if trk.lost > 0:
            continue
        det = trk.det
        score = DISTANCE_WEIGHTS.get(det["distance"], 1.0)
        score += _class_weight(det["label"])
        if trk.approaching:
            score += APPROACH_BONUS
        scored.append((score, trk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [trk for _, trk in scored[:TOP_K]]
