"""
Fuse YOLO detections with MiDaS depth into distance and direction buckets.

Each detection dict is enriched with:
    depth_val  – median normalised inverse-depth inside the bbox (float)
    distance   – "near" | "mid" | "far"
    direction  – "left" | "center" | "right"
"""

import numpy as np

from config import NEAR_THRESHOLD, MID_THRESHOLD


# ── helpers ─────────────────────────────────────────────────────
def _distance_bucket(depth_val: float) -> str:
    if depth_val >= NEAR_THRESHOLD:
        return "near"
    if depth_val >= MID_THRESHOLD:
        return "mid"
    return "far"


def _direction_bucket(cx: float, frame_w: int) -> str:
    third = frame_w / 3.0
    if cx < third:
        return "left"
    if cx < 2 * third:
        return "center"
    return "right"


# ── public API ──────────────────────────────────────────────────
def fuse(detections: list[dict], depth_map: np.ndarray) -> list[dict]:
    """Merge detections and depth into a single fused list."""
    h, w = depth_map.shape[:2]
    fused = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Clamp to frame bounds
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2i <= x1i or y2i <= y1i:
            continue
        roi = depth_map[y1i:y2i, x1i:x2i]
        depth_val = float(np.median(roi))
        cx = (x1 + x2) / 2.0
        fused.append(
            {
                **det,
                "depth_val": depth_val,
                "distance": _distance_bucket(depth_val),
                "direction": _direction_bucket(cx, w),
            }
        )
    return fused
