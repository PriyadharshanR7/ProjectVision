"""
Debug overlay drawn on the video frame.

• Colour-coded bounding boxes: red (near), yellow (mid), green (far)
• Track ID + bucket labels on each box
• Current FPS counter (top-left)
• Red fault banner across the centre when the camera feed is lost
"""

from __future__ import annotations

from typing import List

import cv2

from tracker import Track

# BGR colours for each distance bucket
_BUCKET_COLOURS = {
    "near": (0, 0, 255),     # red
    "mid":  (0, 200, 255),   # yellow-ish
    "far":  (0, 200, 0),     # green
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(
    frame,
    tracks: List[Track],
    fps: float,
    fault: bool = False,
):
    """Annotate *frame* in-place and return it."""

    for trk in tracks:
        if trk.lost > 0:
            continue
        det = trk.det
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        colour = _BUCKET_COLOURS.get(det["distance"], (200, 200, 200))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label
        label = f"#{trk.id} {det['label']} {det['direction']} {det['distance']}"
        if trk.approaching:
            label += " !"

        (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.50, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4), _FONT, 0.50, (0, 0, 0), 1, cv2.LINE_AA
        )

    # FPS (top-left)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 28), _FONT, 0.70, (0, 255, 255), 2, cv2.LINE_AA
    )

    # Fault banner
    if fault:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 200), -1)
        text = "CAMERA FAULT"
        (tw, _), _ = cv2.getTextSize(text, _FONT, 1.0, 2)
        tx = (w - tw) // 2
        cv2.putText(
            frame, text, (tx, h // 2 + 10), _FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )

    return frame
