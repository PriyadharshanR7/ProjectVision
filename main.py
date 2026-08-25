#!/usr/bin/env python3
"""
ObstacleGuard – Phase-0 assistive obstacle-detection prototype.

CLI entry-point that wires capture → detection → depth → fusion →
tracking → priority → speech / beep / overlay into a real-time loop.

Usage examples
--------------
    python main.py                          # default webcam, TTS on
    python main.py --source 1              # second webcam
    python main.py --source road.mp4       # video file
    python main.py --no-speech --beep      # beep only, no TTS
    python main.py --log alerts.txt        # also write alerts to file
    python main.py --no-show               # headless (no window)
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from config import FRAME_WIDTH, FRAME_HEIGHT
from capture import Capture
from detector import Detector
from depth import DepthEstimator
from fusion import fuse
from tracker import Tracker
from priority import rank
from speech import Speaker
from beep import Beeper
from overlay import draw


# ── CLI ─────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ObstacleGuard – assistive obstacle-detection tool"
    )
    p.add_argument(
        "--source",
        default="0",
        help="Webcam index (integer) or path to a video file (default: 0)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Show debug overlay window (default: on)",
    )
    p.add_argument("--no-show", dest="show", action="store_false")

    p.add_argument(
        "--speech",
        action="store_true",
        default=True,
        help="Enable TTS announcements (default: on)",
    )
    p.add_argument("--no-speech", dest="speech", action="store_false")

    p.add_argument(
        "--beep",
        action="store_true",
        default=False,
        help="Enable proximity beep (default: off)",
    )
    p.add_argument("--no-beep", dest="beep", action="store_false")

    p.add_argument(
        "--log",
        default=None,
        metavar="FILE",
        help="Also write timestamped announcements to FILE",
    )
    return p


# ── main loop ───────────────────────────────────────────────────
def main():
    args = _build_parser().parse_args()

    # Interpret --source
    source = int(args.source) if args.source.isdigit() else args.source

    # ── model loading ───────────────────────────────────────────
    print("[init] Loading YOLOv8n …")
    detector = Detector()

    print("[init] Loading MiDaS-small …")
    depth_est = DepthEstimator()

    print("[init] Opening capture source …")
    cap = Capture(source)

    tracker = Tracker()
    speaker = Speaker(enabled=args.speech, log_file=args.log)
    beeper = Beeper(enabled=args.beep)
    beeper.start()

    print("[init] Pipeline running.  Press 'q' to quit.")
    fps: float = 0.0
    fault_spoken = False

    try:
        while True:
            t0 = time.monotonic()

            frame = cap.read()

            # ── handle missing frames / fault ───────────────────
            if frame is None:
                if cap.fault:
                    if not fault_spoken:
                        speaker.fault("Camera feed lost")
                        fault_spoken = True
                    if args.show:
                        blank = np.zeros(
                            (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
                        )
                        draw(blank, [], fps, fault=True)
                        cv2.imshow("ObstacleGuard", blank)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                continue

            fault_spoken = False

            # Ensure consistent resolution
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ── pipeline stages ─────────────────────────────────
            detections = detector.detect(frame)
            depth_map = depth_est.estimate(frame)
            fused = fuse(detections, depth_map)
            tracks = tracker.update(fused)
            top = rank(tracks)

            # ── outputs ─────────────────────────────────────────
            speaker.announce(top)

            # Beep tracks the single nearest active object
            active = [t for t in tracks if t.lost == 0]
            if active:
                nearest = max(active, key=lambda t: t.det.get("depth_val", 0))
                beeper.update(nearest.det["distance"])

            # Timing
            dt = time.monotonic() - t0
            fps = 1.0 / dt if dt > 0 else 0.0

            # Overlay
            if args.show:
                vis = draw(frame, tracks, fps)
                cv2.imshow("ObstacleGuard", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    finally:
        beeper.stop()
        speaker.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print("[done] Pipeline stopped.")


if __name__ == "__main__":
    main()
