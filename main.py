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
    python main.py --gui                   # desktop GUI (video file or webcam)
    python gui.py                          # same GUI, direct launch
"""

from __future__ import annotations

import argparse

import cv2

from capture import Capture
from pipeline import ObstaclePipeline, load_models


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
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop GUI instead of the CLI loop",
    )
    return p


# ── main loop ───────────────────────────────────────────────────
def main():
    args = _build_parser().parse_args()

    if args.gui:
        from gui import run_gui

        run_gui()
        return

    # Interpret --source
    source = int(args.source) if args.source.isdigit() else args.source

    # ── model loading ───────────────────────────────────────────
    print("[init] Loading YOLOv8n …")
    detector, depth_est = load_models(lambda msg: print(f"[init] {msg}"))

    print("[init] Opening capture source …")
    cap = Capture(source)
    pipe = ObstaclePipeline(
        detector,
        depth_est,
        speech=args.speech,
        beep=args.beep,
        log_file=args.log,
    )

    print("[init] Pipeline running.  Press 'q' to quit.")
    fault_spoken = False

    try:
        while True:
            frame = cap.read()

            # ── handle missing frames / fault / EOF ─────────────
            if frame is None:
                if cap.eof:
                    print("[info] End of video.")
                    break
                if cap.fault:
                    if not fault_spoken:
                        pipe.speaker.fault("Camera feed lost")
                        fault_spoken = True
                    if args.show:
                        result = pipe.process_fault()
                        cv2.imshow("ObstacleGuard", result.vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                continue

            fault_spoken = False
            result = pipe.process_frame(frame)

            if args.show:
                cv2.imshow("ObstacleGuard", result.vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    finally:
        pipe.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print("[done] Pipeline stopped.")


if __name__ == "__main__":
    main()
