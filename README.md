# ObstacleGuard – Phase-0 Prototype

> **Assistive obstacle-detection tool for blind / visually-impaired users.**
> Real-time webcam pipeline that combines object detection with monocular
> depth estimation to announce nearby obstacles via speech and/or audio beeps.

---

## How It Works

```
Webcam
  │
  ├─► YOLOv8n (object detection)  ──┐
  │                                  ├─► Fusion (distance + direction buckets)
  └─► MiDaS-small (depth map)   ──┘
                                      │
                                      ▼
                              IOU Tracker (persistent IDs, approach detection)
                                      │
                                      ▼
                              Priority Scorer (rank by danger)
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                       Speech      Beep      Overlay
                    (pyttsx3)   (sounddevice) (OpenCV)
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Capture | `capture.py` | Webcam / video-file reader with fault timer |
| Detection | `detector.py` | YOLOv8n (ultralytics) wrapper |
| Depth | `depth.py` | MiDaS-small monocular inverse-depth |
| Fusion | `fusion.py` | Assigns *distance* (near/mid/far) and *direction* (left/center/right) buckets |
| Tracking | `tracker.py` | IOU-based multi-object tracker with rolling area history |
| Priority | `priority.py` | Scores & ranks tracked objects for announcement |
| Speech | `speech.py` | Offline pyttsx3 TTS with per-object cooldown |
| Beep | `beep.py` | Proximity beep (sine wave via sounddevice) |
| Overlay | `overlay.py` | Debug visualisation (colour-coded boxes, FPS, fault banner) |
| Config | `config.py` | All tuneable constants in one place |
| Entry | `main.py` | CLI interface and main pipeline loop |

---

## Quick Start

### 1. Install dependencies

```bash
# (Recommended) create a virtual environment first
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** On first run the YOLOv8n and MiDaS-small model weights are
> downloaded automatically (~25 MB + ~30 MB).

### 2. Run

```bash
# Default webcam, TTS on, overlay on
python main.py

# Use a video file instead of a webcam
python main.py --source path/to/video.mp4

# Beep mode, no speech, log to file
python main.py --no-speech --beep --log alerts.txt

# Headless (no OpenCV window)
python main.py --no-show
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index (int) or video file path |
| `--show` / `--no-show` | `--show` | Toggle debug overlay window |
| `--speech` / `--no-speech` | `--speech` | Toggle TTS announcements |
| `--beep` / `--no-beep` | `--no-beep` | Toggle proximity beep |
| `--log FILE` | off | Write timestamped announcements to a text file |

---

## Bucket Definitions

### Distance (from per-frame min-max normalised depth)

| Bucket | Normalised depth value |
|--------|----------------------|
| **near** | ≥ 0.66 |
| **mid** | ≥ 0.33 and < 0.66 |
| **far** | < 0.33 |

### Direction (from bounding-box centre x)

| Bucket | Horizontal position |
|--------|-------------------|
| **left** | centre x < ⅓ frame width |
| **center** | centre x < ⅔ frame width |
| **right** | centre x ≥ ⅔ frame width |

---

## Priority Scoring

Each active tracked object is scored as:

```
score = distance_weight + class_weight + approaching_bonus
```

| Factor | Values |
|--------|--------|
| Distance weight | near = 3, mid = 2, far = 1 |
| Class weight | vehicles = 3, person/cyclist = 2, static = 1 |
| Approaching bonus | +1.5 (if bbox area is growing over the rolling window) |

Only the **top 1–2** objects are spoken per cycle, with a **3-second per-object
cooldown** that can be interrupted early **only** if the object escalates to a
nearer bucket.

---

## Debug Overlay

| Element | Meaning |
|---------|---------|
| 🟥 Red box | Object is **near** |
| 🟨 Yellow box | Object is at **mid** range |
| 🟩 Green box | Object is **far** |
| `!` after label | Object is **approaching** |
| FPS counter | Pipeline throughput |
| Red banner | Camera fault (no frame > 1.5 s) |

---

## Known Limitations

> [!WARNING]
> **This is a research prototype and assistive supplement.  It is NOT a
> safety-certified medical device and does NOT replace a white cane,
> guide dog, or professional mobility training.**

1. **Depth is relative per-frame, not metric.**
   MiDaS-small produces an *inverse relative depth* map that is min-max
   normalised per frame.  The bucket thresholds are therefore adaptive:
   "near" means "nearer than most things in *this* frame", not a fixed
   distance in metres.  Absolute ranging requires stereo cameras or LiDAR.

2. **CPU-only performance target is ≥ 5 FPS at 640×480.**
   A GPU (CUDA) will significantly improve throughput, but the prototype
   is designed to be usable on a CPU-only laptop.

3. **IOU tracker is lightweight.**
   It can lose track of fast-moving or occluded objects.  Production
   systems should use DeepSORT, ByteTrack, or similar.

4. **TTS latency.**
   pyttsx3 runs the system TTS engine; latency varies by platform and
   voice.  On Windows the SAPI5 engine is generally responsive.

5. **Beep requires a working audio output device.**
   If `sounddevice` cannot open the default output, beep is silently
   disabled.

---

## Project Structure

```
VisImpProj/
├── main.py            # CLI entry-point & pipeline loop
├── config.py          # all tuneable constants
├── capture.py         # video-capture + fault detection
├── detector.py        # YOLOv8n wrapper
├── depth.py           # MiDaS-small depth estimation
├── fusion.py          # distance & direction bucket assignment
├── tracker.py         # IOU-based multi-object tracker
├── priority.py        # danger scoring & ranking
├── speech.py          # pyttsx3 TTS with cooldowns
├── beep.py            # proximity beep (sounddevice)
├── overlay.py         # debug visualisation
├── requirements.txt   # Python dependencies
└── README.md          # this file
```

---

## License

This prototype is provided as-is for educational and research purposes.
