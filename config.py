"""
Centralised configuration constants for the obstacle-detection pipeline.
All tuneable knobs live here so every other module stays free of magic numbers.
"""

# ──────────────────────────── capture ────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FAULT_TIMEOUT_SEC = 1.5          # seconds without a good frame → fault

# ──────────────────────────── detector (YOLOv8n) ─────────────────
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.40
YOLO_IOU_NMS = 0.45
YOLO_IMG_SIZE = 640

# ──────────────────────────── depth (MiDaS-small) ────────────────
MIDAS_MODEL_TYPE = "MiDaS_small"

# ──────────────────────────── fusion ─────────────────────────────
# After per-frame min-max normalisation the depth map sits in [0, 1]
# where 1 = closest (MiDaS outputs inverse depth).
NEAR_THRESHOLD = 0.66
MID_THRESHOLD = 0.33

# ──────────────────────────── tracker ────────────────────────────
IOU_THRESHOLD = 0.25             # minimum IOU to match a detection to a track
MAX_LOST_FRAMES = 8              # frames before a lost track is pruned
AREA_HISTORY_LEN = 6             # rolling window for approach detection
APPROACH_RATIO = 1.15            # second-half / first-half area ratio to flag

# ──────────────────────────── priority ───────────────────────────
DISTANCE_WEIGHTS = {"near": 5.0, "mid": 3.0, "far": 0.0}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train", "airplane"}
PERSON_CYCLE_CLASSES = {"person", "bicycle"}
CLASS_WEIGHTS = {"vehicle": 3.0, "person_cycle": 2.0, "static": 1.0}
APPROACH_BONUS = 2.0
TOP_K = 3                        # evaluate top-K objects per cycle

# Direction bias: centre obstacles are the most dangerous
DIRECTION_WEIGHTS = {"center": 1.5, "left": 1.0, "right": 1.0}

# Minimum composite score to consider a track alert-worthy.
# Objects below this are silently ignored (no TTS, no beep).
# Score examples:  near+person+center = 5+2+1.5 = 8.5 ✓
#                  mid+car+center     = 3+3+1.5 = 7.5 ✓
#                  mid+bench+left     = 3+1+1   = 5.0 ✗ (silent)
#                  far+car+right      = 0+3+1   = 4.0 ✗ (silent)
MIN_ALERT_SCORE = 5.5

# Staleness: if a track stays in the same distance bucket for this many
# consecutive frames, a penalty is applied so it fades from alerts.
STALENESS_FRAMES = 15            # frames of unchanged bucket → apply penalty
STALENESS_PENALTY = 3.0          # subtracted from score after staleness

# ──────────────────────────── speech ─────────────────────────────
# Per-distance cooldowns (seconds) — near re-alerts quickly, far rarely.
COOLDOWN_BY_DISTANCE = {"near": 3.0, "mid": 6.0, "far": 12.0}
COOLDOWN_SEC = 3.0               # legacy fallback (used if bucket missing)

# Global minimum gap between *any* two spoken alerts (prevents bursts).
GLOBAL_SPEAK_GAP = 2.0

# When a track's label+direction+distance are unchanged since the last
# announcement, multiply the cooldown by this factor (up to a cap).
SAME_INFO_MULTIPLIER = 2.0

TTS_RATE = 175                   # words per minute (pyttsx3)
TTS_VOLUME = 1.0                 # 0.0–1.0, sent to the default audio device
TTS_QUEUE_SIZE = 4               # max pending phrases (was 12)

# ──────────────────────────── beep ───────────────────────────────
BEEP_SAMPLE_RATE = 22_050
BEEP_FREQS      = {"near": 1000, "mid": 600}                 # Hz (no far)
BEEP_DURATIONS   = {"near": 0.08, "mid": 0.12}               # seconds
BEEP_INTERVALS   = {"near": 0.25, "mid": 0.60}               # pause between beeps
