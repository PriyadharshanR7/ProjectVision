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
DISTANCE_WEIGHTS = {"near": 3.0, "mid": 2.0, "far": 1.0}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train", "airplane"}
PERSON_CYCLE_CLASSES = {"person", "bicycle"}
CLASS_WEIGHTS = {"vehicle": 3.0, "person_cycle": 2.0, "static": 1.0}
APPROACH_BONUS = 1.5
TOP_K = 2                       # speak only the top-K objects per cycle

# ──────────────────────────── speech ─────────────────────────────
COOLDOWN_SEC = 3.0               # per-object cooldown before re-announcing
TTS_RATE = 175                   # words per minute

# ──────────────────────────── beep ───────────────────────────────
BEEP_SAMPLE_RATE = 22_050
BEEP_FREQS      = {"near": 1000, "mid": 600,  "far": 350}   # Hz
BEEP_DURATIONS   = {"near": 0.08, "mid": 0.12, "far": 0.18}  # seconds
BEEP_INTERVALS   = {"near": 0.25, "mid": 0.60, "far": 1.20}  # pause between beeps
