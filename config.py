"""Configuration for ZeroController pose visualization and data collection."""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_TARGET_FPS = 30
CAMERA_BACKEND = "msmf"  # Preferred backend on Windows: msmf or dshow
CAMERA_BUFFER_SIZE = 1
CAMERA_READ_WARMUP_FRAMES = 12

# MediaPipe Pose Landmarker (Tasks API)
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
POSE_NUM_POSES = 1
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Display
WINDOW_NAME = "ZeroController - 33 Point Pose"
SHOW_FPS = True
SHOW_CONNECTIONS = True
SHOW_LANDMARK_IDS = True
MIRROR_VIEW = True

# Drawing style (BGR)
POINT_COLOR = (0, 255, 255)
LINE_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)
POINT_RADIUS = 3

# --- Data collection ---
DATA_DIR = "data/raw"
SEQUENCE_LENGTH = 30          # frames per sample (~1 sec at 30 fps)

# The 9 in-game actions. Order matters: the classifier's output index maps
# back to this list, so keep the order consistent across training runs.
ACTIONS = [
    "idle",         # standing still / neutral
    "left_punch",   # left arm forward punch
    "right_punch",  # right arm forward punch
    "left_kick",    # left leg kick
    "right_kick",   # right leg kick
    "jump",         # jump / leap
    "block",        # guard — arms crossed in front of torso
    "forward",      # step / walk forward
    "backward",     # step / walk backward
]

# --- Training ---
MODEL_SAVE_PATH = "models/action_classifier.pth"
EPOCHS = 120
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DROPOUT = 0.3               # applied in TCN, LSTM, and STGCN heads

# ── Model selection ──────────────────────────────────────────────────────────
# "stgcn" → Spatial-Temporal Graph Convolutional Network (recommended).
#           Uses the body skeleton as a graph — best for skeleton data.
# "tcn"   → Temporal Convolutional Network (fast baseline, parallel over time).
# "lstm"  → Vanilla LSTM (simplest baseline).
MODEL_TYPE = "stgcn"

# LSTM params (used when MODEL_TYPE = "lstm")
HIDDEN_SIZE = 64
NUM_LSTM_LAYERS = 1

# TCN params (used when MODEL_TYPE = "tcn")
# 4 dilated blocks with dilations [1,2,4,8] and kernel=3 give a receptive
# field of 31 frames — covering the full 30-frame window from every position.
TCN_NUM_CHANNELS = 64       # feature channels per TCN block
TCN_KERNEL_SIZE = 3         # temporal conv kernel size
TCN_NUM_LAYERS = 4          # number of dilated blocks

# ST-GCN params (used when MODEL_TYPE = "stgcn")
# A lightweight ST-GCN:
#   * 3 spatial-temporal blocks (keeps params low for small datasets)
#   * channel growth 6 → 32 → 64 → 64
#   * temporal kernel = 9 (covers ~0.3 s of motion around each frame)
# Input is (B, C=6, T=30, V=33):  6 channels = (x,y,z) position + (dx,dy,dz) velocity.
STGCN_CHANNELS    = (32, 64, 64)   # output channels per block (tuple of length = num_blocks)
STGCN_TEMPORAL_K  = 9              # temporal conv kernel size (must be odd)
STGCN_USE_VELOCITY = True          # concat Δposition → 6-channel input (recommended)

# ── Preprocessing ────────────────────────────────────────────────────────────
# Applied identically at train and inference time. Keeping these flags in
# config guarantees the runtime preprocessing never drifts from training.
PREPROCESS_HIP_CENTER = True    # translate so mid-hip (landmarks 23 & 24) is at origin
PREPROCESS_SCALE_NORM = True    # scale so torso length (mid-hip → mid-shoulder) ≈ 1
PREPROCESS_USE_VISIBILITY = True  # weight each joint's (x,y,z) by its visibility score

# ── Inference / Game-control ─────────────────────────────────────────────────
# Confidence gate: only emit a game action when softmax ≥ threshold for
# ≥ STABLE_FRAMES consecutive frames. Kills flicker and accidental triggers.
PREDICT_CONFIDENCE_THRESHOLD = 0.85
PREDICT_STABLE_FRAMES = 3
# After a non-idle action triggers, wait this many frames before allowing
# another non-idle trigger. Prevents the same punch from firing twice.
PREDICT_TRIGGER_COOLDOWN = 10
