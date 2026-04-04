"""
Configuration for the ZeroController - AI Fighting Game Controller
All tunable parameters, key mappings, and thresholds live here.
"""

# ─── Camera Settings ───────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = default webcam, 1 = second camera, etc.
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_TARGET_FPS = 60     # Requested camera FPS (actual depends on webcam/driver)
CAMERA_BACKEND = "msmf"    # Many Windows webcams are stable on MSMF; DSHOW can output black frames
CAMERA_BUFFER_SIZE = 1     # Keep latest frame and reduce lag
AUTO_BACKEND_FALLBACK = True
BLACK_FRAME_MEAN_THRESHOLD = 5.0
BLACK_FRAME_FALLBACK_COUNT = 20

# ─── MediaPipe Pose Settings ──────────────────────────────────────
POSE_MIN_DETECTION_CONFIDENCE = 0.6
POSE_MIN_TRACKING_CONFIDENCE = 0.65  # Raised for more stable landmark tracking (was 0.5)
POSE_MODEL_COMPLEXITY = 1  # 0=Lite, 1=Full, 2=Heavy — Full is much more accurate
PROCESS_SCALE = 1.0        # No downscaling — accuracy over FPS
USE_GPU = True             # Use CUDA (via PyTorch) for frame resize/preprocessing

# ─── Move Detection Thresholds ────────────────────────────────────
# All distance thresholds are in shoulder-widths (relative to body size).
# This makes detection work at any distance from the camera.

# Punch detection: wrist must be significantly ahead of the shoulder
PUNCH_EXTENSION_THRESHOLD = 0.5     # Wrist-to-shoulder extension (in shoulder widths)
PUNCH_HEIGHT_TOLERANCE = 0.5        # Vertical tolerance for punch height
PUNCH_SPEED_THRESHOLD = 0.12        # Minimum wrist speed (shoulder widths per second)

# Kick detection: ankle goes above knee level or extends forward
KICK_HEIGHT_THRESHOLD = 0.25        # Ankle rise above knee (in shoulder widths)
KICK_EXTENSION_THRESHOLD = 0.5      # Ankle horizontal extension (in shoulder widths)

# Block detection: both wrists near face level
BLOCK_WRIST_FACE_THRESHOLD = 0.4    # Max wrist-to-nose distance (in shoulder widths)

# Crouch detection: nose drops below a baseline
CROUCH_THRESHOLD = 0.55             # Nose drop from baseline (was 0.5, widened)
CROUCH_EMA_ALPHA = 0.3              # EMA smoothing factor for nose y position
CROUCH_CONFIRM_FRAMES = 3           # Consecutive frames beyond threshold to trigger crouch

# Jump detection: both ankles rise above baseline
JUMP_THRESHOLD = 0.45               # Ankle rise from baseline (was 0.35, widened)
JUMP_EMA_ALPHA = 0.3                # EMA smoothing factor for ankle y position
JUMP_CONFIRM_FRAMES = 3             # Consecutive frames beyond threshold to trigger jump

# Movement detection: horizontal shift of hips
MOVE_LEFT_THRESHOLD = -0.5          # Widened dead zone (was -0.35)
MOVE_RIGHT_THRESHOLD = 0.5          # Widened dead zone (was 0.35)
MOVE_EMA_ALPHA = 0.3                # EMA smoothing factor for hip position (lower = smoother)
MOVE_CONFIRM_FRAMES = 4             # Consecutive frames beyond threshold to trigger move
MOVE_VELOCITY_THRESHOLD = 0.05      # Min hip speed (shoulder widths/frame) to trigger move

# Calibration settings
CALIBRATION_FRAMES = 60             # Frames to collect for baseline (was 30)
CALIBRATION_OUTLIER_STD = 2.0       # Reject calibration samples beyond N standard deviations

# ─── Cooldowns (seconds) ─────────────────────────────────────────
# Prevents the same move from spamming
PUNCH_COOLDOWN = 0.18
KICK_COOLDOWN = 0.25
BLOCK_COOLDOWN = 0.2
CROUCH_COOLDOWN = 0.2
JUMP_COOLDOWN = 0.35
MOVE_COOLDOWN = 0.3

# ─── Keyboard Mappings (Player 1) ────────────────────────────────
# These map to the keys pressed for each detected move.
# Change these to match your fighting game's controls.
KEY_MAP_P1 = {
    "left_punch":    "u",
    "right_punch":   "i",
    "left_kick":     "j",
    "right_kick":    "k",
    "block":         "o",
    "crouch":        "s",
    "jump":          "w",
    "move_left":     "a",
    "move_right":    "d",
}

# ─── Keyboard Mappings (Player 2 — for future use) ───────────────
KEY_MAP_P2 = {
    "left_punch":    "num4",
    "right_punch":   "num5",
    "left_kick":     "num1",
    "right_kick":    "num2",
    "block":         "num6",
    "crouch":        "down",
    "jump":          "up",
    "move_left":     "left",
    "move_right":    "right",
}

# ─── Display Settings ────────────────────────────────────────────
SHOW_LANDMARKS = True         # Draw pose skeleton + landmarks on frame
SHOW_MOVE_LABEL = True        # Show detected move on screen
WINDOW_NAME = "ZeroController - Player 1"
FPS_DISPLAY = True            # Show FPS counter

# ─── LSTM Model Settings ───────────────────────────────────────
LSTM_MODEL_PATH = "models/lstm_move_classifier.pt"
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_SEQ_LEN = 20             # Training sequence window length
LSTM_CONFIDENCE_THRESHOLD = 0.6  # Min probability to trigger a move
LSTM_INPUT_SIZE = 53           # 13 landmarks × 4 values + 1 body_scale
