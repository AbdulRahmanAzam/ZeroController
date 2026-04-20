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
ACTIONS = ["right_punch", "left_punch"]

# --- Training ---
MODEL_SAVE_PATH = "models/punch_classifier.pth"
EPOCHS = 120
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LSTM_LAYERS = 1
