"""Collect pose-landmark sequences for punch training data.

Controls:
    SPACE  - Start recording a sequence (SEQUENCE_LENGTH frames)
    L      - Cycle through action labels
    Q      - Quit

Each sample is saved as  data/raw/<label>/sample_<number>.npy
with shape (SEQUENCE_LENGTH, 33, 4)  ->  [x, y, z, visibility] per landmark.
"""

import os
import sys

# ── Suppress C++ stderr noise from TFLite / MediaPipe ────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

def _suppress_stderr():
    _old = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    return _old

def _restore_stderr(fd):
    os.dup2(fd, 2)
    os.close(fd)

_saved_fd = _suppress_stderr()

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_restore_stderr(_saved_fd)

from camera_utils import open_camera_with_fallback
from config import (
    ACTIONS,
    CAMERA_BACKEND,
    CAMERA_INDEX,
    DATA_DIR,
    LINE_COLOR,
    MIRROR_VIEW,
    POINT_COLOR,
    POINT_RADIUS,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_PRESENCE_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_PATH,
    POSE_MODEL_URL,
    POSE_NUM_POSES,
    SEQUENCE_LENGTH,
    SHOW_CONNECTIONS,
    TEXT_COLOR,
)
from main import POSE_CONNECTIONS, ensure_pose_model


# ── helpers ──────────────────────────────────────────────────────────────────

def _create_landmarker(model_path):
    _fd = _suppress_stderr()
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=POSE_NUM_POSES,
        min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=POSE_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    _restore_stderr(_fd)
    return landmarker
    return vision.PoseLandmarker.create_from_options(options)


def _landmarks_to_array(landmarks):
    """Convert 33 MediaPipe landmarks → numpy array (33, 4)."""
    row = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        row[i] = [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, "visibility") else 1.0]
    return row


def _count_existing(label_dir):
    """Count .npy files already present in a label directory."""
    if not os.path.isdir(label_dir):
        return 0
    return sum(1 for f in os.listdir(label_dir) if f.endswith(".npy"))


def _draw_skeleton(frame, landmarks):
    """Lightweight skeleton overlay (no IDs, keep it fast)."""
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        v = lm.visibility if hasattr(lm, "visibility") else 1.0
        pts.append((x, y, v))

    if SHOW_CONNECTIONS:
        for a, b in POSE_CONNECTIONS:
            if a >= len(pts) or b >= len(pts):
                continue
            if pts[a][2] < 0.25 or pts[b][2] < 0.25:
                continue
            cv2.line(frame, pts[a][:2], pts[b][:2], LINE_COLOR, 2, cv2.LINE_AA)

    for x, y, v in pts:
        if v < 0.25 or x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(frame, (x, y), POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)


def _draw_collection_hud(frame, label, saved, recording, rec_progress, fps):
    """Draw data-collection status overlay."""
    h, w = frame.shape[:2]

    # Recording flash
    if recording:
        border_color = (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 4)
        cv2.putText(frame, f"REC  {rec_progress}/{SEQUENCE_LENGTH}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 2)
    else:
        cv2.putText(frame, "READY", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)

    cv2.putText(frame, f"Label: {label}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, f"Saved: {saved}", (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

    # Controls
    controls = "SPACE: Record  |  L: Label  |  Q: Quit"
    cv2.putText(frame, controls, (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - DATA COLLECTION")
    print(f"  Actions: {ACTIONS}")
    print(f"  Sequence length: {SEQUENCE_LENGTH} frames")
    print("=" * 60)

    ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    landmarker = _create_landmarker(POSE_MODEL_PATH)

    cap, backend = open_camera_with_fallback(CAMERA_INDEX, CAMERA_BACKEND)
    if cap is None:
        print("[ERROR] No working camera found.")
        landmarker.close()
        sys.exit(1)
    print(f"[CAMERA] Backend: {backend}")

    # State
    label_idx = 0
    label = ACTIONS[label_idx]
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    saved = _count_existing(label_dir)

    recording = False
    buffer = []
    prev_t = time.time()
    fps = 0.0
    frame_ts = 0
    _first_detect = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33

            if _first_detect:
                _fd = _suppress_stderr()
                result = landmarker.detect_for_video(mp_image, frame_ts)
                _restore_stderr(_fd)
                _first_detect = False
            else:
                result = landmarker.detect_for_video(mp_image, frame_ts)

            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

            # Draw skeleton
            if landmarks:
                _draw_skeleton(frame, landmarks)

            # Recording logic
            if recording and landmarks:
                buffer.append(_landmarks_to_array(landmarks))
                if len(buffer) >= SEQUENCE_LENGTH:
                    # Save sequence
                    seq = np.stack(buffer, axis=0)  # (SEQUENCE_LENGTH, 33, 4)
                    sample_path = os.path.join(label_dir, f"sample_{saved:04d}.npy")
                    np.save(sample_path, seq)
                    saved += 1
                    print(f"[SAVED] {sample_path}  ({saved} total)")
                    recording = False
                    buffer = []

            # HUD
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            _draw_collection_hud(frame, label, saved, recording, len(buffer), fps)

            try:
                cv2.imshow("ZeroController - Collect Data", frame)
            except cv2.error:
                print("[WARN] OpenCV GUI unavailable.")
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                if not recording:
                    if landmarks is None:
                        print("[WARN] No pose detected — stand in view before recording.")
                    else:
                        recording = True
                        buffer = []
                        print(f"[REC] Recording {label} sample #{saved}...")

            elif key == ord("l"):
                label_idx = (label_idx + 1) % len(ACTIONS)
                label = ACTIONS[label_idx]
                label_dir = os.path.join(DATA_DIR, label)
                os.makedirs(label_dir, exist_ok=True)
                saved = _count_existing(label_dir)
                print(f"[LABEL] Switched to: {label}  ({saved} existing)")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print("[DONE] Data collection finished.")


if __name__ == "__main__":
    main()
