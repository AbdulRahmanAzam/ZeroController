"""
Data Collector for LSTM Training
=================================
Records MediaPipe pose landmarks with move labels for training the LSTM classifier.

Usage:
  python data_collector.py

Controls:
  Hold a number key to label the current frame:
    1=left_punch  2=right_punch  3=block
    4=left_kick   5=right_kick   6=crouch  7=jump
    8=move_left   9=move_right   0=idle
  No key held = frame is NOT recorded (paused)

  S = Save session to disk
  Q = Quit (prompts to save if unsaved data exists)
  R = Reset (discard current session)

Stand in front of the camera so your full body is visible.
Hold a key and repeatedly perform that move for 30-60 seconds per class.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_TARGET_FPS, CAMERA_BACKEND, CAMERA_BUFFER_SIZE,
    POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_COMPLEXITY,
)

# The 13 landmarks used by the model (same as move_detector.py LM class)
LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Key-to-label mapping (cv2 key codes for number keys)
KEY_LABEL_MAP = {
    ord('0'): 0,   # idle
    ord('1'): 1,   # left_punch
    ord('2'): 2,   # right_punch
    ord('3'): 3,   # block
    ord('4'): 4,   # left_kick
    ord('5'): 5,   # right_kick
    ord('6'): 6,   # crouch
    ord('7'): 7,   # jump
    ord('8'): 8,   # move_left
    ord('9'): 9,   # move_right
}

LABEL_NAMES = {
    0: "IDLE",
    1: "LEFT PUNCH",
    2: "RIGHT PUNCH",
    3: "BLOCK",
    4: "LEFT KICK",
    5: "RIGHT KICK",
    6: "CROUCH",
    7: "JUMP",
    8: "MOVE LEFT",
    9: "MOVE RIGHT",
}

# Colors for each label (BGR)
LABEL_COLORS = {
    0: (150, 150, 150),   # gray
    1: (0, 100, 255),     # orange
    2: (0, 100, 255),     # orange
    3: (255, 200, 0),     # cyan
    4: (0, 255, 255),     # yellow
    5: (0, 255, 255),     # yellow
    6: (255, 0, 150),     # purple
    7: (0, 255, 0),       # green
    8: (255, 150, 0),     # blue
    9: (255, 150, 0),     # blue
}

# Map single label int → three-head labels [upper, lower, movement]
def label_to_heads(label):
    """Convert a single recording label to three-head targets."""
    mapping = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [2, 0, 0],
        3: [3, 0, 0],
        4: [0, 1, 0],
        5: [0, 2, 0],
        6: [0, 3, 0],
        7: [0, 4, 0],
        8: [0, 0, 1],
        9: [0, 0, 2],
    }
    return mapping[label]


def extract_landmarks(pose_landmarks):
    """Extract the 13 relevant landmarks as a (13, 4) numpy array."""
    data = np.zeros((13, 4), dtype=np.float32)
    for i, idx in enumerate(LANDMARK_INDICES):
        lm = pose_landmarks.landmark[idx]
        data[i] = [lm.x, lm.y, lm.z, lm.visibility]
    return data


def open_camera(index, backend_name):
    """Open camera with requested backend."""
    if backend_name == "dshow":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif backend_name == "msmf":
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    except Exception:
        pass
    return cap


def draw_collector_overlay(frame, current_label, frame_count, fps, is_recording):
    """Draw the data collection HUD."""
    h, w = frame.shape[:2]

    # Header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "DATA COLLECTOR", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # Recording status
    if is_recording and current_label is not None:
        label_name = LABEL_NAMES.get(current_label, "?")
        color = LABEL_COLORS.get(current_label, (255, 255, 255))

        # Large move label in center
        text_size = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, label_name, (text_x, h // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Recording indicator
        cv2.circle(frame, (30, 80), 10, (0, 0, 255), -1)  # red dot
        cv2.putText(frame, "RECORDING", (50, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "PAUSED - Hold a number key to record", (15, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

    # Frame counter
    cv2.putText(frame, f"Frames: {frame_count}", (w - 200, 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Key guide at bottom
    guide_y = h - 15
    guide = "0:Idle 1:LPunch 2:RPunch 3:Block 4:LKick 5:RKick 6:Crouch 7:Jump 8:Left 9:Right"
    cv2.putText(frame, guide, (10, guide_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    controls = "S:Save  Q:Quit  R:Reset"
    cv2.putText(frame, controls, (w - 280, guide_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


def save_session(landmarks_list, labels_list, actual_fps):
    """Save recorded data to a .npz file."""
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/session_{timestamp}.npz"

    landmarks_arr = np.array(landmarks_list, dtype=np.float32)  # (T, 13, 4)
    labels_arr = np.array(labels_list, dtype=np.int32)          # (T, 3)

    np.savez_compressed(
        filename,
        landmarks=landmarks_arr,
        labels=labels_arr,
        fps=np.float32(actual_fps),
        timestamp=timestamp,
    )

    print(f"\n[SAVED] {filename}")
    print(f"  Frames: {len(landmarks_list)}")
    print(f"  Shape:  landmarks={landmarks_arr.shape}, labels={labels_arr.shape}")
    print(f"  FPS:    {actual_fps:.1f}")

    # Per-label stats
    single_labels = [l[0] * 100 + l[1] * 10 + l[2] for l in labels_list]
    for label_id in range(10):
        head_labels = labels_arr
        if label_id == 0:
            count = np.sum((head_labels[:, 0] == 0) & (head_labels[:, 1] == 0) & (head_labels[:, 2] == 0))
        elif label_id <= 3:
            count = np.sum(head_labels[:, 0] == {1: 1, 2: 2, 3: 3}[label_id])
        elif label_id <= 7:
            count = np.sum(head_labels[:, 1] == {4: 1, 5: 2, 6: 3, 7: 4}[label_id])
        else:
            count = np.sum(head_labels[:, 2] == {8: 1, 9: 2}[label_id])
        name = LABEL_NAMES[label_id]
        print(f"  {name:15s}: {count:5d} frames")

    return filename


def main():
    print("=" * 60)
    print("  ZERO CONTROLLER — Data Collector")
    print("=" * 60)
    print()
    print("  Hold a number key (0-9) to label and record frames.")
    print("  Release all keys to pause recording.")
    print("  Press S to save, Q to quit, R to reset.")
    print()

    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
        model_complexity=POSE_MODEL_COMPLEXITY,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    # Camera
    cap = open_camera(CAMERA_INDEX, CAMERA_BACKEND.lower())
    if not cap.isOpened():
        print("[ERROR] Could not open camera!")
        return

    print(f"[CAMERA] Opened at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print()

    # Session data
    landmarks_list = []
    labels_list = []
    prev_time = time.time()
    fps = 0
    fps_samples = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            # FPS
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
                fps_samples.append(fps)
            prev_time = now

            # Draw skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # Check which label key is held
            key = cv2.waitKey(1) & 0xFF
            current_label = KEY_LABEL_MAP.get(key, None)
            is_recording = current_label is not None and results.pose_landmarks is not None

            # Record frame if a label key is held AND pose is detected
            if is_recording:
                lm_data = extract_landmarks(results.pose_landmarks)
                head_labels = label_to_heads(current_label)
                landmarks_list.append(lm_data)
                labels_list.append(head_labels)

            # Draw overlay
            frame = draw_collector_overlay(frame, current_label, len(landmarks_list), fps, is_recording)
            cv2.imshow("Data Collector", frame)

            # Handle control keys
            if key == ord('q'):
                if landmarks_list:
                    print(f"\n[QUIT] You have {len(landmarks_list)} unsaved frames.")
                    print("  Saving before exit...")
                    avg_fps = np.mean(fps_samples) if fps_samples else 30.0
                    save_session(landmarks_list, labels_list, avg_fps)
                break
            elif key == ord('s'):
                if landmarks_list:
                    avg_fps = np.mean(fps_samples) if fps_samples else 30.0
                    save_session(landmarks_list, labels_list, avg_fps)
                    landmarks_list = []
                    labels_list = []
                    fps_samples = []
                    print("[RESET] Session saved. Ready for new recording.")
                else:
                    print("[INFO] No frames to save.")
            elif key == ord('r'):
                landmarks_list = []
                labels_list = []
                fps_samples = []
                print("[RESET] Session discarded.")

    except KeyboardInterrupt:
        if landmarks_list:
            avg_fps = np.mean(fps_samples) if fps_samples else 30.0
            save_session(landmarks_list, labels_list, avg_fps)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n[DONE] Data collector closed.")


if __name__ == "__main__":
    main()
