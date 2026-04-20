"""Run the trained punch classifier in real-time with camera.

Shows live camera feed with pose skeleton + predicted action label.

Run:  python run_model.py
"""

import os
import sys

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
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_restore_stderr(_saved_fd)

from camera_utils import open_camera_with_fallback
from config import (
    CAMERA_BACKEND,
    CAMERA_INDEX,
    LINE_COLOR,
    MIRROR_VIEW,
    MODEL_SAVE_PATH,
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
from train_model import PunchLSTM


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


def _landmarks_to_flat(landmarks):
    """33 landmarks → flat array (132,)."""
    row = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        row[i] = [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, "visibility") else 1.0]
    return row.flatten()


def _draw_skeleton(frame, landmarks):
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


def _load_classifier(model_path, device):
    """Load trained PunchLSTM from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = PunchLSTM(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        num_classes=len(ckpt["actions"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["actions"]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - LIVE PUNCH DETECTION")
    print("=" * 60)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] No trained model found at {MODEL_SAVE_PATH}")
        print("[ERROR] Run train_model.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, actions = _load_classifier(MODEL_SAVE_PATH, device)
    print(f"[MODEL] Loaded {MODEL_SAVE_PATH}  |  Actions: {actions}  |  Device: {device}")

    ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    landmarker = _create_landmarker(POSE_MODEL_PATH)

    cap, backend = open_camera_with_fallback(CAMERA_INDEX, CAMERA_BACKEND)
    if cap is None:
        print("[ERROR] No working camera found.")
        landmarker.close()
        sys.exit(1)
    print(f"[CAMERA] Backend: {backend}")

    # Rolling buffer of last SEQUENCE_LENGTH frames
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

    prediction = "..."
    confidence = 0.0
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

            if landmarks:
                _draw_skeleton(frame, landmarks)
                frame_buffer.append(_landmarks_to_flat(landmarks))

                # Predict when buffer is full
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    seq = np.stack(list(frame_buffer), axis=0)  # (30, 132)
                    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, 30, 132)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1)
                        conf, idx = probs.max(dim=1)
                        prediction = actions[idx.item()]
                        confidence = conf.item()

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            # ── HUD ──
            h, w = frame.shape[:2]

            # Prediction box
            label_text = f"{prediction}  ({confidence:.0%})"
            color = (0, 255, 0) if confidence > 0.7 else (0, 180, 255)
            cv2.putText(frame, label_text, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

            # Buffer progress bar
            bar_w = 200
            filled = int(bar_w * len(frame_buffer) / SEQUENCE_LENGTH)
            cv2.rectangle(frame, (15, 55), (15 + bar_w, 70), (80, 80, 80), -1)
            cv2.rectangle(frame, (15, 55), (15 + filled, 70), (0, 200, 0), -1)

            cv2.putText(frame, f"FPS: {fps:.1f}", (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

            cv2.putText(frame, "Q: Quit", (15, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

            try:
                cv2.imshow("ZeroController - Live Detection", frame)
            except cv2.error:
                break

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print("[DONE] Live detection closed.")


if __name__ == "__main__":
    main()
