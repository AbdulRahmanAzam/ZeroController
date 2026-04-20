"""ZeroController - 33-point MediaPipe pose visualization."""

import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera_utils import open_camera_with_fallback
from config import (
    CAMERA_INDEX,
    CAMERA_BACKEND,
    WINDOW_NAME,
    MIRROR_VIEW,
    SHOW_CONNECTIONS,
    SHOW_FPS,
    SHOW_LANDMARK_IDS,
    POSE_MODEL_PATH,
    POSE_MODEL_URL,
    POSE_NUM_POSES,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_PRESENCE_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    POINT_COLOR,
    LINE_COLOR,
    TEXT_COLOR,
    POINT_RADIUS,
)


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]


def ensure_pose_model(model_path, model_url):
    """Download pose model if missing."""
    if os.path.exists(model_path):
        return

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print(f"[MODEL] Downloading pose model to: {model_path}")
    urllib.request.urlretrieve(model_url, model_path)
    print("[MODEL] Download complete")


def create_pose_landmarker(model_path):
    """Create a MediaPipe Pose Landmarker configured for image mode."""
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=POSE_NUM_POSES,
        min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=POSE_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _landmark_visibility(landmark):
    if hasattr(landmark, "visibility"):
        return float(landmark.visibility)
    if hasattr(landmark, "presence"):
        return float(landmark.presence)
    return 1.0


def draw_pose(frame, landmarks):
    """Draw 33 landmarks and skeleton connections."""
    h, w = frame.shape[:2]
    points = []

    for idx, lm in enumerate(landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        v = _landmark_visibility(lm)
        points.append((idx, x, y, v))

    if SHOW_CONNECTIONS:
        for a, b in POSE_CONNECTIONS:
            if a >= len(points) or b >= len(points):
                continue
            _, ax, ay, av = points[a]
            _, bx, by, bv = points[b]
            if av < 0.25 or bv < 0.25:
                continue
            cv2.line(frame, (ax, ay), (bx, by), LINE_COLOR, 2, cv2.LINE_AA)

    visible_count = 0
    for idx, x, y, v in points:
        if v < 0.25:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        visible_count += 1
        cv2.circle(frame, (x, y), POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)

        if SHOW_LANDMARK_IDS:
            cv2.putText(
                frame,
                str(idx),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    return visible_count


def draw_hud(frame, fps, pose_found, visible_points, backend_name):
    """Draw basic status text."""
    status = "POSE: DETECTED" if pose_found else "POSE: NOT DETECTED"
    color = (0, 220, 0) if pose_found else (0, 180, 255)

    cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(
        frame,
        f"VISIBLE POINTS: {visible_points}/33",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
    )
    cv2.putText(
        frame,
        f"BACKEND: {backend_name}",
        (15, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
    )

    if SHOW_FPS:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
        )

    cv2.putText(
        frame,
        "Q: Quit",
        (15, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def show_frame(window_name, frame, ui_enabled):
    """Render frame safely, handling headless OpenCV installations."""
    if not ui_enabled:
        return False, -1

    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return True, key
    except cv2.error as err:
        print("[WARN] OpenCV GUI backend is unavailable.")
        print("[WARN] Install GUI-enabled OpenCV to see landmark rendering window.")
        print(f"[WARN] Details: {err}")
        return False, -1


def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - POSE VISUALIZER")
    print("  Mode: 33-point MediaPipe pose tracking")
    print("=" * 60)

    try:
        ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    except Exception as err:
        print(f"[ERROR] Failed to prepare model: {err}")
        sys.exit(1)

    try:
        landmarker = create_pose_landmarker(POSE_MODEL_PATH)
    except Exception as err:
        print(f"[ERROR] Failed to initialize Pose Landmarker: {err}")
        sys.exit(1)

    cap, backend_name = open_camera_with_fallback(CAMERA_INDEX, CAMERA_BACKEND)
    if cap is None:
        print("[ERROR] Could not open a working camera backend.")
        try:
            landmarker.close()
        except Exception:
            pass
        sys.exit(1)

    print(f"[CAMERA] Active backend: {backend_name}")
    print("[INFO] Stand in front of camera. All 33 points will be drawn when pose is detected.")

    prev_t = time.time()
    fps = 0.0
    ui_enabled = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Camera frame read failed.")
                break

            if MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            pose_found = False
            visible_points = 0
            if result.pose_landmarks:
                pose_found = True
                visible_points = draw_pose(frame, result.pose_landmarks[0])

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            draw_hud(frame, fps, pose_found, visible_points, backend_name)
            ui_enabled, key = show_frame(WINDOW_NAME, frame, ui_enabled)
            if not ui_enabled:
                break

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            landmarker.close()
        except Exception:
            pass

    print("[DONE] Pose visualizer closed.")


if __name__ == "__main__":
    main()
