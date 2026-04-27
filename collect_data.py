"""Collect pose-landmark sequences for multi-action training data.

Controls:
    1-9    - Select action by number (shown in sidebar)
    SPACE  - Start recording a sequence (RECORD_DURATION_SEC seconds)
    A      - Toggle auto-mode (auto-records after a short delay after each save)
    L      - Cycle to next action label
    Q      - Quit

Each sample is saved as  data/raw/<label>/sample_<number>.npy
with shape (SEQUENCE_LENGTH, 33, 4)  ->  [x, y, z, visibility] per landmark.
Recording always captures RECORD_DURATION_SEC of real wall-clock time, then
resamples to exactly SEQUENCE_LENGTH frames — so every sample represents the
same real-world duration regardless of camera frame rate.
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


def _resample_sequence(buffer, target_len):
    """Linearly resample a variable-length frame buffer to exactly target_len frames."""
    n = len(buffer)
    if n == 0:
        return np.zeros((target_len, 33, 4), dtype=np.float32)
    if n == target_len:
        return np.stack(buffer, axis=0)
    arr = np.stack(buffer, axis=0)          # (n, 33, 4)
    src_idx = np.linspace(0, n - 1, target_len)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    t = (src_idx - lo)[:, None, None]      # (target_len, 1, 1) for broadcasting
    return arr[lo] * (1 - t) + arr[hi] * t


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


def _draw_collection_hud(frame, actions, label_idx, saved_counts, recording, rec_elapsed, fps, auto_mode, countdown):
    """Draw data-collection status overlay with sidebar showing all actions."""
    h, w = frame.shape[:2]

    # ── Recording border flash ────────────────────────────────────────────────
    if recording:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(frame, f"REC  {rec_elapsed:.2f}s / {RECORD_DURATION_SEC:.1f}s",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    elif countdown > 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
        cv2.putText(frame, f"GET READY  {countdown}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    else:
        status = "AUTO" if auto_mode else "READY"
        color = (0, 200, 255) if auto_mode else (0, 220, 0)
        cv2.putText(frame, status, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

    # ── Right-side action sidebar ─────────────────────────────────────────────
    sidebar_x = w - 240
    cv2.rectangle(frame, (sidebar_x - 8, 0), (w, len(actions) * 34 + 14), (20, 20, 20), -1)
    cv2.putText(frame, "ACTIONS", (sidebar_x, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    for i, action in enumerate(actions):
        y = 44 + i * 34
        is_active = (i == label_idx)
        bg = (50, 120, 200) if is_active else (40, 40, 40)
        cv2.rectangle(frame, (sidebar_x - 6, y - 20), (w - 2, y + 10), bg, -1)
        count = saved_counts.get(action, 0)
        key_hint = str(i + 1) if i < 9 else "-"
        text = f"[{key_hint}] {action:<12} x{count}"
        text_color = (255, 255, 0) if is_active else TEXT_COLOR
        cv2.putText(frame, text, (sidebar_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1)

    # ── Bottom controls bar ───────────────────────────────────────────────────
    controls = "SPACE:Record  |  1-9:Action  |  L:Next  |  A:Auto  |  Q:Quit"
    cv2.putText(frame, controls, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_COLOR, 1, cv2.LINE_AA)


# ── main ─────────────────────────────────────────────────────────────────────

# How many seconds to wait before auto-recording starts
AUTO_DELAY_SEC = 2.0
# Countdown seconds shown before recording begins
COUNTDOWN_SEC = 3
# Fixed real-world duration of each recorded sample (seconds).
# Frames collected during this window are resampled to SEQUENCE_LENGTH.
RECORD_DURATION_SEC = max(1.0, SEQUENCE_LENGTH / 30.0)


def _switch_label(idx, actions):
    """Return (label, label_dir, saved) for a given action index."""
    label = actions[idx]
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    return label, label_dir


def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - DATA COLLECTION")
    print(f"  Actions ({len(ACTIONS)}): {ACTIONS}")
    print(f"  Sequence length: {SEQUENCE_LENGTH} frames")
    print("=" * 60)
    print("  Keys: 1-9 select action | SPACE record | A auto | L next | Q quit")
    print("=" * 60)

    ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    landmarker = _create_landmarker(POSE_MODEL_PATH)

    cap, backend = open_camera_with_fallback(CAMERA_INDEX, CAMERA_BACKEND)
    if cap is None:
        print("[ERROR] No working camera found.")
        landmarker.close()
        sys.exit(1)
    print(f"[CAMERA] Backend: {backend}")

    # Build saved-count map for all actions
    saved_counts = {}
    for action in ACTIONS:
        d = os.path.join(DATA_DIR, action)
        saved_counts[action] = _count_existing(d)
        os.makedirs(d, exist_ok=True)

    # State
    label_idx = 0
    label, label_dir = _switch_label(label_idx, ACTIONS)

    recording = False
    buffer = []
    rec_start_t = None          # wall-clock time when current recording started

    video_writer = None        # cv2.VideoWriter for the current sample clip
    frame_size = None          # (width, height) captured from first frame

    auto_mode = False          # auto-records repeatedly after each save
    auto_next_t = None         # time.time() when next auto-record should start

    countdown = 0              # seconds remaining in pre-record countdown
    countdown_end_t = None     # time.time() when countdown finishes

    prev_t = time.time()
    fps = 0.0
    _first_detect = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            # Capture frame dimensions once (needed for VideoWriter)
            if frame_size is None:
                fh, fw = frame.shape[:2]
                frame_size = (fw, fh)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts = int(time.time() * 1000)  # real wall-clock ms

            if _first_detect:
                _fd = _suppress_stderr()
                result = landmarker.detect_for_video(mp_image, frame_ts)
                _restore_stderr(_fd)
                _first_detect = False
            else:
                result = landmarker.detect_for_video(mp_image, frame_ts)

            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

            # ── Countdown logic ───────────────────────────────────────────────
            if countdown > 0 and countdown_end_t is not None:
                remaining = countdown_end_t - time.time()
                countdown = max(0, int(remaining) + 1)
                if time.time() >= countdown_end_t:
                    countdown = 0
                    if landmarks is not None:
                        recording = True
                        buffer = []
                        rec_start_t = time.time()
                        # Open a video clip alongside the .npy file
                        n = saved_counts[label]
                        vid_path = os.path.join(label_dir, f"sample_{n:04d}.avi")
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        video_writer = cv2.VideoWriter(
                            vid_path, fourcc, 30.0, frame_size or (1280, 720)
                        )
                        if not video_writer.isOpened():
                            print("[WARN] VideoWriter failed to open — clip will NOT be saved.")
                            video_writer = None
                        print(f"[REC] Recording '{label}' sample #{n}...")
                    else:
                        print("[WARN] No pose at countdown end — skipping.")

            # ── Auto-mode trigger ─────────────────────────────────────────────
            if auto_mode and not recording and countdown == 0 and auto_next_t is not None:
                if time.time() >= auto_next_t:
                    auto_next_t = None
                    countdown = COUNTDOWN_SEC
                    countdown_end_t = time.time() + COUNTDOWN_SEC

            # Draw skeleton
            if landmarks:
                _draw_skeleton(frame, landmarks)

            # ── Recording logic ───────────────────────────────────────────────
            if recording and landmarks:
                buffer.append(_landmarks_to_array(landmarks))
                elapsed = time.time() - rec_start_t
                if elapsed >= RECORD_DURATION_SEC:
                    seq = _resample_sequence(buffer, SEQUENCE_LENGTH)  # (SEQUENCE_LENGTH, 33, 4)
                    n = saved_counts[label]
                    sample_path = os.path.join(label_dir, f"sample_{n:04d}.npy")
                    np.save(sample_path, seq)
                    # Finalize the video clip
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        vid_path = os.path.join(label_dir, f"sample_{n:04d}.avi")
                        print(f"[VIDEO] Saved {vid_path}")
                    saved_counts[label] += 1
                    print(f"[SAVED] {sample_path}  ({len(buffer)} raw frames → {SEQUENCE_LENGTH} resampled, "
                          f"{elapsed:.2f}s)  total {saved_counts[label]} for '{label}'")
                    recording = False
                    buffer = []
                    rec_start_t = None
                    if auto_mode:
                        auto_next_t = time.time() + AUTO_DELAY_SEC

            # ── HUD ───────────────────────────────────────────────────────────
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            rec_elapsed = (now - rec_start_t) if (recording and rec_start_t is not None) else 0.0
            _draw_collection_hud(
                frame, ACTIONS, label_idx, saved_counts,
                recording, rec_elapsed, fps, auto_mode, countdown
            )

            # Write annotated frame (with skeleton + HUD) to clip while recording
            if recording and video_writer is not None:
                video_writer.write(frame)

            try:
                cv2.imshow("ZeroController - Collect Data", frame)
            except cv2.error:
                print("[WARN] OpenCV GUI unavailable.")
                break

            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ──────────────────────────────────────────────────
            if key == ord("q"):
                break

            elif key == ord(" "):
                if not recording and countdown == 0:
                    if landmarks is None:
                        print("[WARN] No pose detected — stand in view first.")
                    else:
                        countdown = COUNTDOWN_SEC
                        countdown_end_t = time.time() + COUNTDOWN_SEC
                        print(f"[COUNTDOWN] {COUNTDOWN_SEC}s before recording '{label}'...")

            elif key == ord("l"):
                # Cycle to next label
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                label_idx = (label_idx + 1) % len(ACTIONS)
                label, label_dir = _switch_label(label_idx, ACTIONS)
                recording = False
                buffer = []
                rec_start_t = None
                countdown = 0
                auto_next_t = None
                print(f"[LABEL] → {label}  ({saved_counts[label]} existing)")

            elif key == ord("a"):
                auto_mode = not auto_mode
                if auto_mode:
                    print("[AUTO] Auto-mode ON — will record repeatedly with countdown.")
                    if landmarks is not None and not recording and countdown == 0:
                        auto_next_t = time.time() + AUTO_DELAY_SEC
                else:
                    print("[AUTO] Auto-mode OFF.")
                    auto_next_t = None

            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")   # '1' → 0, '2' → 1, …
                if idx < len(ACTIONS):
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    label_idx = idx
                    label, label_dir = _switch_label(label_idx, ACTIONS)
                    recording = False
                    buffer = []
                    rec_start_t = None
                    countdown = 0
                    auto_next_t = None
                    print(f"[LABEL] → {label}  ({saved_counts[label]} existing)")

    except KeyboardInterrupt:
        pass
    finally:
        if video_writer is not None:
            video_writer.release()   # discard any incomplete clip
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print("\n[DONE] Collection summary:")
    for action in ACTIONS:
        print(f"  {action:<12} : {saved_counts[action]} samples")


if __name__ == "__main__":
    main()
