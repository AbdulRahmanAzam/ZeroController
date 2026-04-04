"""
ZeroController — Main Entry Point
==================================
AI-Powered Controller-Free Fighting Game (Part 1: Move Detection)

This script:
  1. Opens your webcam
  2. Uses MediaPipe Pose to track your body in real-time
  3. Detects fighting moves (punch, kick, block, crouch, jump, move)
  4. Presses corresponding keyboard keys
  5. Prints detected moves to the console
  6. Shows a visual overlay with skeleton + move labels

Controls:
  - Press 'q' to quit
  - Press 'r' to recalibrate (stand still again)
  - Press 'p' to pause/resume keyboard output
  - Press 'c' to toggle camera view

Stand still for ~1 second when the app starts so it can calibrate your baseline position.
"""

import os
import cv2
import mediapipe as mp
import time
import sys
from datetime import datetime

import numpy as np
import torch

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_TARGET_FPS, CAMERA_BACKEND, CAMERA_BUFFER_SIZE,
    AUTO_BACKEND_FALLBACK, BLACK_FRAME_MEAN_THRESHOLD, BLACK_FRAME_FALLBACK_COUNT,
    POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_COMPLEXITY, PROCESS_SCALE, USE_GPU,
    SHOW_LANDMARKS, SHOW_MOVE_LABEL, WINDOW_NAME, FPS_DISPLAY,
    KEY_MAP_P1, LSTM_MODEL_PATH,
)
from move_detector import MoveDetector
from keyboard_controller import KeyboardController


# ─── Move display config ─────────────────────────────────────────
MOVE_COLORS = {
    "left_punch":   (0, 100, 255),    # Orange
    "right_punch":  (0, 100, 255),    # Orange
    "left_kick":    (0, 255, 255),    # Yellow
    "right_kick":   (0, 255, 255),    # Yellow
    "block":        (255, 200, 0),    # Cyan-ish
    "crouch":       (255, 0, 150),    # Purple
    "jump":         (0, 255, 0),      # Green
    "move_left":    (255, 150, 0),    # Blue
    "move_right":   (255, 150, 0),    # Blue
}

MOVE_EMOJIS = {
    "left_punch":   "👊 LEFT PUNCH",
    "right_punch":  "👊 RIGHT PUNCH",
    "left_kick":    "🦵 LEFT KICK",
    "right_kick":   "🦵 RIGHT KICK",
    "block":        "🛡️  BLOCK",
    "crouch":       "⬇️  CROUCH",
    "jump":         "⬆️  JUMP",
    "move_left":    "◀️  MOVE LEFT",
    "move_right":   "▶️  MOVE RIGHT",
}

# How long (seconds) a detected move stays visible on the HUD
MOVE_DISPLAY_DURATION = 0.8


def show_frame(window_name, frame, ui_enabled):
    """Render frame if GUI backend is available, otherwise switch to headless mode."""
    if not ui_enabled:
        return False, -1

    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return True, key
    except cv2.error as err:
        print("\n[WARN] OpenCV GUI backend is not available. Switching to headless mode.")
        print("[WARN] Install GUI-enabled OpenCV and remove headless builds if you need video preview.")
        print("[WARN] OpenCV key controls are disabled in headless mode. Use Ctrl+C to quit.")
        print(f"[WARN] Details: {err}")
        return False, -1


def open_camera(index, backend_name):
    """Open camera with requested backend and apply capture settings."""
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


def draw_overlay(frame, recent_moves, fps, is_calibrated, keyboard_active, detector):
    """Draw HUD overlay on the camera frame.

    recent_moves: list of (move_name, timestamp) tuples for moves still within display duration.
    """
    h, w = frame.shape[:2]
    now = time.time()

    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "ZERO CONTROLLER", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

    # FPS
    if FPS_DISPLAY:
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Status indicators
    status_y = 80
    if not is_calibrated:
        progress = int((detector.calibration_frames / detector.calibration_target) * 100)
        cv2.putText(frame, f"CALIBRATING... Stand still! ({progress}%)", (15, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        # Progress bar
        bar_w = 300
        bar_h = 15
        bar_x, bar_y = 15, status_y + 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * (detector.calibration_frames / detector.calibration_target))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 255), -1)
    else:
        status_color = (0, 255, 0) if keyboard_active else (0, 150, 255)
        status_text = "KEYBOARD: ON" if keyboard_active else "KEYBOARD: PAUSED"
        cv2.putText(frame, status_text, (15, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Detected moves panel — show recent moves with fade-out
    if recent_moves and SHOW_MOVE_LABEL:
        # Deduplicate: keep the most recent timestamp per move name
        unique = {}
        for move, ts in recent_moves:
            unique[move] = ts
        display_list = list(unique.items())

        panel_y = h - 30 - (len(display_list) * 50)
        # Semi-transparent panel background
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (10, panel_y - 10), (350, h - 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

        for i, (move, ts) in enumerate(display_list):
            y = panel_y + (i * 50) + 30
            age = now - ts
            # Fade opacity: 1.0 when fresh, 0.3 when about to expire
            alpha = max(0.3, 1.0 - (age / MOVE_DISPLAY_DURATION) * 0.7)
            base_color = MOVE_COLORS.get(move, (255, 255, 255))
            color = tuple(int(c * alpha) for c in base_color)

            label = move.replace("_", " ").upper()

            # Draw accent line
            cv2.rectangle(frame, (15, y - 25), (20, y + 5), color, -1)
            # Draw move text
            cv2.putText(frame, label, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Help text at bottom
    help_text = "Q:Quit  R:Recalibrate  P:Pause Keys"
    cv2.putText(frame, help_text, (w - 400, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


def main():
    print("=" * 60)
    print("  ZERO CONTROLLER — AI Fighting Game Input System")
    print("  Part 1: MediaPipe Pose → Move Detection → Keyboard")
    print("=" * 60)
    print()
    print(f"  Camera Index     : {CAMERA_INDEX}")
    print(f"  Resolution       : {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"  Target FPS       : {CAMERA_TARGET_FPS}")
    print(f"  Model Complexity : {POSE_MODEL_COMPLEXITY}")
    print(f"  Process Scale    : {PROCESS_SCALE}")
    print(f"  Camera Backend   : {CAMERA_BACKEND}")
    print()
    print("  INSTRUCTIONS:")
    print("  1. Stand in front of your camera (full body visible)")
    print("  2. Stand STILL for ~1 second (calibration)")
    print("  3. Start fighting! Punch, kick, block, crouch, jump")
    print()
    print("  CONTROLS:")
    print("  Q = Quit | R = Recalibrate | P = Pause/Resume keyboard")
    print()
    print("  KEY MAPPINGS:")
    for move, key in KEY_MAP_P1.items():
        print(f"    {move:15s} → '{key}'")
    print()
    print("-" * 60)
    print()

    # ─── Initialize components ────────────────────────────────────
    if os.path.exists(LSTM_MODEL_PATH):
        from lstm_detector import LSTMMoveDetector
        detector = LSTMMoveDetector(LSTM_MODEL_PATH)
        print(f"[MODEL] LSTM detector loaded from {LSTM_MODEL_PATH}")
    else:
        detector = MoveDetector()
        print("[MODEL] Using rule-based detector (no LSTM model found)")
    keyboard = KeyboardController(KEY_MAP_P1)
    keyboard_active = True

    # ─── Initialize MediaPipe Pose ────────────────────────────────
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

    # ─── Open Camera ──────────────────────────────────────────────
    print("[CAMERA] Opening camera...")
    backend_name = CAMERA_BACKEND.lower()
    cap = open_camera(CAMERA_INDEX, backend_name)

    if not cap.isOpened():
        print("[ERROR] Could not open camera! Check your camera index in config.py")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAMERA] Opened successfully! Resolution: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    # ─── GPU / CUDA setup ─────────────────────────────────────────
    cv2.setUseOptimized(True)
    gpu_device = None
    if USE_GPU and torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        print(f"[PERF] CUDA GPU: {gpu_name}  ({vram_gb} GB)")
        print(f"[PERF] Frame preprocessing will run on GPU")
        # Pre-allocate a reusable CUDA tensor for resize ops
        _tgt_h = int(CAMERA_HEIGHT * PROCESS_SCALE) if PROCESS_SCALE < 1.0 else CAMERA_HEIGHT
        _tgt_w = int(CAMERA_WIDTH  * PROCESS_SCALE) if PROCESS_SCALE < 1.0 else CAMERA_WIDTH
    else:
        if USE_GPU:
            print("[PERF] USE_GPU=True but CUDA not available — falling back to CPU")
        else:
            print("[PERF] GPU disabled. Running on CPU")
        gpu_device = None

    black_frame_count = 0
    print()
    print("[CALIBRATION] Stand still in front of the camera...")
    print()

    # ─── Main Loop ────────────────────────────────────────────────
    prev_frame_time = time.time()
    fps = 0
    move_log = []  # Log all moves for future coaching feature
    recent_moves = []  # (move_name, timestamp) — moves currently visible on HUD
    ui_enabled = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera!")
                break

            # Some backends return valid-sized but all-black frames on specific webcams.
            frame_mean = float(frame.mean())
            if frame_mean < BLACK_FRAME_MEAN_THRESHOLD:
                black_frame_count += 1
            else:
                black_frame_count = 0

            if (AUTO_BACKEND_FALLBACK and black_frame_count >= BLACK_FRAME_FALLBACK_COUNT
                    and backend_name != "msmf"):
                print("[CAMERA] Black frames detected. Falling back to MSMF backend...")
                cap.release()
                backend_name = "msmf"
                cap = open_camera(CAMERA_INDEX, backend_name)
                black_frame_count = 0
                continue

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Resize + BGR→RGB for MediaPipe.
            # When CUDA is available we use torch to resize on the GPU then pull back to CPU.
            if gpu_device is not None and PROCESS_SCALE < 1.0:
                # frame: HxWx3 uint8 BGR numpy array
                t = torch.from_numpy(frame).to(gpu_device, non_blocking=True)   # HxWx3
                t = t.permute(2, 0, 1).unsqueeze(0).float()                     # 1x3xHxW
                t = torch.nn.functional.interpolate(
                    t, scale_factor=PROCESS_SCALE, mode="bilinear", align_corners=False
                )
                t = t.squeeze(0).permute(1, 2, 0).byte()                        # h'xw'x3
                proc_bgr = t.cpu().numpy()
                # Reverse channel order BGR→RGB
                rgb_frame = proc_bgr[:, :, ::-1].copy()
            elif PROCESS_SCALE < 1.0:
                proc_bgr = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                                      interpolation=cv2.INTER_LINEAR)
                rgb_frame = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process with MediaPipe Pose
            results = pose.process(rgb_frame)

            rgb_frame.flags.writeable = True

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_frame_time + 1e-6)
            prev_frame_time = current_time

            detected_moves = []

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw skeleton
                if SHOW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                # Detect moves
                detected_moves = detector.detect(landmarks)

                # Process detected moves
                if detected_moves:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    for move in detected_moves:
                        # Console output with color coding
                        emoji = MOVE_EMOJIS.get(move, move)
                        key = KEY_MAP_P1.get(move, "?")
                        print(f"  [{timestamp}]  {emoji:20s}  →  Key: '{key}'")

                        # Log for future coaching
                        move_log.append({
                            "time": current_time,
                            "move": move,
                            "key": key,
                        })

                        # Add to recent moves for HUD display
                        recent_moves.append((move, current_time))

                    # Press keyboard keys
                    if keyboard_active:
                        keyboard.press_moves(detected_moves)
            else:
                # No pose detected
                if not detector.is_calibrated:
                    pass  # Still waiting for person to appear
                # Optionally show "No pose detected" message

            # Expire old moves from the recent list
            recent_moves = [(m, t) for m, t in recent_moves
                            if current_time - t < MOVE_DISPLAY_DURATION]

            # Draw HUD overlay
            frame = draw_overlay(frame, recent_moves, fps,
                                detector.is_calibrated, keyboard_active, detector)

            # Show frame and read UI key input only when GUI backend is available.
            ui_enabled, key = show_frame(WINDOW_NAME, frame, ui_enabled)
            if ui_enabled:
                if key == ord('q'):
                    print("\n[QUIT] Shutting down...")
                    break
                elif key == ord('r'):
                    print("\n[RECALIBRATE] Stand still...")
                    detector.reset_calibration()
                elif key == ord('p'):
                    keyboard_active = not keyboard_active
                    state = "ON" if keyboard_active else "PAUSED"
                    print(f"\n[KEYBOARD] {state}")

    except KeyboardInterrupt:
        print("\n[QUIT] Interrupted by user")

    finally:
        # Cleanup
        keyboard.release_all()
        cap.release()
        if ui_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # Ignore cleanup error when no GUI backend exists.
                pass
        pose.close()

        # Print session summary
        print()
        print("=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        if move_log:
            move_counts = {}
            for entry in move_log:
                m = entry["move"]
                move_counts[m] = move_counts.get(m, 0) + 1

            total = len(move_log)
            print(f"  Total Moves Detected: {total}")
            print()
            for move, count in sorted(move_counts.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 30)
                print(f"    {move:15s}: {count:4d}  {bar}")
        else:
            print("  No moves detected this session.")
        print()
        print("  Session ended. GG! 🎮")
        print("=" * 60)


if __name__ == "__main__":
    main()
