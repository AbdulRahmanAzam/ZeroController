"""Run the trained action classifier live with the camera.

What this script does, step by step
-----------------------------------
1. Load the trained checkpoint (ST-GCN / TCN / LSTM — auto-detected).
2. Read frames from the camera and run MediaPipe Pose on each.
3. Keep the last 30 frames of 33×4 landmark data in a rolling buffer.
4. When the buffer is full, preprocess and classify → softmax probability.
5. Feed the prediction into an `ActionGate` that only *emits* a trigger when
   the same label stays at high confidence for a few consecutive frames —
   this turns noisy per-frame predictions into clean game inputs.
6. Overlay the current prediction, the stable label, and the last triggered
   action on the camera feed.

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
    PREDICT_CONFIDENCE_THRESHOLD,
    PREDICT_STABLE_FRAMES,
    PREDICT_TRIGGER_COOLDOWN,
    SEQUENCE_LENGTH,
    SHOW_CONNECTIONS,
    TEXT_COLOR,
)
from main import POSE_CONNECTIONS, ensure_pose_model
from train_model import (
    build_model_from_ckpt,
    preprocess_pose_sequence,
)


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


def _landmarks_to_frame(landmarks):
    """33 MediaPipe landmarks → numpy array (33, 4) of [x, y, z, visibility].

    This is the SAME raw layout used by collect_data.py and by the dataset
    loader, so the preprocessing pipeline sees identical data at train time
    and at inference time.
    """
    row = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        row[i] = [
            lm.x,
            lm.y,
            lm.z,
            lm.visibility if hasattr(lm, "visibility") else 1.0,
        ]
    return row


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


# ── Checkpoint loading ────────────────────────────────────────────────────────

def _load_classifier(model_path, device):
    """Load any trained classifier (ST-GCN / TCN / LSTM) from a checkpoint.

    Returns
    -------
    model           : torch.nn.Module (eval mode, on `device`)
    actions         : list[str]       — class labels in logit order
    model_type      : str             — "stgcn" | "tcn" | "lstm"
    preprocess_args : dict            — preprocessing flags saved at train time
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    model_type = ckpt.get("model_type", "lstm")

    # Grab the preprocessing flags used at train time. We fall back to True for
    # old checkpoints that didn't save them, matching the training defaults.
    preprocess_args = {
        "hip_center":     ckpt.get("preprocess_hip_center",     True),
        "scale_norm":     ckpt.get("preprocess_scale_norm",     True),
        "use_visibility": ckpt.get("preprocess_use_visibility", True),
        "use_velocity":   ckpt.get("stgcn_use_velocity",        True),
    }

    print(f"[MODEL] {model_type.upper()} loaded  |  actions: {ckpt['actions']}")
    return model, ckpt["actions"], model_type, preprocess_args


# ── Buffer → model input ─────────────────────────────────────────────────────

def _buffer_to_tensor(buffer, model_type, preprocess_args, device):
    """Turn the rolling landmark buffer into an input tensor the model expects.

    buffer : deque of (33, 4) numpy arrays, length == SEQUENCE_LENGTH
    """
    # Stack into a single (T, 33, 4) sample — same layout as collect_data.py saves.
    seq = np.stack(list(buffer), axis=0)  # (30, 33, 4)

    if model_type == "stgcn":
        # Run the SAME preprocessing the training set went through. Using the
        # flags stored in the checkpoint guarantees we never accidentally drift
        # from the trained distribution.
        arr = preprocess_pose_sequence(
            seq,
            use_velocity   = preprocess_args["use_velocity"],
            hip_center     = preprocess_args["hip_center"],
            scale_norm     = preprocess_args["scale_norm"],
            use_visibility = preprocess_args["use_visibility"],
        )  # (C, 30, 33)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, C, 30, 33)
    else:
        # LSTM / TCN use a flat (T, 132) layout.
        flat = seq.reshape(SEQUENCE_LENGTH, -1).astype(np.float32)
        tensor = torch.from_numpy(flat).unsqueeze(0)  # (1, 30, 132)

    return tensor.to(device)


# ── Action gating ────────────────────────────────────────────────────────────

class ActionGate:
    """Stabilises raw per-frame predictions into game-ready triggers.

    Without gating, a single punch sample can produce 5-10 consecutive
    predictions of "right_punch" as the rolling window slides over it — the
    game would register multiple punches from one motion. The gate solves
    two problems:

      1. Flicker suppression: a label is "stable" only if it survives
         `stable_frames` consecutive frames at confidence ≥ `threshold`.
      2. One-shot trigger: a non-idle action fires exactly once, then a
         `cooldown` blocks further triggers. Returning to idle re-arms it.
    """
    def __init__(self, threshold, stable_frames, cooldown, idle_label="idle"):
        self.threshold     = threshold
        self.stable_frames = stable_frames
        self.cooldown      = cooldown
        self.idle_label    = idle_label
        self._last_pred       = None     # last raw per-frame prediction
        self._stable_count    = 0        # streak length of _last_pred above threshold
        self._cooldown_left   = 0        # frames remaining before a new trigger is allowed
        self._last_triggered  = None     # last action that actually fired (non-idle)

    def step(self, pred_label, confidence):
        """Feed one prediction; returns (stable_label, triggered).

        stable_label : the stabilised label currently held (None while still settling)
        triggered    : True only on the rising edge — the single frame where a
                       new non-idle action actually fires.
        """
        # Count down the cooldown from the last trigger.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # Extend the streak if the high-confidence prediction is unchanged.
        if pred_label == self._last_pred and confidence >= self.threshold:
            self._stable_count += 1
        else:
            # New label, or confidence dropped below threshold — restart counting.
            self._stable_count = 1 if confidence >= self.threshold else 0
        self._last_pred = pred_label

        stable_label = (pred_label
                        if self._stable_count >= self.stable_frames
                        else None)

        triggered = False
        if stable_label == self.idle_label:
            # Returning to idle re-arms the gate so the same action can fire again.
            self._last_triggered = None
        elif (stable_label is not None
              and stable_label != self._last_triggered
              and self._cooldown_left == 0):
            # Rising edge of a fresh non-idle action: fire once, start cooldown.
            triggered = True
            self._last_triggered = stable_label
            self._cooldown_left  = self.cooldown

        return stable_label, triggered


# ── HUD ──────────────────────────────────────────────────────────────────────

def _draw_hud(frame, *, prediction, confidence, stable_label, triggered,
              last_triggered, buffer_fill, fps, model_type):
    h, w = frame.shape[:2]

    # Top-left: current (per-frame) prediction and confidence.
    pred_text = f"{prediction}  ({confidence:.0%})"
    pred_col  = (0, 255, 0) if confidence > 0.7 else (0, 180, 255)
    cv2.putText(frame, pred_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, pred_col, 3, cv2.LINE_AA)

    # Buffer-fill progress bar: lets you see how many frames the model has.
    bar_w = 240
    filled = int(bar_w * buffer_fill / SEQUENCE_LENGTH)
    cv2.rectangle(frame, (15, 55), (15 + bar_w, 70), (80, 80, 80), -1)
    cv2.rectangle(frame, (15, 55), (15 + filled, 70), (0, 200, 0), -1)

    cv2.putText(frame, f"FPS: {fps:.1f}  |  model: {model_type.upper()}",
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

    # Stable label (survived the gate's stability check).
    stable_text = f"STABLE: {stable_label if stable_label else '...'}"
    cv2.putText(frame, stable_text, (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Last actually-triggered action (one-shot game input).
    trig_text = f"TRIGGER: {last_triggered if last_triggered else '—'}"
    trig_col  = (0, 0, 255) if triggered else (180, 180, 180)
    cv2.putText(frame, trig_text, (15, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, trig_col, 2)

    cv2.putText(frame, "Q: Quit", (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - LIVE ACTION DETECTION")
    print("=" * 60)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] No trained model found at {MODEL_SAVE_PATH}")
        print("[ERROR] Run train_model.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, actions, model_type, preprocess_args = _load_classifier(
        MODEL_SAVE_PATH, device,
    )
    print(f"[MODEL] Loaded {MODEL_SAVE_PATH}  |  Device: {device}")
    print(f"[MODEL] Preprocessing flags from ckpt: {preprocess_args}")

    ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    landmarker = _create_landmarker(POSE_MODEL_PATH)

    cap, backend = open_camera_with_fallback(CAMERA_INDEX, CAMERA_BACKEND)
    if cap is None:
        print("[ERROR] No working camera found.")
        landmarker.close()
        sys.exit(1)
    print(f"[CAMERA] Backend: {backend}")

    # Rolling buffer of the last SEQUENCE_LENGTH frames, raw (33, 4) each.
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # Gate turns noisy per-frame predictions into clean game triggers.
    gate = ActionGate(
        threshold     = PREDICT_CONFIDENCE_THRESHOLD,
        stable_frames = PREDICT_STABLE_FRAMES,
        cooldown      = PREDICT_TRIGGER_COOLDOWN,
        idle_label    = "idle",
    )
    print(f"[GATE]  threshold={PREDICT_CONFIDENCE_THRESHOLD}  "
          f"stable_frames={PREDICT_STABLE_FRAMES}  "
          f"cooldown={PREDICT_TRIGGER_COOLDOWN}")

    prediction     = "..."
    confidence     = 0.0
    stable_label   = None
    triggered      = False
    last_triggered = None
    prev_t         = time.time()
    fps            = 0.0
    frame_ts       = 0
    _first_detect  = True

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
                # Store the raw (33, 4) frame — preprocessing happens at inference.
                frame_buffer.append(_landmarks_to_frame(landmarks))

                if len(frame_buffer) == SEQUENCE_LENGTH:
                    tensor = _buffer_to_tensor(
                        frame_buffer, model_type, preprocess_args, device,
                    )
                    with torch.no_grad():
                        logits = model(tensor)
                        probs  = torch.softmax(logits, dim=1)
                        conf, idx = probs.max(dim=1)
                        prediction = actions[idx.item()]
                        confidence = conf.item()

                    # Feed into the gate. `triggered` pulses True on the single
                    # frame where a new game action should actually fire.
                    stable_label, triggered = gate.step(prediction, confidence)
                    if triggered:
                        last_triggered = stable_label
                        print(f"[TRIGGER] {stable_label}  (conf={confidence:.2f})")

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            _draw_hud(
                frame,
                prediction     = prediction,
                confidence     = confidence,
                stable_label   = stable_label,
                triggered      = triggered,
                last_triggered = last_triggered,
                buffer_fill    = len(frame_buffer),
                fps            = fps,
                model_type     = model_type,
            )

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
