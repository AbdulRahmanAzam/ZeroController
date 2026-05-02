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

import argparse
import math
import os
import socket
import sys
import threading

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
    EARLY_ACTION_COOLDOWN_MS,
    EARLY_ACTION_ENABLED,
    EARLY_PUNCH_MIN_EXTENSION_DELTA,
    EARLY_PUNCH_MIN_SPEED,
    EARLY_PUNCH_WINDOW_FRAMES,
    LINE_COLOR,
    LOW_LATENCY_CAMERA,
    MIRROR_VIEW,
    MODEL_SAVE_PATH,
    POINT_COLOR,
    POINT_RADIUS,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_PRESENCE_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_PATH,
    POSE_MODEL_VARIANTS,
    POSE_MODEL_URL,
    POSE_NUM_POSES,
    PREDICT_CONFIDENCE_THRESHOLD,
    PREDICT_STABLE_FRAMES,
    PREDICT_TRIGGER_COOLDOWN_MS,
    SEQUENCE_LENGTH,
    SHOW_CONNECTIONS,
    TEXT_COLOR,
)
from main import POSE_CONNECTIONS, ensure_pose_model
from train_model import (
    build_model_from_ckpt,
    preprocess_pose_sequence,
)


# ── Game bridge constants ───────────────────────────────────────────────────

GAME_ACTION_MAP = {
    "idle": "idle",
    "forward": "move_forward",
    "backward": "move_backward",
    "move_forward": "move_forward",
    "move_backward": "move_backward",
    "jump": "jump",
    "block": "block",
    "left_punch": "left_punch",
    "right_punch": "right_punch",
    "left_kick": "left_kick",
    "right_kick": "right_kick",
}

CONTINUOUS_GAME_ACTIONS = {"idle", "move_forward", "move_backward", "block"}
ONE_SHOT_GAME_ACTIONS = {"jump", "left_punch", "right_punch", "left_kick", "right_kick"}
GAME_ACTION_HOLD_SECONDS = 0.20
POSE_API_DEFAULT_HOST = "127.0.0.1"
POSE_API_DEFAULT_PORT = 8000
POSE_WS_PUSH_INTERVAL_SECONDS = 1.0 / 30.0

LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24

PUNCH_LANDMARKS = {
    "left_punch": (LANDMARK_LEFT_SHOULDER, LANDMARK_LEFT_ELBOW, LANDMARK_LEFT_WRIST),
    "right_punch": (LANDMARK_RIGHT_SHOULDER, LANDMARK_RIGHT_ELBOW, LANDMARK_RIGHT_WRIST),
}


def _to_game_action(label):
    if label is None:
        return "idle"
    return GAME_ACTION_MAP.get(str(label), "idle")


def _now_ms():
    return int(time.monotonic() * 1000)


def _elapsed_ms(start_ns):
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _resolve_pose_model(variant):
    spec = POSE_MODEL_VARIANTS.get(variant)
    if spec is None:
        return POSE_MODEL_PATH, POSE_MODEL_URL
    return spec["path"], spec["url"]


class LatencyStats:
    """Small rolling latency summary for the HUD and browser bridge."""

    def __init__(self, maxlen=180):
        self._maxlen = maxlen
        self._values = {
            "capture": deque(maxlen=maxlen),
            "pose": deque(maxlen=maxlen),
            "preprocess": deque(maxlen=maxlen),
            "inference": deque(maxlen=maxlen),
            "gate": deque(maxlen=maxlen),
            "publish": deque(maxlen=maxlen),
            "pipeline": deque(maxlen=maxlen),
            "captureAge": deque(maxlen=maxlen),
        }

    def add(self, name, value_ms):
        bucket = self._values.setdefault(name, deque(maxlen=self._maxlen))
        bucket.append(float(max(0.0, value_ms)))

    def snapshot(self):
        snap = {}
        for name, values in self._values.items():
            vals = list(values)
            if not vals:
                snap[name] = {"lastMs": 0.0, "p50Ms": 0.0, "p95Ms": 0.0}
                continue
            snap[name] = {
                "lastMs": round(vals[-1], 2),
                "p50Ms": round(_percentile(vals, 50), 2),
                "p95Ms": round(_percentile(vals, 95), 2),
            }
        return snap


class LatestFrameReader:
    """Background webcam reader that keeps only the freshest captured frame."""

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._last_delivered = 0
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="zero-controller-camera-reader",
            daemon=True,
        )

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stopped.is_set():
            ok, frame = self.cap.read()
            capture_ns = time.perf_counter_ns()
            frame_ts_ms = _now_ms()
            if ok and frame is not None:
                with self._lock:
                    self._seq += 1
                    self._latest = (True, frame, capture_ns, frame_ts_ms, self._seq)
            else:
                time.sleep(0.003)

    def read(self, timeout_s=1.0):
        deadline = time.monotonic() + timeout_s
        while not self._stopped.is_set():
            with self._lock:
                latest = self._latest
            if latest is not None and latest[4] != self._last_delivered:
                self._last_delivered = latest[4]
                return latest[:4]
            if time.monotonic() >= deadline:
                return False, None, time.perf_counter_ns(), _now_ms()
            time.sleep(0.001)
        return False, None, time.perf_counter_ns(), _now_ms()

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)


class SequentialFrameReader:
    """Direct frame reader for replay or non-threaded camera mode."""

    def __init__(self, cap, replay_fps=None):
        self.cap = cap
        self.replay_fps = replay_fps
        self.frame_idx = 0

    def read(self):
        ok, frame = self.cap.read()
        capture_ns = time.perf_counter_ns()
        if self.replay_fps:
            frame_ts_ms = int(self.frame_idx * (1000.0 / self.replay_fps))
            self.frame_idx += 1
        else:
            frame_ts_ms = _now_ms()
        return ok, frame, capture_ns, frame_ts_ms

    def stop(self):
        pass


class EarlyActionDetector:
    """Detects fast punch starts from a short landmark history.

    This is intentionally narrow: it only emits punch one-shots. The full
    30-frame classifier remains responsible for stable movement and fallback
    action recognition.
    """

    def __init__(
        self,
        *,
        enabled=True,
        window_frames=5,
        min_speed=0.18,
        min_extension_delta=0.12,
        cooldown_ms=250,
    ):
        self.enabled = enabled
        self.window_frames = max(3, int(window_frames))
        self.min_speed = float(min_speed)
        self.min_extension_delta = float(min_extension_delta)
        self.cooldown_ms = int(cooldown_ms)
        self.frames = deque(maxlen=self.window_frames)
        self.timestamps = deque(maxlen=self.window_frames)
        self.cooldown_until_ms = 0
        self._armed = {action: True for action in PUNCH_LANDMARKS}

    def step(self, landmark_frame, timestamp_ms):
        if not self.enabled:
            return None

        self.frames.append(landmark_frame)
        self.timestamps.append(int(timestamp_ms))
        if len(self.frames) < self.window_frames:
            return None

        candidates = []
        for action in PUNCH_LANDMARKS:
            metrics = self._punch_metrics(action)
            if metrics is None:
                continue

            if metrics["extensionDelta"] < self.min_extension_delta * 0.35:
                self._armed[action] = True

            ready = (
                self._armed[action]
                and metrics["speed"] >= self.min_speed
                and metrics["extensionDelta"] >= self.min_extension_delta
                and metrics["elbowOk"]
            )
            if ready:
                candidates.append((metrics["score"], action, metrics))

        if int(timestamp_ms) < self.cooldown_until_ms or not candidates:
            return None

        score, action, metrics = max(candidates, key=lambda item: item[0])
        self.cooldown_until_ms = int(timestamp_ms) + self.cooldown_ms
        self._armed[action] = False
        return action, float(score), metrics

    def _punch_metrics(self, action):
        shoulder_idx, elbow_idx, wrist_idx = PUNCH_LANDMARKS[action]
        first = self.frames[0]
        last = self.frames[-1]

        visibility = min(
            first[shoulder_idx, 3],
            first[wrist_idx, 3],
            last[shoulder_idx, 3],
            last[elbow_idx, 3],
            last[wrist_idx, 3],
        )
        if visibility < 0.45:
            return None

        scale = _torso_scale(last)
        if scale <= 1e-5:
            return None

        wrist_speeds = []
        for i in range(1, len(self.frames)):
            prev = self.frames[i - 1][wrist_idx, :2]
            cur = self.frames[i][wrist_idx, :2]
            dt_ms = max(1, int(self.timestamps[i]) - int(self.timestamps[i - 1]))
            displacement = float(np.linalg.norm(cur - prev)) / scale
            # Normalize physical velocity back to an equivalent 30 FPS frame step.
            wrist_speeds.append(displacement * (1000.0 / dt_ms) / 30.0)
        speed = max(wrist_speeds) if wrist_speeds else 0.0

        start_extension = _joint_distance(first, shoulder_idx, wrist_idx) / scale
        end_extension = _joint_distance(last, shoulder_idx, wrist_idx) / scale
        extension_delta = end_extension - start_extension

        start_angle = _elbow_angle_degrees(first, shoulder_idx, elbow_idx, wrist_idx)
        end_angle = _elbow_angle_degrees(last, shoulder_idx, elbow_idx, wrist_idx)
        elbow_delta = end_angle - start_angle
        elbow_ok = end_angle >= 125.0 or elbow_delta >= 12.0

        speed_score = min(1.0, speed / max(self.min_speed, 1e-6))
        extension_score = min(1.0, extension_delta / max(self.min_extension_delta, 1e-6))
        elbow_score = min(1.0, max(0.0, (end_angle - 90.0) / 70.0))
        score = 0.50 * speed_score + 0.35 * extension_score + 0.15 * elbow_score

        return {
            "speed": float(speed),
            "extensionDelta": float(extension_delta),
            "elbowAngle": float(end_angle),
            "elbowDelta": float(elbow_delta),
            "elbowOk": bool(elbow_ok),
            "visibility": float(visibility),
            "score": float(min(0.99, max(0.01, score))),
        }


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


def _torso_scale(frame):
    shoulders = (frame[LANDMARK_LEFT_SHOULDER, :2] + frame[LANDMARK_RIGHT_SHOULDER, :2]) * 0.5
    hips = (frame[LANDMARK_LEFT_HIP, :2] + frame[LANDMARK_RIGHT_HIP, :2]) * 0.5
    scale = float(np.linalg.norm(shoulders - hips))
    if scale <= 1e-5:
        scale = float(np.linalg.norm(frame[LANDMARK_LEFT_SHOULDER, :2] - frame[LANDMARK_RIGHT_SHOULDER, :2]))
    return max(scale, 1e-5)


def _joint_distance(frame, a_idx, b_idx):
    return float(np.linalg.norm(frame[a_idx, :2] - frame[b_idx, :2]))


def _elbow_angle_degrees(frame, shoulder_idx, elbow_idx, wrist_idx):
    upper = frame[shoulder_idx, :2] - frame[elbow_idx, :2]
    lower = frame[wrist_idx, :2] - frame[elbow_idx, :2]
    denom = float(np.linalg.norm(upper) * np.linalg.norm(lower))
    if denom <= 1e-8:
        return 0.0
    cos_angle = float(np.dot(upper, lower) / denom)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


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
    def __init__(self, threshold, stable_frames, cooldown_ms, idle_label="idle", fast_labels=None):
        self.threshold     = threshold
        self.stable_frames = stable_frames
        self.cooldown_ms   = cooldown_ms
        self.idle_label    = idle_label
        self.fast_labels   = set(fast_labels or ())
        self._last_pred       = None     # last raw per-frame prediction
        self._stable_count    = 0        # streak length of _last_pred above threshold
        self._cooldown_until_ms = 0      # monotonic ms before a new trigger is allowed
        self._last_triggered  = None     # last action that actually fired (non-idle)

    def step(self, pred_label, confidence, timestamp_ms=None):
        """Feed one prediction; returns (stable_label, triggered).

        stable_label : the stabilised label currently held (None while still settling)
        triggered    : True only on the rising edge — the single frame where a
                       new non-idle action actually fires.
        """
        timestamp_ms = _now_ms() if timestamp_ms is None else int(timestamp_ms)

        # Extend the streak if the high-confidence prediction is unchanged.
        if pred_label == self._last_pred and confidence >= self.threshold:
            self._stable_count += 1
        else:
            # New label, or confidence dropped below threshold — restart counting.
            self._stable_count = 1 if confidence >= self.threshold else 0
        self._last_pred = pred_label

        required_frames = 1 if pred_label in self.fast_labels else self.stable_frames
        stable_label = pred_label if self._stable_count >= required_frames else None

        triggered = False
        if stable_label == self.idle_label:
            # Returning to idle re-arms the gate so the same action can fire again.
            self._last_triggered = None
        elif (stable_label is not None
              and stable_label != self._last_triggered
              and timestamp_ms >= self._cooldown_until_ms):
            # Rising edge of a fresh non-idle action: fire once, start cooldown.
            triggered = True
            self._last_triggered = stable_label
            self._cooldown_until_ms = timestamp_ms + self.cooldown_ms

        return stable_label, triggered


# ── Frontend API bridge ─────────────────────────────────────────────────────

class PoseApiState:
    """Thread-safe latest pose command shared with the browser game."""

    def __init__(self):
        now_ms = int(time.time() * 1000)
        self._lock = threading.Lock()
        self._pulse_action = None
        self._pulse_until = 0.0
        self._pulse_event_id = 0
        self._pulse_confidence = 0.0
        self._pulse_source = None
        self._pulse_prediction = "idle"
        self._event_id = 0
        self._snapshot = {
            "ok": True,
            "status": "booting",
            "message": "Starting ZeroController bridge...",
            "action": "idle",
            "confidence": 0.0,
            "sourceAction": "idle",
            "stableAction": None,
            "triggered": False,
            "triggerSource": None,
            "eventId": 0,
            "timestamp": now_ms,
            "bufferFill": 0,
            "sequenceLength": SEQUENCE_LENGTH,
            "fps": 0.0,
            "modelType": None,
            "captureAgeMs": 0.0,
            "pipelineLatencyMs": 0.0,
            "latencyStats": {},
        }

    def mark_status(self, status, message=None, **fields):
        with self._lock:
            self._snapshot.update({
                "ok": status not in {"error", "camera_error", "model_error"},
                "status": status,
                "timestamp": int(time.time() * 1000),
                **fields,
            })
            if message is not None:
                self._snapshot["message"] = message

    def update_prediction(
        self,
        *,
        prediction,
        confidence,
        stable_label,
        triggered,
        buffer_fill,
        fps,
        model_type,
        trigger_source=None,
        capture_age_ms=0.0,
        pipeline_latency_ms=0.0,
        latency_stats=None,
        status="detecting",
        message="ZeroController is sending Player 1 inputs.",
    ):
        now = time.time()
        stable_game_action = _to_game_action(stable_label)

        if triggered and stable_game_action in ONE_SHOT_GAME_ACTIONS:
            self._event_id += 1
            self._pulse_action = stable_game_action
            self._pulse_until = now + GAME_ACTION_HOLD_SECONDS
            self._pulse_event_id = self._event_id
            self._pulse_confidence = float(confidence)
            self._pulse_source = trigger_source
            self._pulse_prediction = str(prediction)

        pulse_active = self._pulse_action is not None and now < self._pulse_until
        if pulse_active:
            published_action = self._pulse_action
            published_confidence = max(float(confidence), self._pulse_confidence)
            published_source_action = self._pulse_prediction
            published_stable_action = self._pulse_action
            published_triggered = True
            published_trigger_source = self._pulse_source
            published_event_id = self._pulse_event_id
        else:
            self._pulse_action = None
            self._pulse_event_id = self._event_id
            self._pulse_confidence = 0.0
            self._pulse_source = None
            self._pulse_prediction = "idle"
            published_action = stable_game_action if stable_game_action in CONTINUOUS_GAME_ACTIONS else "idle"
            published_confidence = float(confidence)
            published_source_action = prediction
            published_stable_action = _to_game_action(stable_label) if stable_label else None
            published_triggered = bool(triggered)
            published_trigger_source = trigger_source if triggered else None
            published_event_id = int(self._event_id)

        with self._lock:
            self._snapshot.update({
                "ok": True,
                "status": status,
                "message": message,
                "action": published_action,
                "confidence": published_confidence,
                "sourceAction": published_source_action,
                "stableAction": published_stable_action,
                "triggered": published_triggered,
                "triggerSource": published_trigger_source,
                "eventId": published_event_id,
                "timestamp": int(now * 1000),
                "bufferFill": int(buffer_fill),
                "sequenceLength": SEQUENCE_LENGTH,
                "fps": float(fps),
                "modelType": model_type,
                "captureAgeMs": float(capture_age_ms),
                "pipelineLatencyMs": float(pipeline_latency_ms),
                "latencyStats": latency_stats or {},
            })

    def snapshot(self, player_id=1):
        with self._lock:
            snap = dict(self._snapshot)
        snap["playerId"] = player_id
        return snap


def _can_bind(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _start_pose_api_server(api_state, host, port):
    if not _can_bind(host, port):
        print(f"[API] Port {port} is already in use. Stop the old bridge or choose --port.")
        api_state.mark_status(
            "error",
            f"Port {port} is already in use. Stop the old run_model.py or choose another port.",
        )
        return None

    try:
        import asyncio
        import uvicorn
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        print("[API] FastAPI/uvicorn is not installed, so the game bridge is disabled.")
        print("[API] Install with: pip install fastapi uvicorn[standard]")
        api_state.mark_status(
            "error",
            "FastAPI/uvicorn is missing. Run: pip install fastapi uvicorn[standard]",
        )
        return None

    app = FastAPI(title="ZeroController Pose Bridge", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return api_state.snapshot()

    @app.get("/pose/{player_id}")
    async def pose(player_id: int):
        return api_state.snapshot(player_id)

    @app.websocket("/ws/pose/{player_id}")
    async def pose_ws(websocket: WebSocket, player_id: int):
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(api_state.snapshot(player_id))
                await asyncio.sleep(POSE_WS_PUSH_INTERVAL_SECONDS)
        except WebSocketDisconnect:
            pass

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="zero-controller-api", daemon=True)
    thread.start()
    print(f"[API] Pose bridge running at http://{host}:{port}")
    print(f"[API] Frontend polling endpoint: http://{host}:{port}/pose/1")
    print(f"[API] Frontend websocket endpoint: ws://{host}:{port}/ws/pose/1")
    return server


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run live ZeroController action detection.")
    parser.add_argument("--host", default=POSE_API_DEFAULT_HOST, help="Pose API host for the browser game.")
    parser.add_argument("--port", type=int, default=POSE_API_DEFAULT_PORT, help="Pose API port for the browser game.")
    parser.add_argument("--no-api", action="store_true", help="Disable the browser game API bridge.")
    parser.add_argument("--no-window", action="store_true", help="Run detection without the OpenCV preview window.")
    parser.add_argument("--self-test", action="store_true", help="Run bridge/action mapping checks without opening the camera.")
    parser.add_argument(
        "--latency-mode",
        "--latency",
        dest="latency_mode",
        choices=("fast", "balanced", "diagnostic"),
        default="fast",
        help="Latency strategy. fast enables early punch triggers; diagnostic also prints periodic timing.",
    )
    parser.add_argument(
        "--pose-model",
        choices=tuple(POSE_MODEL_VARIANTS.keys()),
        default="full",
        help="MediaPipe Pose Landmarker variant to use.",
    )
    parser.add_argument("--replay", help="Replay a .avi/.mp4 video or .npy landmark sequence instead of opening the webcam.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, mainly for replay/benchmark runs.")
    return parser.parse_args(argv)


def _run_self_test():
    expected = {
        "idle": "idle",
        "forward": "move_forward",
        "backward": "move_backward",
        "move_forward": "move_forward",
        "move_backward": "move_backward",
        "jump": "jump",
        "block": "block",
        "left_punch": "left_punch",
        "right_punch": "right_punch",
        "left_kick": "left_kick",
        "right_kick": "right_kick",
        "unknown": "idle",
    }
    for source, game_action in expected.items():
        assert _to_game_action(source) == game_action, source

    state = PoseApiState()
    state.update_prediction(
        prediction="forward",
        confidence=0.92,
        stable_label="forward",
        triggered=False,
        buffer_fill=SEQUENCE_LENGTH,
        fps=30.0,
        model_type="stgcn",
    )
    assert state.snapshot()["action"] == "move_forward"

    state.update_prediction(
        prediction="right_punch",
        confidence=0.96,
        stable_label="right_punch",
        triggered=True,
        buffer_fill=SEQUENCE_LENGTH,
        fps=30.0,
        model_type="stgcn",
    )
    snap = state.snapshot()
    assert snap["action"] == "right_punch"
    assert snap["triggered"] is True
    assert snap["eventId"] == 1

    state.update_prediction(
        prediction="idle",
        confidence=0.10,
        stable_label=None,
        triggered=False,
        buffer_fill=6,
        fps=30.0,
        model_type="stgcn",
    )
    snap = state.snapshot()
    assert snap["action"] == "right_punch"
    assert snap["triggered"] is True
    assert snap["eventId"] == 1

    time.sleep(GAME_ACTION_HOLD_SECONDS + 0.03)
    state.update_prediction(
        prediction="right_punch",
        confidence=0.96,
        stable_label="right_punch",
        triggered=False,
        buffer_fill=SEQUENCE_LENGTH,
        fps=30.0,
        model_type="stgcn",
    )
    assert state.snapshot()["action"] == "idle"

    gate = ActionGate(threshold=0.8, stable_frames=2, cooldown_ms=70)
    assert gate.step("left_kick", 0.9, 0) == (None, False)
    assert gate.step("left_kick", 0.9, 33) == ("left_kick", True)
    assert gate.step("idle", 0.9, 66) == (None, False)
    assert gate.step("idle", 0.9, 99) == ("idle", False)

    fast_gate = ActionGate(
        threshold=0.8,
        stable_frames=2,
        cooldown_ms=70,
        fast_labels={"forward", "backward", "jump"},
    )
    assert fast_gate.step("forward", 0.9, 0) == ("forward", True)
    assert fast_gate.step("idle", 0.9, 33) == (None, False)
    assert fast_gate.step("jump", 0.9, 100) == ("jump", True)

    def make_pose(right_wrist_x, right_wrist_y=0.52, visibility=1.0):
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[:, 3] = visibility
        pose[LANDMARK_LEFT_SHOULDER] = [0.42, 0.42, 0.0, visibility]
        pose[LANDMARK_RIGHT_SHOULDER] = [0.58, 0.42, 0.0, visibility]
        pose[LANDMARK_LEFT_HIP] = [0.46, 0.75, 0.0, visibility]
        pose[LANDMARK_RIGHT_HIP] = [0.54, 0.75, 0.0, visibility]
        pose[LANDMARK_RIGHT_ELBOW] = [(0.58 + right_wrist_x) * 0.5, 0.46, 0.0, visibility]
        pose[LANDMARK_RIGHT_WRIST] = [right_wrist_x, right_wrist_y, 0.0, visibility]
        return pose

    early = EarlyActionDetector(
        enabled=True,
        window_frames=5,
        min_speed=EARLY_PUNCH_MIN_SPEED,
        min_extension_delta=EARLY_PUNCH_MIN_EXTENSION_DELTA,
        cooldown_ms=EARLY_ACTION_COOLDOWN_MS,
    )
    fast_positions = [0.66, 0.72, 0.80, 0.88, 0.96]
    result = None
    for i, wrist_x in enumerate(fast_positions):
        result = early.step(make_pose(wrist_x), i * 33)
    assert result is not None and result[0] == "right_punch", result
    assert early.step(make_pose(0.98), 165) is None

    slow = EarlyActionDetector(enabled=True, window_frames=5)
    result = None
    for i, wrist_x in enumerate(fast_positions):
        result = slow.step(make_pose(wrist_x), i * 100)
    assert result is None, result

    low_vis = EarlyActionDetector(enabled=True, window_frames=5)
    result = None
    for i, wrist_x in enumerate(fast_positions):
        result = low_vis.step(make_pose(wrist_x, visibility=0.2), i * 33)
    assert result is None, result

    print("[SELF-TEST] run_model bridge checks passed.")


# ── HUD ──────────────────────────────────────────────────────────────────────

def _draw_hud(frame, *, prediction, confidence, stable_label, triggered,
              last_triggered, buffer_fill, fps, model_type, trigger_source,
              latency_snapshot):
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
    source = f" [{trigger_source}]" if trigger_source else ""
    trig_text = f"TRIGGER: {last_triggered if last_triggered else '...'}{source}"
    trig_col  = (0, 0, 255) if triggered else (180, 180, 180)
    cv2.putText(frame, trig_text, (15, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, trig_col, 2)

    pipeline = latency_snapshot.get("pipeline", {})
    pose = latency_snapshot.get("pose", {})
    latency_text = (
        f"LAT p50/p95: {pipeline.get('p50Ms', 0):.0f}/{pipeline.get('p95Ms', 0):.0f} ms"
        f"  pose p95: {pose.get('p95Ms', 0):.0f} ms"
    )
    cv2.putText(frame, latency_text, (15, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.putText(frame, "Q: Quit", (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = _parse_args(argv)
    if args.self_test:
        _run_self_test()
        return

    print("=" * 60)
    print("  ZERO CONTROLLER - LIVE ACTION DETECTION")
    print("=" * 60)

    replay_landmarks = None
    replay_path = args.replay
    if replay_path:
        if not os.path.exists(replay_path):
            print(f"[ERROR] Replay path not found: {replay_path}")
            sys.exit(1)
        if os.path.splitext(replay_path)[1].lower() == ".npy":
            replay_landmarks = np.load(replay_path).astype(np.float32)
            if replay_landmarks.ndim != 3 or replay_landmarks.shape[1:] != (33, 4):
                print(f"[ERROR] Replay .npy must have shape (T, 33, 4), got {replay_landmarks.shape}")
                sys.exit(1)
            print(f"[REPLAY] Loaded landmark sequence: {replay_path} ({len(replay_landmarks)} frames)")

    api_state = PoseApiState()
    if not args.no_api:
        _start_pose_api_server(api_state, args.host, args.port)
    else:
        print("[API] Pose bridge disabled by --no-api")

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] No trained model found at {MODEL_SAVE_PATH}")
        print("[ERROR] Run train_model.py first.")
        api_state.mark_status("model_error", f"No trained model found at {MODEL_SAVE_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api_state.mark_status("loading_model", "Loading trained action classifier...")
    model, actions, model_type, preprocess_args = _load_classifier(
        MODEL_SAVE_PATH, device,
    )
    print(f"[MODEL] Loaded {MODEL_SAVE_PATH}  |  Device: {device}")
    print(f"[MODEL] Preprocessing flags from ckpt: {preprocess_args}")

    pose_model_path, pose_model_url = _resolve_pose_model(args.pose_model)
    landmarker = None
    cap = None
    frame_source = None
    backend = None

    if replay_landmarks is None:
        api_state.mark_status("loading_pose_model", "Loading MediaPipe pose landmarker...")
        ensure_pose_model(pose_model_path, pose_model_url)
        landmarker = _create_landmarker(pose_model_path)
        print(f"[POSE] MediaPipe model: {args.pose_model} ({pose_model_path})")

    if replay_path and replay_landmarks is None:
        api_state.mark_status("opening_replay", "Opening replay video for ZeroController...")
        cap = cv2.VideoCapture(replay_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open replay video: {replay_path}")
            api_state.mark_status("camera_error", f"Could not open replay video: {replay_path}")
            if landmarker:
                landmarker.close()
            sys.exit(1)
        replay_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if replay_fps <= 0:
            replay_fps = 30.0
        frame_source = SequentialFrameReader(cap, replay_fps=replay_fps)
        print(f"[REPLAY] Video: {replay_path}  fps={replay_fps:.1f}")
    elif replay_landmarks is None:
        api_state.mark_status("opening_camera", "Opening webcam for ZeroController...")
        cap, backend = open_camera_with_fallback(CAMERA_INDEX, CAMERA_BACKEND)
        if cap is None:
            print("[ERROR] No working camera found.")
            api_state.mark_status("camera_error", "No working camera found.")
            if landmarker:
                landmarker.close()
            sys.exit(1)
        if LOW_LATENCY_CAMERA:
            frame_source = LatestFrameReader(cap).start()
        else:
            frame_source = SequentialFrameReader(cap)
        print(f"[CAMERA] Backend: {backend}  low_latency={LOW_LATENCY_CAMERA}")

    # Rolling buffer of the last SEQUENCE_LENGTH frames, raw (33, 4) each.
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    latency_stats = LatencyStats()
    early_enabled = EARLY_ACTION_ENABLED and args.latency_mode in {"fast", "balanced", "diagnostic"}
    early_detector = EarlyActionDetector(
        enabled=early_enabled,
        window_frames=EARLY_PUNCH_WINDOW_FRAMES,
        min_speed=EARLY_PUNCH_MIN_SPEED,
        min_extension_delta=EARLY_PUNCH_MIN_EXTENSION_DELTA,
        cooldown_ms=EARLY_ACTION_COOLDOWN_MS,
    )

    # Gate turns noisy per-frame predictions into clean game triggers.
    gate = ActionGate(
        threshold     = PREDICT_CONFIDENCE_THRESHOLD,
        stable_frames = PREDICT_STABLE_FRAMES,
        cooldown_ms   = PREDICT_TRIGGER_COOLDOWN_MS,
        idle_label    = "idle",
        fast_labels   = {"forward", "backward", "move_forward", "move_backward", "jump"},
    )
    print(f"[GATE]  threshold={PREDICT_CONFIDENCE_THRESHOLD}  "
          f"stable_frames={PREDICT_STABLE_FRAMES}  "
          f"cooldown_ms={PREDICT_TRIGGER_COOLDOWN_MS}  "
          f"early_punch={early_enabled}")
    api_state.mark_status(
        "warming_up",
        "Camera bridge is warming up. Step into view until the buffer fills.",
        modelType=model_type,
    )

    prediction     = "..."
    confidence     = 0.0
    stable_label   = None
    triggered      = False
    trigger_source = None
    last_triggered = None
    prev_t         = time.time()
    fps            = 0.0
    _first_detect  = True
    frame_count    = 0
    trigger_counts = {"early": 0, "classifier": 0}
    global_one_shot_cooldown_until_ms = 0

    try:
        while True:
            pipeline_start_ns = time.perf_counter_ns()
            frame = None
            landmarks = None
            landmark_frame = None

            capture_start_ns = time.perf_counter_ns()
            if replay_landmarks is not None:
                if frame_count >= len(replay_landmarks):
                    break
                landmark_frame = replay_landmarks[frame_count]
                capture_ns = time.perf_counter_ns()
                frame_ts_ms = int(frame_count * (1000.0 / 30.0))
                if not args.no_window:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ok, frame, capture_ns, frame_ts_ms = frame_source.read()
                if not ok or frame is None:
                    break
            latency_stats.add("capture", _elapsed_ms(capture_start_ns))

            if frame is not None and MIRROR_VIEW and not replay_path:
                frame = cv2.flip(frame, 1)

            if landmarker is not None and frame is not None:
                pose_start_ns = time.perf_counter_ns()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                if _first_detect:
                    _fd = _suppress_stderr()
                    result = landmarker.detect_for_video(mp_image, frame_ts_ms)
                    _restore_stderr(_fd)
                    _first_detect = False
                else:
                    result = landmarker.detect_for_video(mp_image, frame_ts_ms)
                latency_stats.add("pose", _elapsed_ms(pose_start_ns))

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                if landmarks:
                    landmark_frame = _landmarks_to_frame(landmarks)

            if landmarks and frame is not None:
                _draw_skeleton(frame, landmarks)

            publish_prediction = prediction
            publish_confidence = confidence
            publish_stable_label = stable_label
            triggered = False
            trigger_source = None

            if landmark_frame is not None:
                # Store the raw (33, 4) frame — preprocessing happens at inference.
                frame_buffer.append(landmark_frame)

                gate_start_ns = time.perf_counter_ns()
                early_result = early_detector.step(landmark_frame, frame_ts_ms)
                early_triggered = False
                early_label = None
                early_confidence = 0.0
                if early_result is not None:
                    early_label, early_confidence, _ = early_result
                    early_triggered = frame_ts_ms >= global_one_shot_cooldown_until_ms

                if len(frame_buffer) == SEQUENCE_LENGTH:
                    preprocess_start_ns = time.perf_counter_ns()
                    tensor = _buffer_to_tensor(
                        frame_buffer, model_type, preprocess_args, device,
                    )
                    latency_stats.add("preprocess", _elapsed_ms(preprocess_start_ns))

                    inference_start_ns = time.perf_counter_ns()
                    with torch.no_grad():
                        logits = model(tensor)
                        probs  = torch.softmax(logits, dim=1)
                        conf, idx = probs.max(dim=1)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        prediction = actions[idx.item()]
                        confidence = conf.item()
                    latency_stats.add("inference", _elapsed_ms(inference_start_ns))

                    # Feed into the gate. `triggered` pulses True on the single
                    # frame where a new game action should actually fire.
                    stable_label, classifier_triggered_raw = gate.step(
                        prediction,
                        confidence,
                        frame_ts_ms,
                    )
                    classifier_one_shot = _to_game_action(stable_label) in ONE_SHOT_GAME_ACTIONS
                    classifier_triggered = (
                        classifier_triggered_raw
                        and (not classifier_one_shot or frame_ts_ms >= global_one_shot_cooldown_until_ms)
                    )
                else:
                    classifier_triggered = False

                if early_triggered:
                    publish_prediction = early_label
                    publish_confidence = early_confidence
                    publish_stable_label = early_label
                    triggered = True
                    trigger_source = "early"
                    last_triggered = early_label
                    trigger_counts["early"] += 1
                    global_one_shot_cooldown_until_ms = frame_ts_ms + EARLY_ACTION_COOLDOWN_MS
                    print(f"[TRIGGER][early] {early_label}  (score={early_confidence:.2f})")
                elif classifier_triggered:
                    publish_prediction = prediction
                    publish_confidence = confidence
                    publish_stable_label = stable_label
                    triggered = True
                    trigger_source = "classifier"
                    last_triggered = stable_label
                    trigger_counts["classifier"] += 1
                    if _to_game_action(stable_label) in ONE_SHOT_GAME_ACTIONS:
                        global_one_shot_cooldown_until_ms = frame_ts_ms + PREDICT_TRIGGER_COOLDOWN_MS
                    print(f"[TRIGGER][classifier] {stable_label}  (conf={confidence:.2f})")
                else:
                    publish_prediction = prediction
                    publish_confidence = confidence
                    publish_stable_label = stable_label

                latency_stats.add("gate", _elapsed_ms(gate_start_ns))
            else:
                stable_label = None
                triggered = False
                publish_stable_label = None

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            capture_age_ms = _elapsed_ms(capture_ns)
            pipeline_latency_ms = _elapsed_ms(pipeline_start_ns)
            latency_stats.add("captureAge", capture_age_ms)
            latency_stats.add("pipeline", pipeline_latency_ms)
            latency_snapshot = latency_stats.snapshot()

            publish_start_ns = time.perf_counter_ns()
            if landmark_frame is None:
                api_state.update_prediction(
                    prediction=prediction,
                    confidence=0.0,
                    stable_label=None,
                    triggered=False,
                    buffer_fill=len(frame_buffer),
                    fps=fps,
                    model_type=model_type,
                    capture_age_ms=capture_age_ms,
                    pipeline_latency_ms=pipeline_latency_ms,
                    latency_stats=latency_snapshot,
                    status="no_pose",
                    message="No pose detected. Step into the camera view.",
                )
            elif len(frame_buffer) < SEQUENCE_LENGTH and not triggered:
                api_state.update_prediction(
                    prediction=prediction,
                    confidence=confidence,
                    stable_label=None,
                    triggered=False,
                    buffer_fill=len(frame_buffer),
                    fps=fps,
                    model_type=model_type,
                    capture_age_ms=capture_age_ms,
                    pipeline_latency_ms=pipeline_latency_ms,
                    latency_stats=latency_snapshot,
                    status="warming_up",
                    message="Collecting the first pose frames for Player 1 control.",
                )
            else:
                api_state.update_prediction(
                    prediction=publish_prediction,
                    confidence=publish_confidence,
                    stable_label=publish_stable_label,
                    triggered=triggered,
                    buffer_fill=len(frame_buffer),
                    fps=fps,
                    model_type=model_type,
                    trigger_source=trigger_source,
                    capture_age_ms=capture_age_ms,
                    pipeline_latency_ms=pipeline_latency_ms,
                    latency_stats=latency_snapshot,
                )
            latency_stats.add("publish", _elapsed_ms(publish_start_ns))

            if args.latency_mode == "diagnostic" and frame_count > 0 and frame_count % 30 == 0:
                pipe = latency_snapshot["pipeline"]
                pose = latency_snapshot["pose"]
                print(
                    f"[LATENCY] pipeline p50/p95={pipe['p50Ms']:.1f}/{pipe['p95Ms']:.1f} ms  "
                    f"pose p95={pose['p95Ms']:.1f} ms  fps={fps:.1f}"
                )

            if frame is not None:
                _draw_hud(
                    frame,
                    prediction     = publish_prediction,
                    confidence     = publish_confidence,
                    stable_label   = publish_stable_label,
                    triggered      = triggered,
                    last_triggered = last_triggered,
                    buffer_fill    = len(frame_buffer),
                    fps            = fps,
                    model_type     = model_type,
                    trigger_source = trigger_source,
                    latency_snapshot = latency_snapshot,
                )

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            if args.no_window:
                continue

            try:
                cv2.imshow("ZeroController - Live Detection", frame)
            except cv2.error:
                api_state.mark_status(
                    "detecting",
                    "OpenCV preview is unavailable; the browser bridge is still running.",
                )
                break

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if frame_source is not None:
            frame_source.stop()
        if cap is not None:
            cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        api_state.mark_status("stopped", "ZeroController detection stopped.")

    if replay_path:
        print(
            f"[REPLAY] frames={frame_count}  "
            f"early_triggers={trigger_counts['early']}  "
            f"classifier_triggers={trigger_counts['classifier']}"
        )
    print("[DONE] Live detection closed.")


if __name__ == "__main__":
    main()
