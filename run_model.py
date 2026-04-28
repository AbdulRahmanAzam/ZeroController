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


def _to_game_action(label):
    if label is None:
        return "idle"
    return GAME_ACTION_MAP.get(str(label), "idle")


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


# ── Frontend API bridge ─────────────────────────────────────────────────────

class PoseApiState:
    """Thread-safe latest pose command shared with the browser game."""

    def __init__(self):
        now_ms = int(time.time() * 1000)
        self._lock = threading.Lock()
        self._pulse_action = None
        self._pulse_until = 0.0
        self._snapshot = {
            "ok": True,
            "status": "booting",
            "message": "Starting ZeroController bridge...",
            "action": "idle",
            "confidence": 0.0,
            "sourceAction": "idle",
            "stableAction": None,
            "triggered": False,
            "timestamp": now_ms,
            "bufferFill": 0,
            "sequenceLength": SEQUENCE_LENGTH,
            "fps": 0.0,
            "modelType": None,
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
    ):
        now = time.time()
        stable_game_action = _to_game_action(stable_label)

        if triggered and stable_game_action in ONE_SHOT_GAME_ACTIONS:
            self._pulse_action = stable_game_action
            self._pulse_until = now + GAME_ACTION_HOLD_SECONDS

        pulse_active = self._pulse_action is not None and now < self._pulse_until
        if pulse_active:
            published_action = self._pulse_action
        else:
            self._pulse_action = None
            published_action = stable_game_action if stable_game_action in CONTINUOUS_GAME_ACTIONS else "idle"

        with self._lock:
            self._snapshot.update({
                "ok": True,
                "status": "detecting",
                "message": "ZeroController is sending Player 1 inputs.",
                "action": published_action,
                "confidence": float(confidence),
                "sourceAction": prediction,
                "stableAction": _to_game_action(stable_label) if stable_label else None,
                "triggered": bool(triggered),
                "timestamp": int(now * 1000),
                "bufferFill": int(buffer_fill),
                "sequenceLength": SEQUENCE_LENGTH,
                "fps": float(fps),
                "modelType": model_type,
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
                await asyncio.sleep(0.016)  # ~60 Hz push (was 0.05 / 20 Hz)
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
    assert state.snapshot()["action"] == "right_punch"

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

    gate = ActionGate(threshold=0.8, stable_frames=2, cooldown=2)
    assert gate.step("left_kick", 0.9) == (None, False)
    assert gate.step("left_kick", 0.9) == ("left_kick", True)
    assert gate.step("idle", 0.9) == (None, False)
    assert gate.step("idle", 0.9) == ("idle", False)

    print("[SELF-TEST] run_model bridge checks passed.")


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

def main(argv=None):
    args = _parse_args(argv)
    if args.self_test:
        _run_self_test()
        return

    print("=" * 60)
    print("  ZERO CONTROLLER - LIVE ACTION DETECTION")
    print("=" * 60)

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

    api_state.mark_status("loading_pose_model", "Loading MediaPipe pose landmarker...")
    ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    landmarker = _create_landmarker(POSE_MODEL_PATH)

    api_state.mark_status("opening_camera", "Opening webcam for ZeroController...")
    cap, backend = open_camera_with_fallback(CAMERA_INDEX, CAMERA_BACKEND)
    if cap is None:
        print("[ERROR] No working camera found.")
        api_state.mark_status("camera_error", "No working camera found.")
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
    api_state.mark_status(
        "warming_up",
        "Camera bridge is warming up. Step into view until the buffer fills.",
        modelType=model_type,
    )

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
            else:
                stable_label = None
                triggered = False

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            if not landmarks:
                api_state.mark_status(
                    "no_pose",
                    "No pose detected. Step into the camera view.",
                    action="idle",
                    confidence=0.0,
                    sourceAction=prediction,
                    stableAction=None,
                    triggered=False,
                    bufferFill=len(frame_buffer),
                    sequenceLength=SEQUENCE_LENGTH,
                    fps=fps,
                    modelType=model_type,
                )
            elif len(frame_buffer) < SEQUENCE_LENGTH:
                api_state.mark_status(
                    "warming_up",
                    "Collecting the first pose frames for Player 1 control.",
                    action="idle",
                    confidence=confidence,
                    sourceAction=prediction,
                    stableAction=None,
                    triggered=False,
                    bufferFill=len(frame_buffer),
                    sequenceLength=SEQUENCE_LENGTH,
                    fps=fps,
                    modelType=model_type,
                )
            else:
                api_state.update_prediction(
                    prediction=prediction,
                    confidence=confidence,
                    stable_label=stable_label,
                    triggered=triggered,
                    buffer_fill=len(frame_buffer),
                    fps=fps,
                    model_type=model_type,
                )

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
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        landmarker.close()
        api_state.mark_status("stopped", "ZeroController detection stopped.")

    print("[DONE] Live detection closed.")


if __name__ == "__main__":
    main()
