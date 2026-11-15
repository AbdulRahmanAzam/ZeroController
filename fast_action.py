"""Low-latency action classifier: current pose plus the last 133 ms of motion.

The 1-second window models can only change their answer once most of the window
has changed. This model looks at one pose and how far it moved over a short
lookback, so its answer can change within a few frames of the body arriving.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

LOOKBACK_MS = 133.0
LOOKBACK_FRAMES_AT_30FPS = 4

# Nose, shoulders, elbows, wrists, hips, knees, ankles.
BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# position (3) + position change (3) + visibility (1) per joint, plus hip motion (2).
NUM_FEATURES = len(BODY_JOINTS) * 7 + 2
HIP_MOTION = slice(len(BODY_JOINTS) * 7, len(BODY_JOINTS) * 7 + 2)
MIN_JOINT_VISIBILITY = 0.5

# Optional joint angles, measured at the middle joint of each triple.
ANGLE_JOINTS = [
    (11, 13, 15), (12, 14, 16),  # elbows
    (23, 11, 13), (24, 12, 14),  # shoulders: side of the torso to the upper arm
    (23, 25, 27), (24, 26, 28),  # knees
    (11, 23, 25), (12, 24, 26),  # hips: side of the torso to the thigh
]
# "pose" is the original feature set. "pose+angles" adds each angle and its change
# over the lookback. A checkpoint records which set it was trained on.
FEATURE_SETS = {"pose": NUM_FEATURES, "pose+angles": NUM_FEATURES + 2 * len(ANGLE_JOINTS)}


def _normalized(frames, aspect):
    """Return hip-relative joints in torso units, plus the hip in torso units."""
    pos = frames[..., :3].astype(np.float32).copy()
    # MediaPipe scales x and z with frame width.
    pos[..., 0] *= aspect
    pos[..., 2] *= aspect
    hip = (pos[:, L_HIP] + pos[:, R_HIP]) / 2.0
    shoulder = (pos[:, L_SHOULDER] + pos[:, R_SHOULDER]) / 2.0
    torso = np.maximum(np.linalg.norm((shoulder - hip)[:, :2], axis=-1), 1e-3)[:, None]
    relative = (pos[:, BODY_JOINTS] - hip[:, None]) / torso[:, None]
    return relative, hip[:, :2] / torso


def joint_angles(frames, aspect):
    """Angle at each ANGLE_JOINTS middle joint in the image plane: 0 folded, 1 straight."""
    pos = frames[..., :2].astype(np.float32).copy()
    pos[..., 0] *= aspect
    ends_a, middles, ends_b = (pos[:, [triple[k] for triple in ANGLE_JOINTS]] for k in range(3))
    u, v = ends_a - middles, ends_b - middles
    lengths = np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-6)
    return np.arccos(np.clip((u * v).sum(axis=-1) / lengths, -1.0, 1.0)) / np.pi


def build_features(current, past, aspect, feature_set="pose"):
    """Build (N, FEATURE_SETS[feature_set]) from matching (N, 33, 4) current and past frames."""
    now, hip_now = _normalized(current, aspect)
    then, hip_then = _normalized(past, aspect)
    parts = [
        now.reshape(len(now), -1),
        (now - then).reshape(len(now), -1),
        current[:, BODY_JOINTS, 3],
        hip_now - hip_then,
    ]
    if feature_set == "pose+angles":
        angles_now = joint_angles(current, aspect)
        parts += [angles_now, angles_now - joint_angles(past, aspect)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def sequence_features(seq, aspect, feature_set="pose"):
    """Features for every frame of a 30 fps clip that has a full lookback behind it."""
    lag = LOOKBACK_FRAMES_AT_30FPS
    return build_features(seq[lag:], seq[:-lag], aspect, feature_set)


def body_motion(features):
    """Largest movement over the lookback, in torso units, for each row of features.

    Joints the pose model cannot see are skipped because their positions jitter.
    """
    count = len(BODY_JOINTS)
    change = features[:, count * 3:count * 6].reshape(len(features), count, 3)[:, :, :2]
    visible = features[:, count * 6:count * 7] >= MIN_JOINT_VISIBILITY
    joint_motion = (np.linalg.norm(change, axis=2) * visible).max(axis=1)
    return np.maximum(joint_motion, np.linalg.norm(features[:, HIP_MOTION], axis=1))


def apply_motion_gate(label, confidence, held_label, recent_motion, min_motion):
    """Reject a new action while the body has been still: moves cannot start from rest."""
    if label not in ("idle", held_label) and recent_motion < min_motion:
        return "idle", 0.0
    return label, confidence


def build_network(num_classes, num_features=NUM_FEATURES):
    return nn.Sequential(
        nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, num_classes),
    )


class FastActionClassifier:
    """Stateful per-frame classifier for the live stream, independent of frame rate."""

    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.actions = list(ckpt["actions"])
        self.feature_set = ckpt.get("feature_set", "pose")
        self.network = build_network(len(self.actions), FEATURE_SETS[self.feature_set])
        self.network.load_state_dict(ckpt["model_state"])
        self.network.to(device).eval()
        self.device = device
        self.mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["feature_std"], dtype=np.float32)
        self._frames = deque(maxlen=32)
        self._times = deque(maxlen=32)
        self._motion = deque(maxlen=32)

    def clear(self):
        """Forget history, so a lost pose cannot look like a large movement later."""
        self._frames.clear()
        self._times.clear()
        self._motion.clear()

    def recent_motion(self, timestamp_ms, memory_ms):
        """Largest body motion measured within the last memory_ms."""
        return max((m for t, m in self._motion if timestamp_ms - t < memory_ms), default=0.0)

    @property
    def ready(self):
        return bool(self._times) and self._times[-1] - self._times[0] >= LOOKBACK_MS

    def _past_frame(self, target_ms):
        """Linearly interpolate the landmark frame at target_ms."""
        times = self._times
        for i in range(len(times) - 1, 0, -1):
            if times[i - 1] <= target_ms:
                span = max(times[i] - times[i - 1], 1e-6)
                weight = (target_ms - times[i - 1]) / span
                return self._frames[i - 1] * (1.0 - weight) + self._frames[i] * weight
        return self._frames[0]

    def step(self, landmark_frame, timestamp_ms, aspect):
        """Add one (33, 4) frame. Returns (label, confidence), or None until ready."""
        self._frames.append(landmark_frame)
        self._times.append(float(timestamp_ms))
        if not self.ready:
            return None
        past = self._past_frame(float(timestamp_ms) - LOOKBACK_MS)
        features = build_features(landmark_frame[None], past[None], aspect, self.feature_set)
        self._motion.append((float(timestamp_ms), float(body_motion(features)[0])))
        tensor = torch.from_numpy((features - self.mean) / self.std).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.network(tensor), dim=1)[0]
            confidence, index = probabilities.max(dim=0)
        return self.actions[index.item()], confidence.item()
