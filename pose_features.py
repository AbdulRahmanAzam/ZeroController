"""Camera-independent pose features (preprocess version 2).

MediaPipe reports x as a fraction of frame width and y as a fraction of frame
height, so the same body has different numbers on a 16:9 and a 4:3 camera.
Version 2 undoes that, keeps visibility as its own channel instead of
multiplying it into the coordinates, and keeps hip motion, which hip-centering
would otherwise erase and which is what a jump looks like.
"""

import numpy as np

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# position (3) + velocity (3) + visibility (1) + hip velocity (2)
V2_CHANNELS = 9

# Every clip in data/raw was captured at 1280x720.
TRAINING_ASPECT = 16.0 / 9.0


def preprocess_pose_v2(seq, aspect):
    """Convert a (T, 33, 4) landmark window to a (9, T, 33) float32 array.

    aspect is the width / height of the frame the landmarks came from.
    """
    pos = seq[..., :3].astype(np.float32).copy()
    vis = seq[..., 3:4].astype(np.float32)

    # MediaPipe scales z with frame width, so it needs the same correction as x.
    pos[..., 0] *= aspect
    pos[..., 2] *= aspect

    hip = (pos[:, L_HIP] + pos[:, R_HIP]) / 2.0
    shoulder = (pos[:, L_SHOULDER] + pos[:, R_SHOULDER]) / 2.0
    torso = max(float(np.linalg.norm((shoulder - hip)[:, :2], axis=-1).mean()), 1e-3)

    rel = (pos - hip[:, None]) / torso
    vel = np.zeros_like(rel)
    vel[1:] = rel[1:] - rel[:-1]

    hip_vel = np.zeros((len(seq), 2), dtype=np.float32)
    hip_vel[1:] = (hip[1:, :2] - hip[:-1, :2]) / torso
    hip_vel = np.repeat(hip_vel[:, None], seq.shape[1], axis=1)

    features = np.concatenate([rel, vel, vis, hip_vel], axis=-1)
    return features.transpose(2, 0, 1).astype(np.float32)


def to_model_input(features, model_type):
    """Lay out (C, T, V) features for a model: ST-GCN reads them as is, others per frame."""
    if model_type == "stgcn":
        return features
    return features.transpose(1, 0, 2).reshape(features.shape[1], -1)
