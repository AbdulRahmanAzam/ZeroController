"""Train the camera-robust action classifier used by run_model.py.

Differences from train_model.py:
  * Camera-independent features (pose_features.preprocess_pose_v2).
  * Trains on data/raw plus data/multicam, the same clips re-run through MediaPipe
    under live camera conditions (build with tools/build_multicam_landmarks.py).
  * Random camera and body changes applied to raw landmarks, before feature building,
    plus left-right mirroring for punches and kicks.
  * Optional mixed windows, part idle and part action, like the live stream produces.
  * --holdout-session measures accuracy on a recording session never trained on.
    Random splits of this dataset put near-identical clips on both sides and report
    close to 100%, which says nothing about a new camera setup.

Run:  python train_robust.py                         (final model, all sessions)
      python train_robust.py --holdout-session 3     (accuracy check, saves nothing)
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    ACTIONS, DATA_DIR, DROPOUT, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, RUNTIME_MODEL_PATH, RUNTIME_MODEL_TYPE,
    POSECNN_CHANNELS, POSECNN_KERNEL_SIZE, POSECNN_NUM_LAYERS, STGCN_CHANNELS, STGCN_TEMPORAL_K,
    TCN_KERNEL_SIZE, TCN_NUM_CHANNELS, TCN_NUM_LAYERS,
)
from pose_features import TRAINING_ASPECT, V2_CHANNELS, preprocess_pose_v2, to_model_input
from train_model import ActionGRU, ActionPoseConv1D, ActionSTGCN, ActionTCN, AugmentedDataset

MULTICAM_DIR = os.path.join(os.path.dirname(DATA_DIR.rstrip("/\\")), "multicam")
SESSIONS_PATH = os.path.join(os.path.dirname(DATA_DIR.rstrip("/\\")), "labels", "sessions.json")
TRAIN_CONDITIONS = ["full_169", "lite_169", "lite_43"]
ALL_CONDITIONS = TRAIN_CONDITIONS + ["lite_43_zoom", "lite_43_far", "lite_43_tilt", "full_43"]
SESSION_GAP_SECONDS = 1800
LEG_JOINTS = [25, 26, 27, 28, 29, 30, 31, 32]
IDLE = ACTIONS.index("idle")
MIN_FRAMES_FOR_ACTION_LABEL = 12
MIRROR_LABELS = {"left_punch": "right_punch", "right_punch": "left_punch",
                 "left_kick": "right_kick", "right_kick": "left_kick"}


def recording_sessions():
    """Map (action, sample name) to a recording session index.

    Sessions come from gaps in file times, but a fresh clone (or a Colab copy) resets
    file times. So the sessions found on the recording machine are kept in
    data/labels/sessions.json (tools/label_segments.py writes it), and win here.
    Clips missing from that file are split by file times and numbered after it.
    """
    present = {(os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-4]): p
               for p in glob.glob(os.path.join(DATA_DIR, "*", "*.npy"))}
    saved = {}
    if os.path.exists(SESSIONS_PATH):
        with open(SESSIONS_PATH) as f:
            saved = {tuple(key.split("/", 1)): int(session) for key, session in json.load(f).items()}
    sessions = {key: session for key, session in saved.items() if key in present}
    rows = sorted((os.path.getmtime(path), key) for key, path in present.items() if key not in sessions)
    if rows:
        session, previous = max(saved.values(), default=-1) + 1, rows[0][0]
        for stamp, key in rows:
            if stamp - previous > SESSION_GAP_SECONDS:
                session += 1
            previous = stamp
            sessions[key] = session
    return sessions


def save_sessions(sessions):
    """Keep the session of every clip, so a copy without the original file times can use them."""
    os.makedirs(os.path.dirname(SESSIONS_PATH), exist_ok=True)
    with open(SESSIONS_PATH, "w") as f:
        json.dump({f"{action}/{name}": session for (action, name), session in sorted(sessions.items())}, f, indent=0)


def is_tracked(clip):
    """False for a clip whose torso was never properly tracked."""
    return float(clip[:, [11, 12, 23, 24], 3].mean()) >= 0.5


def load_raw(sessions):
    return [
        (action, session, np.load(os.path.join(DATA_DIR, action, name + ".npy")).astype(np.float32))
        for (action, name), session in sessions.items()
    ]


def load_condition(condition, sessions):
    data = np.load(os.path.join(MULTICAM_DIR, condition + ".npz"))
    clips = []
    for full_name, clip in zip(data["names"], data["x"]):
        action, name = str(full_name).split("__")
        key = (action, name[:-4])
        if key in sessions:
            clips.append((action, sessions[key], clip))
    return clips, float(data["aspect"])


def augment_landmarks(clip, aspect, rng, return_index=False):
    """Random camera and body changes on (T, 33, 4) normalized landmarks.

    With return_index, also returns which source frame (fractional) each output frame
    came from, so per-frame labels can follow the random time stretch.
    """
    clip = clip.copy()
    length = len(clip)
    index = np.arange(length, dtype=np.float64)
    cut = int(rng.integers(0, 7))
    if cut:
        start = int(rng.integers(0, cut + 1))
        part = clip[start:length - (cut - start)]
        idx = np.linspace(0, len(part) - 1, length)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, len(part) - 1)
        t = (idx - lo)[:, None, None]
        clip = (part[lo] * (1 - t) + part[hi] * t).astype(np.float32)
        index = start + idx

    x, y = clip[..., 0] * aspect, clip[..., 1].copy()
    cx, cy = x.mean(), y.mean()
    angle = np.deg2rad(rng.uniform(-8, 8))
    scale = rng.uniform(0.75, 1.3)
    dx, dy = (x - cx) * rng.uniform(0.85, 1.15), y - cy
    clip[..., 0] = ((np.cos(angle) * dx - np.sin(angle) * dy) * scale + cx) / aspect
    clip[..., 1] = (np.sin(angle) * dx + np.cos(angle) * dy) * scale + cy

    clip[..., :3] += rng.normal(0, 0.004, clip[..., :3].shape).astype(np.float32)
    clip[..., 2] += rng.normal(0, 0.03, clip[..., 2].shape).astype(np.float32)
    clip[..., 3] = np.clip(clip[..., 3] * rng.uniform(0.75, 1.05), 0, 1)

    if rng.random() < 0.25:  # legs out of frame or badly tracked
        joints = rng.choice(LEG_JOINTS, int(rng.integers(2, 8)), replace=False)
        clip[:, joints, 3] *= rng.uniform(0, 0.4)
        clip[:, joints, :2] += rng.normal(0, 0.03, clip[:, joints, :2].shape).astype(np.float32)
    return (clip, index) if return_index else clip


def cross_fade(first, second, frames):
    """Join two clips with hips aligned and a short blend between them."""
    second = second.copy()
    second[..., :3] += ((first[-1, 23, :3] + first[-1, 24, :3])
                        - (second[0, 23, :3] + second[0, 24, :3])) / 2
    w = np.linspace(0, 1, frames + 2)[1:-1, None, None]
    blend = (first[-1][None] * (1 - w) + second[0][None] * w).astype(np.float32)
    return np.concatenate([first, blend, second], axis=0)


def mixed_window(clip, aspect, label, pool, rng):
    """A window the live stream really produces: part idle, part action.

    It keeps the action label once the action fills most of the window, and an idle
    clip keeps the idle label while a few frames of the previous action remain.
    """
    by_label = pool[aspect]
    length = len(clip)
    fade = int(rng.integers(3, 8))
    if label != IDLE:
        kept = int(rng.integers(MIN_FRAMES_FOR_ACTION_LABEL, length - fade))
        start = int(rng.integers(0, length - kept + 1))
        action = clip[start:start + kept]
        lead = length - kept - fade
        if lead <= 0:
            return action[-length:]
        idle = by_label[IDLE][rng.integers(len(by_label[IDLE]))]
        return cross_fade(idle[-lead:], action, fade)
    other = int(rng.choice([k for k in by_label if k != IDLE]))
    previous = by_label[other][rng.integers(len(by_label[other]))]
    tail = int(rng.integers(2, 7))
    return cross_fade(previous[-tail:], clip[:length - tail - fade], fade)


def mirror_landmarks(clip):
    """Flip a clip left-to-right in normalized image space and swap joint sides."""
    clip = clip.copy()
    clip[..., 0] = 1.0 - clip[..., 0]
    for left, right in AugmentedDataset._SWAP_PAIRS:
        clip[:, [left, right]] = clip[:, [right, left]]
    return clip


def build_model(model_type):
    frame_features = V2_CHANNELS * 33
    classes = len(ACTIONS)
    if model_type == "stgcn":
        return ActionSTGCN(V2_CHANNELS, classes, STGCN_CHANNELS, STGCN_TEMPORAL_K, DROPOUT)
    if model_type == "tcn":
        return ActionTCN(frame_features, TCN_NUM_CHANNELS, TCN_KERNEL_SIZE, TCN_NUM_LAYERS,
                         classes, DROPOUT)
    if model_type == "gru":
        return ActionGRU(frame_features, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, classes, DROPOUT)
    return ActionPoseConv1D(frame_features, POSECNN_CHANNELS, POSECNN_KERNEL_SIZE,
                            POSECNN_NUM_LAYERS, classes, 0.1)


class RobustDataset(Dataset):
    def __init__(self, items, mixed_prob, seed, model_type, mirror=True):
        self.items = items
        self.model_type = model_type
        self.mixed_prob = mixed_prob
        self.mirror = mirror
        self.rng = np.random.default_rng(seed)
        self.pool = {}
        for clip, aspect, label in items:
            self.pool.setdefault(aspect, {}).setdefault(label, []).append(clip)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        clip, aspect, label = self.items[index]
        if self.mixed_prob and self.rng.random() < self.mixed_prob:
            clip = mixed_window(clip, aspect, label, self.pool, self.rng)
        clip = augment_landmarks(clip, aspect, self.rng)
        name = ACTIONS[label]
        if self.mirror and name in MIRROR_LABELS and self.rng.random() < 0.5:
            clip = mirror_landmarks(clip)
            label = ACTIONS.index(MIRROR_LABELS[name])
        return torch.from_numpy(to_model_input(preprocess_pose_v2(clip, aspect), self.model_type)), label


def evaluate(model, clips, aspect, model_type, device="cpu"):
    """Return (accuracy, confusion matrix) for (action, session, clip) rows."""
    confusion = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    with torch.inference_mode():
        for i in range(0, len(clips), 64):
            batch = clips[i:i + 64]
            x = torch.from_numpy(np.stack(
                [to_model_input(preprocess_pose_v2(c[2], aspect), model_type) for c in batch]
            )).to(device)
            for row, predicted in zip(batch, model(x).argmax(1).tolist()):
                confusion[ACTIONS.index(row[0]), predicted] += 1
    return float(np.trace(confusion) / max(confusion.sum(), 1)), confusion


def main():
    parser = argparse.ArgumentParser(description="Train the camera-robust action classifier.")
    parser.add_argument("--model-type", choices=("stgcn", "tcn", "gru", "poseconv1d"),
                        default=RUNTIME_MODEL_TYPE)
    parser.add_argument("--holdout-session", type=int, default=None,
                        help="Exclude this session from training and report accuracy on it.")
    parser.add_argument("--conditions", choices=("three", "all"), default="all",
                        help="Which multicam conditions to train on.")
    parser.add_argument("--mixed-prob", type=float, default=0.4,
                        help="Chance a sample becomes a part-idle, part-action window.")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--save-path", default=RUNTIME_MODEL_PATH)
    parser.add_argument("--device", default="auto", help="auto picks cuda when a GPU is present, else cpu")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    holdout = {args.holdout_session} if args.holdout_session is not None else set()
    sessions = recording_sessions()
    raw = [row for row in load_raw(sessions) if is_tracked(row[2])]
    labelled = [(clip, TRAINING_ASPECT, ACTIONS.index(action))
                for action, session, clip in raw if session not in holdout]

    tests = {"raw": ([row for row in raw if row[1] in holdout], TRAINING_ASPECT)}
    names = ALL_CONDITIONS if args.conditions == "all" else TRAIN_CONDITIONS
    for condition in ALL_CONDITIONS:
        clips, aspect = load_condition(condition, sessions)
        clips = [row for row in clips if is_tracked(row[2])]
        tests[condition] = ([row for row in clips if row[1] in holdout], aspect)
        if condition in names:
            labelled += [(clip, aspect, ACTIONS.index(action))
                         for action, session, clip in clips if session not in holdout]
    print(f"Training windows: {len(labelled)}  conditions: {names}  holdout: {sorted(holdout)}")

    model = build_model(args.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(RobustDataset(labelled, args.mixed_prob, args.seed, args.model_type, not args.no_mirror),
                        batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        scheduler.step()
        print(f"  epoch {epoch:3d}/{args.epochs}  loss {total / len(labelled):.4f}", flush=True)
    model.eval()

    if holdout:
        results = {name: round(evaluate(model, rows, aspect, args.model_type, device)[0], 4)
                   for name, (rows, aspect) in tests.items() if rows}
        print("HOLDOUT " + json.dumps({"session": sorted(holdout), "accuracy": results}))
        return

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "actions": ACTIONS,
        "model_type": args.model_type,
        "input_size": V2_CHANNELS * 33,
        "stgcn_in_channels": V2_CHANNELS,
        "stgcn_channels": list(STGCN_CHANNELS),
        "stgcn_temporal_k": STGCN_TEMPORAL_K,
        "dropout": DROPOUT,
        "tcn_channels": TCN_NUM_CHANNELS,
        "tcn_kernel_size": TCN_KERNEL_SIZE,
        "tcn_num_layers": TCN_NUM_LAYERS,
        "gru_hidden_size": GRU_HIDDEN_SIZE,
        "gru_num_layers": GRU_NUM_LAYERS,
        "posecnn_channels": POSECNN_CHANNELS,
        "posecnn_kernel_size": POSECNN_KERNEL_SIZE,
        "posecnn_num_layers": POSECNN_NUM_LAYERS,
        "preprocess_version": 2,
    }, args.save_path)
    print(f"Saved {args.save_path}")


if __name__ == "__main__":
    main()
