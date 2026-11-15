"""Train the low-latency classifier in fast_action.py.

Labels come from data/labels/frame_labels.json, made by tools/label_segments.py. It
marks which frames of each clip really show the clip's action. Frames where the
player still stands before a move are taught as idle, frames that are too uncertain
are left out, and clips that show no action are skipped.

Most clips hold the finished pose for the whole second, so they never show how a move
starts. Joins from a standing clip into a held pose stand in for that start: once the
body is COMMIT_FRACTION of the way into the pose, those frames carry the move's label.

Run:  python train_fast.py                                       (final model, all sessions)
      python train_fast.py --holdout-session 3 --save-path X.pth (model for held-out scoring)
      python train_fast.py --labels clip --transitions idle ...  (the previous labelling)
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from config import ACTIONS, DATA_DIR
from fast_action import FEATURE_SETS, HIP_MOTION, LOOKBACK_FRAMES_AT_30FPS, build_network, sequence_features
from pose_features import TRAINING_ASPECT
from train_robust import (
    ALL_CONDITIONS, MIRROR_LABELS, MULTICAM_DIR, augment_landmarks, cross_fade, is_tracked,
    mirror_landmarks, recording_sessions,
)

DEFAULT_SAVE_PATH = "models/action_classifier_fast.pth"
FRAME_LABELS_PATH = os.path.join("data", "labels", "frame_labels.json")
IDLE = ACTIONS.index("idle")
JUMP = ACTIONS.index("jump")
SKIP = -1  # target for frames left out of training
# --labels clip only: jump clips include standing frames, so only frames with real hip
# motion count as jump.
JUMP_MIN_HIP_MOTION = 0.12
AUGMENTED_COPIES = 2
TRANSITIONS_PER_CLIP = 2
# Extra joins for a clip that holds its pose: standing into it, and back to standing.
STARTS_PER_HELD_CLIP = 2
ENDS_PER_HELD_CLIP = 2
# A join from standing into a pose counts as the move once the body is this far along.
COMMIT_FRACTION = 0.3
LAG = LOOKBACK_FRAMES_AT_30FPS
BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


# ── Labels ───────────────────────────────────────────────────────────────────

def frame_targets(marks, label, early):
    """Per-frame class index (or SKIP) from a clip's marks string."""
    marks = np.asarray(list(marks))
    targets = np.full(len(marks), SKIP, dtype=np.int64)
    targets[marks == "a"] = label
    targets[marks == "i"] = IDLE
    if early == "action":
        targets[marks == "e"] = label
    return targets


def clip_targets(clip, aspect, label):
    """The previous labelling: every frame is the clip's action, except still jump frames."""
    targets = np.full(len(clip), label, dtype=np.int64)
    if label == JUMP:
        motion = np.zeros(len(clip))
        motion[LAG:] = np.abs(sequence_features(clip, aspect)[:, HIP_MOTION][:, 1])
        targets[motion <= JUMP_MIN_HIP_MOTION] = SKIP
    targets[:LAG] = SKIP
    return targets


def _body(clip, aspect):
    pos = clip[..., :2].astype(np.float64).copy()
    pos[..., 0] *= aspect
    hip = (pos[:, 23] + pos[:, 24]) / 2
    shoulder = (pos[:, 11] + pos[:, 12]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    rel = (pos[:, BODY_JOINTS] - hip[:, None]) / np.median(torso)
    visible = clip[:, [11, 12, 23, 24], 3].min(axis=1) >= 0.5
    usable = visible & (torso > 0.6 * np.median(torso)) & (torso < 1.6 * np.median(torso))
    return rel, usable


def align_marks(raw_clip, clip, aspect, marks):
    """Move a raw clip's marks onto another tracking of the same video.

    The two trackings can sit a few frames apart, because each was resampled to 30
    frames from a different number of camera frames.
    """
    raw, _ = _body(raw_clip, TRAINING_ASPECT)
    other, usable = _body(clip, aspect)
    length = len(raw)
    best_lag, best_error = 0, np.inf
    for lag in range(-3, 4):
        lo, hi = max(0, -lag), min(length, length - lag)
        error = float(np.abs(other[lo:hi] - raw[lo + lag:hi + lag]).mean())
        if error < best_error - 1e-6:
            best_lag, best_error = lag, error
    marks = np.asarray(list(marks))
    shifted = np.full(length, "x")
    for t in range(length):
        if 0 <= t + best_lag < length:
            shifted[t] = marks[t + best_lag]
    shifted[~usable] = "x"
    return "".join(shifted)


def load_all_clips(use_conditions=True):
    """(clip, aspect, label, session, key, source) for raw clips and multicam versions."""
    sessions = recording_sessions()
    rows = []
    for (action, name), session in sorted(sessions.items()):
        clip = np.load(os.path.join(DATA_DIR, action, name + ".npy")).astype(np.float32)
        rows.append((clip, TRAINING_ASPECT, ACTIONS.index(action), session, f"{action}/{name}", "raw"))
    for condition in (ALL_CONDITIONS if use_conditions else []):
        data = np.load(os.path.join(MULTICAM_DIR, condition + ".npz"))
        aspect = float(data["aspect"])
        for full_name, clip in zip(data["names"], data["x"]):
            action, name = str(full_name).split("__")
            key = (action, name[:-4])
            if key in sessions:
                rows.append((clip.astype(np.float32), aspect, ACTIONS.index(action), sessions[key],
                             f"{action}/{name[:-4]}", condition))
    return [row for row in rows if is_tracked(row[0])]


def labelled_clips(rows, labels, early):
    """Attach per-frame targets: (clip, aspect, label, targets, session, key, source)."""
    if labels == "clip":
        return [(clip, aspect, label, clip_targets(clip, aspect, label), session, key, source)
                for clip, aspect, label, session, key, source in rows]
    with open(FRAME_LABELS_PATH) as f:
        marks = json.load(f)["clips"]
    raw = {key: clip for clip, _, _, _, key, source in rows if source == "raw"}
    out = []
    for clip, aspect, label, session, key, source in rows:
        info = marks.get(key)
        if info is None or info["status"] == "drop":
            continue
        clip_marks = info["marks"] if source == "raw" else align_marks(raw[key], clip, aspect, info["marks"])
        out.append((clip, aspect, label, frame_targets(clip_marks, label, early), session, key, source))
    return out


# ── Training samples ─────────────────────────────────────────────────────────

def clip_samples(clip, aspect, targets, feature_set="pose"):
    """Features and targets for every frame with a full lookback and a target."""
    features = sequence_features(clip, aspect, feature_set)
    rows = targets[LAG:]
    keep = rows != SKIP
    return features[keep], rows[keep]


def join_targets(start, end, fade, transitions):
    """Targets for the blend frames of a join from target `start` into target `end`."""
    if transitions == "idle":  # previous rule: nothing is committed while the body moves
        return np.full(fade, IDLE, dtype=np.int64)
    weight = np.arange(1, fade + 1) / (fade + 1)
    out = np.full(fade, SKIP, dtype=np.int64)
    if SKIP in (start, end):
        return out
    if start == end:
        out[:] = end
    elif start == IDLE:            # standing into a move: the move has started
        out[weight >= COMMIT_FRACTION] = end
    elif end == IDLE:              # back to standing: no new move starts on the way
        out[:] = IDLE
    else:                          # one move into another
        out[weight >= 0.5] = end
    return out


def transition_samples(first, first_targets, second, second_targets, aspect, rng, transitions,
                       feature_set="pose"):
    """Frames around a join from the end of one clip into the start of another."""
    tail = LAG + 4
    fade = int(rng.integers(3, 10))
    joined = cross_fade(first[-tail:], second[:tail], fade)
    blend = join_targets(first_targets[-1], second_targets[0], fade, transitions)
    targets = np.concatenate([first_targets[-tail:], blend, second_targets[:tail]])
    features = sequence_features(joined, aspect, feature_set)
    rows = targets[LAG:]
    start = tail - LAG
    # The old rule kept only the blend. The new one also keeps the frames just after it,
    # whose lookback still sees the movement: the moment a pose is reached.
    stop = start + fade + (0 if transitions == "idle" else LAG)
    features, rows = features[start:stop], rows[start:stop]
    keep = rows != SKIP
    return features[keep], rows[keep]


def augmented(clip, targets, aspect, rng):
    variant, index = augment_landmarks(clip, aspect, rng, return_index=True)
    source = np.clip(np.round(index).astype(int), 0, len(targets) - 1)
    return variant, targets[source]


def build_training_set(clips, seed, transitions="onset", feature_set="pose"):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    pool = {}
    for clip, aspect, label, targets, *_ in clips:
        pool.setdefault(aspect, []).append((clip, label, targets))

    def add(features_targets):
        features, targets = features_targets
        if len(features):
            xs.append(features)
            ys.append(targets)

    for clip, aspect, label, targets, *_ in clips:
        name = ACTIONS[label]
        variants = [(clip, targets)] + [augmented(clip, targets, aspect, rng) for _ in range(AUGMENTED_COPIES)]
        for variant, variant_targets in variants:
            add(clip_samples(variant, aspect, variant_targets, feature_set))
            if name in MIRROR_LABELS:
                mirrored = ACTIONS.index(MIRROR_LABELS[name])
                add(clip_samples(mirror_landmarks(variant), aspect,
                                 np.where(variant_targets == label, mirrored, variant_targets), feature_set))

        same = pool[aspect]
        joins = []
        for _ in range(TRANSITIONS_PER_CLIP):
            other, other_label, other_targets = same[rng.integers(len(same))]
            if JUMP in (label, other_label):
                continue
            ordered = [(clip, targets), (other, other_targets)]
            if rng.random() < 0.5:
                ordered.reverse()
            joins.append(ordered)
        if transitions == "onset" and label not in (IDLE, JUMP) and targets[0] == label:
            idle = [item for item in same if item[1] == IDLE]
            for count, into in ((STARTS_PER_HELD_CLIP, True), (ENDS_PER_HELD_CLIP, False)):
                for _ in range(count):
                    other, _, other_targets = idle[rng.integers(len(idle))]
                    pair = [(other, other_targets), (clip, targets)]
                    joins.append(pair if into else pair[::-1])
        for (first, first_targets), (second, second_targets) in joins:
            first, first_targets = augmented(first, first_targets, aspect, rng)
            second, second_targets = augmented(second, second_targets, aspect, rng)
            add(transition_samples(first, first_targets, second, second_targets, aspect, rng, transitions,
                                   feature_set))
    return np.concatenate(xs), np.concatenate(ys)


def train_network(x, y, epochs, seed, idle_weight, device="cpu"):
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    features = torch.from_numpy((x - mean) / std).to(device)
    labels = torch.from_numpy(y).to(device)
    network = build_network(len(ACTIONS), x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), 2e-3, weight_decay=1e-3)
    # A new player's standing pose is the case most often misread as an action.
    class_weights = torch.ones(len(ACTIONS))
    class_weights[IDLE] = idle_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    batch_size = 512
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 3e-3, total_steps=epochs * (len(features) // batch_size + 1))
    for epoch in range(1, epochs + 1):
        network.train()
        order = torch.randperm(len(features), device=device)
        total = 0.0
        for start in range(0, len(features), batch_size):
            batch = order[start:start + batch_size]
            loss = criterion(network(features[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(batch)
        print(f"  epoch {epoch:2d}/{epochs}  loss {total / len(features):.4f}", flush=True)
    return network.cpu().eval(), mean, std


def main():
    parser = argparse.ArgumentParser(description="Train the low-latency action classifier.")
    parser.add_argument("--holdout-session", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--idle-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--device", default="auto", help="auto picks cuda when a GPU is present, else cpu")
    parser.add_argument("--labels", choices=("frames", "clip"), default="frames",
                        help="frames: marks from tools/label_segments.py. clip: every frame is the clip's action.")
    parser.add_argument("--early", choices=("ignore", "action"), default="action",
                        help="How to train frames marked early (a move under 30%% of the way in).")
    parser.add_argument("--transitions", choices=("onset", "idle"), default="onset",
                        help="onset: a join into a pose starts the move. idle: joins are taught as idle.")
    parser.add_argument("--sources", choices=("all", "raw"), default="all",
                        help="raw: skip the multicam re-trackings of the clips.")
    parser.add_argument("--features", choices=tuple(FEATURE_SETS), default="pose",
                        help="pose+angles adds elbow, shoulder, knee and hip angles.")
    args = parser.parse_args()
    if args.holdout_session is not None and args.save_path == DEFAULT_SAVE_PATH:
        parser.error("--holdout-session needs its own --save-path, so it cannot replace the final model")

    if args.threads:
        torch.set_num_threads(args.threads)
    np.random.seed(args.seed)

    holdout = set(args.holdout_session or [])
    rows = [row for row in load_all_clips(args.sources == "all") if row[3] not in holdout]
    clips = labelled_clips(rows, args.labels, args.early)
    x, y = build_training_set(clips, args.seed, args.transitions, args.features)
    assert x.shape[1] == FEATURE_SETS[args.features]
    print(f"Training frames: {len(x)}  from clips: {len(clips)}  holdout: {sorted(holdout)}", flush=True)
    print("Frames per class: " + json.dumps(dict(zip(ACTIONS, np.bincount(y, minlength=len(ACTIONS)).tolist()))), flush=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    network, mean, std = train_network(x, y, args.epochs, args.seed, args.idle_weight, device)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({
        "model_state": network.state_dict(),
        "actions": ACTIONS,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "holdout_sessions": sorted(holdout),
        "labels": args.labels,
        "early": args.early,
        "transitions": args.transitions,
        "sources": args.sources,
        "feature_set": args.features,
    }, args.save_path)
    print(f"Saved {args.save_path}")


if __name__ == "__main__":
    main()
