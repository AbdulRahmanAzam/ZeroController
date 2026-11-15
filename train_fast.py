"""Train the low-latency classifier in fast_action.py.

It trains on the updated clips in data/clips (tools/build_clips.py): each clip holds
only the frames that show its action, standing frames are idle clips, and unusable
recordings are left out. Every frame of a clip is taught as the clip's action, except
up to 4 lead-in frames that only give the first frames their 133 ms of motion.

Most clips hold the finished pose for the whole second, so they never show how a move
starts. Joins from a standing clip into a held pose stand in for that start: once the
body is COMMIT_FRACTION of the way into the pose, those frames carry the move's label.

Run:  python train_fast.py                                       (final model, all sessions)
      python train_fast.py --holdout-session 3 --save-path X.pth (model for held-out scoring)
      python train_fast.py --data raw --transitions idle ...     (the previous training)
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from config import ACTIONS
from fast_action import FEATURE_SETS, HIP_MOTION, LOOKBACK_FRAMES_AT_30FPS, build_network, sequence_features
from train_robust import (
    ALL_CONDITIONS, MIRROR_LABELS, SKIP, augment_landmarks, cross_fade, load_all_clips,
    load_updated_clips, mirror_landmarks,
)

DEFAULT_SAVE_PATH = "models/action_classifier_fast.pth"
IDLE = ACTIONS.index("idle")
JUMP = ACTIONS.index("jump")
# --data raw only: jump recordings include standing frames, so only frames with real
# hip motion count as jump.
JUMP_MIN_HIP_MOTION = 0.12
AUGMENTED_COPIES = 2
TRANSITIONS_PER_CLIP = 2
# Extra joins for a clip that holds its pose: standing into it, and back to standing.
STARTS_PER_HELD_CLIP = 2
ENDS_PER_HELD_CLIP = 2
# A join from standing into a pose counts as the move once the body is this far along.
COMMIT_FRACTION = 0.3
LAG = LOOKBACK_FRAMES_AT_30FPS


# ── Labels ───────────────────────────────────────────────────────────────────

def clip_targets(clip, aspect, label):
    """The previous labelling: every frame is the clip's action, except still jump frames."""
    targets = np.full(len(clip), label, dtype=np.int64)
    if label == JUMP:
        motion = np.zeros(len(clip))
        motion[LAG:] = np.abs(sequence_features(clip, aspect)[:, HIP_MOTION][:, 1])
        targets[motion <= JUMP_MIN_HIP_MOTION] = SKIP
    targets[:LAG] = SKIP
    return targets


def training_rows(data, use_conditions=True):
    """(clip, aspect, label, targets, session, key, source) rows to train on."""
    if data == "raw":  # the previous training: whole recordings, every frame is the action
        return [(clip, aspect, label, clip_targets(clip, aspect, label), session, key, source)
                for clip, aspect, label, session, key, source in load_all_clips(use_conditions)]
    return load_updated_clips(ALL_CONDITIONS if use_conditions else [])


# ── Training samples ─────────────────────────────────────────────────────────

def clip_samples(clip, aspect, targets, feature_set="pose"):
    """Features and targets for every frame with a full lookback and a target."""
    if len(clip) <= LAG:  # too short for one frame with a full 133 ms behind it
        return np.zeros((0, 1), np.float32), np.zeros(0, np.int64)
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
    parser.add_argument("--data", choices=("clips", "raw"), default="clips",
                        help="clips: the updated clips in data/clips. raw: whole recordings, every frame the action.")
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
    clips = [row for row in training_rows(args.data, args.sources == "all") if row[4] not in holdout]
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
        "data": args.data,
        "transitions": args.transitions,
        "sources": args.sources,
        "feature_set": args.features,
    }, args.save_path)
    print(f"Saved {args.save_path}")


if __name__ == "__main__":
    main()
