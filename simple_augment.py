"""Basic data augmentation for ZeroController action samples.

Two techniques only — both safe for skeleton motion data:
  1. Mirror left<->right (cross-class for forward/backward, same-class for
     jump/block/idle so the action label stays correct).
  2. Tiny uniform scaling (±1-2%) around the mid-hip — body size jitter that
     simulates the user standing slightly closer or further from the camera
     without distorting pose semantics.

Saves results to a separate folder so they can be removed wholesale.

Semantics (from collected data):
  forward  -> right hand + right leg raised
  backward -> left  hand + left  leg raised
  jump     -> both legs distant from ground
  block    -> crossed hands

Mirroring x and swapping left<->right landmarks therefore:
  forward  --mirror-->  backward   (cross-class, free doubling)
  backward --mirror-->  forward
  jump     --mirror-->  jump       (same class)
  block    --mirror-->  block      (same class)
  idle     --mirror-->  idle       (same class)

Augmented samples land in:  data/augmented/<action>/*.npy
Original raw samples in:    data/raw/<action>/*.npy   (untouched)

Run:  python simple_augment.py
Remove all augments:  rmdir /s /q data\augmented   (Windows)
                      rm -rf data/augmented        (POSIX)
"""

import os
from pathlib import Path

import numpy as np

from config import DATA_DIR, SEQUENCE_LENGTH

# Sibling folder to data/raw. Easy to nuke when no longer wanted.
AUGMENTED_DIR = str(Path(DATA_DIR).parent / "augmented")

# MediaPipe 33-landmark left<->right pairs (used to mirror skeleton properly).
_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]

# Cross-class mirror map. forward<->backward swap. Others map to themselves.
MIRROR_TARGETS = {
    "forward":  "backward",
    "backward": "forward",
    "jump":     "jump",
    "block":    "block",
    "idle":     "idle",
}

# Classes to augment with tiny scale (the weak ones plus block).
SCALE_TARGETS = ["forward", "backward", "jump", "block", "idle",
                 "left_punch", "right_punch", "left_kick", "right_kick"]

# Tiny scale factors. Kept very close to 1.0 so motion semantics never change —
# enough variation to fight memorisation, not enough to invent fake samples.
SCALE_FACTORS = (0.98, 1.02)


def mirror_sequence(seq):
    """Flip x-coordinate and swap left<->right landmarks for a (T, 33, 4) clip."""
    out = seq.copy()
    out[:, :, 0] = 1.0 - out[:, :, 0]
    for a, b in _SWAP_PAIRS:
        out[:, a, :], out[:, b, :] = out[:, b, :].copy(), out[:, a, :].copy()
    return out


def scale_skeleton(seq, scale):
    """Uniform deterministic scale around the mid-hip.

    `scale` is a fixed factor (e.g. 0.98 or 1.02) instead of a random range,
    so each augmented file is reproducible and the spread of variants is
    controlled — important when trying to add variety without inviting
    overfitting on synthetic distortions.
    """
    out = seq.copy()
    # Center of mass = mid-hip (landmarks 23, 24) per frame.
    mid_hip = (seq[:, 23, :3] + seq[:, 24, :3]) * 0.5  # (T, 3)
    centered = seq[:, :, :3] - mid_hip[:, None, :]
    out[:, :, :3] = np.clip(centered * scale + mid_hip[:, None, :], 0.0, 1.0)
    return out


def _save(seq, action, base_name, suffix):
    out_dir = Path(AUGMENTED_DIR) / action
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{base_name}_{suffix}.npy", seq.astype(np.float32))


def _load_raw(action):
    raw_dir = Path(DATA_DIR) / action
    if not raw_dir.is_dir():
        return []
    samples = []
    for npy in sorted(raw_dir.glob("*.npy")):
        seq = np.load(npy)
        if seq.shape == (SEQUENCE_LENGTH, 33, 4):
            samples.append((npy.stem, seq))
        else:
            print(f"[SKIP] {npy.name}: shape {seq.shape}")
    return samples


def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - SIMPLE AUGMENTATION")
    print(f"  Output: {AUGMENTED_DIR}")
    print("=" * 60)

    rng_seed = 42
    np.random.seed(rng_seed)

    totals = {}

    # 1) Mirror: cross-class for forward<->backward, same-class for others.
    for src, dst in MIRROR_TARGETS.items():
        samples = _load_raw(src)
        if not samples:
            print(f"[SKIP] {src}: no raw samples")
            continue
        for name, seq in samples:
            mirrored = mirror_sequence(seq)
            _save(mirrored, dst, name, "mirror")
        totals[dst] = totals.get(dst, 0) + len(samples)
        print(f"[MIRROR] {src} -> {dst}: {len(samples)} new")

    # 2) Tiny scale: small deterministic factors per sample for every class.
    # 0.98 and 1.02 = ±2% body size change. Mid-hip stays fixed so the
    # skeleton just shrinks/grows in place — no positional drift, no noise.
    for action in SCALE_TARGETS:
        samples = _load_raw(action)
        if not samples:
            continue
        for name, seq in samples:
            for scale in SCALE_FACTORS:
                tag = f"scale{int(round(scale * 100)):03d}"  # e.g. scale098, scale102
                _save(scale_skeleton(seq, scale), action, name, tag)
        n_new = len(samples) * len(SCALE_FACTORS)
        totals[action] = totals.get(action, 0) + n_new
        print(f"[SCALE]  {action}: {n_new} new  (factors {SCALE_FACTORS})")

    print("-" * 60)
    grand = sum(totals.values())
    for action, n in sorted(totals.items()):
        print(f"  {action:<10} +{n}")
    print(f"  TOTAL      +{grand}")
    print("=" * 60)
    print(f"Done. Remove with: rm -rf {AUGMENTED_DIR}")


if __name__ == "__main__":
    main()
