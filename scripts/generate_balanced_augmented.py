"""Generate a balanced augmented dataset without touching raw data."""

import argparse
import hashlib
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np

from config import ACTIONS, DATA_DIR, SEQUENCE_LENGTH


def _augmented_root(raw_root: str) -> str:
    return os.path.join(os.path.dirname(raw_root.rstrip(os.sep)), "augmented")


def _iter_raw_files(root: str, action: str) -> List[str]:
    action_dir = os.path.join(root, action)
    if not os.path.isdir(action_dir):
        return []
    return [
        os.path.join(action_dir, name)
        for name in sorted(os.listdir(action_dir))
        if name.endswith(".npy")
    ]


def _hash_sequence(arr: np.ndarray, round_decimals: int | None) -> str:
    if round_decimals is not None:
        arr = np.round(arr.astype(np.float32), round_decimals)
    h = hashlib.blake2b(digest_size=16)
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode("utf-8"))
    return h.hexdigest()


def _add_spatial_noise(seq: np.ndarray, rng: np.random.Generator, std: float = 0.01) -> np.ndarray:
    noisy = seq.copy()
    noise = rng.normal(0.0, std, size=seq[:, :, :2].shape).astype(np.float32)
    noisy[:, :, 0:2] = np.clip(seq[:, :, 0:2] + noise, 0.0, 1.0)
    return noisy


def _rotation_jitter(seq: np.ndarray, rng: np.random.Generator, max_angle_deg: float = 6.0) -> np.ndarray:
    angle = rng.uniform(-max_angle_deg, max_angle_deg)
    angle_rad = np.radians(angle)
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    rotated = seq.copy()
    x = seq[:, :, 0]
    y = seq[:, :, 1]
    rotated[:, :, 0] = np.clip(x * cos_a - y * sin_a, 0.0, 1.0)
    rotated[:, :, 1] = np.clip(x * sin_a + y * cos_a, 0.0, 1.0)
    return rotated


def _temporal_jitter(seq: np.ndarray, rng: np.random.Generator, jitter_sigma: float = 0.03) -> np.ndarray:
    T = seq.shape[0]
    time_offsets = rng.normal(0.0, jitter_sigma * T, size=T)
    time_offsets = np.cumsum(time_offsets)
    time_offsets = np.clip(time_offsets, 0, T - 1)

    jittered = np.zeros_like(seq)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            jittered[:, j, c] = np.interp(np.arange(T), time_offsets, seq[:, j, c])

    return jittered


def _time_stretch(seq: np.ndarray, rng: np.random.Generator, stretch_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    T = seq.shape[0]
    stretch = rng.uniform(*stretch_range)
    old_idx = np.arange(T) / stretch
    old_idx = np.clip(old_idx, 0, T - 1)

    stretched = np.zeros_like(seq)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            stretched[:, j, c] = np.interp(np.arange(T), old_idx, seq[:, j, c])

    return stretched


def _visibility_dropout(seq: np.ndarray, rng: np.random.Generator, dropout_prob: float = 0.05) -> np.ndarray:
    dropped = seq.copy()
    mask = rng.random(seq.shape[1]) < dropout_prob
    dropped[:, mask, 3] = 0.0
    return dropped


def _augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    techniques = [
        _add_spatial_noise,
        _rotation_jitter,
        _temporal_jitter,
        _time_stretch,
        _visibility_dropout,
    ]
    num_ops = 2 if rng.random() < 0.5 else 1
    picks = rng.choice(len(techniques), size=num_ops, replace=False)
    out = seq
    for idx in picks:
        out = techniques[int(idx)](out, rng)
    return out.astype(np.float32)


def _load_raw_sequences(raw_root: str, action: str) -> List[np.ndarray]:
    sequences = []
    for path in _iter_raw_files(raw_root, action):
        seq = np.load(path)
        if seq.shape == (SEQUENCE_LENGTH, 33, 4):
            sequences.append(seq.astype(np.float32))
    return sequences


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _generate_for_action(
    action: str,
    raw_sequences: List[np.ndarray],
    out_dir: str,
    target_total: int,
    rng: np.random.Generator,
    round_decimals: int | None,
) -> Dict[str, int]:
    raw_count = len(raw_sequences)
    if raw_count == 0:
        return {"raw": 0, "augmented": 0, "saved": 0, "needed": 0}

    if raw_count >= target_total:
        return {"raw": raw_count, "augmented": 0, "saved": 0, "needed": 0}

    needed = target_total - raw_count
    _ensure_dir(out_dir)

    existing_hashes = set()
    for seq in raw_sequences:
        existing_hashes.add(_hash_sequence(seq, round_decimals))

    saved = 0
    attempts = 0
    max_attempts = needed * 30

    while saved < needed and attempts < max_attempts:
        attempts += 1
        base = raw_sequences[int(rng.integers(0, raw_count))]
        aug = _augment_sequence(base, rng)
        h = _hash_sequence(aug, round_decimals)
        if h in existing_hashes:
            continue
        fname = os.path.join(out_dir, f"aug_{saved:04d}.npy")
        np.save(fname, aug)
        existing_hashes.add(h)
        saved += 1

    return {"raw": raw_count, "augmented": saved, "saved": saved, "needed": needed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a balanced augmented dataset.")
    parser.add_argument("--raw", default=DATA_DIR, help="Raw dataset root (default: config DATA_DIR).")
    parser.add_argument("--aug", default=None, help="Augmented output root (default: sibling 'augmented').")
    parser.add_argument("--target", type=int, default=150, help="Total per-class target (raw + augmented).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round", type=int, default=4, help="Round decimals for duplicate detection.")
    parser.add_argument("--reset", action="store_true", help="Delete existing augmented folder first.")
    args = parser.parse_args()

    raw_root = args.raw
    aug_root = args.aug or _augmented_root(raw_root)

    if args.reset and os.path.isdir(aug_root):
        shutil.rmtree(aug_root)

    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("ZERO CONTROLLER - BALANCED AUGMENTATION")
    print(f"Raw root      : {raw_root}")
    print(f"Augmented root: {aug_root}")
    print(f"Target total  : {args.target} per class")
    print(f"Duplicate hash: rounded({args.round})")
    print("=" * 72)

    summary = {}
    for action in ACTIONS:
        raw_sequences = _load_raw_sequences(raw_root, action)
        out_dir = os.path.join(aug_root, action)
        stats = _generate_for_action(
            action,
            raw_sequences,
            out_dir,
            args.target,
            rng,
            args.round,
        )
        summary[action] = stats
        print(
            f"{action:<12} raw={stats['raw']:3d}  "
            f"augmented={stats['augmented']:3d}  "
            f"target={args.target:3d}"
        )

    print("\n[DONE] Augmented dataset ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
