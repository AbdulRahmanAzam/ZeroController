"""Mirror right-side action samples to generate matching left-side data.

For each action pair in _MIRROR_PAIRS, every .npy in data/raw/right_*/
is flipped (x → 1-x) and left↔right landmark indices are swapped, then
saved to the corresponding data/raw/left_*/ directory.

Run:  python augment_data.py
"""

import os
import sys

import numpy as np

from config import DATA_DIR

# MediaPipe 33-landmark left↔right swap pairs
_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6),    # eyes
    (7, 8),                      # ears
    (9, 10),                     # mouth corners
    (11, 12),                    # shoulders
    (13, 14),                    # elbows
    (15, 16),                    # wrists
    (17, 18),                    # pinkies
    (19, 20),                    # index fingers
    (21, 22),                    # thumbs
    (23, 24),                    # hips
    (25, 26),                    # knees
    (27, 28),                    # ankles
    (29, 30),                    # heels
    (31, 32),                    # foot indices
]


def mirror_sequence(seq):
    """Mirror a (T, 33, 4) landmark sequence: flip x and swap sides."""
    mirrored = seq.copy()

    # Flip x-coordinate (column 0)
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]

    # Swap left↔right landmarks
    for a, b in _SWAP_PAIRS:
        mirrored[:, a, :], mirrored[:, b, :] = (
            mirrored[:, b, :].copy(),
            mirrored[:, a, :].copy(),
        )

    return mirrored


# Action pairs to mirror: (source, destination)
_MIRROR_PAIRS = [
    ("right_punch", "left_punch"),
    ("right_kick",  "left_kick"),
]


def _augment_pair(src_action, dst_action):
    """Mirror all .npy samples from src_action into dst_action."""
    src_dir = os.path.join(DATA_DIR, src_action)
    dst_dir = os.path.join(DATA_DIR, dst_action)

    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_action}: directory not found.")
        return 0

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npy"))
    if not files:
        print(f"[SKIP] {src_action}: no .npy files found.")
        return 0

    os.makedirs(dst_dir, exist_ok=True)
    created = 0
    for fname in files:
        dst_path = os.path.join(dst_dir, f"mirror_{fname}")
        if os.path.exists(dst_path):
            continue
        seq = np.load(os.path.join(src_dir, fname))
        np.save(dst_path, mirror_sequence(seq))
        created += 1

    print(f"[AUG] {src_action} → {dst_action}:  {created} new  "
          f"/ {len(files)} source samples")
    return created


def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - DATA AUGMENTATION (Mirror L↔R)")
    print("=" * 60)
    total = 0
    for src, dst in _MIRROR_PAIRS:
        total += _augment_pair(src, dst)
    print(f"[DONE] {total} mirrored sample(s) created.")


if __name__ == "__main__":
    main()
