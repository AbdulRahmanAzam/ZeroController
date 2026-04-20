"""Mirror right_punch samples to create left_punch data.

For each .npy in data/raw/right_punch/:
  1. Flip x-coordinates  (x → 1.0 - x)
  2. Swap left/right landmark indices
  3. Save to data/raw/left_punch/

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


def main():
    src_dir = os.path.join(DATA_DIR, "right_punch")
    dst_dir = os.path.join(DATA_DIR, "left_punch")

    if not os.path.isdir(src_dir):
        print(f"[ERROR] Source directory not found: {src_dir}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npy"))
    if not files:
        print(f"[ERROR] No .npy files in {src_dir}")
        sys.exit(1)

    os.makedirs(dst_dir, exist_ok=True)
    existing = sum(1 for f in os.listdir(dst_dir) if f.endswith(".npy"))

    created = 0
    for fname in files:
        dst_name = f"mirror_{fname}"
        dst_path = os.path.join(dst_dir, dst_name)
        if os.path.exists(dst_path):
            continue

        seq = np.load(os.path.join(src_dir, fname))
        mirrored = mirror_sequence(seq)
        np.save(dst_path, mirrored)
        created += 1

    total = existing + created
    print(f"[DONE] {created} new left_punch samples created ({total} total in {dst_dir})")


if __name__ == "__main__":
    main()
