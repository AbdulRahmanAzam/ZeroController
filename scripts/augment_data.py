"""Comprehensive data augmentation for pose action sequences.

Techniques:
1. Mirror left/right (existing functionality)
2. Spatial noise (±2% Gaussian on coordinates)
3. Skeleton scaling (±15% uniform scale)
4. Temporal jittering (speed up/slow down)
5. Visibility dropout (simulate occlusion)
6. Rotation jitter (±15° around z-axis)
7. Time stretching (compress/expand motion)
8. Mixup with same-action samples

Generates 4-5x more training samples from collected data.

Run:  python augment_data.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

from config import DATA_DIR, ACTIONS, SEQUENCE_LENGTH

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

    print(f"[AUG] {src_action} -> {dst_action}:  {created} new  "
          f"/ {len(files)} source samples")
    return created


def main():
    print("=" * 70)
    print("  ZERO CONTROLLER - DATA AUGMENTATION")
    print("=" * 70)
    
    # Phase 1: Mirror left/right actions
    print("\n[Phase 1] Mirroring left/right actions...")
    total_mirrored = 0
    for src, dst in _MIRROR_PAIRS:
        total_mirrored += _augment_pair(src, dst)
    
    # Phase 2: Generate diverse augmentations
    print("\n[Phase 2] Generating diverse augmentations...")
    total_original = 0
    total_generated = 0
    
    for action in ACTIONS:
        originals = load_action_samples(action)
        if not originals:
            print(f"  [{action}] No samples found.")
            continue
        
        print(f"  [{action}] {len(originals)} originals → ", end='', flush=True)
        total_original += len(originals)
        
        aug_per_sample = 4  # 4x multiplier
        generated = 0
        
        for sample_idx, original in enumerate(originals):
            for aug_idx in range(aug_per_sample):
                # First augmentation per sample is mixup
                if aug_idx == 0 and len(originals) > 1:
                    other = originals[np.random.randint(len(originals))]
                    augmented = _mixup_with_same_action(original, other)
                else:
                    augmented = augment_sequence(original, action)
                
                global_idx = sample_idx * aug_per_sample + aug_idx
                save_augmented_sequence(augmented, action, global_idx)
                generated += 1
        
        total_generated += generated
        print(f"generated {generated} → Total: {len(originals) + generated}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Mirrored samples:   {total_mirrored}")
    print(f"Original samples:   {total_original}")
    print(f"Generated samples:  {total_generated}")
    print(f"Total samples:      {total_original + total_generated + total_mirrored}")
    if total_original > 0:
        multiplier = (total_original + total_generated + total_mirrored) / total_original
        print(f"Augmentation rate:  {multiplier:.1f}x")
    print(f"\nAugmented data saved to: {Path(DATA_DIR) / 'augmented'}/")
    print("=" * 70)


def _add_spatial_noise(seq, std=0.02):
    """Add Gaussian noise to coordinates (x, y, z)."""
    noisy = seq.copy()
    noise = np.random.normal(0, std, size=seq[:, :, :3].shape)
    noisy[:, :, :3] = np.clip(seq[:, :, :3] + noise, 0, 1)
    return noisy


def _scale_skeleton(seq, scale_range=(0.85, 1.15)):
    """Scale all joints uniformly."""
    scaled = seq.copy()
    scale = np.random.uniform(*scale_range)
    scaled[:, :, :3] = np.clip(seq[:, :, :3] * scale, 0, 1)
    return scaled


def _temporal_jitter(seq, jitter_sigma=0.05):
    """Resample sequence with time jitter."""
    T = seq.shape[0]
    time_offsets = np.random.normal(0, jitter_sigma * T, size=T)
    time_offsets = np.cumsum(time_offsets)
    time_offsets = np.clip(time_offsets, 0, T - 1)
    
    jittered = np.zeros_like(seq)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            jittered[:, j, c] = np.interp(np.arange(T), time_offsets, seq[:, j, c])
    
    return jittered


def _visibility_dropout(seq, dropout_prob=0.1):
    """Randomly occlude joints (set visibility to 0)."""
    dropped = seq.copy()
    mask = np.random.rand(seq.shape[1]) < dropout_prob
    dropped[:, mask, 3] = 0
    return dropped


def _rotation_jitter(seq, max_angle_deg=15):
    """Rotate skeleton around z-axis."""
    angle = np.random.uniform(-max_angle_deg, max_angle_deg)
    angle_rad = np.radians(angle)
    
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated = seq.copy()
    x, y = seq[:, :, 0], seq[:, :, 1]
    
    rotated[:, :, 0] = np.clip(x * cos_a - y * sin_a, 0, 1)
    rotated[:, :, 1] = np.clip(x * sin_a + y * cos_a, 0, 1)
    
    return rotated


def _time_stretch(seq, stretch_factor=None):
    """Stretch/compress motion in time."""
    if stretch_factor is None:
        stretch_factor = np.random.uniform(0.8, 1.2)
    
    T = seq.shape[0]
    old_indices = np.arange(T) / stretch_factor
    old_indices = np.clip(old_indices, 0, T - 1)
    
    stretched = np.zeros_like(seq)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            stretched[:, j, c] = np.interp(np.arange(T), old_indices, seq[:, j, c])
    
    return stretched


def _mixup_with_same_action(seq1, seq2, alpha=None):
    """Blend two sequences from same action class."""
    if alpha is None:
        alpha = np.random.uniform(0.3, 0.7)
    
    mixed = alpha * seq1 + (1 - alpha) * seq2
    return mixed.astype(np.float32)


def augment_sequence(seq, action):
    """Apply a random augmentation."""
    techniques = [
        _add_spatial_noise,
        _scale_skeleton,
        _temporal_jitter,
        _rotation_jitter,
        _visibility_dropout,
        _time_stretch,
    ]
    
    technique = np.random.choice(techniques)
    try:
        return technique(seq).astype(np.float32)
    except Exception as e:
        print(f"[WARN] Augmentation failed: {e}, returning original")
        return seq


def load_action_samples(action_name):
    """Load all .npy files for an action."""
    action_dir = Path(DATA_DIR) / action_name
    if not action_dir.exists():
        return []
    
    sequences = []
    for npy_file in sorted(action_dir.glob('*.npy')):
        try:
            seq = np.load(npy_file)
            if seq.shape == (SEQUENCE_LENGTH, 33, 4):
                sequences.append(seq)
        except Exception as e:
            print(f"[WARN] Failed to load {npy_file}: {e}")
    
    return sequences


def save_augmented_sequence(seq, action_name, index):
    """Save augmented sequence."""
    augmented_dir = Path(DATA_DIR) / 'augmented'
    action_dir = augmented_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)
    
    filename = action_dir / f'aug_{index:04d}.npy'
    np.save(filename, seq)


if __name__ == "__main__":
    main()
