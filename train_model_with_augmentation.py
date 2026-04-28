"""Example: Training with augmented data.

This example shows how to load both original and augmented samples,
then train your ST-GCN model on the expanded dataset.

Run: python train_model_with_augmentation.py
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import ACTIONS, SEQUENCE_LENGTH, DATA_DIR


def load_dataset_with_augmentation(data_dir='data/raw', include_augmented=True, test_size=0.2):
    """Load original + augmented samples for training.
    
    Args:
        data_dir: Path to data directory (contains 'raw' and 'augmented' subdirs)
        include_augmented: Whether to include augmented samples
        test_size: Fraction of ORIGINAL data to reserve for testing
                   (ensures test set is on real, not augmented data)
    
    Returns:
        X_train, X_test, y_train, y_test: numpy arrays
    """
    raw_dir = Path(data_dir)
    aug_dir = raw_dir / 'augmented'
    
    sequences_original = []
    labels_original = []
    sequences_augmented = []
    labels_augmented = []
    
    print(f"Loading from {raw_dir}...")
    
    # Load original samples
    for action_idx, action in enumerate(ACTIONS):
        action_dir = raw_dir / action
        count = 0
        
        if action_dir.exists():
            for npy_file in sorted(action_dir.glob('*.npy')):
                try:
                    seq = np.load(npy_file)
                    if seq.shape == (SEQUENCE_LENGTH, 33, 4):
                        sequences_original.append(seq)
                        labels_original.append(action_idx)
                        count += 1
                except Exception as e:
                    print(f"  [WARN] Failed to load {npy_file}: {e}")
        
        print(f"  {action}: {count} original samples")
    
    # Load augmented samples (optional)
    if include_augmented and aug_dir.exists():
        print(f"\nLoading augmented from {aug_dir}...")
        for action_idx, action in enumerate(ACTIONS):
            aug_action_dir = aug_dir / action
            count = 0
            
            if aug_action_dir.exists():
                for npy_file in sorted(aug_action_dir.glob('*.npy')):
                    try:
                        seq = np.load(npy_file)
                        if seq.shape == (SEQUENCE_LENGTH, 33, 4):
                            sequences_augmented.append(seq)
                            labels_augmented.append(action_idx)
                            count += 1
                    except Exception as e:
                        print(f"  [WARN] Failed to load {npy_file}: {e}")
            
            if count > 0:
                print(f"  {action}: {count} augmented samples")
    
    X_original = np.array(sequences_original)
    y_original = np.array(labels_original)
    
    print(f"\nTotal original samples: {len(X_original)}")
    if include_augmented:
        print(f"Total augmented samples: {len(sequences_augmented)}")
    
    # Split ONLY original data to ensure test set is on real data
    X_train_original, X_test, y_train_original, y_test = train_test_split(
        X_original, y_original, test_size=test_size, stratify=y_original, random_state=42
    )
    
    # Combine training original + augmented
    if include_augmented and sequences_augmented:
        X_augmented = np.array(sequences_augmented)
        y_augmented = np.array(labels_augmented)
        
        X_train = np.concatenate([X_train_original, X_augmented], axis=0)
        y_train = np.concatenate([y_train_original, y_augmented], axis=0)
    else:
        X_train = X_train_original
        y_train = y_train_original
    
    print(f"\nTraining set:  {len(X_train)} samples")
    print(f"Test set:      {len(X_test)} samples (original data only)")
    print(f"Multiplier:    {len(X_train) / len(X_train_original):.1f}x")
    
    return X_train, X_test, y_train, y_test


def main():
    """Example training workflow."""
    print("=" * 70)
    print("  TRAINING WITH AUGMENTED DATA")
    print("=" * 70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_dataset_with_augmentation(
        include_augmented=True,
        test_size=0.2
    )
    
    # Optional: Standardize data (for certain models)
    print("\nStandardizing data...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train = X_train_flat.reshape(X_train.shape[0], SEQUENCE_LENGTH, 33, 4)
    X_test = X_test_flat.reshape(X_test.shape[0], SEQUENCE_LENGTH, 33, 4)
    
    print("✓ Data loaded and standardized")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    
    # ── Now use X_train, X_test, y_train, y_test with your model ──
    print("\n" + "=" * 70)
    print("  Ready for training!")
    print("=" * 70)
    print("\nExample (PyTorch):")
    print("""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from train_model import build_model_from_ckpt
    
    # Create PyTorch datasets
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    
    # Load model, train, evaluate...
    model = build_model_from_ckpt(...)
    # ... training loop ...
    """)
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = main()
