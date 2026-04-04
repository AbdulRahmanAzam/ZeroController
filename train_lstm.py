"""
LSTM Training Script
=====================
Trains the ThreeHeadLSTM model on recorded pose data.

Usage:
  python train_lstm.py

Expects .npz files in data/raw/ (created by data_collector.py).
Saves trained model to models/lstm_move_classifier.pt
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lstm_model import ThreeHeadLSTM, LABEL_NAMES
from config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_SEQ_LEN, LSTM_INPUT_SIZE, LSTM_MODEL_PATH,
)

# The 13 landmark indices used (must match data_collector.py)
LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Indices within the 13-landmark array for shoulders and hips
IDX_LEFT_SHOULDER = 1   # LANDMARK_INDICES[1] = 11
IDX_RIGHT_SHOULDER = 2  # LANDMARK_INDICES[2] = 12
IDX_LEFT_HIP = 7        # LANDMARK_INDICES[7] = 23
IDX_RIGHT_HIP = 8       # LANDMARK_INDICES[8] = 24


def normalize_landmarks(landmarks):
    """
    Normalize a sequence of landmarks for translation and scale invariance.

    Args:
        landmarks: numpy array (T, 13, 4) — x, y, z, visibility per landmark

    Returns:
        features: numpy array (T, 53) — normalized and flattened
    """
    T = landmarks.shape[0]
    features = np.zeros((T, 53), dtype=np.float32)

    for t in range(T):
        frame = landmarks[t].copy()  # (13, 4)

        # Hip midpoint for translation normalization
        hip_mid_x = (frame[IDX_LEFT_HIP, 0] + frame[IDX_RIGHT_HIP, 0]) / 2
        hip_mid_y = (frame[IDX_LEFT_HIP, 1] + frame[IDX_RIGHT_HIP, 1]) / 2
        hip_mid_z = (frame[IDX_LEFT_HIP, 2] + frame[IDX_RIGHT_HIP, 2]) / 2

        # Shoulder width for scale normalization
        sx = frame[IDX_LEFT_SHOULDER, 0] - frame[IDX_RIGHT_SHOULDER, 0]
        sy = frame[IDX_LEFT_SHOULDER, 1] - frame[IDX_RIGHT_SHOULDER, 1]
        body_scale = max(np.sqrt(sx**2 + sy**2), 0.01)

        # Normalize: subtract hip center, divide by shoulder width
        frame[:, 0] = (frame[:, 0] - hip_mid_x) / body_scale
        frame[:, 1] = (frame[:, 1] - hip_mid_y) / body_scale
        frame[:, 2] = (frame[:, 2] - hip_mid_z) / body_scale
        # visibility (column 3) stays unchanged

        # Flatten to 52 + 1 body_scale = 53
        features[t, :52] = frame.flatten()
        features[t, 52] = body_scale

    return features


def augment_noise(features, sigma=0.005):
    """Add Gaussian noise to coordinate features (not visibility or body_scale)."""
    noise = np.random.randn(*features.shape).astype(np.float32) * sigma
    # Zero out noise for visibility columns (indices 3, 7, 11, ..., 51) and body_scale (52)
    for i in range(13):
        noise[:, i * 4 + 3] = 0  # visibility
    noise[:, 52] = 0  # body_scale
    return features + noise


def augment_scale(features, low=0.9, high=1.1):
    """Random scale jitter on coordinate features."""
    scale = np.random.uniform(low, high)
    aug = features.copy()
    for i in range(13):
        aug[:, i * 4:i * 4 + 3] *= scale  # x, y, z
    # body_scale also scales
    aug[:, 52] *= scale
    return aug


def augment_horizontal_flip(features, labels):
    """Mirror x-coordinates and swap left/right labels."""
    aug_f = features.copy()
    aug_l = labels.copy()

    # Flip x coordinates (every 4th starting from 0)
    for i in range(13):
        aug_f[:, i * 4] *= -1  # negate x

    # Swap left/right landmark pairs in the feature vector:
    # Left shoulder (idx 1) <-> Right shoulder (idx 2)
    # Left elbow (3) <-> Right elbow (4)
    # Left wrist (5) <-> Right wrist (6)
    # Left hip (7) <-> Right hip (8)
    # Left knee (9) <-> Right knee (10)
    # Left ankle (11) <-> Right ankle (12)
    swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    for a, b in swap_pairs:
        a_start, b_start = a * 4, b * 4
        aug_f[:, a_start:a_start + 4], aug_f[:, b_start:b_start + 4] = \
            aug_f[:, b_start:b_start + 4].copy(), aug_f[:, a_start:a_start + 4].copy()

    # Swap labels: left_punch(1) <-> right_punch(2) in upper head
    upper = aug_l[:, 0].copy()
    upper[aug_l[:, 0] == 1] = 2
    upper[aug_l[:, 0] == 2] = 1
    aug_l[:, 0] = upper

    # left_kick(1) <-> right_kick(2) in lower head
    lower = aug_l[:, 1].copy()
    lower[aug_l[:, 1] == 1] = 2
    lower[aug_l[:, 1] == 2] = 1
    aug_l[:, 1] = lower

    # move_left(1) <-> move_right(2) in movement head
    movement = aug_l[:, 2].copy()
    movement[aug_l[:, 2] == 1] = 2
    movement[aug_l[:, 2] == 2] = 1
    aug_l[:, 2] = movement

    return aug_f, aug_l


def augment_visibility_dropout(features, prob=0.1):
    """Randomly zero out landmark visibility and coordinates to simulate occlusion."""
    aug = features.copy()
    for i in range(13):
        if np.random.rand() < prob:
            aug[:, i * 4:i * 4 + 4] = 0  # zero x,y,z,vis
    return aug


class MoveSequenceDataset(Dataset):
    """Dataset of sliding-window sequences for LSTM training."""

    def __init__(self, sessions, seq_len=20, augment=False):
        """
        Args:
            sessions: list of (features, labels) tuples per session.
                      features: (T, 53), labels: (T, 3)
            seq_len: sliding window length.
            augment: whether to apply data augmentation.
        """
        self.seq_len = seq_len
        self.augment = augment
        self.windows = []
        self.targets = []

        for features, labels in sessions:
            T = features.shape[0]
            if T < seq_len:
                continue

            for start in range(T - seq_len + 1):
                end = start + seq_len
                self.windows.append(features[start:end])
                # Label = last frame in the window
                self.targets.append(labels[end - 1])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        features = self.windows[idx].copy()
        labels = self.targets[idx].copy()

        if self.augment:
            # Apply augmentations with probability
            if np.random.rand() < 0.5:
                features = augment_noise(features)
            if np.random.rand() < 0.5:
                features = augment_scale(features)
            if np.random.rand() < 0.5:
                features, labels = augment_horizontal_flip(features, labels)
            if np.random.rand() < 0.3:
                features = augment_visibility_dropout(features)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels[0], dtype=torch.long),  # upper
            torch.tensor(labels[1], dtype=torch.long),  # lower
            torch.tensor(labels[2], dtype=torch.long),  # movement
        )


def load_sessions(data_dir="data/raw"):
    """Load all .npz sessions and return normalized features + labels."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        print(f"[ERROR] No .npz files found in {data_dir}/")
        return []

    sessions = []
    total_frames = 0

    for f in files:
        data = np.load(f)
        landmarks = data["landmarks"]  # (T, 13, 4)
        labels = data["labels"]        # (T, 3)

        features = normalize_landmarks(landmarks)  # (T, 53)
        sessions.append((features, labels))
        total_frames += features.shape[0]

        print(f"  Loaded {os.path.basename(f)}: {features.shape[0]} frames")

    print(f"\n  Total: {len(sessions)} sessions, {total_frames} frames")
    return sessions


def compute_class_weights(sessions):
    """Compute class weights for each head (inverse frequency, capped)."""
    upper_counts = np.zeros(4)
    lower_counts = np.zeros(5)
    movement_counts = np.zeros(3)

    for _, labels in sessions:
        for i in range(4):
            upper_counts[i] += np.sum(labels[:, 0] == i)
        for i in range(5):
            lower_counts[i] += np.sum(labels[:, 1] == i)
        for i in range(3):
            movement_counts[i] += np.sum(labels[:, 2] == i)

    def weights_from_counts(counts):
        total = counts.sum()
        if total == 0:
            return np.ones(len(counts))
        w = total / (len(counts) * counts + 1e-6)
        w = np.clip(w, 0.3, 3.0)  # cap weights to prevent instability
        return w

    return (
        weights_from_counts(upper_counts),
        weights_from_counts(lower_counts),
        weights_from_counts(movement_counts),
    )


def evaluate(model, dataloader, device, upper_weights, lower_weights, movement_weights):
    """Evaluate model on a dataset, return loss and per-head accuracy."""
    model.eval()

    upper_ce = nn.CrossEntropyLoss(weight=torch.tensor(upper_weights, dtype=torch.float32).to(device))
    lower_ce = nn.CrossEntropyLoss(weight=torch.tensor(lower_weights, dtype=torch.float32).to(device))
    movement_ce = nn.CrossEntropyLoss(weight=torch.tensor(movement_weights, dtype=torch.float32).to(device))

    total_loss = 0
    upper_correct = lower_correct = movement_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y_upper, y_lower, y_movement = [b.to(device) for b in batch]

            upper_logits, lower_logits, movement_logits, _ = model(x)

            loss = (upper_ce(upper_logits, y_upper) +
                    lower_ce(lower_logits, y_lower) +
                    movement_ce(movement_logits, y_movement))

            total_loss += loss.item() * x.size(0)
            upper_correct += (upper_logits.argmax(1) == y_upper).sum().item()
            lower_correct += (lower_logits.argmax(1) == y_lower).sum().item()
            movement_correct += (movement_logits.argmax(1) == y_movement).sum().item()
            total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    upper_acc = upper_correct / max(total, 1)
    lower_acc = lower_correct / max(total, 1)
    movement_acc = movement_correct / max(total, 1)

    return avg_loss, upper_acc, lower_acc, movement_acc


def train():
    print("=" * 60)
    print("  LSTM Training — ThreeHeadLSTM Move Classifier")
    print("=" * 60)
    print()

    # Load data
    print("[DATA] Loading sessions...")
    sessions = load_sessions()
    if not sessions:
        return

    # Train/val split by session
    np.random.seed(42)
    indices = np.random.permutation(len(sessions))
    split = max(1, int(len(sessions) * 0.8))
    train_sessions = [sessions[i] for i in indices[:split]]
    val_sessions = [sessions[i] for i in indices[split:]]

    # If only 1 session, use it for both train and val
    if not val_sessions:
        val_sessions = train_sessions
        print("  [WARN] Only 1 session — using same data for train & val")

    print(f"\n  Train sessions: {len(train_sessions)}, Val sessions: {len(val_sessions)}")

    # Create datasets
    train_dataset = MoveSequenceDataset(train_sessions, seq_len=LSTM_SEQ_LEN, augment=True)
    val_dataset = MoveSequenceDataset(val_sessions, seq_len=LSTM_SEQ_LEN, augment=False)

    print(f"  Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("[ERROR] Not enough data for training! Record more sessions.")
        return

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Compute class weights
    upper_w, lower_w, movement_w = compute_class_weights(train_sessions)
    print(f"\n  Class weights:")
    print(f"    Upper:    {upper_w}")
    print(f"    Lower:    {lower_w}")
    print(f"    Movement: {movement_w}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Model
    model = ThreeHeadLSTM(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Loss functions with class weights
    upper_ce = nn.CrossEntropyLoss(weight=torch.tensor(upper_w, dtype=torch.float32).to(device))
    lower_ce = nn.CrossEntropyLoss(weight=torch.tensor(lower_w, dtype=torch.float32).to(device))
    movement_ce = nn.CrossEntropyLoss(weight=torch.tensor(movement_w, dtype=torch.float32).to(device))

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 20
    max_epochs = 150
    history = []

    print(f"\n{'='*60}")
    print(f"  Training for up to {max_epochs} epochs (early stopping patience={max_patience})")
    print(f"{'='*60}\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0
        batches = 0

        for batch in train_loader:
            x, y_upper, y_lower, y_movement = [b.to(device) for b in batch]

            optimizer.zero_grad()
            upper_logits, lower_logits, movement_logits, _ = model(x)

            loss = (upper_ce(upper_logits, y_upper) +
                    lower_ce(lower_logits, y_lower) +
                    movement_ce(movement_logits, y_movement))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_train_loss = epoch_loss / max(batches, 1)

        # Validation
        val_loss, upper_acc, lower_acc, movement_acc = evaluate(
            model, val_loader, device, upper_w, lower_w, movement_w
        )
        avg_acc = (upper_acc + lower_acc + movement_acc) / 3

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "upper_acc": upper_acc,
            "lower_acc": lower_acc,
            "movement_acc": movement_acc,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Acc: U={upper_acc:.3f} L={lower_acc:.3f} M={movement_acc:.3f} Avg={avg_acc:.3f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": LSTM_INPUT_SIZE,
                "hidden_size": LSTM_HIDDEN_SIZE,
                "num_layers": LSTM_NUM_LAYERS,
                "dropout": LSTM_DROPOUT,
            }, LSTM_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n  [EARLY STOP] No improvement for {max_patience} epochs.")
                break

    # Save training history
    history_path = os.path.join(os.path.dirname(LSTM_MODEL_PATH), "training_log.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation with best model
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE — Final Evaluation")
    print(f"{'='*60}\n")

    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, upper_acc, lower_acc, movement_acc = evaluate(
        model, val_loader, device, upper_w, lower_w, movement_w
    )
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print(f"  Upper body acc:   {upper_acc:.3f}")
    print(f"  Lower body acc:   {lower_acc:.3f}")
    print(f"  Movement acc:     {movement_acc:.3f}")
    print(f"  Average acc:      {(upper_acc + lower_acc + movement_acc) / 3:.3f}")
    print(f"\n  Model saved:      {LSTM_MODEL_PATH}")
    print(f"  History saved:    {history_path}")
    print(f"  Parameters:       {param_count:,}")


if __name__ == "__main__":
    train()
