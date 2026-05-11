"""Train and evaluate all existing models with an in-memory train/test split."""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    ACTIONS,
    BATCH_SIZE,
    DATA_DIR,
    DROPOUT,
    EPOCHS,
    GRU_HIDDEN_SIZE,
    GRU_NUM_LAYERS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LSTM_LAYERS,
    POSECNN_CHANNELS,
    POSECNN_KERNEL_SIZE,
    POSECNN_NUM_LAYERS,
    PREPROCESS_HIP_CENTER,
    PREPROCESS_SCALE_NORM,
    PREPROCESS_USE_VISIBILITY,
    SEQUENCE_LENGTH,
    STGCN_CHANNELS,
    STGCN_TEMPORAL_K,
    STGCN_USE_VELOCITY,
    TCN_KERNEL_SIZE,
    TCN_NUM_CHANNELS,
    TCN_NUM_LAYERS,
)
from train_model import (
    ActionGRU,
    ActionLSTM,
    ActionPoseConv1D,
    ActionSTGCN,
    ActionTCN,
    preprocess_pose_sequence,
)


MODEL_TYPES = ["lstm", "gru", "tcn", "stgcn", "poseconv1d"]


def _iter_raw_files(root: str, action: str) -> List[str]:
    action_dir = os.path.join(root, action)
    if not os.path.isdir(action_dir):
        return []
    return [
        os.path.join(action_dir, name)
        for name in sorted(os.listdir(action_dir))
        if name.endswith(".npy")
    ]


def _load_raw_sequences(root: str) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    labels = []
    for label_idx, action in enumerate(ACTIONS):
        for path in _iter_raw_files(root, action):
            seq = np.load(path)
            if seq.shape != (SEQUENCE_LENGTH, 33, 4):
                continue
            sequences.append(seq.astype(np.float32))
            labels.append(label_idx)
    if not sequences:
        return np.empty((0, SEQUENCE_LENGTH, 33, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(sequences, axis=0), np.array(labels, dtype=np.int64)


def _balance_min(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    indices = []
    min_count = None
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if min_count is None or len(cls_idx) < min_count:
            min_count = len(cls_idx)
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:min_count].tolist())
    rng.shuffle(indices)
    return X[indices], y[indices]


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_ratio))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


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


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        model_type: str,
        augment: bool,
        rng: np.random.Generator,
    ):
        self.sequences = sequences
        self.labels = labels
        self.model_type = model_type
        self.augment = augment
        self.rng = rng

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        if self.augment:
            seq = _augment_sequence(seq, self.rng)
        if self.model_type == "stgcn":
            sample = preprocess_pose_sequence(
                seq,
                use_velocity=STGCN_USE_VELOCITY,
                hip_center=PREPROCESS_HIP_CENTER,
                scale_norm=PREPROCESS_SCALE_NORM,
                use_visibility=PREPROCESS_USE_VISIBILITY,
            )
        else:
            sample = seq.reshape(SEQUENCE_LENGTH, -1).astype(np.float32)
        return torch.from_numpy(sample), torch.tensor(self.labels[idx], dtype=torch.int64)


def _build_model(model_type: str, input_size: int, num_classes: int) -> nn.Module:
    if model_type == "stgcn":
        in_ch = 6 if STGCN_USE_VELOCITY else 3
        return ActionSTGCN(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=STGCN_CHANNELS,
            temporal_k=STGCN_TEMPORAL_K,
            dropout=DROPOUT,
        )
    if model_type == "tcn":
        return ActionTCN(
            input_size,
            TCN_NUM_CHANNELS,
            TCN_KERNEL_SIZE,
            TCN_NUM_LAYERS,
            num_classes,
            DROPOUT,
        )
    if model_type == "gru":
        return ActionGRU(input_size, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, num_classes, DROPOUT)
    if model_type == "poseconv1d":
        return ActionPoseConv1D(
            input_size,
            POSECNN_CHANNELS,
            POSECNN_KERNEL_SIZE,
            POSECNN_NUM_LAYERS,
            num_classes,
            DROPOUT,
        )
    return ActionLSTM(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS, num_classes, DROPOUT)


def _save_checkpoint(path: str, model: nn.Module, model_type: str, input_size: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "actions": ACTIONS,
        "input_size": input_size,
        "model_type": model_type,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LSTM_LAYERS,
        "tcn_channels": TCN_NUM_CHANNELS,
        "tcn_kernel_size": TCN_KERNEL_SIZE,
        "tcn_num_layers": TCN_NUM_LAYERS,
        "gru_hidden_size": GRU_HIDDEN_SIZE,
        "gru_num_layers": GRU_NUM_LAYERS,
        "posecnn_channels": POSECNN_CHANNELS,
        "posecnn_kernel_size": POSECNN_KERNEL_SIZE,
        "posecnn_num_layers": POSECNN_NUM_LAYERS,
        "stgcn_in_channels": 6 if STGCN_USE_VELOCITY else 3,
        "stgcn_channels": list(STGCN_CHANNELS),
        "stgcn_temporal_k": STGCN_TEMPORAL_K,
        "stgcn_use_velocity": STGCN_USE_VELOCITY,
        "preprocess_hip_center": PREPROCESS_HIP_CENTER,
        "preprocess_scale_norm": PREPROCESS_SCALE_NORM,
        "preprocess_use_visibility": PREPROCESS_USE_VISIBILITY,
        "dropout": DROPOUT,
        "sequence_length": SEQUENCE_LENGTH,
    }, path)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total else 0.0


def train_one(
    model_type: str,
    train_seq: np.ndarray,
    train_labels: np.ndarray,
    test_seq: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    augment: bool,
    rng: np.random.Generator,
    save_dir: str,
) -> Dict[str, float]:
    input_size = 6 if model_type == "stgcn" and STGCN_USE_VELOCITY else (3 if model_type == "stgcn" else 33 * 4)
    model = _build_model(model_type, input_size, len(ACTIONS)).to(device)

    train_ds = SequenceDataset(train_seq, train_labels, model_type, augment, rng)
    test_ds = SequenceDataset(test_seq, test_labels, model_type, False, rng)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        if epoch % 10 == 0 or epoch == 1:
            train_acc = correct / total if total else 0.0
            train_loss = total_loss / total if total else 0.0
            print(
                f"  Epoch {epoch:3d}/{epochs}  |  loss: {train_loss:.4f}  "
                f"train_acc: {train_acc:.2%}"
            )
        scheduler.step()

    test_acc = _evaluate(model, test_loader, device)
    save_path = os.path.join(save_dir, f"action_classifier_{model_type}_split.pth")
    _save_checkpoint(save_path, model, model_type, input_size)

    return {"test_acc": test_acc, "save_path": save_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate models with a train/test split.")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="*", default=MODEL_TYPES)
    parser.add_argument("--augment", action="store_true", help="Apply mild augmentation to training only.")
    parser.add_argument("--balance", choices=("min", "none"), default="min")
    parser.add_argument("--save-dir", default=os.path.join(os.path.dirname(DATA_DIR), "models"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 72)
    print("ZERO CONTROLLER - TRAIN/TEST SPLIT")
    print(f"Test ratio : {args.test_ratio:.2f}")
    print(f"Balance    : {args.balance}")
    print(f"Augment    : {args.augment}")
    print(f"Device     : {device}")
    print("=" * 72)

    X, y = _load_raw_sequences(DATA_DIR)
    if len(y) == 0:
        print("[ERROR] No raw samples found.")
        return 1

    if args.balance == "min":
        X, y = _balance_min(X, y, rng)

    train_seq, train_labels, test_seq, test_labels = _stratified_split(
        X, y, args.test_ratio, rng
    )

    print(f"[DATA] Raw samples used: {len(y)}  |  Train: {len(train_labels)}  Test: {len(test_labels)}")

    results = []
    for model_type in args.models:
        if model_type not in MODEL_TYPES:
            print(f"[SKIP] Unknown model: {model_type}")
            continue
        print("\n" + "-" * 60)
        print(f"Model: {model_type}")
        stats = train_one(
            model_type,
            train_seq,
            train_labels,
            test_seq,
            test_labels,
            device,
            args.epochs,
            args.batch_size,
            args.augment,
            rng,
            args.save_dir,
        )
        print(f"[RESULT] test_acc: {stats['test_acc']:.2%}  saved: {stats['save_path']}")
        results.append((model_type, stats["test_acc"], stats["save_path"]))

    print("\n" + "=" * 72)
    print("SUMMARY")
    for model_type, acc, path in results:
        print(f"  {model_type:<10} test_acc={acc:.2%}  {path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
