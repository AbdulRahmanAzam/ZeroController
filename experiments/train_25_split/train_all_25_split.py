"""Train all five models with a 75/25 raw-only train/val split.

This script saves new checkpoints into experiments/train_25_split/models
so the main models/ folder remains untouched.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    train_val_split,
)

MODEL_TYPES = ["stgcn", "tcn", "lstm", "gru", "poseconv1d"]


def save_config_snapshot(path, args):
    payload = {
        "experiment": "train_25_split",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_types": MODEL_TYPES,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "seed": 42,
        "data_dir": DATA_DIR,
        "sequence_length": SEQUENCE_LENGTH,
        "actions": ACTIONS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "lstm": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LSTM_LAYERS,
        },
        "gru": {
            "hidden_size": GRU_HIDDEN_SIZE,
            "num_layers": GRU_NUM_LAYERS,
        },
        "poseconv1d": {
            "channels": POSECNN_CHANNELS,
            "kernel_size": POSECNN_KERNEL_SIZE,
            "num_layers": POSECNN_NUM_LAYERS,
        },
        "tcn": {
            "channels": TCN_NUM_CHANNELS,
            "kernel_size": TCN_KERNEL_SIZE,
            "num_layers": TCN_NUM_LAYERS,
        },
        "stgcn": {
            "channels": list(STGCN_CHANNELS),
            "temporal_k": STGCN_TEMPORAL_K,
            "use_velocity": STGCN_USE_VELOCITY,
        },
        "preprocess": {
            "hip_center": PREPROCESS_HIP_CENTER,
            "scale_norm": PREPROCESS_SCALE_NORM,
            "use_visibility": PREPROCESS_USE_VISIBILITY,
        },
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)


def _empty_dataset(model_type):
    if model_type == "stgcn":
        c = 6 if STGCN_USE_VELOCITY else 3
        empty_x = np.empty((0, c, SEQUENCE_LENGTH, 33), dtype=np.float32)
    else:
        empty_x = np.empty((0, SEQUENCE_LENGTH, 33 * 4), dtype=np.float32)
    return empty_x, np.empty(0, dtype=np.int64)


def load_raw_dataset(model_type):
    X_list, y_list = [], []

    for label_idx, action in enumerate(ACTIONS):
        action_dir = os.path.join(DATA_DIR, action)
        if not os.path.isdir(action_dir):
            print(f"[WARN] Missing directory: {action_dir}")
            continue

        for fname in sorted(os.listdir(action_dir)):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(action_dir, fname)
            seq = np.load(fpath)
            if seq.shape != (SEQUENCE_LENGTH, 33, 4):
                print(f"[SKIP] {fname} has unexpected shape {seq.shape}")
                continue

            if model_type == "stgcn":
                sample = preprocess_pose_sequence(
                    seq,
                    use_velocity=STGCN_USE_VELOCITY,
                    hip_center=PREPROCESS_HIP_CENTER,
                    scale_norm=PREPROCESS_SCALE_NORM,
                    use_visibility=PREPROCESS_USE_VISIBILITY,
                )
            else:
                sample = seq.reshape(SEQUENCE_LENGTH, -1).astype(np.float32)

            X_list.append(sample)
            y_list.append(label_idx)

    if not X_list:
        return _empty_dataset(model_type)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def build_model(model_type, input_size, num_classes, device):
    if model_type == "stgcn":
        in_ch = 6 if STGCN_USE_VELOCITY else 3
        model = ActionSTGCN(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=STGCN_CHANNELS,
            temporal_k=STGCN_TEMPORAL_K,
            dropout=DROPOUT,
        )
    elif model_type == "tcn":
        model = ActionTCN(
            input_size,
            TCN_NUM_CHANNELS,
            TCN_KERNEL_SIZE,
            TCN_NUM_LAYERS,
            num_classes,
            DROPOUT,
        )
    elif model_type == "gru":
        model = ActionGRU(input_size, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, num_classes, DROPOUT)
    elif model_type == "poseconv1d":
        model = ActionPoseConv1D(
            input_size,
            POSECNN_CHANNELS,
            POSECNN_KERNEL_SIZE,
            POSECNN_NUM_LAYERS,
            num_classes,
            DROPOUT,
        )
    else:
        model = ActionLSTM(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS, num_classes, DROPOUT)

    return model.to(device)


def benchmark_inference(model, input_size, model_type, device, runs=200):
    model.eval()
    cpu = torch.device("cpu")
    model_cpu = model.to(cpu)

    if model_type == "stgcn":
        x = torch.randn(1, input_size, SEQUENCE_LENGTH, 33)
    else:
        x = torch.randn(1, SEQUENCE_LENGTH, input_size)

    with torch.no_grad():
        for _ in range(20):
            model_cpu(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model_cpu(x)
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / runs) * 1000.0
    print(f"[BENCH] CPU inference: {avg_ms:.2f} ms/sample over {runs} runs")
    model.to(device)
    return avg_ms


def train_one(model_type, save_path, val_ratio, epochs, bench):
    print("=" * 60)
    print("ZERO CONTROLLER - TRAIN ACTION CLASSIFIER")
    print(f"Model type: {model_type}")
    print(f"Save path : {save_path}")
    print("=" * 60)

    X, y = load_raw_dataset(model_type)
    class_counts = dict(zip(ACTIONS, np.bincount(y, minlength=len(ACTIONS))))
    print(f"[DATA] Raw samples: {len(y)}  |  Classes: {class_counts}")

    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=val_ratio, seed=42)
    print(f"[DATA] Train: {len(y_train)}  |  Val: {len(y_val)}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    if model_type == "stgcn":
        input_size = 6 if STGCN_USE_VELOCITY else 3
    else:
        input_size = 33 * 4

    model = build_model(model_type, input_size, len(ACTIONS), device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        train_loss = total_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                v_correct += (preds == yb).sum().item()
                v_total += len(yb)

        val_acc = v_correct / v_total if v_total > 0 else 0.0

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}  |  loss: {train_loss:.4f}  "
                f"train_acc: {train_acc:.2%}  val_acc: {val_acc:.2%}"
            )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
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
                },
                save_path,
            )

        scheduler.step()

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.2%}")
    print(f"[SAVED] {save_path}")

    if bench:
        benchmark_inference(model, input_size, model_type, device)


def _parse_args():
    p = argparse.ArgumentParser(description="Train all models with a 25% val split.")
    p.add_argument("--val-ratio", type=float, default=0.25, help="Validation split ratio.")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs per model.")
    p.add_argument("--bench", action="store_true", help="Measure CPU latency per model.")
    return p.parse_args()


def main():
    args = _parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    config_path = os.path.join(base_dir, "config_snapshot.json")

    save_config_snapshot(config_path, args)
    print(f"[CONFIG] Wrote {config_path}")

    for model_type in MODEL_TYPES:
        save_path = os.path.join(models_dir, f"action_classifier_{model_type}.pth")
        train_one(model_type, save_path, args.val_ratio, args.epochs, args.bench)


if __name__ == "__main__":
    main()
