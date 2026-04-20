"""Train a lightweight LSTM classifier on collected punch data.

Loads all .npy sequences from data/raw/<action>/, trains a small LSTM,
prints per-epoch loss/accuracy, and saves the model to models/punch_classifier.pth.

Run:  python train_model.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ACTIONS,
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_LSTM_LAYERS,
    SEQUENCE_LENGTH,
)


# ── Model ────────────────────────────────────────────────────────────────────

class PunchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (hn, _) = self.lstm(x)       # hn: (num_layers, batch, hidden)
        out = self.fc(hn[-1])           # last layer hidden state
        return out


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset():
    """Load all .npy files, return (X, y) tensors."""
    X_list, y_list = [], []

    for label_idx, action in enumerate(ACTIONS):
        action_dir = os.path.join(DATA_DIR, action)
        if not os.path.isdir(action_dir):
            print(f"[WARN] Missing directory: {action_dir}")
            continue

        files = sorted(f for f in os.listdir(action_dir) if f.endswith(".npy"))
        for fname in files:
            seq = np.load(os.path.join(action_dir, fname))  # (30, 33, 4)
            if seq.shape != (SEQUENCE_LENGTH, 33, 4):
                print(f"[SKIP] {fname} has unexpected shape {seq.shape}")
                continue
            # Flatten landmarks per frame: (30, 132)
            seq_flat = seq.reshape(SEQUENCE_LENGTH, -1)
            X_list.append(seq_flat)
            y_list.append(label_idx)

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 30, 132)
    y = np.array(y_list, dtype=np.int64)              # (N,)
    return X, y


def train_val_split(X, y, val_ratio=0.25, seed=42):
    """Stratified-ish split for small datasets."""
    random.seed(seed)
    indices = list(range(len(y)))
    random.shuffle(indices)

    val_count = max(1, int(len(y) * val_ratio))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - TRAIN PUNCH CLASSIFIER")
    print("=" * 60)

    X, y = load_dataset()
    print(f"[DATA] Total samples: {len(y)}  |  Classes: {dict(zip(ACTIONS, np.bincount(y)))}")

    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print(f"[DATA] Train: {len(y_train)}  |  Val: {len(y_val)}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    input_size = 33 * 4  # 132
    model = PunchLSTM(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS, len(ACTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
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

        train_loss = total_loss / total
        train_acc = correct / total

        # ── Validate ──
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
            print(f"  Epoch {epoch:3d}/{EPOCHS}  |  loss: {train_loss:.4f}  "
                  f"train_acc: {train_acc:.2%}  val_acc: {val_acc:.2%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "actions": ACTIONS,
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LSTM_LAYERS,
                "sequence_length": SEQUENCE_LENGTH,
            }, MODEL_SAVE_PATH)

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.2%}")
    print(f"[SAVED] {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
