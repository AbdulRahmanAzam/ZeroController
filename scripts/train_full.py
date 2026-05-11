"""Train a single model on 100% of the raw data (no train/test split).

Use after train_all_split.py identifies the best model. Saves the checkpoint
to data/models/action_classifier_<model>_final.pth (path printable for use as
MODEL_SAVE_PATH in config.py).
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    ACTIONS,
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
)
from train_all_split import (
    MODEL_TYPES,
    SequenceDataset,
    _balance_min,
    _build_model,
    _load_raw_sequences,
    _save_checkpoint,
)


def train_full(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    augment: bool,
    rng: np.random.Generator,
    save_dir: str,
) -> str:
    from config import STGCN_USE_VELOCITY

    input_size = (
        6 if model_type == "stgcn" and STGCN_USE_VELOCITY
        else (3 if model_type == "stgcn" else 33 * 4)
    )
    model = _build_model(model_type, input_size, len(ACTIONS)).to(device)

    ds = SequenceDataset(X, y, model_type, augment, rng)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
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
            print(
                f"  Epoch {epoch:3d}/{epochs}  |  loss: {total_loss / total:.4f}  "
                f"train_acc: {correct / total:.2%}"
            )
        scheduler.step()

    save_path = os.path.join(save_dir, f"action_classifier_{model_type}_final.pth")
    _save_checkpoint(save_path, model, model_type, input_size)
    return save_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one model on 100% raw data.")
    parser.add_argument("--model", required=True, choices=MODEL_TYPES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--balance", choices=("min", "none"), default="min")
    parser.add_argument("--save-dir", default=os.path.join(os.path.dirname(DATA_DIR), "models"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 72)
    print("ZERO CONTROLLER - FULL-DATA TRAIN")
    print(f"Model      : {args.model}")
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

    print(f"[DATA] Samples used: {len(y)} (100% — no test split)")

    save_path = train_full(
        args.model, X, y, device,
        args.epochs, args.batch_size, args.augment, rng, args.save_dir,
    )
    print(f"\n[DONE] Saved final checkpoint: {save_path}")
    print("Update config.py MODEL_SAVE_PATH to point to this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
