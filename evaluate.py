"""
Evaluate — Rule-Based vs LSTM Comparison
==========================================
Runs both detectors on test data and prints comparative metrics.

Usage:
  python evaluate.py
"""

import os
import glob
import numpy as np
import torch
from collections import defaultdict

from lstm_model import ThreeHeadLSTM, UPPER_CLASSES, LOWER_CLASSES, MOVEMENT_CLASSES
from train_lstm import normalize_landmarks, MoveSequenceDataset
from config import LSTM_MODEL_PATH, LSTM_SEQ_LEN


ALL_MOVES = [
    "idle", "left_punch", "right_punch", "block",
    "left_kick", "right_kick", "crouch", "jump",
    "move_left", "move_right",
]


def compute_metrics(y_true, y_pred, class_names):
    """Compute per-class precision, recall, F1 and macro averages."""
    n_classes = len(class_names)
    metrics = {}

    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[class_names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    # Macro averages
    prec_vals = [m["precision"] for m in metrics.values() if m["support"] > 0]
    rec_vals = [m["recall"] for m in metrics.values() if m["support"] > 0]
    f1_vals = [m["f1"] for m in metrics.values() if m["support"] > 0]

    metrics["_macro_avg"] = {
        "precision": np.mean(prec_vals) if prec_vals else 0,
        "recall": np.mean(rec_vals) if rec_vals else 0,
        "f1": np.mean(f1_vals) if f1_vals else 0,
    }

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)
    metrics["_accuracy"] = accuracy

    return metrics


def confusion_matrix(y_true, y_pred, n_classes):
    """Build confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def print_metrics(metrics, class_names, head_name):
    """Pretty-print per-class metrics."""
    print(f"\n  --- {head_name} ---")
    print(f"  {'Class':20s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    print(f"  {'-'*52}")

    for name in class_names:
        m = metrics.get(name, {})
        if m.get("support", 0) > 0:
            print(f"  {name:20s} {m['precision']:8.3f} {m['recall']:8.3f} "
                  f"{m['f1']:8.3f} {m['support']:8d}")

    macro = metrics.get("_macro_avg", {})
    print(f"  {'-'*52}")
    print(f"  {'Macro avg':20s} {macro.get('precision', 0):8.3f} "
          f"{macro.get('recall', 0):8.3f} {macro.get('f1', 0):8.3f}")
    print(f"  {'Accuracy':20s} {metrics.get('_accuracy', 0):8.3f}")


def print_confusion(cm, class_names, head_name):
    """Pretty-print confusion matrix."""
    print(f"\n  Confusion Matrix — {head_name}")
    # Header
    header = "  " + " " * 15 + " ".join(f"{n[:6]:>6s}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{cm[i][j]:6d}" for j in range(len(class_names)))
        print(f"  {name:15s}{row}")


def main():
    print("=" * 60)
    print("  EVALUATE — LSTM Model Performance")
    print("=" * 60)

    # Load test data
    files = sorted(glob.glob("data/raw/*.npz"))
    if not files:
        print("\n  [ERROR] No data files found in data/raw/")
        return

    sessions = []
    total = 0
    for f in files:
        data = np.load(f)
        features = normalize_landmarks(data["landmarks"])
        labels = data["labels"]
        sessions.append((features, labels))
        total += features.shape[0]
        print(f"  Loaded {os.path.basename(f)}: {features.shape[0]} frames")

    print(f"\n  Total: {total} frames across {len(sessions)} sessions")

    # Create dataset (no augmentation for eval)
    dataset = MoveSequenceDataset(sessions, seq_len=LSTM_SEQ_LEN, augment=False)
    print(f"  Windows: {len(dataset)}")

    if len(dataset) == 0:
        print("  [ERROR] Not enough data!")
        return

    # Load model
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"\n  [ERROR] Model not found at {LSTM_MODEL_PATH}")
        print("  Run train_lstm.py first!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=True)
    model = ThreeHeadLSTM(
        input_size=checkpoint.get("input_size", 53),
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run predictions
    upper_true, upper_pred = [], []
    lower_true, lower_pred = [], []
    movement_true, movement_pred = [], []

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x, y_u, y_l, y_m = [b.to(device) for b in batch]
            u_logits, l_logits, m_logits, _ = model(x)

            upper_true.extend(y_u.cpu().tolist())
            upper_pred.extend(u_logits.argmax(1).cpu().tolist())
            lower_true.extend(y_l.cpu().tolist())
            lower_pred.extend(l_logits.argmax(1).cpu().tolist())
            movement_true.extend(y_m.cpu().tolist())
            movement_pred.extend(m_logits.argmax(1).cpu().tolist())

    # Compute and print metrics for each head
    upper_metrics = compute_metrics(upper_true, upper_pred, UPPER_CLASSES)
    print_metrics(upper_metrics, UPPER_CLASSES, "Upper Body (punch/block)")
    cm_u = confusion_matrix(upper_true, upper_pred, len(UPPER_CLASSES))
    print_confusion(cm_u, UPPER_CLASSES, "Upper Body")

    lower_metrics = compute_metrics(lower_true, lower_pred, LOWER_CLASSES)
    print_metrics(lower_metrics, LOWER_CLASSES, "Lower Body (kick/crouch/jump)")
    cm_l = confusion_matrix(lower_true, lower_pred, len(LOWER_CLASSES))
    print_confusion(cm_l, LOWER_CLASSES, "Lower Body")

    movement_metrics = compute_metrics(movement_true, movement_pred, MOVEMENT_CLASSES)
    print_metrics(movement_metrics, MOVEMENT_CLASSES, "Movement (left/right)")
    cm_m = confusion_matrix(movement_true, movement_pred, len(MOVEMENT_CLASSES))
    print_confusion(cm_m, MOVEMENT_CLASSES, "Movement")

    # Overall summary
    overall_f1 = np.mean([
        upper_metrics["_macro_avg"]["f1"],
        lower_metrics["_macro_avg"]["f1"],
        movement_metrics["_macro_avg"]["f1"],
    ])
    overall_acc = np.mean([
        upper_metrics["_accuracy"],
        lower_metrics["_accuracy"],
        movement_metrics["_accuracy"],
    ])

    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Average Macro F1:  {overall_f1:.3f}")
    print(f"  Average Accuracy:  {overall_acc:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
