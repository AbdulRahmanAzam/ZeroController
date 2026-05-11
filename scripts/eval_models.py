"""Evaluate saved action classifiers on a held-out validation split."""

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ACTIONS, DATA_DIR, SEQUENCE_LENGTH
from train_model import build_model_from_ckpt, preprocess_pose_sequence, train_val_split


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_checkpoint(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_model(path: str, device: torch.device) -> Tuple[torch.nn.Module, str, List[str], Dict, int]:
    ckpt = _load_checkpoint(path, device)
    model = build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    model_type = ckpt.get("model_type", "lstm")
    actions = ckpt.get("actions", ACTIONS)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))

    preprocess_args = {
        "hip_center": ckpt.get("preprocess_hip_center", True),
        "scale_norm": ckpt.get("preprocess_scale_norm", True),
        "use_visibility": ckpt.get("preprocess_use_visibility", True),
        "use_velocity": ckpt.get("stgcn_use_velocity", True),
    }

    return model, model_type, actions, preprocess_args, seq_len


def _iter_sequence_files(actions: List[str], data_dir: str, include_augmented: bool) -> List[Tuple[str, str]]:
    augmented_root = os.path.join(os.path.dirname(data_dir.rstrip(os.sep)), "augmented")
    files = []
    for action in actions:
        action_dir = os.path.join(data_dir, action)
        if not os.path.isdir(action_dir):
            continue
        candidates = [action_dir]
        if include_augmented:
            aug_dir = os.path.join(augmented_root, action)
            if os.path.isdir(aug_dir):
                candidates.append(aug_dir)
        for folder in candidates:
            for name in sorted(os.listdir(folder)):
                if name.endswith(".npy"):
                    files.append((action, os.path.join(folder, name)))
    return files


def _load_dataset(
    actions: List[str],
    model_type: str,
    preprocess_args: Dict,
    data_dir: str,
    include_augmented: bool,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    X_list = []
    y_list = []
    skipped_shape = 0
    expected_features = 33 * 4

    files = _iter_sequence_files(actions, data_dir, include_augmented)
    for label_idx, action in enumerate(actions):
        for action_name, path in files:
            if action_name != action:
                continue
            seq = np.load(path)
            if seq.shape != (seq_len, 33, 4):
                skipped_shape += 1
                continue

            if model_type == "stgcn":
                sample = preprocess_pose_sequence(
                    seq,
                    use_velocity=preprocess_args["use_velocity"],
                    hip_center=preprocess_args["hip_center"],
                    scale_norm=preprocess_args["scale_norm"],
                    use_visibility=preprocess_args["use_visibility"],
                )
            else:
                sample = seq.reshape(seq_len, -1).astype(np.float32)
                if sample.shape[1] != expected_features:
                    skipped_shape += 1
                    continue

            X_list.append(sample)
            y_list.append(label_idx)

    if not X_list:
        if model_type == "stgcn":
            c = 6 if preprocess_args.get("use_velocity", True) else 3
            X = np.empty((0, c, seq_len, 33), dtype=np.float32)
        else:
            X = np.empty((0, seq_len, expected_features), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

    counts = {action: int(np.sum(y == idx)) for idx, action in enumerate(actions)}
    counts["__skipped__"] = skipped_shape
    return X, y, counts


def _evaluate(model: torch.nn.Module, X_val: np.ndarray, y_val: np.ndarray, device: torch.device, batch_size: int) -> float:
    if len(y_val) == 0:
        return 0.0
    ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total else 0.0


def _discover_models(models_dir: str) -> List[str]:
    pattern = os.path.join(models_dir, "action_classifier*.pth")
    return sorted(glob.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved action classifiers.")
    parser.add_argument("--models", nargs="*", default=None, help="List of model .pth files.")
    parser.add_argument("--models-dir", default=os.path.join(_project_root(), "models"))
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--include-augmented", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_paths = args.models if args.models else _discover_models(args.models_dir)
    if not model_paths:
        print("[ERROR] No model checkpoints found.")
        return 1

    mode = "raw+aug" if args.include_augmented else "raw-only"
    print("=" * 72)
    print("ZERO CONTROLLER - MODEL ACCURACY REPORT")
    print(f"Dataset: {mode}  |  Val ratio: {args.val_ratio:.2f}  |  Seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 72)

    for path in model_paths:
        name = os.path.basename(path)
        model, model_type, actions, preprocess_args, seq_len = _load_model(path, device)
        X, y, counts = _load_dataset(
            actions=actions,
            model_type=model_type,
            preprocess_args=preprocess_args,
            data_dir=args.data,
            include_augmented=args.include_augmented,
            seq_len=seq_len,
        )

        if len(y) == 0:
            print(f"[SKIP] {name}  |  no samples found for actions: {actions}")
            continue

        X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=args.val_ratio, seed=args.seed)
        val_acc = _evaluate(model, X_val, y_val, device, args.batch_size)

        sample_total = len(y)
        skipped = counts.get("__skipped__", 0)
        print(
            f"{name:30s}  type={model_type:9s}  val_acc={val_acc:.2%}  "
            f"samples={sample_total}  val={len(y_val)}  skipped={skipped}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
