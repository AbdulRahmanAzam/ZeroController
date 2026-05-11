"""Dataset integrity checks for ZeroController pose samples."""

import argparse
import hashlib
import os
from typing import Dict, List, Tuple

import numpy as np

from config import ACTIONS, DATA_DIR, SEQUENCE_LENGTH


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _augmented_root(raw_root: str) -> str:
    return os.path.join(os.path.dirname(raw_root.rstrip(os.sep)), "augmented")


def _iter_npy_files(root: str, actions: List[str]) -> List[Tuple[str, str]]:
    files = []
    for action in actions:
        action_dir = os.path.join(root, action)
        if not os.path.isdir(action_dir):
            continue
        for name in sorted(os.listdir(action_dir)):
            if name.endswith(".npy"):
                files.append((action, os.path.join(action_dir, name)))
    return files


def _count_ext_files(root: str, actions: List[str], ext: str) -> int:
    total = 0
    for action in actions:
        action_dir = os.path.join(root, action)
        if not os.path.isdir(action_dir):
            continue
        for name in os.listdir(action_dir):
            if name.endswith(ext):
                total += 1
    return total


def _hash_sequence(arr: np.ndarray, round_decimals: int | None) -> str:
    if round_decimals is not None:
        arr = np.round(arr.astype(np.float32), round_decimals)
    h = hashlib.blake2b(digest_size=16)
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode("utf-8"))
    return h.hexdigest()


def _scan_dataset(
    root: str,
    actions: List[str],
    seq_len: int,
    round_decimals: int | None,
) -> Dict:
    files = _iter_npy_files(root, actions)
    counts = {action: 0 for action in actions}
    valid_counts = {action: 0 for action in actions}
    bad_shape = 0
    hash_to_paths: Dict[str, List[str]] = {}

    for action, path in files:
        counts[action] += 1
        seq = np.load(path)
        if seq.shape != (seq_len, 33, 4):
            bad_shape += 1
            continue
        valid_counts[action] += 1
        h = _hash_sequence(seq, round_decimals)
        hash_to_paths.setdefault(h, []).append(path)

    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    return {
        "root": root,
        "total_files": len(files),
        "bad_shape": bad_shape,
        "counts": counts,
        "valid_counts": valid_counts,
        "hashes": set(hash_to_paths.keys()),
        "duplicates": duplicates,
    }


def _balance_summary(counts: Dict[str, int]) -> Dict:
    values = list(counts.values())
    if not values:
        return {"min": 0, "max": 0, "delta": 0, "under": {}, "over": {}}
    min_count = min(values)
    max_count = max(values)
    under = {a: max_count - c for a, c in counts.items() if c < max_count}
    over = {a: c - min_count for a, c in counts.items() if c > min_count}
    return {
        "min": min_count,
        "max": max_count,
        "delta": max_count - min_count,
        "under": under,
        "over": over,
    }


def _print_counts(title: str, counts: Dict[str, int]) -> None:
    print(f"{title}:")
    for action in ACTIONS:
        print(f"  {action:<12} : {counts.get(action, 0)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check raw vs augmented dataset integrity.")
    parser.add_argument("--raw", default=DATA_DIR, help="Raw data root (default: config DATA_DIR).")
    parser.add_argument("--aug", default=None, help="Augmented data root (default: sibling 'augmented').")
    parser.add_argument("--round", type=int, default=None, help="Round decimals before hashing (approx duplicate check).")
    parser.add_argument("--strict-balance", action="store_true", help="Exit non-zero if any class counts differ.")
    args = parser.parse_args()

    raw_root = args.raw
    aug_root = args.aug or _augmented_root(raw_root)

    mode = "exact" if args.round is None else f"rounded({args.round})"

    print("=" * 72)
    print("ZERO CONTROLLER - DATASET INTEGRITY REPORT")
    print(f"Raw root      : {raw_root}")
    print(f"Augmented root: {aug_root}")
    print(f"Hashing mode  : {mode}")
    print(f"Expected shape: ({SEQUENCE_LENGTH}, 33, 4)")
    print("=" * 72)

    raw = _scan_dataset(raw_root, ACTIONS, SEQUENCE_LENGTH, args.round)
    aug_exists = os.path.isdir(aug_root)
    aug = _scan_dataset(aug_root, ACTIONS, SEQUENCE_LENGTH, args.round) if aug_exists else None

    print("RAW DATA")
    print(f"  .npy files   : {raw['total_files']}")
    print(f"  bad shape    : {raw['bad_shape']}")
    print(f"  .avi clips   : {_count_ext_files(raw_root, ACTIONS, '.avi')}")
    _print_counts("  samples per action", raw["valid_counts"])

    raw_balance = _balance_summary(raw["valid_counts"])
    print(f"  balance      : min={raw_balance['min']} max={raw_balance['max']} delta={raw_balance['delta']}")
    if raw_balance["under"]:
        print(f"  need + to reach max: {raw_balance['under']}")
    if raw_balance["over"]:
        print(f"  need - to reach min: {raw_balance['over']}")

    raw_dup = raw["duplicates"]
    print(f"  exact duplicates: {sum(len(v) - 1 for v in raw_dup.values())}")

    if aug is None:
        print("\nAUGMENTED DATA")
        print("  (augmented folder not found)")
    else:
        print("\nAUGMENTED DATA")
        print(f"  .npy files   : {aug['total_files']}")
        print(f"  bad shape    : {aug['bad_shape']}")
        _print_counts("  samples per action", aug["valid_counts"])

        aug_balance = _balance_summary(aug["valid_counts"])
        print(f"  balance      : min={aug_balance['min']} max={aug_balance['max']} delta={aug_balance['delta']}")
        if aug_balance["under"]:
            print(f"  need + to reach max: {aug_balance['under']}")
        if aug_balance["over"]:
            print(f"  need - to reach min: {aug_balance['over']}")

        aug_dup = aug["duplicates"]
        print(f"  exact duplicates: {sum(len(v) - 1 for v in aug_dup.values())}")

        overlap = raw["hashes"].intersection(aug["hashes"])
        raw_pct = (len(overlap) / max(1, len(raw["hashes"]))) * 100.0
        aug_pct = (len(overlap) / max(1, len(aug["hashes"]))) * 100.0
        print("\nRAW vs AUGMENTED")
        print(f"  shared samples (by hash): {len(overlap)}")
        print(f"  overlap vs raw: {raw_pct:.2f}%  |  overlap vs aug: {aug_pct:.2f}%")
        if len(overlap) == len(aug["hashes"]):
            print("  WARNING: augmented data appears identical to raw (no new samples).")

    if args.strict_balance:
        if raw_balance["delta"] != 0:
            print("\n[ERROR] Raw dataset is imbalanced.")
            return 2
        if aug is not None and _balance_summary(aug["valid_counts"])["delta"] != 0:
            print("\n[ERROR] Augmented dataset is imbalanced.")
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
