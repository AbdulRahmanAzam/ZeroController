"""Merge two or more trained action-classifier checkpoints into a single model.

How it works
------------
Both TCN and LSTM architectures share this structure:

    backbone  (all layers except the final FC)
    FC head   Linear(channels, num_classes)

Merging strategy
~~~~~~~~~~~~~~~~
1. **Backbone**: parameters are averaged across all input models.
   Models trained on the same input distribution (skeleton keypoints)
   tend to learn similar feature representations, so averaging
   produces a good shared feature extractor.

2. **FC head**: weight rows are taken per-action from whichever model
   trained on that action.  If the same action appears in multiple
   models, its FC rows are averaged across them.

3. **Optional fine-tune** (``--finetune``): re-trains just the FC
   layer for a few epochs on all available data.  This re-calibrates
   the merged head to the averaged backbone's feature space and
   typically recovers most accuracy lost by averaging.

Usage
-----
# Merge two models, save to default path
python merge_models.py models/punch.pth models/kick.pth

# Specify output path
python merge_models.py models/punch.pth models/kick.pth --output models/merged.pth

# Merge + re-calibrate FC (recommended)
python merge_models.py models/punch.pth models/kick.pth --finetune

# Merge 3+ models
python merge_models.py models/punch.pth models/kick.pth models/movement.pth --finetune
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import LEARNING_RATE, SEQUENCE_LENGTH
from train_model import ActionLSTM, ActionTCN, build_model_from_ckpt, load_dataset


# ── Merge logic ───────────────────────────────────────────────────────────────

def merge_checkpoints(paths: list):
    """Load, validate, and merge checkpoints.

    Returns
    -------
    merged_model   : nn.Module  (CPU, eval mode)
    merged_actions : list[str]
    base_ckpt      : dict       first checkpoint (used for metadata)
    """
    ckpts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)
        ckpts.append(torch.load(p, map_location="cpu", weights_only=True))
        print(f"[LOAD]  {p}  →  actions: {ckpts[-1]['actions']}")

    # ── Validate architecture compatibility ───────────────────────────────────
    ref = ckpts[0]
    for i, c in enumerate(ckpts[1:], 1):
        if c.get("model_type", "lstm") != ref.get("model_type", "lstm"):
            print(f"[ERROR] Model {i} has a different model_type — cannot merge.")
            print(f"        Got '{c.get('model_type')}', expected '{ref.get('model_type')}'.")
            sys.exit(1)
        if c["input_size"] != ref["input_size"]:
            print(f"[ERROR] Model {i} has a different input_size — cannot merge.")
            sys.exit(1)

    # ── Collect unique actions (preserve insertion order) ─────────────────────
    merged_actions: list = []
    for c in ckpts:
        for a in c["actions"]:
            if a not in merged_actions:
                merged_actions.append(a)

    print(f"\n[MERGE] {len(ckpts)} models  →  {len(merged_actions)} total action(s)")
    print(f"[MERGE] Actions: {merged_actions}")

    # ── Average backbone parameters ───────────────────────────────────────────
    backbone_avg: dict = {}
    all_backbone_keys: set = set()
    for c in ckpts:
        for k in c["model_state"]:
            if not k.startswith("fc."):
                all_backbone_keys.add(k)

    for key in all_backbone_keys:
        tensors = [c["model_state"][key].float()
                   for c in ckpts if key in c["model_state"]]
        backbone_avg[key] = torch.stack(tensors).mean(dim=0)

    # ── Build deduplicated FC rows ─────────────────────────────────────────────
    # Same action in multiple models → average its FC weight rows.
    action_fc: dict = {}
    for c in ckpts:
        for local_i, action in enumerate(c["actions"]):
            if action not in action_fc:
                action_fc[action] = {"weights": [], "biases": []}
            action_fc[action]["weights"].append(c["model_state"]["fc.weight"][local_i])
            action_fc[action]["biases"].append( c["model_state"]["fc.bias"][local_i])

    fc_weight = torch.stack([
        torch.stack(action_fc[a]["weights"]).mean(dim=0) for a in merged_actions
    ])  # (num_merged_actions, channels)
    fc_bias = torch.stack([
        torch.stack(action_fc[a]["biases"]).mean(dim=0) for a in merged_actions
    ])  # (num_merged_actions,)

    # ── Assemble merged model ─────────────────────────────────────────────────
    merged = build_model_from_ckpt(ref, num_classes=len(merged_actions))
    merged.load_state_dict({**backbone_avg, "fc.weight": fc_weight, "fc.bias": fc_bias})
    merged.eval()
    return merged, merged_actions, ref


# ── Optional FC fine-tune ─────────────────────────────────────────────────────

def finetune_fc(model: nn.Module, merged_actions: list,
                device: torch.device, epochs: int = 30) -> None:
    """Re-calibrate the merged FC on all available data (backbone frozen).

    After backbone averaging the feature distribution shifts slightly.
    Re-training just the FC for a few epochs re-calibrates the decision
    boundaries without changing the shared backbone.
    """
    X, y = load_dataset(actions=merged_actions)

    if len(X) == 0:
        print("[FINETUNE] No data found — skipping.")
        return

    print(f"\n[FINETUNE] {len(X)} samples  |  {epochs} epochs  |  FC only (backbone frozen)")

    # Freeze backbone
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=16, shuffle=True,
    )
    model.to(device).train()
    opt       = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}  acc: {correct / total:.2%}")

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    model.eval()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two or more ZeroController action-classifier .pth files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Two or more paths to .pth checkpoint files.",
    )
    parser.add_argument(
        "--output", "-o", default="models/merged_classifier.pth",
        help="Output path for the merged checkpoint  (default: models/merged_classifier.pth).",
    )
    parser.add_argument(
        "--finetune", action="store_true",
        help="Re-calibrate FC on all available data after merging (recommended).",
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=30, metavar="N",
        help="Number of FC fine-tune epochs  (default: 30).",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("At least two input .pth files are required.")

    print("=" * 60)
    print("  ZERO CONTROLLER - MODEL MERGE")
    print("=" * 60)

    merged_model, merged_actions, base_ckpt = merge_checkpoints(args.inputs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    if args.finetune:
        finetune_fc(merged_model, merged_actions, device, args.finetune_epochs)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Copy every hyperparameter from the base checkpoint, then overwrite
    # the fields that the merge actually changes (state_dict, actions).
    # Using dict-spread preserves any future metadata without extra code.
    merged_ckpt = {
        **base_ckpt,
        "model_state": merged_model.state_dict(),
        "actions":     merged_actions,
    }
    # Ensure defaults exist for older checkpoints that might miss some fields.
    merged_ckpt.setdefault("model_type",       "lstm")
    merged_ckpt.setdefault("hidden_size",        64)
    merged_ckpt.setdefault("num_layers",          1)
    merged_ckpt.setdefault("tcn_channels",       64)
    merged_ckpt.setdefault("tcn_kernel_size",     3)
    merged_ckpt.setdefault("tcn_num_layers",      4)
    merged_ckpt.setdefault("dropout",           0.2)
    merged_ckpt.setdefault("sequence_length", SEQUENCE_LENGTH)
    torch.save(merged_ckpt, args.output)

    print(f"\n[SAVED]   {args.output}")
    print(f"[ACTIONS] {merged_actions}")
    print()
    print("  To use the merged model:")
    print(f"    1. Set MODEL_SAVE_PATH = \"{args.output}\" in config.py")
    print(f"    2. Set ACTIONS = {merged_actions} in config.py")
    print(f"    3. python run_model.py")


if __name__ == "__main__":
    main()
