"""Cut the recorded clips down to the frames worth training on: data/clips/.

data/raw keeps the original one-second recordings untouched. Most of them are not
all action: many start with the player standing still, every jump starts and ends
standing, and some frames are tracking errors. The models train on data/clips:

  * An action clip keeps only the frames where the move is under way. When the move
    starts partway through a recording, up to CONTEXT frames (133 ms) from just
    before it are kept as lead-in, because the fast model measures motion over the
    last 133 ms. Lead-in frames are marked as context and never taught as the move.
  * Standing frames before a move become idle clips: the model must learn that
    standing still is idle.
  * Frames the tracker lost, the way back after a move, landing after a jump, and
    clips with no usable action are left out.

The cuts come from the marks in data/labels/frame_labels.json, so run
tools/label_segments.py first. Outputs data/clips/<action>/*.npy and
data/clips/index.csv (one row per clip: its source recording and frame range).

Run from the repo root:  python tools/build_clips.py
"""

import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import ACTIONS, DATA_DIR  # noqa: E402
from fast_action import LOOKBACK_FRAMES_AT_30FPS  # noqa: E402
from label_segments import body_frame  # noqa: E402
from pose_features import TRAINING_ASPECT  # noqa: E402

FRAME_LABELS_PATH = os.path.join(ROOT, "data", "labels", "frame_labels.json")
CLIPS_DIR = os.path.join(ROOT, "data", "clips")
INDEX_PATH = os.path.join(CLIPS_DIR, "index.csv")
CONTEXT = LOOKBACK_FRAMES_AT_30FPS
MIN_ACTION_FRAMES = 3   # shorter action runs are left out
MIN_STANDING_FRAMES = 8  # shorter standing runs are left out
INDEX_FIELDS = ["file", "action", "source", "session", "start", "end", "context"]


def runs(marks, wanted):
    """(first, last) frame of each maximal run of marks that are in `wanted`."""
    found, start = [], None
    for i, mark in enumerate(marks + "\0"):
        if mark in wanted and start is None:
            start = i
        elif mark not in wanted and start is not None:
            found.append((start, i - 1))
            start = None
    return found


def lead_in(usable, start):
    """Well-tracked frames right before `start`, at most CONTEXT."""
    count = 0
    while count < CONTEXT and start - count - 1 >= 0 and usable[start - count - 1]:
        count += 1
    return count


def cut_clip(action, name, marks, usable):
    """Pieces of one recording as (folder, file stem, action, start, end, context)."""
    pieces = []
    whole = len(marks)
    if action != "idle":
        for start, end in runs(marks, "ae"):
            if end - start + 1 < MIN_ACTION_FRAMES:
                continue
            stem = name if (start, end) == (0, whole - 1) else f"{name}_f{start:02d}-{end:02d}"
            pieces.append((action, stem, action, start, end, lead_in(usable, start)))
    for start, end in runs(marks, "i"):
        if end - start + 1 < MIN_STANDING_FRAMES:
            continue
        if action == "idle":
            stem = name if (start, end) == (0, whole - 1) else f"{name}_f{start:02d}-{end:02d}"
        else:
            stem = f"{action}_{name}_f{start:02d}-{end:02d}"
        pieces.append(("idle", stem, "idle", start, end, 0))
    return pieces


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    with open(FRAME_LABELS_PATH) as f:
        labels = json.load(f)["clips"]

    shutil.rmtree(CLIPS_DIR, ignore_errors=True)
    for action in ACTIONS:
        os.makedirs(os.path.join(CLIPS_DIR, action), exist_ok=True)

    index, dropped = [], 0
    for key, info in sorted(labels.items()):
        if info["status"] == "drop":
            dropped += 1
            continue
        action, name = key.split("/")
        recording = np.load(os.path.join(ROOT, DATA_DIR, action, name + ".npy"))
        usable = body_frame(recording, TRAINING_ASPECT)[2]
        for folder, stem, piece_action, start, end, context in cut_clip(action, name, info["marks"], usable):
            file = f"{folder}/{stem}.npy"
            np.save(os.path.join(CLIPS_DIR, file), recording[start - context:end + 1])
            index.append({"file": file, "action": piece_action, "source": key, "session": info["session"],
                          "start": start, "end": end, "context": context})

    with open(INDEX_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index)

    kept = sum(row["end"] - row["start"] + 1 for row in index)
    total = 30 * len(labels)
    print(f"{'action':12s} {'clips':>5s} {'frames':>6s}  (frames are the ones taught; lead-in not counted)")
    for action in ACTIONS:
        rows = [row for row in index if row["action"] == action]
        print(f"{action:12s} {len(rows):5d} {sum(r['end'] - r['start'] + 1 for r in rows):6d}")
    print(f"Recordings used: {len(labels) - dropped} of {len(labels)}. "
          f"Frames kept: {kept} of {total} ({kept / total:.0%}). Wrote {os.path.relpath(INDEX_PATH, ROOT)}")


if __name__ == "__main__":
    main()
