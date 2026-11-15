"""Mark the frames of every recorded clip that really show the clip's action.

Each clip is one second long, but the action does not fill that second:
  * Most punch, kick and block clips hold the finished pose for the whole second.
  * Some clips start from standing and perform the move partway through.
  * Every jump starts and ends standing on the ground.
  * A few clips show only an arm or a face close to the camera.

train_fast.py used to give every frame the clip's action, so standing frames before
a move were taught as that move. This script gives every frame one mark:

  a  action  the body is in the move
  e  early   the body has started moving into the move (under 30% of the way)
  i  idle    the body is still in the standing pose the move started from
  x  ignore  too uncertain to train on: the frame just before a move starts, the
             way back after a move, landing after a jump, or a pose the tracker lost

Clips that cannot be trusted get status "drop" and are not trained on. Clips worth
a human look get status "review" and are still used.

Outputs, in data/labels/:
  frame_labels.json  marks for every clip, read by train_fast.py
  clip_review.csv    one row per clip with the numbers behind its marks
  overrides.csv      hand corrections, applied last (created with a header if missing)
  sessions.json      the recording session of every clip, from file times. Run this on
                     the machine that recorded new clips, before copying them anywhere.

Run from the repo root:  python tools/label_segments.py
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import ACTIONS, DATA_DIR  # noqa: E402
from pose_features import TRAINING_ASPECT  # noqa: E402
from train_robust import recording_sessions, save_sessions  # noqa: E402

LABELS_DIR = os.path.join(ROOT, "data", "labels")
FRAME_LABELS_PATH = os.path.join(LABELS_DIR, "frame_labels.json")
REVIEW_PATH = os.path.join(LABELS_DIR, "clip_review.csv")
OVERRIDES_PATH = os.path.join(LABELS_DIR, "overrides.csv")

ACTION, EARLY, IDLE, IGNORE = "a", "e", "i", "x"
MARK_NAMES = {ACTION: "action", EARLY: "early", IDLE: "idle", IGNORE: "ignore"}

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
ARM_JOINTS = [13, 14, 15, 16]  # elbows and wrists
LEG_JOINTS = [25, 26, 27, 28]  # knees and ankles
MOVING_JOINTS = {
    "left_punch": ARM_JOINTS, "right_punch": ARM_JOINTS, "block": ARM_JOINTS,
    "left_kick": LEG_JOINTS, "right_kick": LEG_JOINTS,
    "forward": LEG_JOINTS, "backward": LEG_JOINTS,
}
# (joints that should move, joints of the other side) for one-sided actions. Kicks are
# left out: MediaPipe often swaps the left and right leg when the legs cross.
SIDE_JOINTS = {"left_punch": ([13, 15], [14, 16]), "right_punch": ([14, 16], [13, 15])}

# All distances are in torso lengths (mid-shoulder to mid-hip).
MIN_MOVE = 0.3           # a clip whose limbs move less than this holds one pose
START_RATIO = 0.6        # a move must start at most this far from standing, relative to its peak
ONSET_FRACTION = 0.1     # the move starts where it leaves this fraction of its full travel
EARLY_FRACTION = 0.3     # frames before this fraction of the travel are "early"
PEAK_FRACTION = 0.9
RETURN_FRACTION = 0.4    # after the peak, below this fraction the limb is on its way back
MIN_JUMP = 0.3           # hip rise or dip, in torso lengths, that counts as a jump
JUMP_LAND_FRACTION = 0.25
MAX_TORSO_HEIGHT = 0.5   # a torso taller than half the frame is a close-up, not a full body
STANDING_LIKE = 0.45     # a held pose this close to standing probably shows no action


def smooth(values):
    padded = np.concatenate([values[:1], values, values[-1:]])
    return np.convolve(padded, [0.25, 0.5, 0.25], mode="valid")


def body_frame(clip, aspect):
    """Hip-centred 2D joints and the hip, both in torso lengths, plus usable frames."""
    pos = clip[..., :2].astype(np.float64).copy()
    pos[..., 0] *= aspect  # MediaPipe x is a fraction of the width
    hip = (pos[:, L_HIP] + pos[:, R_HIP]) / 2
    shoulder = (pos[:, L_SHOULDER] + pos[:, R_SHOULDER]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    visible = clip[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], 3].min(axis=1) >= 0.5
    ref = float(np.median(torso[visible] if visible.any() else torso))
    usable = visible & (torso > 0.6 * ref) & (torso < 1.6 * ref)
    # The tracker sometimes jumps to a wrong body for a frame.
    step = np.r_[0.0, np.linalg.norm(np.diff(hip, axis=0), axis=1)] / ref
    usable &= step < 0.5
    return (pos - hip[:, None]) / ref, hip / ref, usable


def clip_problem(clip):
    """Reason to drop a clip outright, or None."""
    if float(clip[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], 3].mean()) < 0.5:
        return "torso not tracked"
    shoulder_y = (clip[:, L_SHOULDER, 1] + clip[:, R_SHOULDER, 1]) / 2
    hip_y = (clip[:, L_HIP, 1] + clip[:, R_HIP, 1]) / 2
    if float(np.median(np.abs(hip_y - shoulder_y))) > MAX_TORSO_HEIGHT:
        return "close-up: no full body in view"
    return None


def first_at_least(values, level, start=0):
    hits = np.nonzero(values[start:] >= level)[0]
    return int(hits[0]) + start if len(hits) else None


def rise_onset(travel, level_index):
    """First frame of the rise that reaches travel[level_index]."""
    full = travel.max()
    quiet = np.nonzero(travel[:level_index + 1] <= ONSET_FRACTION * full)[0]
    return int(quiet[-1]) + 1 if len(quiet) else 0


def mark_move(marks, onset, commit):
    marks[:max(onset - 1, 0)] = IDLE
    marks[max(onset - 1, 0):onset] = IGNORE
    marks[onset:commit] = EARLY


def label_action(action, rel, standing, force_move=False):
    """Marks and numbers for a punch, kick, block or step clip.

    force_move skips the test that the clip starts from standing, for clips a person
    checked by eye.
    """
    joints = MOVING_JOINTS[action]
    start = rel[:3, joints].mean(axis=0)
    travel = smooth(np.linalg.norm(rel[:, joints] - start, axis=2).max(axis=1))
    from_standing = smooth(np.linalg.norm(rel[:, joints] - standing[joints], axis=2).max(axis=1))
    move = float(travel.max())
    peak = first_at_least(travel, PEAK_FRACTION * move)
    info = {"move": move, "start_dev": float(from_standing[:3].mean()),
            "peak_dev": float(from_standing[peak]), "hold_dev": float(np.median(from_standing))}
    marks = np.full(len(rel), ACTION)
    if not force_move and (move < MIN_MOVE or info["start_dev"] > START_RATIO * info["peak_dev"]):
        info["kind"] = "held"
        return marks, info
    commit = first_at_least(travel, EARLY_FRACTION * move)
    onset = rise_onset(travel, commit)
    mark_move(marks, onset, commit)
    back = first_at_least(-travel, -RETURN_FRACTION * move, peak)
    if back is not None:
        marks[back:] = IGNORE
    info.update(kind="move", onset=onset, commit=commit, peak=peak, end=(back or len(rel)) - 1)
    return marks, info


def label_jump(hip):
    """Marks and numbers for a jump clip, from the hip height."""
    height = smooth(np.median(hip[:3, 1]) - hip[:, 1])  # image y points down
    travel = np.abs(height)  # the dip before take-off counts as the start
    move = float(travel.max())
    marks = np.full(len(hip), ACTION)
    info = {"move": move, "kind": "jump"}
    if move < MIN_JUMP:
        info["note"] = "no jump found"
        return marks, info
    commit = first_at_least(travel, EARLY_FRACTION * move)
    onset = rise_onset(travel, commit)
    peak = first_at_least(travel, PEAK_FRACTION * move)
    land = int(np.nonzero(travel >= JUMP_LAND_FRACTION * move)[0][-1])
    mark_move(marks, onset, commit)
    marks[land + 1:] = IGNORE
    info.update(onset=onset, commit=commit, peak=peak, end=land)
    return marks, info


def side_check(action, rel, marks, standing):
    """Note when the other arm or leg moved more than the one the label names."""
    if action not in SIDE_JOINTS:
        return None
    frames = marks == ACTION
    if not frames.any():
        return None
    named, other = SIDE_JOINTS[action]
    named_dev = np.linalg.norm(rel[frames][:, named] - standing[named], axis=2).max(axis=1).mean()
    other_dev = np.linalg.norm(rel[frames][:, other] - standing[other], axis=2).max(axis=1).mean()
    if other_dev > 1.2 * named_dev:
        return f"other side moved more ({other_dev:.2f} vs {named_dev:.2f})"
    return None


def load_overrides():
    if not os.path.exists(OVERRIDES_PATH):
        with open(OVERRIDES_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["action", "sample", "status", "kind", "marks", "note"])
        return {}
    with open(OVERRIDES_PATH, newline="") as f:
        return {(row["action"], row["sample"]): row for row in csv.DictReader(f)}


def standing_poses(clips, problems):
    """Median standing pose of each session, from its idle clips."""
    poses = {}
    for session in sorted({session for session, _ in clips.values()}):
        frames = [body_frame(clip, TRAINING_ASPECT)[0]
                  for (action, name), (s, clip) in clips.items()
                  if action == "idle" and s == session and (action, name) not in problems]
        poses[session] = np.median(np.concatenate(frames), axis=0)
    return poses


def build_labels():
    sessions = recording_sessions()
    clips = {key: (session, np.load(os.path.join(DATA_DIR, key[0], key[1] + ".npy")).astype(np.float32))
             for key, session in sorted(sessions.items())}
    problems = {key: reason for key, (_, clip) in clips.items() if (reason := clip_problem(clip))}
    standing = standing_poses(clips, problems)
    overrides = load_overrides()

    labels, review = {}, []
    for (action, name), (session, clip) in clips.items():
        override = overrides.get((action, name), {})
        rel, hip, usable = body_frame(clip, TRAINING_ASPECT)
        if action == "idle":
            marks, info = np.full(len(clip), IDLE), {"kind": "idle"}
        elif action == "jump":
            marks, info = label_jump(hip)
        else:
            marks, info = label_action(action, rel, standing[session], override.get("kind") == "move")
        marks[~usable] = IGNORE

        notes = [info.pop("note")] if "note" in info else []
        status = "ok"
        if (action, name) in problems:
            status, notes = "drop", notes + [problems[(action, name)]]
        else:
            side = side_check(action, rel, marks, standing[session])
            if side:
                notes.append(side)
            if info.get("kind") == "held" and info["hold_dev"] < STANDING_LIKE:
                notes.append("held pose is close to standing")
            if notes:
                status = "review"
        if (marks == IGNORE).mean() > 0.5 and status != "drop":
            status, notes = "drop", notes + ["most frames unusable"]

        if override:
            status = override.get("status") or status
            if override.get("marks"):
                marks = np.array(list(override["marks"]))
                info["kind"] = "hand"
            if override.get("note"):
                notes.append("checked by eye: " + override["note"])

        mark_string = "".join(marks)
        labels[f"{action}/{name}"] = {"session": session, "status": status, "kind": info["kind"],
                                      "marks": mark_string, "note": "; ".join(notes)}
        review.append({
            "action": action, "sample": name, "session": session, "status": status,
            "kind": info["kind"], "marks": mark_string,
            **{k: info.get(k, "") for k in ("onset", "commit", "peak", "end")},
            **{k: f"{info[k]:.2f}" if k in info else "" for k in ("move", "start_dev", "peak_dev", "hold_dev")},
            "note": "; ".join(notes),
        })
    return labels, review


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    os.makedirs(LABELS_DIR, exist_ok=True)
    save_sessions(recording_sessions())
    labels, review = build_labels()
    with open(FRAME_LABELS_PATH, "w") as f:
        json.dump({"version": 1, "marks": MARK_NAMES, "clips": labels}, f, indent=1)
    with open(REVIEW_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)

    print(f"{'action':12s} {'clips':>5s} {'drop':>5s} {'review':>6s} {'held':>5s} {'move':>5s}"
          f" {'action':>7s} {'early':>6s} {'idle':>5s} {'ignore':>6s}")
    for action in ACTIONS:
        rows = [r for r in review if r["action"] == action]
        used = [r for r in rows if r["status"] != "drop"]
        marks = "".join(r["marks"] for r in used)
        print(f"{action:12s} {len(rows):5d} {sum(r['status'] == 'drop' for r in rows):5d}"
              f" {sum(r['status'] == 'review' for r in rows):6d}"
              f" {sum(r['kind'] == 'held' for r in used):5d}"
              f" {sum(r['kind'] in ('move', 'jump') for r in used):5d}"
              f" {marks.count(ACTION):7d} {marks.count(EARLY):6d} {marks.count(IDLE):5d} {marks.count(IGNORE):6d}")
    print(f"Wrote {os.path.relpath(FRAME_LABELS_PATH, ROOT)} and {os.path.relpath(REVIEW_PATH, ROOT)}")


if __name__ == "__main__":
    main()
