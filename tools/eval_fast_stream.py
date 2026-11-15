"""Score the fast classifier the way the game uses it: frame by frame, through the gate.

Per-frame accuracy says little about play. This replays held-out recording sessions
as a stream (standing, one move, standing again), runs the classifier and the same
gate as run_model.py on every frame, and reports for each move:
  hit     the right action fired during the move
  delay   ms from the start of the move to the trigger
  wrong   triggers of any other action in the stream
  repeat  the same move done twice, 270 ms apart, fired twice
and, on a stream of only standing clips, false triggers per minute.

Moves that a clip shows from a standing start are timed from that start (the onset
in data/labels/frame_labels.json). A clip that holds its pose gets a joined start:
100 to 230 ms from a standing clip into the pose. Training uses the same kind of join,
so delays measured on real starts are the ones to trust.

Run:  python tools/eval_fast_stream.py --model X.pth --session 3 [--source lite_43]
      python tools/eval_fast_stream.py --model "fast_s{session}.pth" --session 0 1 2 3
      (one held-out model per session, results pooled)
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (  # noqa: E402
    ACTIONS, FAST_CONFIDENCE_THRESHOLD, FAST_INSTANT_CONFIDENCE, FAST_MIN_MOTION,
    FAST_MOTION_MEMORY_MS, FAST_STABLE_FRAMES, PREDICT_TRIGGER_COOLDOWN_MS,
)
from fast_action import FastActionClassifier, apply_motion_gate  # noqa: E402
from train_robust import FRAME_LABELS_PATH, align_marks, cross_fade, load_all_clips  # noqa: E402

FRAME_MS = 1000.0 / 30.0
REAL_START_KINDS = ("move", "jump")
REPEAT_ACTIONS = ("left_punch", "right_punch", "left_kick", "right_kick")
REPEAT_GAP_FRAMES = 8


def make_gate():
    from run_model import ActionGate  # heavy imports, so only when scoring
    return ActionGate(FAST_CONFIDENCE_THRESHOLD, FAST_STABLE_FRAMES, PREDICT_TRIGGER_COOLDOWN_MS,
                      fast_labels=(), instant_confidence=FAST_INSTANT_CONFIDENCE)


def run_stream(classifier, stream, aspect, motion_gate=True):
    """Triggers as (frame index, label), the way run_model.py fires them."""
    classifier.clear()
    gate = make_gate()
    triggers = []
    for t, frame in enumerate(stream):
        stamp = t * FRAME_MS
        result = classifier.step(frame, stamp, aspect)
        if result is None:
            continue
        label, confidence = result
        if motion_gate:
            label, confidence = apply_motion_gate(
                label, confidence, gate.last_triggered,
                classifier.recent_motion(stamp, FAST_MOTION_MEMORY_MS), FAST_MIN_MOTION)
        stable, fired = gate.step(label, confidence, stamp)
        if fired:
            triggers.append((t, stable))
    return triggers


def session_clips(session, source):
    """{key: (clip, aspect, marks, kind)} for one held-out session and tracking source."""
    with open(FRAME_LABELS_PATH) as f:
        labels = json.load(f)["clips"]
    rows = load_all_clips(source != "raw")
    raw = {row[4]: row[0] for row in rows if row[5] == "raw"}
    out = {}
    for clip, aspect, _, clip_session, key, row_source in rows:
        info = labels.get(key)
        if row_source != source or clip_session != session or info is None or info["status"] == "drop":
            continue
        marks = info["marks"] if source == "raw" else align_marks(raw[key], clip, aspect, info["marks"])
        out[key] = (clip, aspect, marks, info["kind"])
    return out


def move_span(marks):
    """First frame of the move (early or action) and last action frame, or None."""
    moving = [i for i, m in enumerate(marks) if m in "ea"]
    acting = [i for i, m in enumerate(marks) if m == "a"]
    if not moving or not acting:
        return None
    return moving[0], acting[-1]


def event_stream(clip, marks, kind, before, after, rng):
    """Standing, the move, standing. Returns (stream, onset frame, end frame) or None."""
    span = move_span(marks)
    if span is None:
        return None
    if kind in REAL_START_KINDS:
        fade = 2
        onset = len(before) + fade + span[0]
    else:
        fade = int(rng.integers(3, 8))
        onset = len(before)
    stream = cross_fade(before, clip, fade)
    end = len(before) + fade + span[1]
    stream = cross_fade(stream, after, int(rng.integers(3, 8)))
    return stream, onset, end


def score_session(classifier, clips, rng, motion_gate=True):
    """Per-move results and the standing-stream false trigger rate."""
    idle = [key for key, value in clips.items() if key.startswith("idle/")]
    results = []
    for key, (clip, aspect, marks, kind) in sorted(clips.items()):
        action = key.split("/")[0]
        if action == "idle" or not idle:
            continue
        before = clips[idle[rng.integers(len(idle))]][0]
        after = clips[idle[rng.integers(len(idle))]][0]
        built = event_stream(clip, marks, kind, before, after, rng)
        if built is None:
            continue
        stream, onset, end = built
        triggers = run_stream(classifier, stream, aspect, motion_gate)
        right = [t for t, label in triggers if label == action and onset - 3 <= t <= end + 6]
        result = {
            "key": key, "action": action, "real_start": kind in REAL_START_KINDS,
            "hit": bool(right), "delay_ms": (right[0] - onset) * FRAME_MS if right else None,
            "wrong": sum(1 for _, label in triggers if label != action),
        }
        if action in REPEAT_ACTIONS:
            start, stop = move_span(marks)
            part = clip[start:]
            gap = clips[idle[rng.integers(len(idle))]][0][:REPEAT_GAP_FRAMES]
            twice = cross_fade(cross_fade(cross_fade(cross_fade(before, part, 3), gap, 3), part, 3), after, 4)
            result["repeat"] = sum(1 for _, label in run_stream(classifier, twice, aspect, motion_gate)
                                   if label == action) >= 2
        results.append(result)

    frames, false = 0, 0
    for _ in range(3):
        order = [clips[idle[i]] for i in rng.permutation(len(idle))]
        stream = order[0][0]
        for clip, *_ in order[1:]:
            stream = cross_fade(stream, clip, 4)
        false += len(run_stream(classifier, stream, order[0][1], motion_gate))
        frames += len(stream)
    return results, {"idle_minutes": frames / 30 / 60, "idle_false": false}


def summarize(results, idle):
    def stats(rows):
        delays = np.array([r["delay_ms"] for r in rows if r["hit"]])
        repeats = [r["repeat"] for r in rows if "repeat" in r]
        return {
            "moves": len(rows),
            "hit": float(np.mean([r["hit"] for r in rows])) if rows else float("nan"),
            "delay_median": float(np.median(delays)) if len(delays) else float("nan"),
            "delay_p90": float(np.percentile(delays, 90)) if len(delays) else float("nan"),
            "wrong_per_move": float(np.mean([r["wrong"] for r in rows])) if rows else float("nan"),
            "repeat": float(np.mean(repeats)) if repeats else float("nan"),
        }
    summary = {"all": stats(results),
               "real_start": stats([r for r in results if r["real_start"]]),
               "joined_start": stats([r for r in results if not r["real_start"]])}
    for action in ACTIONS:
        rows = [r for r in results if r["action"] == action]
        if rows:
            summary[action] = stats(rows)
    summary["idle_false_per_min"] = idle["idle_false"] / max(idle["idle_minutes"], 1e-9)
    return summary


def print_summary(summary):
    print(f"{'group':14s} {'moves':>5s} {'hit':>6s} {'delay med':>9s} {'p90':>6s} {'wrong/move':>10s} {'repeat':>7s}")
    for group, s in summary.items():
        if not isinstance(s, dict):
            continue
        print(f"{group:14s} {s['moves']:5d} {s['hit']:6.0%} {s['delay_median']:7.0f}ms {s['delay_p90']:4.0f}ms"
              f" {s['wrong_per_move']:10.2f} {s['repeat']:7.0%}")
    print(f"false triggers while standing: {summary['idle_false_per_min']:.1f} per minute")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True,
                        help="Checkpoint. {session} in the path is replaced by each session number.")
    parser.add_argument("--session", type=int, nargs="+", required=True)
    parser.add_argument("--source", default="raw", help="raw, or a multicam condition such as lite_43")
    parser.add_argument("--no-motion-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results, idle = [], {"idle_minutes": 0.0, "idle_false": 0}
    for session in args.session:
        classifier = FastActionClassifier(args.model.format(session=session))
        part, part_idle = score_session(classifier, session_clips(session, args.source),
                                        np.random.default_rng(args.seed + session), not args.no_motion_gate)
        results += part
        idle["idle_minutes"] += part_idle["idle_minutes"]
        idle["idle_false"] += part_idle["idle_false"]
    print_summary(summarize(results, idle))


if __name__ == "__main__":
    main()
