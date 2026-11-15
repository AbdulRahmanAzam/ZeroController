"""Rebuild data/multicam/*.npz: landmarks from the recorded clips under live camera conditions.

The clips exist only in git history, so this reads them from there. Each condition
reprocesses every clip with a real MediaPipe pass, which gives the classifier the
landmark noise a different camera or the lite pose model actually produces.

collect_data.py saved each video frame after drawing its skeleton on it. Live camera
frames have no drawing, and the drawn lines shift what MediaPipe finds, so each frame
is cleaned first: the drawn lines are found by colour near where the recorded
landmarks put them, and painted over from the surrounding pixels.

Run from the repo root:  python tools/build_multicam_landmarks.py
"""

import argparse
import glob
import multiprocessing
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import ACTIONS, DATA_DIR  # noqa: E402
from main import POSE_CONNECTIONS  # noqa: E402

# HSV ranges of the colours collect_data.py drew with: LINE_COLOR green, POINT_COLOR yellow.
DRAWN_COLOURS = [((45, 120, 90), (80, 255, 255)), ((22, 120, 120), (38, 255, 255))]

# condition name -> pose model. Names say model, frame shape, and any extra change.
CONDITIONS = {
    "full_169": "full",
    "lite_169": "lite",
    "lite_43": "lite",
    "lite_43_zoom": "lite",
    "lite_43_far": "lite",
    "lite_43_tilt": "lite",
    "full_43": "full",
}


def transform(condition, img, tilt):
    h, w = img.shape[:2]
    if condition.endswith("_169"):
        if condition.startswith("lite"):
            return cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        return img
    if "zoom" in condition:
        # Close to the camera: lower legs leave the frame.
        crop_w, crop_h = 720, 540
        x0 = max(0, min(w - crop_w, int(0.47 * w - crop_w / 2)))
        return cv2.resize(img[:crop_h, x0:x0 + crop_w], (640, 480), interpolation=cv2.INTER_AREA)
    x0 = (w - 960) // 2
    crop = cv2.resize(img[:, x0:x0 + 960], (640, 480), interpolation=cv2.INTER_AREA)
    if "far" in condition:
        matrix = np.float32([[0.7, 0, 640 * 0.15], [0, 0.7, 480 * 0.2]])
        return cv2.warpAffine(crop, matrix, (640, 480), borderMode=cv2.BORDER_REPLICATE)
    if "tilt" in condition:
        angle, gamma, shift = tilt
        matrix = cv2.getRotationMatrix2D((320, 240), angle, 1.0)
        matrix[0, 2] += shift
        out = cv2.warpAffine(crop, matrix, (640, 480), borderMode=cv2.BORDER_REPLICATE)
        return np.clip((out / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
    return crop


def remove_drawn_skeleton(img, landmarks):
    """Paint out the skeleton the recorder drew, given landmarks near where it was drawn."""
    h, w = img.shape[:2]
    near = np.zeros((h, w), np.uint8)
    points = [(int(x * w), int(y * h)) for x, y in landmarks[:, :2]]
    for a, b in POSE_CONNECTIONS:
        cv2.line(near, points[a], points[b], 255, 26)
    for point in points:
        cv2.circle(near, point, 16, 255, -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    drawn = np.zeros((h, w), np.uint8)
    for low, high in DRAWN_COLOURS:
        drawn |= cv2.inRange(hsv, low, high)
    mask = cv2.dilate(drawn, np.ones((3, 3), np.uint8), iterations=2) & near
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def recorded_landmarks(name):
    """The clip's saved (30, 33, 4) landmarks, or None. name looks like action__sample_0001.avi."""
    action, sample = name[:-4].split("__")
    path = os.path.join(ROOT, DATA_DIR, action, sample + ".npy")
    return np.load(path) if os.path.exists(path) else None


def resample(frames, length=30):
    idx = np.linspace(0, len(frames) - 1, length)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(frames) - 1)
    t = (idx - lo)[:, None, None]
    return (frames[lo] * (1 - t) + frames[hi] * t).astype(np.float32)


def process_clip(job):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    condition, model, path = job
    name = os.path.basename(path)
    rng = np.random.default_rng(sum(map(ord, name)))
    tilt = (float(rng.uniform(-7, 7)), float(rng.uniform(0.8, 1.6)), float(rng.uniform(-30, 30)))
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=os.path.join(ROOT, f"models/pose_landmarker_{model}.task")
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    cap = cv2.VideoCapture(path)
    count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
    recorded = recorded_landmarks(name)
    points, index, aspect = [], 0, None
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if recorded is not None:
                # The saved clip spans the same second, resampled to its own frame count.
                t = index * (len(recorded) - 1) / (count - 1)
                lo = min(int(t), len(recorded) - 1)
                hi = min(lo + 1, len(recorded) - 1)
                img = remove_drawn_skeleton(img, recorded[lo] * (1 - (t - lo)) + recorded[hi] * (t - lo))
            frame = transform(condition, img, tilt)
            aspect = frame.shape[1] / frame.shape[0]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(index * 33.33)
            )
            index += 1
            if result.pose_landmarks:
                points.append([[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]])
    cap.release()
    clip = resample(np.array(points, np.float32)) if len(points) >= 8 else None
    return name, aspect, clip


def extract_clips(out_dir):
    """Write the newest version of each recorded .avi from git history to out_dir."""
    listing = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.splitlines()
    seen = {}
    for line in listing:
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].endswith(".avi"):
            segments = parts[1].split("/")
            if len(segments) >= 3 and segments[-2] in ACTIONS:
                seen.setdefault((segments[-2], segments[-1]), parts[0])
    for (action, name), sha in seen.items():
        blob = subprocess.run(["git", "cat-file", "blob", sha], capture_output=True, cwd=ROOT, check=True)
        with open(os.path.join(out_dir, f"{action}__{name}"), "wb") as f:
            f.write(blob.stdout)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "multicam"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as clips_dir:
        print(f"Extracted {extract_clips(clips_dir)} clips from git history")
        paths = sorted(glob.glob(os.path.join(clips_dir, "*.avi")))
        with multiprocessing.Pool(args.workers) as pool:
            for condition, model in CONDITIONS.items():
                target = os.path.join(args.out, condition + ".npz")
                if os.path.exists(target):
                    continue
                names, clips, aspect = [], [], None
                jobs = [(condition, model, path) for path in paths]
                for name, clip_aspect, clip in pool.imap(process_clip, jobs, chunksize=4):
                    aspect = clip_aspect or aspect
                    if clip is not None:
                        names.append(name)
                        clips.append(clip)
                np.savez_compressed(target, names=np.array(names), x=np.stack(clips), aspect=aspect)
                print(f"{condition}: {len(names)} clips")


if __name__ == "__main__":
    main()
