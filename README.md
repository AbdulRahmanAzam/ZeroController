# ZeroController - Pose Visualizer

This project is now focused only on one goal:

- Detect your body pose from webcam in real-time
- Draw all 33 MediaPipe pose landmarks on your body
- Draw skeleton connections between landmarks

No training pipeline, no move classification, and no keyboard control are included.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Controls

- `Q` quit the app

## Notes

- On first run, the app downloads `pose_landmarker_full.task` into the `models/` folder.
- If camera startup fails on one backend, it automatically tries fallback backends.
