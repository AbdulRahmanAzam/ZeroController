# ZeroController — Real-time pose visualizer & action classifier

Brief: a small toolkit to detect MediaPipe 33-point poses from webcam, collect
labeled pose-sequences, augment data, train a lightweight LSTM classifier, and
run live inference for simple actions (e.g., punches).

What the project provides (high level):
- Real-time pose visualizer (`main.py`) using MediaPipe pose landmarker.
- Interactive data collection tool (`collect_data.py`) that saves (T,33,4) samples.
- Simple augmentation helper (`augment_data.py`) to mirror right/left samples.
- Training script (`train_model.py`) for a small LSTM action classifier.
- Live inference runner (`run_model.py`) that shows predicted action on camera.

Quick setup
1. Install Python 3.8+ and a GPU build of PyTorch if desired.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run (examples)
- Visualize pose: `python main.py`
- Collect labeled sequences: `python collect_data.py`  (controls in the file header)
- Mirror right→left samples: `python augment_data.py`
- Train classifier: `python train_model.py`
- Run live detector with trained model: `python run_model.py`

Files (brief, bullet points)
- `config.py`: global constants — camera options, MediaPipe model path/URL,
  data paths, sequence length, and training hyperparameters.
- `camera_utils.py`: robust webcam open + backend fallback helpers (Windows-ready).
- `main.py`: pose visualizer, model downloader (`ensure_pose_model`), drawing HUD.
- `collect_data.py`: interactive recorder that saves sequences to `data/raw/<label>/`.
- `augment_data.py`: mirrors sequences (flip x + swap left/right landmark indices).
- `train_model.py`: small LSTM (`PunchLSTM`), data loader, training loop, saves checkpoint.
- `run_model.py`: loads checkpoint + MediaPipe landmarker, runs live prediction overlay.
- `requirements.txt`: required Python packages (mediapipe, opencv-contrib-python, numpy, torch).
- `models/`: contains the MediaPipe `.task` and any saved PyTorch checkpoints.
- `data/raw/`: folder for collected .npy sequences organized by label.

Progress (what's done)
- Pose visualizer implemented and downloads `pose_landmarker_full.task` when needed.
- Data collection UI and HUD implemented; sample files exist under `data/raw/`.
- `augment_data.py` available and mirrors `right_punch` → `left_punch`.
- Training script and live-runner exist; `models/punch_classifier.pth` found in repo.

Remaining / recommended next steps
- Gather more labeled sequences for each action (raise dataset size & balance).
- Confirm training config vs saved checkpoint paths (resolve `MODEL_SAVE_PATH` mismatch).
- Add evaluation metrics and a validation script / confusion matrix output.
- Improve robustness: model export (TorchScript/ONNX), lower-latency inference.
- Add unit tests or a simple demo video + instructions for reproducibility.

Notes & quick tips
- On Windows prefer `CAMERA_BACKEND = "msmf"`; `camera_utils.py` will fallback.
- If OpenCV GUI calls fail, install a GUI-enabled OpenCV (`opencv-contrib-python`) not headless.
- The collect script header documents keyboard controls (SPACE, 1-9, A, L, Q).

If you want, I can also:
- run a quick static check and fix small config typos, or
- open a short CONTRIBUTING or USAGE example with screenshots.

