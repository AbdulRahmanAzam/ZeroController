# ZeroController — Real-time pose visualizer & action classifier

Brief: a small toolkit to detect MediaPipe 33-point poses from webcam, collect
labeled pose-sequences, augment data, train lightweight action classifiers
(ST-GCN/TCN/LSTM/GRU/PoseConv1D), and run live inference for game control.

What the project provides (high level):
- Real-time pose visualizer (`main.py`) using MediaPipe pose landmarker.
- Interactive data collection tool (`collect_data.py`) that saves (T,33,4) samples.
- Simple augmentation helper (`augment_data.py`) to mirror right/left samples.
- Training script (`train_model.py`) for ST-GCN, TCN, LSTM, GRU, and PoseConv1D classifiers.
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
- Train classifier (defaults from `config.py`): `python train_model.py`
- Train a specific architecture: `python train_model.py --model-type stgcn` (or `tcn`, `lstm`, `gru`, `poseconv1d`)
- Run live detector with trained model: `python run_model.py`

Run with the 2D game
1. From the project root, start the pose bridge and live classifier:

```bash
python run_model.py
```

This opens the camera preview and also serves Player 1 actions at
`http://localhost:8000` and `ws://localhost:8000/ws/pose/1`.

2. In another terminal, start the game:

```bash
cd "2D Game"
npm install
npm run dev
```

3. In the game menu, choose a match mode, then choose either `KEYBOARD` or
`ZERO CONTROLLER` for Player 1. If `ZERO CONTROLLER` is selected, the arena
wait screen will stay locked until `run_model.py` is connected.

Files (brief, bullet points)
- `config.py`: global constants — camera options, MediaPipe model path/URL,
  data paths, sequence length, and training hyperparameters.
- `camera_utils.py`: robust webcam open + backend fallback helpers (Windows-ready).
- `main.py`: pose visualizer, model downloader (`ensure_pose_model`), drawing HUD.
- `collect_data.py`: interactive recorder that saves sequences to `data/raw/<label>/`.
- `augment_data.py`: mirrors sequences (flip x + swap left/right landmark indices).
- `train_model.py`: model definitions (`ActionSTGCN`, `ActionTCN`, `ActionLSTM`, `ActionGRU`, `ActionPoseConv1D`), data loader, training loop, checkpoint save.
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

Final report
- A PDF-ready final semester report draft grounded in this repository is available at:
  - `FINAL_SEMESTER_REPORT.md`
