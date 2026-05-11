# ZeroController — Real-time pose visualizer & action classifier

Brief: a small toolkit to detect MediaPipe 33-point poses from webcam, collect
labeled pose-sequences, augment data, train a lightweight LSTM classifier, and
run live inference for simple actions (e.g., punches).

What the project provides (high level):
- Real-time pose visualizer (`scripts/main.py`) using MediaPipe pose landmarker.
- Interactive data collection tool (`scripts/collect_data.py`) that saves (T,33,4) samples.
- Simple augmentation helper (`scripts/augment_data.py`) to mirror right/left samples.
- Training script (`scripts/train_model.py`) for a small LSTM action classifier.
- Live inference runner (`scripts/run_model.py`) that shows predicted action on camera.

Quick setup
1. Install Python 3.8+ and a GPU build of PyTorch if desired.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run (examples)
- Visualize pose: `python scripts/main.py`
- Collect labeled sequences: `python scripts/collect_data.py`  (controls in the file header)
- Mirror right→left samples: `python scripts/augment_data.py`
- Train classifier: `python scripts/train_model.py`
- Run live detector with trained model: `python scripts/run_model.py`

Run with the 2D game
1. From the project root, start the pose bridge and live classifier:

```bash
python scripts/run_model.py
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
wait screen will stay locked until `scripts/run_model.py` is connected.

Files (brief, bullet points)
- `config.py`: global constants — camera options, MediaPipe model path/URL, data paths, sequence length, and training hyperparameters.
- `camera_utils.py`: robust webcam open + backend fallback helpers (Windows-ready).
- `scripts/main.py`: pose visualizer, model downloader (`ensure_pose_model`), drawing HUD.
- `scripts/collect_data.py`: interactive recorder that saves sequences to `data/raw/<label>/`.
- `scripts/augment_data.py`: mirrors sequences (flip x + swap left/right landmark indices).
- `scripts/train_model.py`: small LSTM (`PunchLSTM`), data loader, training loop, saves checkpoint.
- `scripts/run_model.py`: loads checkpoint + MediaPipe landmarker, runs live prediction overlay.
- `requirements.txt`: required Python packages (mediapipe, opencv-contrib-python, numpy, torch).
- `models/`: contains the MediaPipe `.task` and any saved PyTorch checkpoints.
- `data/raw/`: folder for collected .npy sequences organized by label.

Notes & quick tips
- On Windows prefer `CAMERA_BACKEND = "msmf"`; `camera_utils.py` will fallback.
- If OpenCV GUI calls fail, install a GUI-enabled OpenCV (`opencv-contrib-python`) not headless.
- The collect script header documents keyboard controls (SPACE, 1-9, A, L, Q).
