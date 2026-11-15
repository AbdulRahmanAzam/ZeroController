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

Classifiers
- `run_model.py` uses the low-latency classifier by default (`fast_action.py`,
  `models/action_classifier_fast.pth`). It reads the current pose plus the last
  133 ms of motion. Retrain with `python train_fast.py`.
- Frame labels: a 1-second clip is not all action. Many clips hold the finished
  pose, some start from standing, and every jump starts and ends on the ground.
  `python tools/label_segments.py` marks each frame as action, early, idle or
  ignore, and drops clips with no usable action. It writes
  `data/labels/frame_labels.json` (read by `train_fast.py`) and
  `data/labels/clip_review.csv` (one row per clip). Put hand corrections in
  `data/labels/overrides.csv` and rerun it after recording new clips.
- Score a model the way the game uses it, on a recording session it never saw:
  `python train_fast.py --holdout-session 3 --save-path models/fast_s3.pth`, then
  `python tools/eval_fast_stream.py --model models/fast_s3.pth --session 3`.
  It reports moves that fired, delay from the start of each move, wrong
  triggers, and false triggers per minute while standing.
- `python train_fast.py --labels clip --transitions idle` reproduces the old
  labelling (every frame is the clip's action), for comparison.

Train on Google Colab (no laptop compute)
- `notebooks/train_on_colab.ipynb` clones the repository, rebuilds the frame
  labels, trains, and downloads `action_classifier_fast.pth` (with a copy in
  Google Drive). Optional settings score the method on held-out sessions and
  train the 1-second models on a GPU. Open it in Colab (File > Open notebook >
  GitHub), check the settings cell, then choose Runtime > Run all.
- Colab clones from GitHub, so push the branch first.
- Recording sessions come from file times, which a clone resets. After recording,
  run `python tools/label_segments.py` on the recording laptop: it saves the
  sessions in `data/labels/sessions.json`. Commit that file with the new clips.
- `data/multicam/` comes from the repository if committed, otherwise from Google
  Drive, otherwise the notebook rebuilds it from the videos in git history (about
  2-3 hours on a free Colab CPU, once).
- `python run_model.py --classifier window` runs the older 1-second model.
- A new action only fires if the body moved in the last 270 ms. If moves are
  missed or fire by themselves, tune `FAST_MIN_MOTION` and
  `FAST_CONFIDENCE_THRESHOLD` in `config.py`.
- Stand with your full body in frame, since kicks and movement need the legs.

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

