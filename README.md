# ZeroController 🎮

## AI-Powered Controller-Free Fighting Game Input System

Use your body as a game controller! Stand in front of your webcam, throw punches and kicks, and the system translates your real movements into keyboard inputs for any fighting game.

## Part 1: Move Detection System

### Supported Moves

| Move | Detection Method | Default Key (P1) |
|------|-----------------|-------------------|
| Left Punch | Left arm fully extended + fast motion | `U` |
| Right Punch | Right arm fully extended + fast motion | `I` |
| Left Kick | Left ankle rises above knee level | `J` |
| Right Kick | Right ankle rises above knee level | `K` |
| Block | Both wrists raised near face | `O` |
| Crouch | Head drops below standing baseline | `S` |
| Jump | Both ankles rise above standing baseline | `W` |
| Move Left | Hips shift left from baseline | `A` |
| Move Right | Hips shift right from baseline | `D` |

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Recalibrate (stand still again) |
| `P` | Pause/Resume keyboard output |

### How It Works

1. **Calibration** — Stand still for ~1 second when the app starts. The system records your standing baseline position.
2. **Detection** — MediaPipe Pose tracks 33 body landmarks in real-time. The move detector analyzes joint angles, distances, and speeds to classify your movements.
3. **Output** — Detected moves are printed to the console and translated into keyboard presses using `pynput`.

### File Structure

```
zeroController/
├── main.py                # Main entry point — camera loop + display
├── move_detector.py       # Core move detection logic using landmarks
├── keyboard_controller.py # Translates moves → keyboard key presses
├── config.py              # All tunable settings, thresholds, key maps
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Tips

- Make sure your **full body** is visible in the camera
- Stand ~6-8 feet away from the camera for best results
- Good lighting helps a lot!
- If detection feels off, press `R` to recalibrate
- Adjust thresholds in `config.py` to tune sensitivity

### Troubleshooting

- If you see an OpenCV error like `cvShowImage` / `cvDestroyAllWindows` not implemented, your environment likely has a headless OpenCV build.
- Fix it with:

```bash
pip uninstall -y opencv-python-headless
pip install --upgrade opencv-python
```

- The app now falls back to headless mode automatically (no camera window). In headless mode, OpenCV hotkeys (`Q`, `R`, `P`) are unavailable, so use `Ctrl+C` to stop.
