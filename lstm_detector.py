"""
LSTM Move Detector — Inference Wrapper
=======================================
Drop-in replacement for MoveDetector that uses the trained LSTM model.
Same interface: detect(landmarks) → list[str]
"""

import time
import math
import numpy as np
import torch

from lstm_model import ThreeHeadLSTM, UPPER_TO_MOVE, LOWER_TO_MOVE, MOVEMENT_TO_MOVE
from config import (
    LSTM_CONFIDENCE_THRESHOLD, LSTM_MODEL_PATH,
    PUNCH_COOLDOWN, KICK_COOLDOWN, BLOCK_COOLDOWN,
    CROUCH_COOLDOWN, JUMP_COOLDOWN, MOVE_COOLDOWN,
    CALIBRATION_FRAMES, CALIBRATION_OUTLIER_STD,
)

# Same 13 landmark indices as everything else
LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
IDX_LEFT_SHOULDER = 1
IDX_RIGHT_SHOULDER = 2
IDX_LEFT_HIP = 7
IDX_RIGHT_HIP = 8


class LSTMMoveDetector:
    """
    LSTM-based fighting move detector.
    Same interface as MoveDetector — a drop-in replacement.
    """

    def __init__(self, model_path=LSTM_MODEL_PATH, device=None):
        # Device selection
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = ThreeHeadLSTM(
            input_size=checkpoint.get("input_size", 53),
            hidden_size=checkpoint.get("hidden_size", 64),
            num_layers=checkpoint.get("num_layers", 2),
            dropout=0.0,  # No dropout at inference time
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # LSTM hidden state (persists across frames)
        self.hidden = self.model.init_hidden(1, self.device)

        # Calibration (same as rule-based: collect baselines for normalization)
        self.baseline_nose_y = None
        self.baseline_ankle_y = None
        self.baseline_hip_x = None
        self.calibration_frames = 0
        self.calibration_target = CALIBRATION_FRAMES
        self._nose_y_samples = []
        self._ankle_y_samples = []
        self._hip_x_samples = []

        # Cooldown timers
        self.last_move_times = {
            "left_punch": 0, "right_punch": 0,
            "left_kick": 0, "right_kick": 0,
            "block": 0, "crouch": 0, "jump": 0,
            "move_left": 0, "move_right": 0,
        }
        self.cooldowns = {
            "left_punch": PUNCH_COOLDOWN, "right_punch": PUNCH_COOLDOWN,
            "left_kick": KICK_COOLDOWN, "right_kick": KICK_COOLDOWN,
            "block": BLOCK_COOLDOWN, "crouch": CROUCH_COOLDOWN,
            "jump": JUMP_COOLDOWN,
            "move_left": MOVE_COOLDOWN, "move_right": MOVE_COOLDOWN,
        }

        self.confidence_threshold = LSTM_CONFIDENCE_THRESHOLD

    def _can_trigger(self, move_name):
        now = time.time()
        elapsed = now - self.last_move_times[move_name]
        return elapsed >= self.cooldowns[move_name]

    def _trigger(self, move_name):
        self.last_move_times[move_name] = time.time()

    def _extract_and_normalize(self, landmarks):
        """Extract 13 landmarks, normalize, return (53,) feature vector."""
        raw = np.zeros((13, 4), dtype=np.float32)
        for i, idx in enumerate(LANDMARK_INDICES):
            lm = landmarks[idx]
            raw[i] = [lm.x, lm.y, lm.z, lm.visibility]

        # Hip midpoint
        hip_mid_x = (raw[IDX_LEFT_HIP, 0] + raw[IDX_RIGHT_HIP, 0]) / 2
        hip_mid_y = (raw[IDX_LEFT_HIP, 1] + raw[IDX_RIGHT_HIP, 1]) / 2
        hip_mid_z = (raw[IDX_LEFT_HIP, 2] + raw[IDX_RIGHT_HIP, 2]) / 2

        # Shoulder width
        sx = raw[IDX_LEFT_SHOULDER, 0] - raw[IDX_RIGHT_SHOULDER, 0]
        sy = raw[IDX_LEFT_SHOULDER, 1] - raw[IDX_RIGHT_SHOULDER, 1]
        body_scale = max(math.sqrt(sx**2 + sy**2), 0.01)

        # Normalize coordinates
        norm = raw.copy()
        norm[:, 0] = (norm[:, 0] - hip_mid_x) / body_scale
        norm[:, 1] = (norm[:, 1] - hip_mid_y) / body_scale
        norm[:, 2] = (norm[:, 2] - hip_mid_z) / body_scale

        features = np.zeros(53, dtype=np.float32)
        features[:52] = norm.flatten()
        features[52] = body_scale

        return features

    def _calibrate(self, landmarks):
        """Collect baseline samples with outlier rejection. Also warms up LSTM."""
        nose = landmarks[0]
        l_ankle, r_ankle = landmarks[27], landmarks[28]
        l_hip, r_hip = landmarks[23], landmarks[24]

        self._nose_y_samples.append(nose.y)
        self._ankle_y_samples.append((l_ankle.y + r_ankle.y) / 2)
        self._hip_x_samples.append((l_hip.x + r_hip.x) / 2)
        self.calibration_frames += 1

        # Feed frame through LSTM to warm up hidden state
        features = self._extract_and_normalize(landmarks)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, self.hidden = self.model(x, self.hidden)

        if self.calibration_frames >= self.calibration_target:
            # Robust averaging: reject outliers beyond N standard deviations
            self.baseline_nose_y = self._robust_mean(self._nose_y_samples)
            self.baseline_ankle_y = self._robust_mean(self._ankle_y_samples)
            self.baseline_hip_x = self._robust_mean(self._hip_x_samples)
            print(f"[CALIBRATION DONE] LSTM detector ready. "
                  f"Baseline nose_y={self.baseline_nose_y:.3f}, "
                  f"ankle_y={self.baseline_ankle_y:.3f}, hip_x={self.baseline_hip_x:.3f}")
            return True
        return False

    @staticmethod
    def _robust_mean(samples):
        """Compute mean after rejecting outliers beyond CALIBRATION_OUTLIER_STD std devs."""
        arr = np.array(samples)
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-8:
            return float(mean)
        mask = np.abs(arr - mean) <= CALIBRATION_OUTLIER_STD * std
        if mask.sum() == 0:
            return float(mean)
        return float(np.mean(arr[mask]))

    @property
    def is_calibrated(self):
        return self.baseline_nose_y is not None

    def detect(self, landmarks):
        """
        Classify the current frame's landmarks using the LSTM model.

        Args:
            landmarks: MediaPipe pose landmarks list (33 landmarks).

        Returns:
            list of move name strings, e.g. ["right_punch", "move_left"]
        """
        if not self.is_calibrated:
            self._calibrate(landmarks)
            return []

        # Extract and normalize features
        features = self._extract_and_normalize(landmarks)

        # LSTM inference: single timestep, stateful
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            upper_logits, lower_logits, movement_logits, self.hidden = self.model(x, self.hidden)

        # Convert logits to probabilities
        upper_probs = torch.softmax(upper_logits, dim=1).squeeze(0).cpu().numpy()
        lower_probs = torch.softmax(lower_logits, dim=1).squeeze(0).cpu().numpy()
        movement_probs = torch.softmax(movement_logits, dim=1).squeeze(0).cpu().numpy()

        detected_moves = []

        # Upper body head
        upper_class = int(upper_probs.argmax())
        if upper_class > 0 and upper_probs[upper_class] >= self.confidence_threshold:
            move = UPPER_TO_MOVE.get(upper_class)
            if move and self._can_trigger(move):
                detected_moves.append(move)
                self._trigger(move)

        # Lower body head
        lower_class = int(lower_probs.argmax())
        if lower_class > 0 and lower_probs[lower_class] >= self.confidence_threshold:
            move = LOWER_TO_MOVE.get(lower_class)
            if move and self._can_trigger(move):
                detected_moves.append(move)
                self._trigger(move)

        # Movement head
        movement_class = int(movement_probs.argmax())
        if movement_class > 0 and movement_probs[movement_class] >= self.confidence_threshold:
            move = MOVEMENT_TO_MOVE.get(movement_class)
            if move and self._can_trigger(move):
                detected_moves.append(move)
                self._trigger(move)

        return detected_moves

    def reset_calibration(self):
        """Reset calibration and LSTM hidden state."""
        self.baseline_nose_y = None
        self.baseline_ankle_y = None
        self.baseline_hip_x = None
        self.calibration_frames = 0
        self._nose_y_samples = []
        self._ankle_y_samples = []
        self._hip_x_samples = []
        self.hidden = self.model.init_hidden(1, self.device)
        print("[CALIBRATION] Reset — LSTM hidden state cleared. Stand still...")
