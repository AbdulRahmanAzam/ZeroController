"""
Move Detector Module
====================
Uses MediaPipe Pose landmarks to detect fighting game moves:
  - Left/Right Punch
  - Left/Right Kick
  - Block
  - Crouch
  - Jump
  - Move Left / Move Right

Each move has a cooldown to prevent spamming.
"""

import time
import math
import numpy as np
from config import (
    PUNCH_EXTENSION_THRESHOLD, PUNCH_HEIGHT_TOLERANCE, PUNCH_SPEED_THRESHOLD,
    KICK_HEIGHT_THRESHOLD, KICK_EXTENSION_THRESHOLD,
    BLOCK_WRIST_FACE_THRESHOLD,
    CROUCH_THRESHOLD, JUMP_THRESHOLD,
    CROUCH_EMA_ALPHA, CROUCH_CONFIRM_FRAMES,
    JUMP_EMA_ALPHA, JUMP_CONFIRM_FRAMES,
    MOVE_LEFT_THRESHOLD, MOVE_RIGHT_THRESHOLD,
    MOVE_EMA_ALPHA, MOVE_CONFIRM_FRAMES, MOVE_VELOCITY_THRESHOLD,
    PUNCH_COOLDOWN, KICK_COOLDOWN, BLOCK_COOLDOWN,
    CROUCH_COOLDOWN, JUMP_COOLDOWN, MOVE_COOLDOWN,
    CALIBRATION_FRAMES, CALIBRATION_OUTLIER_STD,
)

# Minimum landmark visibility to trust the detection
MIN_VISIBILITY = 0.7


# MediaPipe Pose landmark indices
class LM:
    """Landmark indices for easy reference."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


def _distance(p1, p2):
    """Euclidean distance between two landmark points (x, y)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _angle(a, b, c):
    """
    Calculate angle at point b, formed by points a-b-c.
    Returns angle in degrees.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


class MoveDetector:
    """
    Detects fighting game moves from MediaPipe Pose landmarks.
    
    Landmarks are expected as a list of objects with .x, .y, .z, .visibility
    attributes (as returned by MediaPipe).
    """

    def __init__(self):
        # Standing baseline (calibrated on first few frames)
        self.baseline_nose_y = None
        self.baseline_ankle_y = None
        self.baseline_hip_x = None
        self.calibration_frames = 0
        self.calibration_target = CALIBRATION_FRAMES

        # Track previous wrist positions for speed detection
        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.prev_time = None

        # ── Movement smoothing & confirmation state ──
        self.ema_hip_x = None               # EMA-smoothed hip x position
        self.prev_ema_hip_x = None           # Previous EMA value (for velocity)
        self.move_confirm_count = 0          # Consecutive frames beyond threshold
        self.move_confirm_direction = None   # "left", "right", or None

        # ── Jump smoothing & confirmation state ──
        self.ema_ankle_y = None              # EMA-smoothed average ankle y position
        self.jump_confirm_count = 0          # Consecutive frames beyond threshold

        # ── Crouch smoothing & confirmation state ──
        self.ema_nose_y = None               # EMA-smoothed nose y position
        self.crouch_confirm_count = 0        # Consecutive frames beyond threshold

        # Cooldown timers — last time each move was triggered
        self.last_move_times = {
            "left_punch": 0,
            "right_punch": 0,
            "left_kick": 0,
            "right_kick": 0,
            "block": 0,
            "crouch": 0,
            "jump": 0,
            "move_left": 0,
            "move_right": 0,
        }

        # Cooldown durations
        self.cooldowns = {
            "left_punch": PUNCH_COOLDOWN,
            "right_punch": PUNCH_COOLDOWN,
            "left_kick": KICK_COOLDOWN,
            "right_kick": KICK_COOLDOWN,
            "block": BLOCK_COOLDOWN,
            "crouch": CROUCH_COOLDOWN,
            "jump": JUMP_COOLDOWN,
            "move_left": MOVE_COOLDOWN,
            "move_right": MOVE_COOLDOWN,
        }

        # Accumulate values for baseline calibration
        self._nose_y_samples = []
        self._ankle_y_samples = []
        self._hip_x_samples = []

    def _can_trigger(self, move_name):
        """Check if enough time has passed since the last trigger of this move."""
        now = time.time()
        elapsed = now - self.last_move_times[move_name]
        return elapsed >= self.cooldowns[move_name]

    def _trigger(self, move_name):
        """Record that a move was just triggered."""
        self.last_move_times[move_name] = time.time()

    def _get_point(self, landmarks, idx):
        """Extract (x, y) tuple from a landmark."""
        lm = landmarks[idx]
        return (lm.x, lm.y)

    def _is_visible(self, landmarks, *indices):
        """Return True only if all given landmark indices have sufficient visibility."""
        return all(landmarks[i].visibility >= MIN_VISIBILITY for i in indices)

    def _calibrate(self, landmarks):
        """Collect samples for baseline standing position with outlier rejection."""
        nose = self._get_point(landmarks, LM.NOSE)
        left_ankle = self._get_point(landmarks, LM.LEFT_ANKLE)
        right_ankle = self._get_point(landmarks, LM.RIGHT_ANKLE)
        left_hip = self._get_point(landmarks, LM.LEFT_HIP)
        right_hip = self._get_point(landmarks, LM.RIGHT_HIP)

        self._nose_y_samples.append(nose[1])
        self._ankle_y_samples.append((left_ankle[1] + right_ankle[1]) / 2)
        self._hip_x_samples.append((left_hip[0] + right_hip[0]) / 2)
        self.calibration_frames += 1

        if self.calibration_frames >= self.calibration_target:
            # Robust averaging: reject outliers beyond N standard deviations
            self.baseline_nose_y = self._robust_mean(self._nose_y_samples)
            self.baseline_ankle_y = self._robust_mean(self._ankle_y_samples)
            self.baseline_hip_x = self._robust_mean(self._hip_x_samples)

            # Initialize EMA values with calibrated baselines
            self.ema_hip_x = self.baseline_hip_x
            self.prev_ema_hip_x = self.baseline_hip_x
            self.ema_ankle_y = self.baseline_ankle_y
            self.ema_nose_y = self.baseline_nose_y

            print(f"[CALIBRATION DONE] Baseline nose_y={self.baseline_nose_y:.3f}, "
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
            return float(mean)  # fallback: all rejected, use raw mean
        return float(np.mean(arr[mask]))

    @property
    def is_calibrated(self):
        return self.baseline_nose_y is not None

    def detect(self, landmarks):
        """
        Analyze landmarks and return a list of detected moves.
        
        Args:
            landmarks: MediaPipe pose landmarks list
            
        Returns:
            list of move name strings, e.g. ["right_punch", "move_left"]
        """
        if not self.is_calibrated:
            self._calibrate(landmarks)
            return []

        now = time.time()
        detected_moves = []

        # Extract key points
        nose = self._get_point(landmarks, LM.NOSE)
        l_shoulder = self._get_point(landmarks, LM.LEFT_SHOULDER)
        r_shoulder = self._get_point(landmarks, LM.RIGHT_SHOULDER)
        l_elbow = self._get_point(landmarks, LM.LEFT_ELBOW)
        r_elbow = self._get_point(landmarks, LM.RIGHT_ELBOW)
        l_wrist = self._get_point(landmarks, LM.LEFT_WRIST)
        r_wrist = self._get_point(landmarks, LM.RIGHT_WRIST)
        l_hip = self._get_point(landmarks, LM.LEFT_HIP)
        r_hip = self._get_point(landmarks, LM.RIGHT_HIP)
        l_knee = self._get_point(landmarks, LM.LEFT_KNEE)
        r_knee = self._get_point(landmarks, LM.RIGHT_KNEE)
        l_ankle = self._get_point(landmarks, LM.LEFT_ANKLE)
        r_ankle = self._get_point(landmarks, LM.RIGHT_ANKLE)

        mid_hip_x = (l_hip[0] + r_hip[0]) / 2
        avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2

        # Body scale: shoulder width makes all thresholds distance-invariant.
        # Whether you're close or far from camera, ratios stay the same.
        body_scale = max(_distance(l_shoulder, r_shoulder), 0.01)

        # Calculate wrist speeds (normalized by body scale)
        l_wrist_speed = 0
        r_wrist_speed = 0
        if self.prev_left_wrist is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                l_wrist_speed = _distance(l_wrist, self.prev_left_wrist) / dt / body_scale
                r_wrist_speed = _distance(r_wrist, self.prev_right_wrist) / dt / body_scale

        # ─── BLOCK Detection ──────────────────────────────────────
        # Both wrists near face = blocking
        if self._is_visible(landmarks, LM.LEFT_WRIST, LM.RIGHT_WRIST, LM.NOSE):
            l_wrist_to_nose = _distance(l_wrist, nose) / body_scale
            r_wrist_to_nose = _distance(r_wrist, nose) / body_scale
            if (l_wrist_to_nose < BLOCK_WRIST_FACE_THRESHOLD and
                r_wrist_to_nose < BLOCK_WRIST_FACE_THRESHOLD):
                if self._can_trigger("block"):
                    detected_moves.append("block")
                    self._trigger("block")
            else:
                # ─── PUNCH Detection ──────────────────────────────────
                # Left punch: left arm extended forward (wrist far from shoulder)
                if self._is_visible(landmarks, LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST):
                    l_arm_angle = _angle(l_shoulder, l_elbow, l_wrist)
                    if l_arm_angle > 140 and l_wrist_speed > PUNCH_SPEED_THRESHOLD:
                        l_extension = abs(l_wrist[0] - l_shoulder[0]) / body_scale
                        if l_extension > PUNCH_EXTENSION_THRESHOLD:
                            if self._can_trigger("left_punch"):
                                detected_moves.append("left_punch")
                                self._trigger("left_punch")

                # Right punch: right arm extended forward
                if self._is_visible(landmarks, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST):
                    r_arm_angle = _angle(r_shoulder, r_elbow, r_wrist)
                    if r_arm_angle > 140 and r_wrist_speed > PUNCH_SPEED_THRESHOLD:
                        r_extension = abs(r_wrist[0] - r_shoulder[0]) / body_scale
                        if r_extension > PUNCH_EXTENSION_THRESHOLD:
                            if self._can_trigger("right_punch"):
                                detected_moves.append("right_punch")
                                self._trigger("right_punch")

        # ─── KICK Detection ───────────────────────────────────────
        # Left kick: left ankle rises above knee or extends forward significantly
        if self._is_visible(landmarks, LM.LEFT_KNEE, LM.LEFT_ANKLE, LM.LEFT_HIP):
            l_knee_ankle_diff = (l_knee[1] - l_ankle[1]) / body_scale
            l_kick_extension = abs(l_ankle[0] - l_hip[0]) / body_scale
            if (l_knee_ankle_diff > KICK_HEIGHT_THRESHOLD or
                (l_kick_extension > KICK_EXTENSION_THRESHOLD and
                 l_ankle[1] < l_knee[1])):  # ankle higher than knee
                if self._can_trigger("left_kick"):
                    detected_moves.append("left_kick")
                    self._trigger("left_kick")

        # Right kick
        if self._is_visible(landmarks, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_HIP):
            r_knee_ankle_diff = (r_knee[1] - r_ankle[1]) / body_scale
            r_kick_extension = abs(r_ankle[0] - r_hip[0]) / body_scale
            if (r_knee_ankle_diff > KICK_HEIGHT_THRESHOLD or
                (r_kick_extension > KICK_EXTENSION_THRESHOLD and
                 r_ankle[1] < r_knee[1])):
                if self._can_trigger("right_kick"):
                    detected_moves.append("right_kick")
                    self._trigger("right_kick")

        # ─── CROUCH Detection ─────────────────────────────────────
        # Uses EMA smoothing + multi-frame confirmation to prevent false triggers.
        if self._is_visible(landmarks, LM.NOSE):
            self.ema_nose_y = CROUCH_EMA_ALPHA * nose[1] + (1.0 - CROUCH_EMA_ALPHA) * self.ema_nose_y
            nose_drop = (self.ema_nose_y - self.baseline_nose_y) / body_scale
            if nose_drop > CROUCH_THRESHOLD:
                self.crouch_confirm_count += 1
            else:
                self.crouch_confirm_count = 0
            if self.crouch_confirm_count >= CROUCH_CONFIRM_FRAMES:
                if self._can_trigger("crouch"):
                    detected_moves.append("crouch")
                    self._trigger("crouch")
                    self.crouch_confirm_count = 0

        # ─── JUMP Detection ──────────────────────────────────────
        # Uses EMA smoothing + multi-frame confirmation to prevent false triggers.
        if self._is_visible(landmarks, LM.LEFT_ANKLE, LM.RIGHT_ANKLE):
            self.ema_ankle_y = JUMP_EMA_ALPHA * avg_ankle_y + (1.0 - JUMP_EMA_ALPHA) * self.ema_ankle_y
            ankle_rise = (self.baseline_ankle_y - self.ema_ankle_y) / body_scale
            if ankle_rise > JUMP_THRESHOLD:
                self.jump_confirm_count += 1
            else:
                self.jump_confirm_count = 0
            if self.jump_confirm_count >= JUMP_CONFIRM_FRAMES:
                if self._can_trigger("jump"):
                    detected_moves.append("jump")
                    self._trigger("jump")
                    self.jump_confirm_count = 0

        # ─── MOVE LEFT / RIGHT Detection ─────────────────────────
        # Uses EMA smoothing, velocity gating, and multi-frame confirmation
        # to eliminate false triggers from landmark jitter.
        if self._is_visible(landmarks, LM.LEFT_HIP, LM.RIGHT_HIP):
            # 1) EMA-smooth the hip x position
            alpha = MOVE_EMA_ALPHA
            self.prev_ema_hip_x = self.ema_hip_x
            self.ema_hip_x = alpha * mid_hip_x + (1.0 - alpha) * self.ema_hip_x

            # 2) Compute shift from baseline using the smoothed value
            hip_shift = (self.ema_hip_x - self.baseline_hip_x) / body_scale

            # 3) Velocity gate: only consider movement if hip is actively moving
            hip_velocity = abs(self.ema_hip_x - self.prev_ema_hip_x) / body_scale
            velocity_ok = hip_velocity > MOVE_VELOCITY_THRESHOLD

            # 4) Determine direction (or idle)
            if hip_shift < MOVE_LEFT_THRESHOLD and velocity_ok:
                direction = "left"
            elif hip_shift > MOVE_RIGHT_THRESHOLD and velocity_ok:
                direction = "right"
            else:
                direction = None

            # 5) Multi-frame confirmation
            if direction is not None and direction == self.move_confirm_direction:
                self.move_confirm_count += 1
            elif direction is not None:
                # New direction — reset counter
                self.move_confirm_direction = direction
                self.move_confirm_count = 1
            else:
                # Back in dead zone — reset
                self.move_confirm_count = 0
                self.move_confirm_direction = None

            # 6) Only trigger after N consecutive confirmed frames
            if self.move_confirm_count >= MOVE_CONFIRM_FRAMES:
                move_name = f"move_{self.move_confirm_direction}"
                if self._can_trigger(move_name):
                    detected_moves.append(move_name)
                    self._trigger(move_name)
                    # Reset counter so it must re-confirm for next trigger
                    self.move_confirm_count = 0

        # Update previous positions
        self.prev_left_wrist = l_wrist
        self.prev_right_wrist = r_wrist
        self.prev_time = now

        return detected_moves

    def reset_calibration(self):
        """Reset calibration to re-establish baseline."""
        self.baseline_nose_y = None
        self.baseline_ankle_y = None
        self.baseline_hip_x = None
        self.calibration_frames = 0
        self._nose_y_samples = []
        self._ankle_y_samples = []
        self._hip_x_samples = []
        # Reset movement smoothing state
        self.ema_hip_x = None
        self.prev_ema_hip_x = None
        self.move_confirm_count = 0
        self.move_confirm_direction = None
        # Reset jump/crouch smoothing state
        self.ema_ankle_y = None
        self.jump_confirm_count = 0
        self.ema_nose_y = None
        self.crouch_confirm_count = 0
        print("[CALIBRATION] Reset — stand still for calibration...")
