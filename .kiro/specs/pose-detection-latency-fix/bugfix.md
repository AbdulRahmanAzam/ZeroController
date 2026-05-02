# Bugfix Requirements Document

## Introduction

The camera-based pose detection system for the 2D fighting game exhibits excessive latency (~1 second) between when a user performs an action (e.g., punch, kick, jump) and when the system detects and triggers that action in the game. This latency makes the game unresponsive and unplayable, as users must perform actions slowly and deliberately for approximately 1 second before detection occurs.

The root cause is a combination of architectural bottlenecks in the action recognition pipeline:
1. **Buffer Latency**: The system requires a full 30-frame sequence (~1 second at 30 FPS) before any prediction can occur
2. **Stable Frame Gate**: Requires 2 consecutive high-confidence predictions (adds ~66ms)
3. **Conservative Cooldown**: 5-frame cooldown after trigger (~166ms)
4. **No Early Exit**: System always waits for full buffer even when action is obvious from initial frames

Industry standard for real-time gaming is 200-300ms latency. The current system operates at 3-5x this threshold, making responsive gameplay impossible.

**Impact**: The game is currently unplayable with camera-based controls. Users report frustration and inability to perform combos or react to opponent actions.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a user performs a quick action (punch, kick, jump) in front of the camera THEN the system requires approximately 1 second of continuous motion before detecting and triggering the action

1.2 WHEN the pose sequence buffer contains fewer than 30 frames (SEQUENCE_LENGTH) THEN the system makes no prediction and outputs "idle" regardless of the action being performed

1.3 WHEN a user performs a rapid sequence of different actions (e.g., punch followed immediately by kick) THEN the system fails to detect the first action before the second action begins, resulting in missed inputs

1.4 WHEN the ActionGate receives high-confidence predictions (≥0.75) for the same action THEN it still requires 2 consecutive frames (PREDICT_STABLE_FRAMES) before considering the action "stable", adding ~66ms of latency

1.5 WHEN an action is triggered THEN the system enforces a 5-frame cooldown (PREDICT_TRIGGER_COOLDOWN = 5, ~166ms at 30 FPS) before allowing another non-idle action, preventing rapid action sequences

1.6 WHEN the model processes a full 30-frame buffer THEN it performs hip-centering and scale normalization preprocessing on every frame in the sequence, adding computational overhead

1.7 WHEN the camera captures frames THEN the CAMERA_BUFFER_SIZE = 1 may cause frame drops during processing spikes, contributing to inconsistent latency

### Expected Behavior (Correct)

2.1 WHEN a user performs a quick action (punch, kick, jump) in front of the camera THEN the system SHALL detect and trigger the action within 200-300ms (industry standard for real-time gaming)

2.2 WHEN the pose sequence buffer contains a partial sequence (e.g., 10-15 frames) and the action is recognizable with high confidence THEN the system SHALL make an early prediction without waiting for the full 30-frame buffer

2.3 WHEN a user performs a rapid sequence of different actions THEN the system SHALL detect each action independently with minimal latency between detections

2.4 WHEN the ActionGate receives high-confidence predictions (≥0.85) for the same action THEN it SHALL consider the action "stable" after 1 frame or implement adaptive gating based on confidence level

2.5 WHEN an action is triggered THEN the system SHALL enforce a minimal cooldown (1-2 frames, ~33-66ms) or implement action-specific cooldowns (e.g., longer for jumps, shorter for punches)

2.6 WHEN the model processes pose sequences THEN it SHALL optimize preprocessing by caching normalized reference points (hip center, torso scale) or performing incremental updates rather than full recomputation

2.7 WHEN the camera captures frames THEN the system SHALL use CAMERA_BUFFER_SIZE ≥ 3 to prevent frame drops and maintain consistent frame delivery

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the system detects an action with confidence below the threshold (currently 0.75) THEN the system SHALL CONTINUE TO suppress that prediction to avoid false positives

3.2 WHEN a user is standing idle or performing no recognizable action THEN the system SHALL CONTINUE TO output "idle" without triggering game actions

3.3 WHEN the pose landmarker fails to detect a person in the camera frame THEN the system SHALL CONTINUE TO output "idle" and display "No pose detected" status

3.4 WHEN the model makes predictions THEN the system SHALL CONTINUE TO use the same trained model weights and action classification logic to maintain accuracy

3.5 WHEN preprocessing is applied to pose sequences THEN the system SHALL CONTINUE TO use hip-centering and scale-normalization to maintain position and scale invariance (but optimize the implementation)

3.6 WHEN the WebSocket pushes action updates to the frontend THEN the system SHALL CONTINUE TO send updates at ~60 Hz (0.016s interval) to maintain smooth UI updates

3.7 WHEN the ActionGate prevents duplicate triggers for one-shot actions (punch, kick, jump) THEN the system SHALL CONTINUE TO ensure each physical action triggers exactly once in the game

3.8 WHEN the system processes continuous actions (idle, move_forward, move_backward, block) THEN the system SHALL CONTINUE TO allow these actions to be held continuously without cooldown restrictions
