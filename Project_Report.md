# Project Report: ZeroController

## 1. Task Definition
ZeroController addresses **real-time skeleton-based action recognition** for controlling a 2D fighting game using a webcam. The system detects a 33‑point human pose per frame, aggregates a **30‑frame sequence**, and classifies it into one of **nine discrete actions** (idle, punches, kicks, jump, block, forward, backward). The predicted action is then gated and mapped to game controls in real time.

## 2. Dataset Description
**Format:**
- Training samples are stored as **NumPy arrays** in `data/raw/<label>/`.
- Each sample has shape **(T, 33, 4)** where **T = 30** frames, **33** is the MediaPipe BlazePose joint count, and the last dimension is **(x, y, z, visibility)**.

**Labels:**
- The label set is defined in `config.py` as nine action classes:
  `idle, left_punch, right_punch, left_kick, right_kick, jump, block, forward, backward`.

**Augmented data:**
- Offline augmentation produces additional samples (mirrors + synthetic variants) and is saved under an **augmented** sibling directory (loaded automatically if present).

## 3. Data Pre-processing
Preprocessing is shared between training and inference via `preprocess_pose_sequence`:
1. **Hip-centering:** subtract mid‑hip position to remove absolute screen position.
2. **Scale normalization:** divide by torso length (mid‑hip to mid‑shoulder) to normalize body size and camera distance.
3. **Visibility weighting:** multiply (x, y, z) by MediaPipe visibility to down‑weight occluded joints.
4. **Velocity stream (optional, enabled by default):** append first‑order temporal differences to create **6 channels** (xyz + dxdydz).
5. **Layout transform:** output is **channel‑first (C, T, V)** for ST‑GCN.

Augmentation is applied in two places:
- **Offline augmentation (`augment_data.py`):** mirror left/right, spatial noise, uniform scaling, temporal jitter, visibility dropout, rotation jitter, time stretching, and same‑class mixup to expand the dataset.
- **Online augmentation (`AugmentedDataset`):** Gaussian noise, random temporal crop + resize, and mirror flips for symmetric actions during training (disabled for validation).

## 4. Network Architecture
**Primary model (default): ActionSTGCN**
- **Input:** (B, 6, 30, 33) when velocity stream is enabled.
- **Graph structure:** fixed 33‑joint skeleton graph with self‑loops, symmetrically normalized adjacency matrix `Â`.
- **Pipeline:**
  1. **Input BatchNorm** over joint‑channel features.
  2. **3 ST‑GCN blocks** (channels 6 → 32 → 64 → 64):
     - Spatial graph convolution (1×1 conv + `Â` aggregation).
     - Temporal convolution (kernel size 9).
     - Dropout + residual connection + ReLU.
  3. **Global average pooling** over time and joints.
  4. **Linear classifier** to 9 logits (softmax used at inference).
- **Parameter scale:** ~94k parameters (lightweight for real‑time use).

**Alternative architectures (configurable):**
- **TCN:** dilated 1‑D temporal convolutions.
- **LSTM / GRU:** recurrent baselines.
- **PoseConv1D:** causal depthwise‑separable 1‑D CNN for lowest latency.

## 5. Loss Function
Training uses **single‑term multi‑class cross‑entropy**:
- **Loss:** `CrossEntropyLoss` over the 9 action classes.
- **Number of terms:** 1 (no auxiliary losses).
- **Weights:** all terms implicitly weighted **1.0**; no class weighting or regularization terms are added in the loss.

## 6. Hyperparameters
**Core training settings (from `config.py` / `train_model.py`):**
- **Sequence length:** 30 frames
- **Batch size:** 8
- **Epochs:** 120
- **Learning rate:** 1e‑3
- **Optimizer:** Adam
- **LR schedule:** CosineAnnealingLR (`T_max=EPOCHS`, `eta_min=1e‑5`)
- **Dropout:** 0.3
- **ST‑GCN channels:** (32, 64, 64)
- **ST‑GCN temporal kernel:** 9
- **Velocity stream:** enabled (`STGCN_USE_VELOCITY=True`)
- **Preprocess flags:** hip‑center, scale‑norm, visibility weighting all enabled
- **Train/val split:** 75/25 stratified split with fixed seed 42

**Selection methodology:**
- Hyperparameters are tuned for a **small, manually collected dataset** and for **real‑time inference**. The lightweight ST‑GCN (three blocks, ~94k params) balances accuracy with low latency. Comments in `Project.md` explicitly justify these values as latency‑friendly and data‑efficient defaults, with easy ablation via config flags.

## 7. SOTA Comparison
**Qualitative comparison (from project rationale):**
- The repo positions **ST‑GCN** as the standard baseline for skeleton‑based action recognition and notes that newer SOTA models (e.g., 2s‑AGCN, CTR‑GCN, MS‑G3D, transformer hybrids) achieve higher accuracy but at **much higher computational cost** and latency, making them a poor fit for real‑time game control.
- The project favors **real‑time responsiveness** and **data efficiency** over peak accuracy, explicitly citing small dataset size and CPU‑bound latency constraints.

**Quantitative comparison available in the repo:**
- **ZeroController ST‑GCN:** ~94k parameters, **5–8 ms** forward time, and end‑to‑end pipeline latency **~20 ms** per frame (MediaPipe + preprocessing + ST‑GCN + gating).
- **Heavier SOTA variants:** documented as **20–110 ms** latency in the project’s design notes, which would exceed the 30 fps budget.

**Accuracy comparison:**
- The repository does not include benchmark accuracy numbers against public datasets or SOTA models. Therefore, a **quantitative accuracy comparison cannot be reported from the current codebase**. To complete this section with accuracy metrics, the project would need evaluations on a standard skeleton‑action dataset (e.g., NTU RGB+D) or a controlled internal benchmark with reported validation accuracy and confusion matrices.
