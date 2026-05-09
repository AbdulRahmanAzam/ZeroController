# ZeroController — Final Semester Project Report (PDF-Ready)

**Repository:** `AbdulRahmanAzam/ZeroController`  
**Project type:** Real-time skeleton-based action recognition for game control  
**Primary implementation files:** `collect_data.py`, `augment_data.py`, `train_model.py`, `run_model.py`, `config.py`, `main.py`

---

## 1) Task Definition

This project solves a **real-time human action recognition** task (skeleton-based sequence classification) and maps recognized actions to control events in a 2D fighting game.

- Input stream: webcam frames processed by MediaPipe Pose (33 landmarks/frame).
- Model task: classify a short temporal window into one of 9 action classes.
- Output usage: `run_model.py` converts stable predictions into game actions (`idle`, `move_forward`, `move_backward`, `jump`, `block`, `left_punch`, `right_punch`, `left_kick`, `right_kick`).

Formally, for a sequence window \(X \in \mathbb{R}^{T\times 33\times 4}\), predict class \(y \in \{0,\dots,8\}\).

---

## 2) Dataset Description

### 2.1 Data format

Data is collected by `collect_data.py` and saved as NumPy files:

- Path convention: `data/raw/<action>/sample_XXXX.npy`
- Tensor shape per sample: `(SEQUENCE_LENGTH, 33, 4)`
- Last dimension: `[x, y, z, visibility]`
- Default sequence length (`config.py`): `SEQUENCE_LENGTH = 30`

### 2.2 Label space

Action labels are defined in `config.py` (`ACTIONS`) and currently include 9 classes:

1. `idle`
2. `left_punch`
3. `right_punch`
4. `left_kick`
5. `right_kick`
6. `jump`
7. `block`
8. `forward`
9. `backward`

### 2.3 Dataset generation process

- Recording is interactive (`collect_data.py`) with keyboard controls.
- Variable captured frame count is resampled to exactly 30 frames (`_resample_sequence`).
- Augmentation is provided in `augment_data.py`:
  - left/right mirroring
  - spatial noise
  - temporal jitter
  - scale perturbation
  - visibility dropout
  - rotation jitter
  - time stretch
  - same-class mixup

---

## 3) Data Preprocessing

The core model preprocessing is centralized in `train_model.py::preprocess_pose_sequence` and reused at inference time via `run_model.py` (loaded from checkpoint flags):

1. **Hip-centering** (`PREPROCESS_HIP_CENTER`)  
   Translate all joints by midpoint of left/right hips (indices 23, 24).
2. **Scale normalization** (`PREPROCESS_SCALE_NORM`)  
   Normalize by torso length (mid-hip → mid-shoulder magnitude).
3. **Visibility weighting** (`PREPROCESS_USE_VISIBILITY`)  
   Multiply coordinates by per-joint visibility score.
4. **Velocity stream** (`STGCN_USE_VELOCITY`)  
   Append temporal derivative \(\Delta x,\Delta y,\Delta z\).

### Output representation

- ST-GCN input becomes `(C, T, V)` where:
  - \(C=6\) when velocity is enabled; otherwise \(C=3\)
  - \(T=30\), \(V=33\)
- Non-graph models (TCN/LSTM/GRU/PoseConv1D) use flattened `(T, 132)`.

---

## 4) Network Architecture

### 4.1 Implemented model family in repository

`train_model.py` includes multiple architectures:

- `ActionSTGCN` (default in `config.py`)
- `ActionTCN`
- `ActionLSTM`
- `ActionGRU`
- `ActionPoseConv1D`

The default/recommended architecture for this project is **ST-GCN** (`MODEL_TYPE = "stgcn"`).

### 4.2 ST-GCN architecture detail (actual implementation)

- Skeleton graph built from MediaPipe bone connections (`build_adjacency`), with:
  - self-loops
  - symmetric normalization: \(\hat{A}=D^{-1/2}(A+I)D^{-1/2}\)
- Input normalization: `BatchNorm1d(in_channels * num_joints)`
- 3 ST-GCN residual blocks (`_STGCNBlock`), each:
  - spatial graph conv (`_SpatialGCN`)
  - temporal conv (`Conv2d` with kernel `(STGCN_TEMPORAL_K, 1)`)
  - batch norm + dropout + residual connection
- Global average pooling over time and joints
- Final linear classifier head

Default hyperparameters from `config.py`:

- `STGCN_CHANNELS = (32, 64, 64)`
- `STGCN_TEMPORAL_K = 9`
- `DROPOUT = 0.3`

### 4.3 Diagram description (for report figure)

Use this data-flow diagram in the PDF:

```text
Webcam Frame
   -> MediaPipe Pose (33 landmarks)
   -> Rolling Buffer (30 frames)
   -> preprocess_pose_sequence
      [hip center, scale normalize, visibility weighting, velocity]
   -> Tensor (B, C, T, V)
   -> Input BatchNorm
   -> ST-GCN Block 1 (C -> 32)
   -> ST-GCN Block 2 (32 -> 64)
   -> ST-GCN Block 3 (64 -> 64)
   -> Global Average Pool (T,V)
   -> FC (64 -> 9 logits)
   -> Softmax
   -> ActionGate (stability + confidence + cooldown)
   -> Game action event
```

---

## 5) Loss Function

Training objective in `train_model.py`:

- **Criterion:** `nn.CrossEntropyLoss()`
- **Terms:** single-term multiclass classification loss
- **Weighting:** one term only, effective weight = 1.0

Mathematically:
\[
\mathcal{L} = \mathcal{L}_{CE}(\mathbf{z}, y)
\]
where \(\mathbf{z}\) are model logits and \(y\) is ground-truth class index.

No auxiliary losses (e.g., center loss, contrastive loss, localization term) are used in current implementation.

---

## 6) Hyperparameters and Selection Method

### 6.1 Core training hyperparameters (`config.py`)

- `EPOCHS = 120`
- `BATCH_SIZE = 8`
- `LEARNING_RATE = 1e-3`
- `DROPOUT = 0.3`
- `SEQUENCE_LENGTH = 30`
- Optimizer: `Adam`
- Scheduler: `CosineAnnealingLR(T_max=EPOCHS, eta_min=1e-5)`
- Train/validation split: stratified 75/25 (`train_val_split(..., val_ratio=0.25, seed=42)`)

### 6.2 Inference control hyperparameters (`config.py`)

- `PREDICT_CONFIDENCE_THRESHOLD = 0.75`
- `PREDICT_STABLE_FRAMES = 2`
- `PREDICT_TRIGGER_COOLDOWN_MS = 170`
- Early action path enabled: `EARLY_ACTION_ENABLED = True`

### 6.3 Selection methodology

The repository design reflects **manual engineering + literature-aligned defaults**:

- Adam with LR `1e-3` is a stable default for small/medium deep models.
- Batch size `8` and dropout `0.3` target limited manually collected data.
- ST-GCN selected as default due to strong inductive bias for skeleton graphs.
- Inference gate thresholds/cooldowns are tuned for real-time game usability (reduced false triggers and repeated actions).

---

## 7) SOTA Comparison (Quantitative + Qualitative)

### 7.1 What is compared

This repository supports direct internal baseline comparison by changing `--model-type`:

- LSTM (recurrent baseline)
- GRU (lighter recurrent baseline)
- TCN (temporal convolution baseline)
- PoseConv1D (causal lightweight CNN)
- ST-GCN (graph-temporal baseline; default)

This is valuable because all models use the same data pipeline and labels.

### 7.2 Quantitative comparison protocol (repo-aligned)

Run training for each model type and report:

1. Best validation accuracy (from training logs / saved checkpoint stage)
2. CPU inference latency (`python train_model.py --model-type <type> --bench`)
3. Parameter count (printed by `train_model.py` as `Trainable parameters`)

Recommended table for final PDF:

| Model | Best Val Accuracy (%) | CPU Latency (ms/sample) | Trainable Params | Notes |
|---|---:|---:|---:|---|
| LSTM | [fill from run] | [fill] | [fill] | sequential baseline |
| GRU | [fill] | [fill] | [fill] | lighter recurrent baseline |
| TCN | [fill] | [fill] | [fill] | temporal conv baseline |
| PoseConv1D | [fill] | [fill] | [fill] | fastest causal CNN baseline |
| ST-GCN | [fill] | [fill] | [fill] | best topology-aware model |

### 7.3 Qualitative SOTA context

Compared with recent skeleton-action literature trends:

- ST-GCN remains a strong baseline for topology-aware skeleton modeling.
- Newer GCN/Transformer hybrids can exceed ST-GCN on large benchmarks but are typically heavier and more data-hungry.
- For this project’s constraints (manual dataset, real-time CPU game control), the implemented ST-GCN + action gating is a pragmatic balance of accuracy and latency.

---

## 8) README-Style Reproducibility Instructions (Training + Inference)

### 8.1 Environment setup

```bash
pip install -r requirements.txt
```

### 8.2 Verify pose detection

```bash
python main.py
```

### 8.3 Collect data

```bash
python collect_data.py
```

### 8.4 Augment data

```bash
python augment_data.py
```

### 8.5 Train model

```bash
# default model from config.py (currently ST-GCN)
python train_model.py

# explicit model selection
python train_model.py --model-type stgcn
python train_model.py --model-type tcn
python train_model.py --model-type lstm
python train_model.py --model-type gru
python train_model.py --model-type poseconv1d
```

### 8.6 Live inference / game bridge

```bash
# normal live mode (camera + API bridge)
python run_model.py

# useful variants
python run_model.py --latency-mode balanced
python run_model.py --pose-model lite
python run_model.py --no-api
python run_model.py --self-test
```

### 8.7 Run with 2D game frontend

```bash
# terminal 1
python run_model.py

# terminal 2
cd "2D Game"
npm install
npm run dev
```

---

## 9) Implementation Traceability (where each report item maps in code)

- Task pipeline & game bridge: `run_model.py`
- Data collection format: `collect_data.py`
- Data augmentation: `augment_data.py`
- Preprocessing + model definitions + training objective: `train_model.py`
- Hyperparameters/constants: `config.py`
- Pose model download and skeleton connections: `main.py`

---

## 10) Conclusion

ZeroController implements a complete end-to-end real-time action-control pipeline:

1. webcam pose extraction,
2. temporal action classification,
3. robustness gating,
4. live game-action publishing.

The codebase already supports strong comparative experimentation across model families. For final submission, attach your measured metrics table (Section 7.2) produced by the provided training/benchmark commands and export this document as one PDF.

