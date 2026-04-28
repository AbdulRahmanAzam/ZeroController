"""Train an action classifier (ST-GCN, TCN, or LSTM) on collected pose-sequence data.

Loads all .npy sequences from data/raw/<action>/, trains the model selected
by MODEL_TYPE in config.py, prints per-epoch loss/accuracy, and saves the
best checkpoint to MODEL_SAVE_PATH.

MODEL_TYPE = "stgcn"  →  Spatial-Temporal Graph Convolutional Network (recommended)
MODEL_TYPE = "tcn"    →  Temporal Convolutional Network
MODEL_TYPE = "lstm"   →  Vanilla LSTM baseline

Run:  python train_model.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ACTIONS,
    BATCH_SIZE,
    DATA_DIR,
    DROPOUT,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    MODEL_TYPE,
    NUM_LSTM_LAYERS,
    PREPROCESS_HIP_CENTER,
    PREPROCESS_SCALE_NORM,
    PREPROCESS_USE_VISIBILITY,
    SEQUENCE_LENGTH,
    STGCN_CHANNELS,
    STGCN_TEMPORAL_K,
    STGCN_USE_VELOCITY,
    TCN_KERNEL_SIZE,
    TCN_NUM_CHANNELS,
    TCN_NUM_LAYERS,
)


# ── Models ───────────────────────────────────────────────────────────────────

class ActionLSTM(nn.Module):
    """Vanilla LSTM classifier — simple baseline."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(self.drop(hn[-1]))


# Keep old name as alias so existing checkpoints / run_model.py still load fine
PunchLSTM = ActionLSTM


class _TCNBlock(nn.Module):
    """One dilated residual 1-D convolution block."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2   # 'same' symmetric padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.drop(self.relu(self.bn(self.conv(x)))) + self.skip(x)


class ActionTCN(nn.Module):
    """Temporal Convolutional Network — recommended for skeleton action recognition.

    Stack of dilated residual 1-D conv blocks with exponentially growing
    dilations [1, 2, 4, 8, ...].  With kernel_size=3 and 4 layers the
    receptive field spans 31 frames — covering the full 30-frame window
    from every time position simultaneously.

    Why TCN beats LSTM here:
      - Sees the WHOLE sequence at once (no hidden-state bottleneck)
      - Fully parallelisable → 2-3x faster training
      - Fewer parameters → less overfitting on small datasets
      - Stable gradients (no vanishing/exploding gradient)
    """
    def __init__(self, input_size, num_channels, kernel_size, num_layers,
                 num_classes, dropout=0.2):
        super().__init__()
        layers = []
        in_ch  = input_size
        for i in range(num_layers):
            layers.append(_TCNBlock(in_ch, num_channels, kernel_size, 2 ** i, dropout))
            in_ch = num_channels
        self.net = nn.Sequential(*layers)
        self.fc  = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        # x : (batch, seq_len, features)
        x = x.permute(0, 2, 1)    # → (batch, features, seq_len)  for Conv1d
        x = self.net(x)            # → (batch, channels, seq_len)
        x = x.mean(dim=2)          # global average pool over time
        return self.fc(x)


# ── Skeleton graph (for ST-GCN) ──────────────────────────────────────────────
#
# MediaPipe Pose gives us 33 keypoints. The human body is a graph: joints
# are connected by bones. ST-GCN exploits this structure directly so the
# model does not have to learn "wrist and elbow are related" from scratch.
#
# Each tuple below is an edge (bone) of the skeleton graph. Kept as a local
# copy so we don't have to import main.py (which pulls in MediaPipe).

_POSE_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # right side of face
    (0, 4), (4, 5), (5, 6), (6, 8),       # left side of face
    (9, 10),                              # mouth
    (11, 12),                             # shoulders
    (11, 13), (13, 15),                   # left arm
    (15, 17), (15, 19), (15, 21), (17, 19),  # left hand
    (12, 14), (14, 16),                   # right arm
    (16, 18), (16, 20), (16, 22), (18, 20),  # right hand
    (11, 23), (12, 24), (23, 24),         # torso
    (23, 25), (24, 26),                   # thighs
    (25, 27), (26, 28),                   # shins
    (27, 29), (28, 30),                   # ankles → heels
    (29, 31), (30, 32),                   # heels → foot tips
    (27, 31), (28, 32),                   # ankles → foot tips
]
_NUM_JOINTS = 33


def build_adjacency(num_joints=_NUM_JOINTS, bones=_POSE_BONES):
    """Build the symmetric-normalized adjacency matrix used by the GCN.

    The GCN layer computes  y = A_hat @ x  where A_hat is the graph's
    adjacency matrix (which joints are neighbours) after two standard tweaks:

      1) Add self-loops (A + I):  each joint also keeps its own feature,
         otherwise the GCN would throw the central node's info away.
      2) Symmetric normalisation D^(-1/2) (A+I) D^(-1/2)  (Kipf & Welling 2016):
         stops well-connected joints (e.g. the torso) from dominating and
         keeps activations at a stable scale across layers.

    Returns a (V, V) float32 tensor, where V = num_joints = 33.
    """
    A = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for a, b in bones:
        # Undirected graph — connection goes both ways.
        A[a, b] = 1.0
        A[b, a] = 1.0
    A = A + torch.eye(num_joints)              # self-loops
    degree = A.sum(dim=1)                      # number of neighbours per joint
    d_inv_sqrt = torch.diag(degree.pow(-0.5))  # D^(-1/2)
    return d_inv_sqrt @ A @ d_inv_sqrt         # A_hat


# ── Pose preprocessing (shared by train and inference) ───────────────────────
#
# Raw MediaPipe output per sample has shape (T, 33, 4) = (frames, joints,
# [x, y, z, visibility]). Before feeding the ST-GCN we apply three transforms:
#
#   1. Hip-center      — make the model invariant to camera distance.
#   2. Scale-normalise — make it invariant to body size.
#   3. Velocity stream — append Δposition so the model sees motion explicitly.
#
# These transforms MUST match between training and inference. Keeping them
# in one function (used by both train_model.py and run_model.py) guarantees
# they never drift apart.

# MediaPipe BlazePose landmark indices we rely on for normalisation.
_L_HIP, _R_HIP = 23, 24
_L_SHOULDER, _R_SHOULDER = 11, 12


def preprocess_pose_sequence(seq_raw,
                             use_velocity=True,
                             hip_center=True,
                             scale_norm=True,
                             use_visibility=True):
    """Convert one raw MediaPipe pose sequence to an ST-GCN input tensor.

    Parameters
    ----------
    seq_raw : np.ndarray of shape (T, 33, 4)
        Last axis = [x, y, z, visibility], exactly what collect_data.py saves.
    use_velocity : bool
        If True, append first-order differences (Δposition) as extra channels.
        Result has C = 6 channels (xyz + dxdydz). If False, C = 3.
    hip_center / scale_norm / use_visibility : bool
        Toggle the three preprocessing stages individually (useful for ablation).

    Returns
    -------
    np.ndarray of shape (C, T, 33), dtype float32
        Channel-first layout that Conv2d / ST-GCN expects.
    """
    xyz = seq_raw[..., :3].astype(np.float32).copy()   # (T, 33, 3) positions
    vis = seq_raw[..., 3:4].astype(np.float32).copy()  # (T, 33, 1) visibility

    # ── 1) Hip-centering ────────────────────────────────────────────────────
    # The mid-point between the two hip joints is a stable body origin.
    # Subtracting it removes where-the-person-stands from the features,
    # so the classifier learns actions, not camera position.
    if hip_center:
        mid_hip = (xyz[:, _L_HIP, :] + xyz[:, _R_HIP, :]) / 2.0   # (T, 3)
        xyz = xyz - mid_hip[:, None, :]

    # ── 2) Scale normalisation ─────────────────────────────────────────────
    # Torso length (mid-hip → mid-shoulder) is the most stable body scale.
    # Dividing by it makes a tall adult and a short kid look "the same size".
    # We average the torso length over all frames so one noisy frame can't
    # blow up the whole sample.
    if scale_norm:
        mid_sho = (xyz[:, _L_SHOULDER, :] + xyz[:, _R_SHOULDER, :]) / 2.0  # (T, 3)
        torso = float(np.linalg.norm(mid_sho, axis=-1).mean())
        xyz = xyz / max(torso, 1e-3)   # guard against division by zero

    # ── 3) Visibility weighting ────────────────────────────────────────────
    # MediaPipe reports a [0,1] visibility score per joint. If a joint is
    # occluded, its (x,y,z) is often a guess — muting it with the score
    # prevents those guesses from misleading the classifier.
    if use_visibility:
        xyz = xyz * vis   # broadcasts (T,33,1) over the 3 coord channels

    # ── 4) Velocity (temporal derivative) ──────────────────────────────────
    # Action recognition is fundamentally about motion. Feeding raw positions
    # alone forces the model to infer motion from differences it sees across
    # the time axis; giving it Δposition directly is a strong shortcut.
    if use_velocity:
        vel = np.zeros_like(xyz)
        vel[1:] = xyz[1:] - xyz[:-1]          # first frame: zero vel (no prev)
        features = np.concatenate([xyz, vel], axis=-1)  # (T, 33, 6)
    else:
        features = xyz                         # (T, 33, 3)

    # ST-GCN (and any Conv2d over "time × joint") expects channel-first order.
    # (T, V, C) → (C, T, V)
    return features.transpose(2, 0, 1).astype(np.float32)


# ── ST-GCN model (recommended) ───────────────────────────────────────────────

class _SpatialGCN(nn.Module):
    """One graph-convolution step over the skeleton.

    Computes   y_v = sum over neighbour u of  A_hat[v,u] * (x_u @ W)

    Implemented efficiently as:
      (a) a 1x1 2-D convolution = per-joint linear map (no temporal mixing);
      (b) an einsum along the joint axis that sums features from neighbours
          weighted by the pre-computed normalised adjacency matrix.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Conv2d with kernel 1×1 behaves as a Linear layer applied to every
        # (time, joint) position — same weights W shared across all 30×33 cells.
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, A):
        # x : (B, C_in, T, V)
        x = self.proj(x)                         # → (B, C_out, T, V)
        # For each (b, c, t), replace the V-length vector with A @ vector.
        # That is exactly one graph convolution step on the joint graph.
        return torch.einsum('bctv,vw->bctw', x, A)


class _STGCNBlock(nn.Module):
    """Spatial GCN  →  Temporal 1-D conv  →  residual connection.

    Conceptually: "mix information between connected joints, then between
    neighbouring frames, then add the input back so gradients flow freely."

    Using a residual (skip) connection is what makes deep stacks trainable —
    the standard trick from ResNet, applied here to graph + temporal layers.
    """
    def __init__(self, in_ch, out_ch, temporal_k=9, dropout=0.3):
        super().__init__()
        # Spatial: share info across connected joints in every frame.
        self.gcn = _SpatialGCN(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)

        # Temporal: 1-D conv over the time axis only (kernel_size × 1).
        # "same" padding keeps the sequence length at T=30 throughout.
        pad = (temporal_k - 1) // 2
        self.tcn = nn.Conv2d(out_ch, out_ch,
                             kernel_size=(temporal_k, 1),
                             padding=(pad, 0))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout)

        # Residual: if channel count changes, a 1×1 conv aligns the shapes.
        self.res = (nn.Conv2d(in_ch, out_ch, kernel_size=1)
                    if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        r = self.res(x)
        x = self.relu(self.bn1(self.gcn(x, A)))   # spatial step
        x = self.drop(self.bn2(self.tcn(x)))      # temporal step
        return self.relu(x + r)                   # residual + non-linearity


class ActionSTGCN(nn.Module):
    """Lightweight Spatial-Temporal Graph Convolutional Network.

    Input shape : (B, C, T, V)
        B = batch, C = 3 or 6 (positions, optionally + velocities),
        T = 30 frames, V = 33 joints.

    Pipeline
    --------
      input BN
        ↓
      ST-GCN block 1  (C → 32 channels)
      ST-GCN block 2  (32 → 64 channels)
      ST-GCN block 3  (64 → 64 channels)
        ↓
      global average pool over time+joints  → feature vector (B, 64)
        ↓
      Linear(64 → num_classes)  → logits

    Why this shape? Three blocks = enough spatial receptive field to reach
    every joint from every other joint (≥ 3 graph hops cover the skeleton),
    and enough temporal receptive field (3 × kernel_9 ≈ 25 frames) to see
    the whole 30-frame sample. Keeping it to three blocks holds the parameter
    count around 80k — small enough to train on a few hundred samples.
    """
    def __init__(self, in_channels, num_classes,
                 channels=(32, 64, 64),
                 temporal_k=9, dropout=0.3,
                 num_joints=_NUM_JOINTS):
        super().__init__()

        # register_buffer → tensor is moved with .to(device) but is not a
        # learnable parameter. Perfect for a fixed graph.
        self.register_buffer('A', build_adjacency(num_joints))

        # Input normalisation: stabilises training when data scales differ
        # per joint (e.g. head joints vs. foot joints move by different amounts).
        self.bn_in = nn.BatchNorm1d(in_channels * num_joints)

        # Stack of ST-GCN blocks — channel dimension grows, T and V stay 30×33.
        blocks = []
        prev = in_channels
        for out_ch in channels:
            blocks.append(_STGCNBlock(prev, out_ch, temporal_k, dropout))
            prev = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape

        # ── Input BN: normalise each (channel, joint) time-series ──
        # Re-layout into (B, C*V, T) so BN1d treats every joint/channel as a
        # separate feature and normalises it along the T (and batch) axis.
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.bn_in(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()   # → (B, C, T, V)

        # ── Stack of blocks ──
        for block in self.blocks:
            x = block(x, self.A)

        # ── Global average pool over time AND joints ──
        # Collapses the (T, V) grid to a single vector per sample. An action
        # label summarises the whole clip, so pooling over both axes is natural.
        x = x.mean(dim=(2, 3))    # (B, C_last)

        return self.fc(x)


# ── Model factory (used by train, merge, and run scripts) ────────────────────

def build_model_from_ckpt(ckpt, num_classes=None):
    """Reconstruct an ActionSTGCN / ActionTCN / ActionLSTM from a saved checkpoint.

    All model-type-specific hyperparameters live in the checkpoint dict, so a
    single function is enough for train, merge, and inference scripts.

    Parameters
    ----------
    ckpt        : dict   loaded with torch.load(...)
    num_classes : int | None   override output size (e.g. when expanding FC)
    """
    mt = ckpt.get("model_type", "lstm")
    nc = num_classes if num_classes is not None else len(ckpt["actions"])

    if mt == "stgcn":
        # in_channels is whatever was used at training time (3 or 6).
        return ActionSTGCN(
            in_channels = ckpt.get("stgcn_in_channels",
                                   6 if STGCN_USE_VELOCITY else 3),
            num_classes = nc,
            channels    = tuple(ckpt.get("stgcn_channels",  STGCN_CHANNELS)),
            temporal_k  = ckpt.get("stgcn_temporal_k",      STGCN_TEMPORAL_K),
            dropout     = ckpt.get("dropout",               DROPOUT),
        )

    if mt == "tcn":
        return ActionTCN(
            input_size   = ckpt["input_size"],
            num_channels = ckpt.get("tcn_channels",    TCN_NUM_CHANNELS),
            kernel_size  = ckpt.get("tcn_kernel_size", TCN_KERNEL_SIZE),
            num_layers   = ckpt.get("tcn_num_layers",  TCN_NUM_LAYERS),
            num_classes  = nc,
            dropout      = ckpt.get("dropout",         DROPOUT),
        )

    return ActionLSTM(
        input_size  = ckpt["input_size"],
        hidden_size = ckpt.get("hidden_size",  HIDDEN_SIZE),
        num_layers  = ckpt.get("num_layers",   NUM_LSTM_LAYERS),
        num_classes = nc,
        dropout     = ckpt.get("dropout",      DROPOUT),
    )


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(actions=None, model_type=None):
    """Load all .npy files and return (X, y) tensors shaped for the given model.

    The same raw samples (30, 33, 4) are reshaped / preprocessed differently
    depending on which model will consume them:

      - "stgcn" → (C, T, V) = (6 or 3, 30, 33)   with normalisation + velocity
      - "tcn"   → (T, F)    = (30, 132)          flat landmarks per frame
      - "lstm"  → (T, F)    = (30, 132)          flat landmarks per frame

    Parameters
    ----------
    actions : list[str] | None
        Action labels to load in order. Defaults to ACTIONS from config.
    model_type : str | None
        "stgcn", "tcn", or "lstm". Defaults to MODEL_TYPE from config.
    """
    if actions is None:
        actions = ACTIONS
    if model_type is None:
        model_type = MODEL_TYPE

    X_list, y_list = [], []

    for label_idx, action in enumerate(actions):
        action_dir = os.path.join(DATA_DIR, action)
        if not os.path.isdir(action_dir):
            print(f"[WARN] Missing directory: {action_dir}")
            continue

        files = sorted(f for f in os.listdir(action_dir) if f.endswith(".npy"))
        for fname in files:
            seq = np.load(os.path.join(action_dir, fname))  # (30, 33, 4)
            if seq.shape != (SEQUENCE_LENGTH, 33, 4):
                print(f"[SKIP] {fname} has unexpected shape {seq.shape}")
                continue

            if model_type == "stgcn":
                # Normalised + velocity-augmented, channel-first tensor.
                sample = preprocess_pose_sequence(
                    seq,
                    use_velocity   = STGCN_USE_VELOCITY,
                    hip_center     = PREPROCESS_HIP_CENTER,
                    scale_norm     = PREPROCESS_SCALE_NORM,
                    use_visibility = PREPROCESS_USE_VISIBILITY,
                )  # (C, 30, 33)
            else:
                # Legacy flat layout for LSTM and TCN: one feature vector per frame.
                sample = seq.reshape(SEQUENCE_LENGTH, -1).astype(np.float32)

            X_list.append(sample)
            y_list.append(label_idx)

    if not X_list:
        # Return empty arrays whose shape matches what the model expects,
        # so downstream code can still work out tensor dimensions safely.
        if model_type == "stgcn":
            c = 6 if STGCN_USE_VELOCITY else 3
            empty_x = np.empty((0, c, SEQUENCE_LENGTH, _NUM_JOINTS), dtype=np.float32)
        else:
            empty_x = np.empty((0, SEQUENCE_LENGTH, 33 * 4), dtype=np.float32)
        return empty_x, np.empty(0, dtype=np.int64)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def train_val_split(X, y, val_ratio=0.25, seed=42):
    """True per-class stratified split.

    For each class, val_ratio of its samples go to val and the rest to train.
    This guarantees every class is equally represented in both sets, which is
    critical with small datasets (~50 samples/class) where a random global
    shuffle can accidentally put most of one class into val.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_ratio))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    # Shuffle both sets so batches see mixed classes
    rng.shuffle(train_idx := np.array(train_idx))
    rng.shuffle(val_idx   := np.array(val_idx))

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ── Online augmentation ──────────────────────────────────────────────────────

class AugmentedDataset(torch.utils.data.Dataset):
    """Wrap a numpy (X, y) pair and apply stochastic augmentations each epoch.

    Augmentations applied on the fly (ST-GCN channel-first tensors C×T×V):

    1. Gaussian joint noise   — tiny random perturbations to each landmark
       coordinate, simulating detector jitter and slight pose variation.
       Keeps the action recognisable while making the model noise-robust.

    2. Temporal crop + resize — randomly drop up to 20% of leading or
       trailing frames, then linearly interpolate back to T frames. Forces
       the model to recognise actions that start/end at different phases,
       rather than relying on a fixed temporal alignment.

    3. Random mirror flip     — flip x-coordinates and swap left/right joint
       pairs. Doubles effective dataset for symmetric actions (punches, kicks).
       Only applied to actions that exist in a symmetric pair so e.g. "jump"
       and "block" are not artificially mirrored.

    Augmentations are disabled at val time — always pass augment=False for
    the validation dataset.
    """

    # MediaPipe left/right landmark index pairs — same as augment_data.py
    _SWAP_PAIRS = [
        (1, 4), (2, 5), (3, 6),
        (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16),
        (17, 18), (19, 20), (21, 22),
        (23, 24), (25, 26), (27, 28),
        (29, 30), (31, 32),
    ]
    # Actions where left↔right mirror makes sense (both sides present in dataset)
    _MIRROR_CLASSES = None   # filled lazily from ACTIONS in __init__

    def __init__(self, X, y, augment=True,
                 noise_std=0.005,
                 crop_ratio=0.20,
                 mirror_prob=0.5):
        self.X          = torch.from_numpy(X)   # (N, C, T, V)
        self.y          = torch.from_numpy(y)
        self.augment    = augment
        self.noise_std  = noise_std
        self.crop_ratio = crop_ratio
        self.mirror_prob= mirror_prob

        # Determine which class indices are symmetric (both sides exist)
        pairs = [("left_punch","right_punch"), ("left_kick","right_kick")]
        sym = set()
        for a, b in pairs:
            if a in ACTIONS and b in ACTIONS:
                sym.add(ACTIONS.index(a))
                sym.add(ACTIONS.index(b))
        self._sym_classes = sym

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()   # (C, T, V)
        label = self.y[idx]

        if not self.augment:
            return x, label

        C, T, V = x.shape

        # 1. Gaussian noise on position channels
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # 2. Temporal crop: randomly shorten the sequence, then interpolate back
        max_crop = max(1, int(T * self.crop_ratio))
        crop = random.randint(0, max_crop)
        if crop > 0:
            start = random.randint(0, crop)
            end   = T - (crop - start)
            # x[:, start:end, :] is the cropped window; interpolate back to T frames.
            # F.interpolate 'linear' mode needs 3D input (N, C, L), so flatten
            # the spatial (C, V) axes into one before interpolating.
            seg  = x[:, start:end, :]                      # (C, T', V)
            Tp   = seg.shape[1]
            seg  = seg.permute(0, 2, 1).reshape(C * V, Tp) # (C*V, T')
            seg  = torch.nn.functional.interpolate(
                seg.unsqueeze(0), size=T, mode='linear', align_corners=False
            ).squeeze(0)                                   # (C*V, T)
            x = seg.reshape(C, V, T).permute(0, 2, 1)     # (C, T, V)

        # 3. Mirror flip — only for symmetric action pairs
        if label.item() in self._sym_classes and random.random() < self.mirror_prob:
            # Flip x-coordinate (channel 0 of position, channel 3 of velocity)
            x[0] = 1.0 - x[0]
            if C == 6:         # velocity stream present
                x[3] = -x[3]  # flip velocity x too
            # Swap left↔right joint indices along V dimension
            for a, b in self._SWAP_PAIRS:
                if a < V and b < V:
                    x[:, :, a], x[:, :, b] = x[:, :, b].clone(), x[:, :, a].clone()
            # Remap label to its mirror counterpart
            action_name = ACTIONS[label.item()]
            mirror_map  = {"left_punch": "right_punch", "right_punch": "left_punch",
                           "left_kick":  "right_kick",  "right_kick":  "left_kick"}
            if action_name in mirror_map and mirror_map[action_name] in ACTIONS:
                label = torch.tensor(ACTIONS.index(mirror_map[action_name]),
                                     dtype=label.dtype)

        return x, label


# ── Model resume / expand helpers ────────────────────────────────────────────

def _build_fresh(input_size, num_classes, device):
    """Instantiate a brand-new model from current config settings.

    `input_size` is used by LSTM / TCN (flat 132 features per frame) and
    ignored by ST-GCN (which infers its shape from in_channels instead).
    """
    if MODEL_TYPE == "stgcn":
        in_ch = 6 if STGCN_USE_VELOCITY else 3
        m = ActionSTGCN(
            in_channels = in_ch,
            num_classes = num_classes,
            channels    = STGCN_CHANNELS,
            temporal_k  = STGCN_TEMPORAL_K,
            dropout     = DROPOUT,
        )
        print(f"[MODEL] ActionSTGCN  in_ch={in_ch}  channels={STGCN_CHANNELS}  "
              f"temporal_k={STGCN_TEMPORAL_K}  dropout={DROPOUT}")
    elif MODEL_TYPE == "tcn":
        m = ActionTCN(input_size, TCN_NUM_CHANNELS, TCN_KERNEL_SIZE,
                      TCN_NUM_LAYERS, num_classes, DROPOUT)
        print(f"[MODEL] ActionTCN  ch={TCN_NUM_CHANNELS}  k={TCN_KERNEL_SIZE}  "
              f"layers={TCN_NUM_LAYERS}  dropout={DROPOUT}")
    else:
        m = ActionLSTM(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS, num_classes, DROPOUT)
        print(f"[MODEL] ActionLSTM  hidden={HIDDEN_SIZE}  "
              f"layers={NUM_LSTM_LAYERS}  dropout={DROPOUT}")
    return m.to(device)


def _resume_or_build(input_size, device):
    """Build model fresh, or resume / expand from an existing checkpoint.

    Three cases
    -----------
    1. No checkpoint exists           → build from scratch.
    2. Checkpoint actions == ACTIONS  → load weights and resume.
    3. ACTIONS has new classes        → expand FC; backbone is preserved.
       Returns ``expanded=True`` so the caller can use differential LR.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        print("[MODEL] No checkpoint — building from scratch.")
        return _build_fresh(input_size, len(ACTIONS), device), False

    ckpt        = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    old_actions = ckpt["actions"]

    removed = [a for a in old_actions if a not in ACTIONS]
    new_act  = [a for a in ACTIONS     if a not in old_actions]

    if removed:
        print(f"[WARN] Checkpoint has actions not in config: {removed}")
        print("[WARN] Building fresh model to avoid class mismatch.")
        return _build_fresh(input_size, len(ACTIONS), device), False

    if not new_act:
        print(f"[RESUME] Continuing training  |  {len(old_actions)} action(s): {old_actions}")
        m = build_model_from_ckpt(ckpt)
        m.load_state_dict(ckpt["model_state"])
        return m.to(device), False

    # ── Expand: new actions detected ──────────────────────────────────────────
    print(f"[EXPAND] Existing actions : {old_actions}")
    print(f"[EXPAND] New actions added: {new_act}")
    print(f"[EXPAND] Full action list : {ACTIONS}")

    old_m = build_model_from_ckpt(ckpt, len(old_actions))
    old_m.load_state_dict(ckpt["model_state"])

    new_m  = build_model_from_ckpt(ckpt, len(ACTIONS))   # FC has expanded size
    old_sd = old_m.state_dict()
    new_sd = new_m.state_dict()

    # Copy backbone weights (everything except the FC head)
    for k in new_sd:
        if not k.startswith("fc."):
            new_sd[k] = old_sd[k].clone()

    # Preserve FC rows for known actions; new action rows keep default init
    for new_i, action in enumerate(ACTIONS):
        if action in old_actions:
            old_i = old_actions.index(action)
            new_sd["fc.weight"][new_i] = old_sd["fc.weight"][old_i].clone()
            new_sd["fc.bias"][new_i]   = old_sd["fc.bias"][old_i].clone()

    new_m.load_state_dict(new_sd)
    return new_m.to(device), True


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ZERO CONTROLLER - TRAIN ACTION CLASSIFIER")
    print("=" * 60)

    X, y = load_dataset()
    print(f"[DATA] Total samples: {len(y)}  |  Classes: {dict(zip(ACTIONS, np.bincount(y)))}")

    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print(f"[DATA] Train: {len(y_train)}  |  Val: {len(y_val)}")

    train_ds = AugmentedDataset(X_train, y_train, augment=(MODEL_TYPE == "stgcn"))
    val_ds   = AugmentedDataset(X_val,   y_val,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # input_size means different things per architecture:
    #   - LSTM / TCN: number of flat features per frame (33 landmarks × 4 = 132)
    #   - ST-GCN    : number of input channels (3 positions, or 6 with velocity)
    # We store it in the checkpoint either way so `build_model_from_ckpt` can
    # reconstruct the model without needing the config file.
    if MODEL_TYPE == "stgcn":
        input_size = 6 if STGCN_USE_VELOCITY else 3
    else:
        input_size = 33 * 4  # 132
    model, expanded = _resume_or_build(input_size, device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    if expanded:
        # Backbone already knows motion features — use low LR to preserve them.
        # New FC rows are randomly initialised — use full LR so they learn fast.
        backbone_p = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        fc_p       = [p for n, p in model.named_parameters() if     n.startswith("fc.")]
        optimizer  = torch.optim.Adam([
            {"params": backbone_p, "lr": LEARNING_RATE * 0.1},
            {"params": fc_p,       "lr": LEARNING_RATE},
        ])
        print(f"[EXPAND] LR → backbone={LEARNING_RATE * 0.1:.1e}  FC={LEARNING_RATE:.1e}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cosine annealing decays LR smoothly → better final accuracy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        train_loss = total_loss / total
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                v_correct += (preds == yb).sum().item()
                v_total += len(yb)

        val_acc = v_correct / v_total if v_total > 0 else 0.0

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  |  loss: {train_loss:.4f}  "
                  f"train_acc: {train_acc:.2%}  val_acc: {val_acc:.2%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            # Save every hyperparameter needed to rebuild the model later,
            # plus the preprocessing flags used at train time — that way
            # inference can replay the *exact* same pipeline.
            torch.save({
                "model_state":     model.state_dict(),
                "actions":         ACTIONS,
                "input_size":      input_size,
                "model_type":      MODEL_TYPE,
                # LSTM params
                "hidden_size":     HIDDEN_SIZE,
                "num_layers":      NUM_LSTM_LAYERS,
                # TCN params
                "tcn_channels":    TCN_NUM_CHANNELS,
                "tcn_kernel_size": TCN_KERNEL_SIZE,
                "tcn_num_layers":  TCN_NUM_LAYERS,
                # ST-GCN params
                "stgcn_in_channels":  6 if STGCN_USE_VELOCITY else 3,
                "stgcn_channels":     list(STGCN_CHANNELS),
                "stgcn_temporal_k":   STGCN_TEMPORAL_K,
                "stgcn_use_velocity": STGCN_USE_VELOCITY,
                # Preprocessing flags (must match at inference time)
                "preprocess_hip_center":     PREPROCESS_HIP_CENTER,
                "preprocess_scale_norm":     PREPROCESS_SCALE_NORM,
                "preprocess_use_visibility": PREPROCESS_USE_VISIBILITY,
                # Generic
                "dropout":         DROPOUT,
                "sequence_length": SEQUENCE_LENGTH,
            }, MODEL_SAVE_PATH)

        scheduler.step()

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.2%}")
    print(f"[SAVED] {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
