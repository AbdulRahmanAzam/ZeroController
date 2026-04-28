# Data Augmentation Guide

## Overview

The augmentation pipeline generates **5.1x more training samples** from your existing pose action data. Starting with **467 original samples**, it creates **1,868 augmented samples** for a total of **2,364 samples** across 9 action classes.

## Techniques Used

### 1. **Spatial Noise** (±2%)
Adds small Gaussian noise to joint coordinates (x, y, z), simulating natural sensor jitter.

### 2. **Skeleton Scaling** (±15%)
Uniformly scales all joints, simulating persons closer or farther from the camera.

### 3. **Temporal Jittering**
Subtly shifts frame timing via interpolation, creating natural speed variations without changing the action.

### 4. **Visibility Dropout**
Randomly marks joints as occluded (visibility = 0), simulating partial body visibility.

### 5. **Rotation Jitter** (±15°)
Rotates the skeleton around the z-axis (vertical), simulating camera angle variations.

### 6. **Time Stretching** (0.8–1.2x)
Compresses or expands the motion timeline while resampling to original 30-frame length.

### 7. **Mixup** (Same-Action Blending)
Blends two samples from the same action class: `new_seq = α × seq₁ + (1-α) × seq₂` with α ∈ [0.3, 0.7].
- Captures intermediate poses between two instances of the same action
- Adds natural motion continuity

### 8. **Horizontal Flipping** (Left ↔ Right Actions)
Mirrors x-coordinates and swaps symmetric joints for left/right punch and kick actions.

## Generated Dataset

| Action | Original | Augmented | Total |
|--------|----------|-----------|-------|
| idle | 51 | 204 | 255 |
| left_punch | 71 | 284 | 355 |
| right_punch | 41 | 164 | 205 |
| left_kick | 71 | 284 | 355 |
| right_kick | 42 | 168 | 210 |
| jump | 53 | 212 | 265 |
| block | 53 | 212 | 265 |
| forward | 42 | 168 | 210 |
| backward | 43 | 172 | 215 |
| **TOTAL** | **467** | **1,868** | **2,364** |

**Overall Multiplier: 5.1x**

## Running the Pipeline

```bash
python augment_data.py
```

The script will:
1. Mirror right_punch → left_punch and right_kick → left_kick (if not already present)
2. Generate 4 augmented samples per original for each action
3. Save augmented data to `data/raw/augmented/`

## Using Augmented Data for Training

To train with augmented data, update `train_model.py`:

```python
from pathlib import Path
import numpy as np

def load_dataset_with_augmentation(data_dir='data/raw', include_augmented=True):
    """Load original + augmented samples."""
    raw_dir = Path(data_dir)
    aug_dir = raw_dir / 'augmented'
    
    sequences, labels = [], []
    
    # Load original samples
    for action_idx, action in enumerate(ACTIONS):
        action_dir = raw_dir / action
        for npy_file in sorted(action_dir.glob('*.npy')):
            seq = np.load(npy_file)
            sequences.append(seq)
            labels.append(action_idx)
    
    # Load augmented samples (if available and requested)
    if include_augmented and aug_dir.exists():
        for action_idx, action in enumerate(ACTIONS):
            aug_action_dir = aug_dir / action
            if aug_action_dir.exists():
                for npy_file in sorted(aug_action_dir.glob('*.npy')):
                    seq = np.load(npy_file)
                    sequences.append(seq)
                    labels.append(action_idx)
    
    return np.array(sequences), np.array(labels)

# In main training code:
X, y = load_dataset_with_augmentation(include_augmented=True)
# Continue with train_test_split and model training...
```

## Augmentation Strategies & When to Use Each

### For Small Datasets (< 200 samples/action)
✅ Use **all augmentation techniques** (current approach)
- Spatial noise + scaling + temporal jitter + rotation

### For Medium Datasets (200–500 samples/action)  
✅ Reduce to **selective techniques**:
- Spatial noise (always safe)
- Temporal jittering (for subtle speed variations)
- Visibility dropout (simulates real sensor noise)

### For Large Datasets (> 500 samples/action)
⚠️ **Disable augmentation or use minimal**:
- Risk of overfitting to augmentation artifacts
- Focus on model regularization instead (dropout, L2)

## Advanced: Conditional VAE (Alternative Approach)

If you need **learned**, **semantic** augmentation:

```bash
pip install torch torchvision
# Then implement conditional_vae.py (available on request)
```

Benefits:
- Generates realistic novel sequences
- Learns action-specific distribution
- Better for downstream tasks (transfer learning)

Drawbacks:
- Requires training time (~1–2 hours)
- Needs more compute (GPU recommended)
- More complex pipeline

## Recommended Training Workflow

1. **Phase 1: Train on Original + Augmented Data**
   ```bash
   python augment_data.py  # Generate 5.1x augmented samples
   python train_model.py   # Train with --include-augmented
   ```

2. **Phase 2: Evaluate on Test Set**
   - Keep test set from **original data only** (no augmented samples in test)
   - Ensures fair evaluation of model generalization

3. **Phase 3: Fine-tune on Real Data** (Optional)
   - After initial training on augmented + original
   - Fine-tune on collected real gameplay data
   - Reduces domain gap

## Notes

- All augmented samples are saved with prefix `aug_` to distinguish from originals
- Visibility scores (4th channel) are preserved and updated correctly
- Coordinates remain normalized [0, 1] range throughout augmentation
- Mirror operation preserves skeleton connectivity

## Troubleshooting

**Q: "No augmented samples generated"**
- A: Check that `data/raw/{action}/*.npy` files exist
- Verify SEQUENCE_LENGTH in `config.py` matches saved data

**Q: "Augmentation too aggressive"**
- A: Adjust technique parameters in augment_data.py:
  - `std=0.02` → `std=0.01` for less spatial noise
  - `scale_range=(0.85, 1.15)` → `(0.9, 1.1)` for subtler scaling
  - `max_angle_deg=15` → `max_angle_deg=8` for smaller rotations

**Q: "Model underfits even with augmented data"**
- A: Check if augmented data is actually being loaded in training
- Try reducing model regularization (dropout, L2) if overfitting isn't present
- Verify augmented data quality with visualization

## References

- MediaPipe Pose: https://mediapipe.dev/solutions/pose
- Data Augmentation for Action Recognition: https://arxiv.org/abs/1904.04998
- ST-GCN Training Best Practices: https://arxiv.org/abs/1801.07455
