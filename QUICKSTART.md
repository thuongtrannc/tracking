# Quick Start Guide: Model-Based IMM

This guide helps you quickly get started with the Model-Based Interactive Multiple Model (IMM) tracking system.

## Overview

The Model-Based IMM uses a GRU neural network to predict which motion model (CA, CV, or CT) is most appropriate at each timestep, instead of using a fixed transition matrix.

## Quick Test

### 1. Test Without Neural Network (Likelihood-based)

```bash
cd /workspace/tracking
python -c "
from core.imm_model_based import ModelBasedIMM
import numpy as np

# Load data
data = np.loadtxt('data/imm_single.txt', delimiter=',')
observations = data[:100, [0, 1, 6, 7]]  # First 100 samples
ground_truth = data[:100, :8]

# Initialize without trained model (uses likelihood)
imm = ModelBasedIMM('configs/imm.json', model_path=None, device='cpu')

# Run tracking
estimates = []
for t in range(len(observations)):
    imm.update(observations[t])
    estimates.append(imm.get_estimate())

estimates = np.array(estimates)
errors = estimates - ground_truth
rmse = np.sqrt(np.mean(errors ** 2))
print(f'RMSE: {rmse:.6f}')
"
```

### 2. Train the Neural Network

```bash
# Train for 50 epochs (quick test)
python tools/train.py --epochs 50 --batch-size 64 --lr 0.001

# Train for full training (200 epochs)
python tools/train.py --epochs 200 --batch-size 32 --lr 0.001
```

### 3. Test with Trained Model

```bash
python tools/train.py --test-only --model-path models/imm_gru_model.pth
```

## Training Details

The training process:
1. Generates features from all 2001 samples in `data/imm_single.txt`
2. Splits into 80% training, 20% validation
3. Trains GRU network to predict model probabilities
4. Saves best model based on validation loss

## Input Features

For each model (CA, CV, CT), we compute:
- **δy_tilde**: Innovation (observation - prediction)
- **δy**: Observation difference (temporal)
- **δx_tilde**: State evolution difference
- **δx**: Update correction

Total input: `(3 models, 24 features per model)`

## Architecture Options

### Simple GRU (Default - Recommended)
```
Input (3×24) → Flatten → GRU(72→128) → FC → Softmax → (3,)
```
- Faster training
- Good generalization
- ~185K parameters

### Complex GRU
```
Input (3×24) → 3×GRU(24→64) → Fusion → FC → Softmax → (3,)
```
- Model-specific processing
- Better for complex scenarios
- More parameters

Use with: `python tools/train.py --simple-gru False`

## Troubleshooting

### NaN Values During Testing

If you see NaN values, the neural network might be producing extreme features. The system will automatically fall back to likelihood-based updates with this message:
```
Warning: NaN/Inf in features, using likelihood-based update
```

### Covariance Matrix Explosion

This can happen with aggressive neural network predictions. Solutions:
1. Train for more epochs with smaller learning rate
2. Add regularization to covariance matrices (already implemented)
3. Use likelihood-based mode (no neural network)

### Model Not Loading

If you get attribute errors when loading, retrain the model:
```bash
rm models/imm_gru_model.pth
python tools/train.py --epochs 100
```

## Performance Metrics

Expected performance:
- **Likelihood-based IMM**: RMSE ~35-40
- **Model-based IMM (untrained)**: Similar to likelihood
- **Model-based IMM (trained)**: Should improve with proper training

## Files Structure

```
core/
  ├── imm_model_based.py      # Main implementation
  └── kalman_filter.py         # Kalman filter for each model

tools/
  └── train.py                 # Training script with GRU networks

configs/
  └── imm.json                 # Motion model configurations

data/
  └── imm_single.txt           # Training/testing data

test/
  └── test_imm_model_based.py  # Test scripts
```

## Next Steps

1. Run the test without neural network to verify base system works
2. Train the neural network with your desired parameters
3. Evaluate performance on test data
4. Adjust hyperparameters as needed

## Advanced Usage

### Custom Neural Network

```python
from tools.train import SimpleGRUModelNet
from core.imm_model_based import ModelBasedIMM
import torch

# Create custom model
model = SimpleGRUModelNet(
    obs_dim=4,
    state_dim=8, 
    hidden_dim=128,
    num_models=3,
    num_layers=2
)

# Train or load weights
# model.load_state_dict(torch.load('your_weights.pth'))

# Use with IMM
imm = ModelBasedIMM('configs/imm.json', model_path=None)
imm.set_model(model)
```

### Batch Processing

```python
from tools.train import generate_training_data

# Generate features for analysis
features, labels = generate_training_data(
    'data/imm_single.txt',
    'configs/imm.json'
)

print(f"Features shape: {features.shape}")  # (2001, 3, 24)
print(f"Labels shape: {labels.shape}")      # (2001, 3)
```

## References

- KalmanNet: Neural Network Aided Kalman Filtering
- Interactive Multiple Model (IMM) Algorithm
- GRU: Gated Recurrent Units for Sequence Modeling
