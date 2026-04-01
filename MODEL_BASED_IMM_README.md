# Model-Based Interactive Multiple Model (IMM) Tracking

## Overview

This implementation extends the traditional IMM tracking framework with a neural network-based approach inspired by KalmanNet. Instead of using a fixed transition matrix to determine model probabilities, we use a GRU-based neural network that learns to predict the most appropriate motion model based on observation and state differences.

## Key Features

- **Data-driven model selection**: Neural network learns optimal model probabilities from data
- **KalmanNet-inspired features**: Uses innovation, observation, and state differences
- **Multiple motion models**: Supports Constant Acceleration (CA), Constant Velocity (CV), and Constant Turn (CT)
- **End-to-end differentiable**: Entire pipeline can be trained using gradient descent

## Architecture

### 1. Input Features

For each of the 3 motion models (CA, CV, CT), we compute 4 key difference terms at each timestep:

1. **δy_tilde** (Innovation): `z - H·x_pred`
   - Difference between actual observation and predicted observation
   - Dimension: `obs_dim = 4` (x, y, width, length)

2. **δy** (Observation difference): `z_t - z_{t-1}`
   - Temporal change in observations
   - Dimension: `obs_dim = 4`

3. **δx_tilde** (Forward evolution difference): `x_pred - x_{t-1}`
   - How the state evolved according to the motion model
   - Dimension: `state_dim = 8` (x, y, v, dv, w, dw, width, length)

4. **δx** (Forward update difference): `x_hat - x_pred`
   - Correction applied during Kalman filter update
   - Dimension: `state_dim = 8`

**Total input per model**: `2 × obs_dim + 2 × state_dim = 2×4 + 2×8 = 24`

**Network input shape**: `(batch_size, num_models=3, feature_dim=24)`

### 2. Neural Network Architecture

We provide two GRU-based architectures:

#### SimpleGRUModelNet (Recommended)
```
Input (3, 24) → Flatten → (72,)
       ↓
GRU(72 → 128) with 2 layers
       ↓
FC(128 → 64) → ReLU → Dropout
       ↓
FC(64 → 3) → Softmax
       ↓
Output: u ∈ R³ (model probabilities)
```

#### GRUModelNet (Advanced)
```
Input (3, 24)
       ↓
Separate GRU for each model (24 → 64)
       ↓
Per-model FC layers (64 → 32 → 16)
       ↓
Fusion layer: Concat → FC(48 → 64) → FC(64 → 32)
       ↓
Output layer: FC(32 → 3) → Softmax
       ↓
Output: u ∈ R³ (model probabilities)
```

### 3. Model-Based IMM Algorithm

```python
for each observation z_t:
    # 1. Compute features for all models
    for i in [CA, CV, CT]:
        features[i] = compute_model_differences(z_t, model_i)
    
    # 2. Predict model probabilities using neural network
    u = GRU_network(features)  # Shape: (3,)
    
    # 3. State mixing (interaction)
    for i in [CA, CV, CT]:
        X_mix[i] = weighted_sum(all model states, using u)
        P_mix[i] = weighted_covariance(all model covariances, using u)
    
    # 4. Update each model with mixed states
    for i in [CA, CV, CT]:
        model[i].update(z_t, P_mix[i], X_mix[i])
    
    # 5. Compute final estimate
    X_hat = u[0]·X_CA + u[1]·X_CV + u[2]·X_CT
```

## Usage

### Training

```bash
# Train with default settings
python tools/train.py

# Train with custom parameters
python tools/train.py --epochs 300 --batch-size 64 --lr 0.0005

# Use complex GRU architecture
python tools/train.py --simple-gru False

# Test only (requires pre-trained model)
python tools/train.py --test-only --model-path models/imm_gru_model.pth
```

### Testing

```python
from core.imm_model_based import ModelBasedIMM
import numpy as np

# Load data
data = np.loadtxt('data/imm_single.txt', delimiter=',')
observations = data[:, [0, 1, 6, 7]]  # x, y, width, length

# Initialize model-based IMM with trained model
imm = ModelBasedIMM(
    config_file='configs/imm.json',
    model_path='models/imm_gru_model.pth',
    device='cpu'
)

# Run tracking
estimates = []
model_probs = []

for t in range(len(observations)):
    z = observations[t]
    imm.update(z)
    estimates.append(imm.get_estimate())
    model_probs.append(imm.get_model_prob().copy())

estimates = np.array(estimates)
model_probs = np.array(model_probs)
```

### Without Trained Model (Likelihood-based)

```python
# Initialize without model (falls back to traditional IMM)
imm = ModelBasedIMM(
    config_file='configs/imm.json',
    model_path=None,  # No neural network
    device='cpu'
)

# Will use likelihood-based model probability updates
for t in range(len(observations)):
    z = observations[t]
    imm.update(z)
```

## Training Data Format

The training data (`data/imm_single.txt`) contains simulated trajectories with format:
```
x, y, v, vx, vy, w, width, length
```

Each row represents one timestep with:
- `x, y`: Position
- `v`: Velocity magnitude
- `vx, vy`: Velocity components
- `w`: Heading angle
- `width, length`: Object dimensions

Observations are extracted as: `[x, y, width, length]`

## Training Process

1. **Feature Generation**: Process all observations through each motion model independently to generate features
2. **Label Generation**: Compute model probabilities based on ground truth tracking errors
3. **Network Training**: Train GRU network using KL divergence loss
4. **Validation**: Monitor validation loss and save best model
5. **Testing**: Evaluate on full dataset and compute RMSE

### Loss Function

We use KL divergence to compare predicted and target probability distributions:
```python
loss = KL_divergence(log(predicted_probs), target_probs)
```

## Motion Models

### 1. Constant Velocity (CV)
```
x[k+1] = x[k] + v·cos(w)·dt
y[k+1] = y[k] + v·sin(w)·dt
v[k+1] = v[k]
w[k+1] = w[k]
```

### 2. Constant Acceleration (CA)
```
x[k+1] = x[k] + v·cos(w)·dt + 0.5·dv·cos(w)·dt²
y[k+1] = y[k] + v·sin(w)·dt + 0.5·dv·sin(w)·dt²
v[k+1] = v[k] + dv·dt
w[k+1] = w[k]
```

### 3. Constant Turn (CT)
```
x[k+1] = x[k] + (v/dw)·(-sin(w) + sin(w + dw·dt))
y[k+1] = y[k] + (v/dw)·(cos(w) - cos(w + dw·dt))
v[k+1] = v[k]
w[k+1] = w[k] + dw·dt
```

## Files

- `core/imm_model_based.py`: Model-based IMM implementation
- `tools/train.py`: Training script with GRU architectures
- `test/test_imm_model_based.py`: Testing and comparison scripts
- `configs/imm.json`: Configuration for motion models and parameters
- `data/imm_single.txt`: Training/testing data

## Performance Metrics

The system reports:
- **RMSE per state dimension**: Position, velocity, acceleration, etc.
- **Overall RMSE**: Average across all state dimensions
- **Model probabilities**: Average probability for each motion model
- **Training/validation loss**: KL divergence during training

## Advantages over Traditional IMM

1. **Adaptive**: Learns optimal model switching from data
2. **No manual tuning**: Transition matrix not required
3. **Better generalization**: Can adapt to unseen motion patterns
4. **End-to-end optimization**: All parameters learned jointly

## References

This implementation is inspired by:
- KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics
- Traditional Interactive Multiple Model (IMM) filtering
- GRU-based sequence modeling

## License

See main repository license.
Input: (batch_size, num_models, feature_dim)
  ↓
3 parallel GRU layers (one per model)
  - Input: (batch_size, seq_len, feature_dim)
  - Hidden: 64 units
  ↓
Cross-model attention layer
  - Multi-head attention (4 heads)
  - Allows models to share information
  ↓
3 parallel FC layers
  - Hidden: 32 units → 1 output per model
  ↓
Concatenate → Softmax
  ↓
Output: (batch_size, 3) - model probabilities
```

#### Key Features
- **Separate GRU for each model**: Captures temporal dynamics specific to each motion model
- **Cross-attention mechanism**: Enables information sharing between models
- **Softmax output layer**: Ensures probabilities sum to 1

### 3. Training Pipeline (`tools/train.py`)

#### Data Generation
```python
features, labels = generate_training_data(data_file, config_file)
```
- Reads ground truth from `data/imm_single.txt`
- Extracts observations: `[x, y, width, length]`
- Computes model differences for each time step
- Generates labels based on model-specific tracking errors

#### Training Process
1. **Data Split**: 80% training, 20% validation
2. **Loss Function**: KL Divergence (for probability distributions)
3. **Optimizer**: Adam with learning rate scheduling
4. **Batch Size**: 32
5. **Epochs**: 200 (configurable)

#### Training Command
```bash
cd /home/thuong/source/tracking
python tools/train.py
```

## Usage

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Training a New Model

```python
from tools.train import train_model

# Train with custom parameters
model, history = train_model(
    config_file="configs/imm.json",
    data_file="data/imm_single.txt",
    epochs=200,
    batch_size=32,
    learning_rate=0.001,
    save_path="models/imm_gru_model.pth"
)
```

### Using the Trained Model for Tracking

```python
from core.imm_model_based import ModelBasedIMM
import numpy as np

# Initialize IMM with trained model
imm = ModelBasedIMM(
    config_file="configs/imm.json",
    model_path="models/imm_gru_model.pth",
    device='cpu'
)

# Process observations
observations = np.loadtxt("data/imm_single.txt", delimiter=',')
obs_sequence = observations[:, [0, 1, 6, 7]]  # [x, y, width, length]

estimates = []
model_probs = []

for z in obs_sequence:
    imm.update(z)
    estimates.append(imm.get_estimate())
    model_probs.append(imm.get_model_prob())

estimates = np.array(estimates)
model_probs = np.array(model_probs)
```

### Testing the Model

```python
from tools.train import test_model

# Evaluate model performance
estimates, model_probs = test_model(
    config_file="configs/imm.json",
    data_file="data/imm_single.txt",
    model_path="models/imm_gru_model.pth"
)
```

## Data Format

### Input Data (`data/imm_single.txt`)
Each line contains 8 comma-separated values:
```
x, y, v, vx, vy, w, width, length
```
- `x, y`: Position
- `v, vx, vy`: Velocity components
- `w`: Yaw rate
- `width, length`: Object dimensions

### Model Configuration (`configs/imm.json`)
Defines the motion models, initial conditions, and noise parameters:
- **CV (Constant Velocity)**: Simple linear motion
- **CA (Constant Acceleration)**: Motion with acceleration
- **CT (Coordinated Turn)**: Curved trajectory motion

## Motion Models

1. **CA (Constant Acceleration)**: Index 0
   - Assumes constant acceleration in velocity
   - Good for acceleration/deceleration phases

2. **CV (Constant Velocity)**: Index 1
   - Assumes constant velocity
   - Good for straight-line motion

3. **CT (Coordinated Turn)**: Index 2
   - Assumes constant turn rate
   - Good for curved trajectories

## Architecture Comparison

### Traditional IMM
```
Model Probabilities ← Fixed Transition Matrix × Likelihood
```

### Model-Based IMM (This Implementation)
```
Model Probabilities ← Neural Network(Model Differences)
```

## Advantages of Model-Based Approach

1. **Adaptive**: Learns optimal model switching from data
2. **Context-aware**: Uses multiple difference terms to capture dynamics
3. **Improved accuracy**: Can learn complex relationships between observations and models
4. **No manual tuning**: Transition matrix learned from data

## Output

After training:
- **Model file**: `models/imm_gru_model.pth`
- **Training history**: `models/imm_gru_model_history.json`
- **Performance metrics**: RMSE per state dimension, average model probabilities

## References

This implementation is inspired by:
- KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics
- Interactive Multiple Model (IMM) algorithm for maneuvering target tracking

## File Structure

```
tracking/
├── core/
│   ├── imm.py                    # Traditional IMM implementation
│   ├── imm_model_based.py        # Model-based IMM (NEW)
│   └── kalman_filter.py          # Kalman filter for each model
├── tools/
│   └── train.py                  # Training script (NEW)
├── configs/
│   └── imm.json                  # Configuration file
├── data/
│   └── imm_single.txt            # Training data
├── models/                       # Saved models (created during training)
├── requirements.txt              # Python dependencies (NEW)
└── MODEL_BASED_IMM_README.md     # This file (NEW)
```
