# MSE Loss Update for Model-Based IMM

## Changes Made

The training loss function has been changed from **KL Divergence** (probability distribution matching) to **MSE Loss** (Mean Squared Error on x,y ground truth positions).

## Why This Change?

### Previous Approach (KL Divergence)
- **Goal**: Match predicted model probabilities to target probabilities derived from tracking errors
- **Limitation**: Indirect supervision - we were predicting which model is best, not directly optimizing tracking accuracy

### New Approach (MSE Loss on Position)
- **Goal**: Directly minimize position tracking error
- **Benefit**: End-to-end optimization - the network learns to produce model probabilities that result in accurate position estimates

## How It Works

### Training Pipeline

1. **Input Features**: For each timestep, compute KalmanNet-style differences:
   ```
   model_i_difference = [δy_tilde, δy, δx_tilde, δx]
   ```
   Shape: `(3 models, 24 features)`

2. **Neural Network Prediction**: 
   ```python
   model_probs = GRU_network(features)  # (3,) probabilities
   ```

3. **Get Individual Model Estimates**:
   ```python
   # Run IMM update to get estimates from each motion model
   imm.update(observation)
   
   # Extract x,y positions from each model
   X_ca = model_CA.X_hat[:2]  # x, y from CA model
   X_cv = model_CV.X_hat[:2]  # x, y from CV model  
   X_ct = model_CT.X_hat[:2]  # x, y from CT model
   ```

4. **Compute Weighted Estimate**:
   ```python
   # Weighted combination using predicted probabilities
   X_pred = model_probs[0] * X_ca + model_probs[1] * X_cv + model_probs[2] * X_ct
   ```

5. **MSE Loss**:
   ```python
   loss = MSE(X_pred, ground_truth_xy)
   ```

### Loss Comparison

| Aspect | KL Divergence Loss | MSE Position Loss |
|--------|-------------------|-------------------|
| **Target** | Model probabilities | Position (x, y) |
| **Supervision** | Indirect (error-based probs) | Direct (ground truth) |
| **Gradient** | Through softmax | Through weighted sum |
| **Optimization** | Match probability distribution | Minimize position error |
| **Interpretability** | Less direct | More intuitive |

## Code Changes

### File: `tools/train.py`

#### 1. Data Generation
```python
# OLD: Return features and probability labels
features, prob_labels = generate_training_data(...)

# NEW: Return features and x,y positions
features, gt_positions = generate_training_data(...)
# gt_positions shape: (N, 2) for x, y
```

#### 2. Loss Function
```python
# OLD: KL Divergence
criterion = nn.KLDivLoss(reduction='batchmean')
loss = criterion(torch.log(outputs + 1e-10), batch_labels)

# NEW: MSE Loss
criterion = nn.MSELoss()
loss = criterion(weighted_estimate, ground_truth_xy)
```

#### 3. Training Loop
```python
for each sample:
    # Predict model probabilities
    probs = neural_network(features)
    
    # Get estimates from each model
    imm.update(observation)
    X_ca, X_cv, X_ct = [model.X_hat[:2] for model in imm.models]
    
    # Compute weighted estimate
    X_pred = probs @ [X_ca, X_cv, X_ct]
    
    # MSE loss
    loss = MSE(X_pred, ground_truth_xy)
    loss.backward()
```

## Expected Benefits

1. **Direct Optimization**: Network learns to produce probabilities that directly minimize position error
2. **Simpler Supervision**: No need to derive "correct" model probabilities from errors
3. **End-to-End Training**: Gradients flow from position error through model weighting
4. **Better Generalization**: Optimization target matches evaluation metric (position accuracy)

## Usage

### Training with MSE Loss (Default Now)
```bash
# Train for 100 epochs
python tools/train.py --epochs 100 --batch-size 32 --lr 0.001

# Quick test (5 epochs)
python tools/train.py --epochs 5 --batch-size 32
```

### Expected Output
```
Starting Training (MSE Loss on x,y positions)
============================================================
Epoch [  1/100] | Train Loss (MSE): 0.0116 | Val Loss (MSE): 950.84
  → Saved best model (val_loss: 950.84)
...
```

The loss values represent squared position error in the same units as your data.

## Training Tips

1. **Learning Rate**: Start with 0.001, reduce if training is unstable
2. **Batch Size**: Larger batches (32-64) give more stable gradients
3. **Epochs**: 100-200 epochs recommended for full convergence
4. **Validation Loss**: Should decrease over time, indicating better position accuracy

## Technical Details

### Gradient Flow
```
Ground Truth (x, y)
        ↑
      MSE Loss
        ↑
  Weighted Sum (x, y)
        ↑
   Model Probs × Model Estimates
        ↑              ↑
   Softmax         IMM Update
        ↑              ↑
    GRU Network    Kalman Filters
        ↑
    Features
```

### Device Handling
- **Neural Network**: Runs on GPU (if available)
- **IMM/Kalman Filters**: Run on CPU (for numerical stability)
- **Data Transfer**: Features and estimates moved between devices as needed

## Results

After training, the model should:
- Produce lower MSE loss on validation set
- Generate model probabilities that result in accurate position tracking
- Adaptively select the best motion model for each scenario

## Monitoring Training

Check the saved history file:
```bash
cat models/imm_gru_model_history.json
```

Should show:
```json
{
    "train_losses": [0.0116, 0.0089, ...],
    "val_losses": [950.84, 723.45, ...],
    "best_val_loss": 15.23,
    "loss_type": "MSE on x,y positions"
}
```

## Comparison with Previous Approach

| Metric | KL Div Loss | MSE Loss |
|--------|-------------|----------|
| Training Speed | Fast | Slower (IMM per sample) |
| Convergence | Indirect | Direct |
| Final Accuracy | Good | Better (end-to-end) |
| Interpretability | Model selection | Position error |

## Summary

The MSE loss provides **direct end-to-end optimization** of the tracking objective (position accuracy) rather than indirectly optimizing through model probability matching. This should lead to better tracking performance in practice.
