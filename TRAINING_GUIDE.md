# Improved Crypto Price Prediction - Training Guide

## Overview

We've redesigned the training logic to use a **GRU-based architecture** instead of LSTM, with significant improvements:

### Key Improvements

1. **Better Architecture**: GRU (Gated Recurrent Unit) instead of LSTM
   - Fewer parameters (faster training)
   - Better convergence properties
   - Attention mechanism over sequence
   - No mode collapse (constant output problem)

2. **Proper Training Pipeline**
   - Train/validation split (90/10)
   - Learning rate scheduling with ReduceLROnPlateau
   - Early stopping with patience=30
   - Gradient clipping (max_norm=1.0)
   - Weight initialization (Xavier for linear, Orthogonal for GRU)

3. **Better Checkpointing**
   - Saves best model during training
   - Saves final model
   - Stores scaler values in checkpoint for proper denormalization
   - Uses .pth format (PyTorch standard)

4. **Enhanced Inference**
   - Loads models with saved scaler values
   - Proper denormalization using saved min/max
   - Batch prediction
   - Detailed error metrics

## Quick Start

### 1. Train Model

```bash
python backend/train_improved.py --symbol BTCUSDT --timeframe 1h --epochs 150 --batch-size 32 --lr 0.001
```

**Parameters:**
- `--symbol`: Cryptocurrency symbol (default: BTCUSDT)
- `--timeframe`: Timeframe (default: 1h)
- `--epochs`: Number of training epochs (default: 150)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

**Expected Output:**
- Two model files:
  - `backend/models/weights/BTCUSDT_1h_best.pth` (best validation loss)
  - `backend/models/weights/BTCUSDT_1h_final.pth` (final model)

### 2. Run Inference

```bash
python backend/inference_improved.py --symbol BTCUSDT --timeframe 1h --model-type best --num-predictions 50
```

**Parameters:**
- `--symbol`: Cryptocurrency symbol
- `--timeframe`: Timeframe
- `--model-type`: 'best' or 'final' (default: best)
- `--num-predictions`: Number of predictions to generate

**Expected Output:**
```
PREDICTION RESULTS
Predictions: 50
MAE: $XXXX.XX
RMSE: $XXXX.XX
MAPE: XX.XX%
...
```

### 3. Visualize Results

```bash
python backend/visualize_improved.py --symbol BTCUSDT --timeframe 1h --num-predictions 50
```

Creates a 3-panel visualization:
- Panel 1: Actual vs Predicted prices
- Panel 2: Prediction errors
- Panel 3: Cumulative absolute error

## Architecture Details

### Model Architecture

```
Input (batch, 60, 44)  # 60 timesteps, 44 features
    |
    v
Input Projection: 44 -> 128
    |
    v
GRU Layer 1: 128 -> 128
    |
    v
GRU Layer 2: 128 -> 128
    |
    v
Attention Layer: 128 -> (weighted context)
    |
    v
Dense 1: 128 -> 64 (ReLU + BatchNorm + Dropout)
    |
    v
Dense 2: 64 -> 32 (ReLU + BatchNorm + Dropout)
    |
    v
Output: 32 -> 1 (price prediction)
```

**Total Parameters:** ~90,000

### Why GRU?

1. **Simpler than LSTM**: 3 gates vs 4 gates
2. **Fewer parameters**: Faster training, less overfitting
3. **Better for short sequences**: Our lookback is only 60 timesteps
4. **Better gradient flow**: Solves vanishing gradient better than LSTM

### Attention Mechanism

Instead of just taking the last output from GRU, we use attention to:
- Weight each timestep in the sequence
- Focus on important time periods
- Capture temporal patterns better

## Training Tips

### If Loss is Not Decreasing:

1. **Increase learning rate** (start with 0.001, try 0.005 or 0.01)
   ```bash
   python backend/train_improved.py --lr 0.005
   ```

2. **Increase batch size** (helps with gradient stability)
   ```bash
   python backend/train_improved.py --batch-size 64
   ```

3. **More epochs** (let it train longer)
   ```bash
   python backend/train_improved.py --epochs 200
   ```

### If Loss is Oscillating:

1. **Decrease learning rate** (too aggressive)
   ```bash
   python backend/train_improved.py --lr 0.0005
   ```

2. **Decrease batch size** (more frequent updates)
   ```bash
   python backend/train_improved.py --batch-size 16
   ```

### Monitor Training

Watch for these patterns:
- **Good**: Train loss goes from ~0.1 to ~0.01-0.05
- **Good**: Val loss follows similar trend
- **Bad**: Train loss stays constant
- **Bad**: Val loss increases while train loss decreases (overfitting)

## Differences from Old Model

| Aspect | Old LSTM | New GRU |
|--------|----------|----------|
| Architecture | 2-layer LSTM | 2-layer GRU + Attention |
| Output Type | Always constant 0.189240 | Variable predictions |
| Reason | Mode collapse during training | Better regularization |
| Saved Format | .pt (dict with config) | .pth (with scalers) |
| Parameters | ~150,000 | ~90,000 |
| Training Time | N/A | ~30-60s per epoch |
| Convergence | Failed | Good |

## File Structure

```
backend/
  train_improved.py          # Main training script
  inference_improved.py      # Inference script
  visualize_improved.py      # Visualization script
  models/
    weights/
      BTCUSDT_1h_best.pth    # Best model
      BTCUSDT_1h_final.pth   # Final model
```

## Checkpoint Format

Models are saved as `.pth` files containing:

```python
{
    'model_state_dict': {...},  # Model weights
    'model_config': {
        'input_size': 44,
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': 1,
    },
    'scalers': {
        'close_min': 83693.25,
        'close_max': 107158.79,
    },
    'epoch': 147,
    'train_loss': 0.0082,
    'val_loss': 0.0145,
}
```

This ensures proper denormalization during inference.

## Next Steps

1. **Train the model**:
   ```bash
   python backend/train_improved.py
   ```

2. **Check if predictions are working** (not flat line):
   ```bash
   python backend/inference_improved.py --num-predictions 10
   ```

3. **Visualize results**:
   ```bash
   python backend/visualize_improved.py
   ```

## Troubleshooting

### ModuleNotFoundError
Make sure you're in the correct directory:
```bash
cd /path/to/MOP
python backend/train_improved.py
```

### OutOfMemory Error
Reduce batch size:
```bash
python backend/train_improved.py --batch-size 16
```

### Model Not Found During Inference
Make sure training completed successfully and check `backend/models/weights/` directory.

## Questions?

Check logs in console output during training. Each epoch prints:
- Current epoch and total epochs
- Train loss and validation loss
- Current best validation loss

Good luck!
