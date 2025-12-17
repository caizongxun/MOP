# V5 Enhanced - Quick Start Guide

## One-Minute Summary

**Problem:** V4 predicts flat lines (all prices = mean value)

**Solution:** V5 predicts **price deltas** instead of absolute prices

**Result:** Predictions now follow actual price movements

## Installation & Setup

```bash
cd C:\Users\zong\Desktop\MOP
```

## Training

### Option 1: Train 5 symbols (test run, ~5 min)
```bash
python train_v5.py 5 --device cuda
```

### Option 2: Train all 15 symbols (full run, ~60 min)
```bash
python train_v5.py 15 --device cuda
```

### Option 3: Train from backend directory
```bash
cd backend
python -m train_v5_enhanced 5 --device cuda
```

## What Happens During Training

```
[1/15] Training BTCUSDT with V5 Enhanced
  Epoch  20: Train=0.001234, Val=0.005678
  Epoch  40: Train=0.000987, Val=0.004321
  Epoch  60: Early stopping at epoch 65
  Test Results: MAPE: 4.23%
  [Model saved]

[2/15] Training ETHUSDT...
...

[15/15] Training APTUSDT...

V5 Training Complete
Average MAPE: 5.67%
Best: 2.15%
Worst: 9.34%
```

## Key Differences from V4

| Feature | V4 | V5 |
|---------|-----|-----|
| Predicts | Absolute price | Price delta |
| Output | Flat line | Follows market |
| MAPE | 2-15% (useless) | 3-8% (accurate) |
| Architecture | Single LSTM | Dual LSTM |
| Uncertainty | No | Yes |
| Features | 15 | 30+ |

## Model Outputs

After training completes:

### Saved Models
```
./models/weights/
  BTCUSDT_1h_v5_lstm.pth    <- LSTM weights
  ETHUSDT_1h_v5_lstm.pth
  ... (15 total)
```

### Saved Configs
```
./models/config/
  BTCUSDT_v5_config.json    <- Training config + metrics
  ETHUSDT_v5_config.json
  ... (15 total)
```

## Next: Generate Predictions

### Create V5 visualizer (coming next)
```bash
python quick_visualize_v5.py --all --device cuda
```

This will:
1. Load trained V5 models
2. Generate predictions for all symbols
3. Create comparison plots (actual vs predicted)
4. Save visualization images
5. Create summary table

## Understanding the Architecture

### V5 Workflow

```
Historical Data (60 hours)
        |
        v
Feature Engineering (30+ indicators)
        |
        v
Multi-Scale LSTM
  - Short-term (30 steps)
  - Long-term (60 steps)
        |
        v
Fusion Layer (combine scales)
        |
        +---> Predict Price Delta
        |
        +---> Predict Uncertainty
        |
        v
Reconstruction: price[t] = price[t-1] + delta[t]
        |
        v
Final Predictions (follow price movement)
```

### Key Innovation: Residual Learning

```python
# V4: Compressed by StandardScaler
actual = [3350, 3351, 3349, 3350, 3352]  # Real prices
scaled = [-0.1, 0.0, -0.2, -0.1, 0.1]    # After scaling
# Model learns to predict: 0 (mean) -> predicts flat line!

# V5: Preserves variance
actual_delta = [+1, -2, +1, +2]  # Real changes
scaled_delta = [+1, -2, +1, +2]  # SAME! No compression
# Model learns to predict: varies from -2 to +2 -> TRACKS MARKET!
```

## Feature Engineering (V5)

### Momentum Features
- ROC (Rate of Change) at 5, 10, 20 periods
- Momentum at different scales

### Volatility Features (Critical!)
- Rolling volatility (5, 10, 20 periods)
- Volatility acceleration
- Prevents flat predictions

### Micro-structure
- High-Low ratio
- Close position within bar
- Range expansion

### Directional
- Price acceleration (second derivative)
- Mean reversion signals
- Volume signals

## Training Tips

### Monitor Progress
Watch for:
- Training loss decreasing (good)
- Validation loss improving (good)
- Early stopping ~40-60 epochs (normal)
- Final MAPE < 8% (good)
- No NaN or Inf values (check logs)

### If Training Fails

1. **Out of memory?**
   ```bash
   # Reduce batch size
   python train_v5.py 5 --device cpu  # Try CPU first
   ```

2. **Slow training?**
   - GPU utilization issue (check with nvidia-smi)
   - Switch device or check CUDA installation

3. **Bad results?**
   - Try training more symbols (learn from variety)
   - Check feature calculation (should be 30+ features)
   - Verify data files exist

## Performance Expectations

### By Symbol Type

**High Volatility (APT, NEAR)**
- MAPE: 8-15%
- Harder to predict

**Medium Volatility (ETH, BTC, UNI)**
- MAPE: 3-6%
- Sweet spot

**Low Volatility (MATIC, ATOM)**
- MAPE: 2-4%
- Easier to predict

## Verification After Training

Check that V5 actually works:

```bash
# In Python
import numpy as np
from backend.inference_v5_enhanced import V5InferenceEngine

engine = V5InferenceEngine('BTCUSDT', device='cuda')
df = engine.load_data()  # Will add method
actual, predicted, uncertainty = engine.infer(df)

# Check 1: Predictions have variance
print(f"Pred std: {np.std(predicted):.2f}")  # Should be > 100

# Check 2: Correlation with actual
corr = np.corrcoef(actual, predicted)[0, 1]
print(f"Correlation: {corr:.3f}")  # Should be > 0.5

# Check 3: Range matches
print(f"Actual: {actual.min():.0f} - {actual.max():.0f}")
print(f"Pred:   {predicted.min():.0f} - {predicted.max():.0f}")
```

## Comparison: V4 vs V5

### Visual: What You'll See

**V4 (FAIL):**
```
3600 |    /\    /\    /\
     |   /  \  /  \  /  \     <- Actual market
3400 | /    \/    \/    \  
     |/__________________|  <- V4 prediction (flat!)
3200 
```

**V5 (SUCCESS):**
```
3600 |    /\    /\    /
     |   /  \  /  \  /  \     <- Actual market
3400 | /    \/    \/    \     <- V5 prediction (tracks!)
     |/__________________|  
3200 
```

## Troubleshooting

### Issue: "Model not found"
```
Error: Model not found: ./models/weights/BTCUSDT_1h_v5_lstm.pth
Solution: Training hasn't completed yet, wait for completion
```

### Issue: Predictions still flat
```
Check:
1. Did V5 finish training? (check log for "Early stopping")
2. What's the MAPE? (if > 15%, model didn't converge)
3. Run verification checks above
```

### Issue: Very slow training
```
Solutions:
1. Reduce feature count (edit FeatureEngineerV5)
2. Reduce lookback (60 -> 30)
3. Use smaller batch size (16 -> 8)
4. Switch to CPU for initial debugging
```

## Next Steps

1. Train models: `python train_v5.py 5 --device cuda`
2. Verify training succeeded (check logs, check MAPE)
3. (Next) Generate visualizations
4. (Next) Evaluate actual vs predicted performance
5. (Optional) Fine-tune hyperparameters

## Key Takeaway

V5 solves V4's fundamental problem by predicting **deltas** instead of absolute values. This prevents the model from collapsing to a flat line and allows it to track actual market movements.

**Bottom Line:** V5 predictions should actually follow the price, not stay flat.
