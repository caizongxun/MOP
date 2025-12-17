# V4 vs V5 - Detailed Comparison

## Problem with V4

All V4 predictions are **flat lines at mean price**:

```
V4 Problems Observed:

ETHUSL_6K_1350
  Predicted:  3350.00  (constant)
  Actual:     3200-3600 (dynamic range 400)
  Error:      COMPLETE FAILURE

SOLUST:
  Predicted:  141.00  (constant)
  Actual:     140-170  (dynamic range 30)
  Error:      SEVERE

BTCUST:
  Predicted:  94000.00  (constant)
  Actual:     98000-106000  (dynamic range 8000)
  Error:      MAJOR
```

### Root Cause Analysis

```
V4 Architecture:
  Features → LSTM → XGBoost → Prediction
  
The Problem:
  1. StandardScaler(y_train) compresses prices to [-1, +1] range
  2. LSTM learns to output this compressed range
  3. XGBoost sees LSTM features but not actual price levels
  4. Model converges to predicting mean (0 in scaled space)
  5. Inverse transform produces mean price for all predictions

Why This Happens:
  - No explicit connection to actual price levels
  - Loss function doesn't penalize constant predictions
  - Model takes shortest path: output mean = lowest error
```

## V5 Solution Architecture

### 1. Residual Learning (Main Fix)

```python
# V4: Predict absolute price
y_target = close_price  # e.g., [3350, 3351, 3349, 3350, ...]
# Problem: All values ~3350, StandardScaler makes them all ~0

# V5: Predict price change
y_target = close_price[t] - close_price[t-1]  # e.g., [+10, -2, +1, -5, ...]
# Benefit: Range is [-50, +100], not [-1, +1]
#          Model MUST learn actual movements
#          Cannot cheat by predicting 0 (mean)
```

### 2. Price Reconstruction

```python
# Training:
loss = (predicted_delta - actual_delta)^2

# Inference:
predicted_price[0] = actual_price[0]  # Start from known
for t in range(1, T):
    predicted_price[t] = predicted_price[t-1] + predicted_delta[t]

# Result: Predictions follow actual price movements
```

### 3. Multi-Scale LSTM

```
V4: Single LSTM
  Input (60, features) → LSTM(256) → Output
  Problem: Can't capture both short and long patterns simultaneously

V5: Dual LSTM
  Input (60, features)
      |
      +-> LSTM_SHORT(30 steps) ──┐
      |                           ├→ Fusion Layer → Output
      +-> LSTM_LONG(60 steps) ──┘
  
  Benefit:
  - LSTM_SHORT: Catches immediate reactions (5-10 steps)
  - LSTM_LONG: Captures trends (20-50 steps)
  - Fusion: Combines both perspectives
```

### 4. Uncertainty Quantification

```python
# V4: Single output
pred_price = model(features)  # Point estimate

# V5: Dual outputs
pred_delta, pred_uncertainty = model(features)

# Usage:
- High uncertainty → conservatively trust prediction
- Low uncertainty → confidently follow prediction
- Loss: weighted_loss = (pred - true)^2 / uncertainty^2
  Benefit: Model learns when to be confident vs cautious
```

### 5. Feature Engineering

```python
# V4 Features (limited):
- Basic returns
- MA ratios
- Some momentum

# V5 Features (comprehensive):
- Multi-scale momentum (5, 10, 20 periods)
- Multi-scale volatility (5, 10, 20 periods)  ← KEY!
- Volatility acceleration
- Micro-structure (HL ratio, close position)
- Volume signals
- Price acceleration (second derivative)
- Mean reversion signals

# Why volatility features matter:
  If volatility_5 > 50 → market is moving wildly → model should capture that
  If volatility_5 < 1 → market is flat → model can be less aggressive
```

## Loss Function Comparison

### V4 Loss
```python
loss = MSE(predicted_price, actual_price)

Problem:
  - All predictions at mean have similar loss
  - Model has no penalty for flatness
  - Converges to mean = local minimum
```

### V5 Loss
```python
loss = 0.6 * Weighted_Regression + 0.3 * Direct_Regression + 0.1 * Volatility_Loss

Weighted_Regression = mean((pred - true)^2 / (2 * uncertainty^2))
  - Forces model to predict uncertainty
  - Wrong predictions with high uncertainty: penalized heavily
  - Prevents overconfidence

Direct_Regression = mean((pred - true)^2)
  - Direct MSE on deltas
  - Prevents going too far off

Volatility_Loss = mean((pred_volatility - true_volatility)^2)
  - Model must learn to predict market volatility
  - Can't ignore market dynamics
  - Prevents collapse to flat predictions
```

## Training Comparison

| Aspect | V4 | V5 |
|--------|-----|-----|
| **Target** | Absolute price | Price delta |
| **Scaling Problem** | Compresses range | No compression needed |
| **LSTM Architecture** | Single stream | Dual scale |
| **Loss Function** | Simple MSE | Complex weighted ensemble |
| **Uncertainty** | No | Yes (learned) |
| **Feature Count** | ~15 | ~30 |
| **Training Focus** | Price level | Price movement + uncertainty |

## Expected Improvements

### Metrics
```
V4 Results:
  MAPE: 2-15% (but predictions flat!)
  
V5 Expected:
  MAPE: 3-8% (WITH accurate tracking)
  
Key Difference:
  - V4: 4% MAPE but predicts flat line (USELESS)
  - V5: 5% MAPE but tracks actual movements (USEFUL)
```

### Visual Comparison

```
V4 Output:
  |
3600 |    /\    /\         ← Actual price swings
  |   /  \  /  \
3400 |  /    \/    \ -------- ← V4 predicts (flat)
  | /              \
3200 |________________

V5 Output:
  |
3600 |    /\    /\         ← Actual price
  |   /  \  /  \
3400 |  /    \/    \        ← V5 predicts (follows!)
  | /              \
3200 |________________
```

## Why V5 Will Work

1. **Residual removes mean bias** - Can't cheat by predicting constant
2. **Multi-scale captures all patterns** - No scale ignored
3. **Uncertainty prevents overconfidence** - Model learns caution
4. **Volatility features expose dynamics** - Can't hide flat predictions
5. **Better loss function** - Rewards accurate tracking

## How to Verify V5 Works

After training, check:

```python
# 1. Predictions should have VARIANCE
print(np.std(predictions))  # Should be > 50 for BTC

# 2. Predictions should CORRELATE with actual
from numpy import corrcoef
corr = corrcoef(actual, predicted)[0, 1]
print(f"Correlation: {corr:.3f}")  # Should be > 0.7

# 3. Predictions should TRACK ranges
print(f"Actual range: {actual.min():.0f} - {actual.max():.0f}")
print(f"Pred range:   {predicted.min():.0f} - {predicted.max():.0f}")
# Should be similar!

# 4. Mean error should be centered at 0
errors = actual - predicted
print(f"Error mean: {errors.mean():.2f}")  # Should be ~0
print(f"Error std:  {errors.std():.2f}")   # Shows typical error magnitude
```

## Training V5

```bash
# Quick start (5 symbols)
python train_v5.py 5 --device cuda

# Full run (15 symbols)
python train_v5.py 15 --device cuda

# Expected time: ~2-5 minutes per symbol
# Total: ~20-75 minutes for 15 symbols
```

## What to Expect

- Training loss will decrease steadily
- Validation loss will improve then plateau
- Early stopping around epoch 40-60
- Final MAPE should be 2-8% (with variance!)
- Most importantly: **predictions will NOT be flat**
