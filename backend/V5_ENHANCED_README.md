# V5 Enhanced - Next Generation Predictor

## Key Improvements Over V4

### 1. Residual Learning (Delta Prediction)
**Problem in V4:** Model predicted absolute prices, which got squeezed to mean value by StandardScaler

**V5 Solution:** 
- Predict **price deltas (changes)** instead of absolute prices
- Reconstruct final price by cumulative sum: `pred_price[t] = pred_price[t-1] + pred_delta[t]`
- Removes the bias toward mean, allows tracking real fluctuations

### 2. Multi-Scale LSTM Architecture
**Problem in V4:** Single LSTM couldn't capture both short and long-term patterns

**V5 Solution:**
```python
- lstm_short: Processes first 30% of sequence (quick reactions)
- lstm_long: Processes full sequence (trend understanding)
- Fusion layer: Combines both scales for better representation
```

### 3. Uncertainty Quantification
**New Capability:**
- Model predicts **confidence intervals** alongside predictions
- Predicts **volatility** to understand uncertainty level
- Loss function weights predictions by uncertainty
- High uncertainty → lower prediction weight → more conservative

### 4. Advanced Feature Engineering

#### Multi-Scale Momentum
```python
momentum_5, momentum_10, momentum_20
rate_of_change_5, rate_of_change_10, rate_of_change_20
```

#### Volatility Features (Critical)
```python
volatility_5, volatility_10, volatility_20  # Rolling volatility
vol_change_5, vol_change_10, vol_change_20  # Volatility acceleration
```
These help the model avoid predicting flat lines.

#### Micro-Structure
```python
hl_ratio: (High-Low) / Close  # Intra-bar volatility
close_position: How close is the bar to its range
range_expansion: Is the bar getting bigger or smaller
```

#### Price Acceleration
```python
acceleration_5, acceleration_10  # Second derivative of price
```
Detects momentum changes before they happen.

### 5. Advanced Loss Function

```python
Total Loss = 0.6 * Weighted_Regression + 0.3 * Direct_Regression + 0.1 * Volatility_Loss

Weighted_Regression = mean((pred - true)^2 / (2 * uncertainty^2))
  → Forces model to learn uncertainty
  → High uncertainty predictions get penalized less if wrong
  
Direct_Regression = mean((pred - true)^2)
  → Ensures reasonable predictions
  
Volatility_Loss = mean((pred_vol - true_vol)^2)
  → Model learns to predict volatility accurately
```

### 6. Better Regularization
- Weight decay: 1e-5 (prevents overfitting)
- Gradient clipping: max_norm=1.0 (stable training)
- Dropout: 0.2 (prevents co-adaptation)
- Learning rate scheduler: ReduceLROnPlateau (adaptive learning)

## Training Strategy

### Data Split
- 70% Training (for learning)
- 15% Validation (for early stopping)
- 15% Test (for final evaluation)

### Hyperparameters
```python
Lookback Window: 60 steps (60 hours of history)
Batch Size: 16
Learning Rate: 0.0005
Epochs: 150 (with early stopping at 25 patience)
Hidden Size: 192
Num LSTM Layers: 2
```

## Expected Performance

### MAPE Targets
- **Excellent:** < 2%
- **Good:** 2-5%
- **Acceptable:** 5-8%
- **Needs Work:** > 8%

### Why V5 Should Be Better
1. **Residual learning** removes mean-bias
2. **Multi-scale** captures all patterns
3. **Uncertainty** makes predictions adaptive
4. **Volatility features** prevent flat predictions
5. **Better loss function** trains model properly

## Training Command

```bash
cd C:\Users\zong\Desktop\MOP\backend

# Train 5 symbols
python train_v5_enhanced.py 5 --device cuda

# Train all 15 symbols
python train_v5_enhanced.py 15 --device cuda
```

## Monitoring Training

Watch the logs for:
1. **Epoch Loss Decreasing** - Good sign
2. **Validation Loss Improving** - Model generalizing
3. **Early Stop** - After 25 epochs without improvement (normal)
4. **Final MAPE** - Should be < 8% for most symbols

## What to Expect

### From V4 to V5
```
V4 Problems:
- ETHUSDT: Flat line at $3350, actual: $3200-3600 (FAIL)
- SOLUSDT: Flat line at $141, actual: $140-170 (FAIL)
- BTCUSDT: Flat line at $94k, actual: $98k-106k (FAIL)

V5 Should Achieve:
- ETHUSDT: Tracks 3200-3600 range (PASS)
- SOLUSDT: Tracks 140-170 range (PASS)
- BTCUSDT: Tracks 98k-106k range (PASS)
```

## Next Steps After Training

1. **Verify predictions stick to price range**
2. **Check error distribution** (should be normal, centered at 0)
3. **Analyze failures** (high volatility symbols might still struggle)
4. **Fine-tune** if needed:
   - Add more volatility features if still flat
   - Increase lookback if missing patterns
   - Reduce batch size if overfitting

## Architecture Diagram

```
Input (60, features)
       |
       +-----> LSTM_SHORT (30 steps) ---->
       |                                  |
       +-----> LSTM_LONG (60 steps) ----> Fusion Layer
       |                                  |
       +---> Process Full Sequence -----> |
                                         |
                                    Output Heads:
                                    - Price Delta
                                    - Uncertainty
```

## Key Success Factor

The **residual learning** is the game-changer. Instead of asking "what is the price?", V5 asks "what is the change in price?" This fundamentally changes what the model can learn.
