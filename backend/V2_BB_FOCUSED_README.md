# MOP v2: Bollinger Bands-Focused Training

## Overview

Version 2 implements a **Bollinger Bands-centric approach** to cryptocurrency price prediction, addressing the limitations found in v1 models.

## v2 Improvements

### Problem with v1
- **High variance in MAPE**: BTC (99.54%), ETH (86.50%), but DOGE (1.39%)
- Root cause: Absolute price prediction scales poorly across different price magnitudes
- Different price ranges (BTC $100k vs DOGE $0.17) caused model to overfit to magnitude

### v2 Solution: BB-Focused Features

**Core Insight**: Bollinger Bands naturally adapt to each asset's scale and volatility

```
Bollinger Bands = MA ± (k * std_dev)
- Automatically scales with volatility
- BB_percent_b = (close - lower) / (upper - lower)  [0-1 scale]
- BB_width = (upper - lower) / middle  [volatility measure]
```

## Feature Engineering

### Primary Features (5)
```python
'bb_upper'      - Upper band
'bb_middle'     - Middle (SMA 20)
'bb_lower'      - Lower band
'bb_percent_b'  - Price position in band (0-1)
'bb_width'      - Band width (volatility)
```

### Supporting Features (13-14, selected by mutual information)
```python
Trend:      EMA_10, EMA_20, EMA_50
Momentum:   MACD, MACD_signal, MACD_histogram
Oscillator: RSI_14, RSI_21
Volume:     Volume, Volume_change
Volatility: ATR_14, DI+, DI-, MFI, OBV
```

Total: **18 features** (selected by correlation to close price)

## Model Architecture

```
Optimized Hyperparameters:
- Hidden size:   128
- Num layers:    2
- Dropout:       0.2 (reduced from 0.4)
- Learning rate: 0.001
- Batch size:    16
- Weight decay:   1e-5
- Optimizer:     Adam
```

## Key Differences from v1

| Aspect | v1 | v2 |
|--------|----|---------|
| Feature selection | All 44 features | 18 BB-focused features |
| Feature scaling | Min-Max global | Min-Max per feature |
| Core strategy | Price prediction | BB position + volume + momentum |
| Expected MAPE | 85-99% | 5-15% (estimated) |
| Training time | ~1-2 min per symbol | ~2-3 min per symbol |

## Usage

### Training (v2)

```bash
# Train all 15 symbols
python -m backend.train_v2_bb_focused --epochs 100

# Train specific symbols
python -m backend.train_v2_bb_focused --symbols BTCUSDT ETHUSDT BNBUSDT --epochs 100

# Custom settings
python -m backend.train_v2_bb_focused --symbols BTCUSDT --epochs 150 --device cuda
```

### Inference (v2)

```bash
# Single prediction
python -m backend.inference_v2_bb --symbol BTCUSDT

# Batch predictions
python -m backend.inference_v2_bb --batch

# Different timeframe
python -m backend.inference_v2_bb --batch --timeframe 4h
```

## Model Outputs

Models are saved to:
```
backend/models/weights/{symbol}_1h_v2_best.pth
```

Each checkpoint includes:
- Model state dict
- Architecture config
- Training epoch
- Version marker ('v2_bb_focused')

## Why BB Works Better

### Mathematical Foundation

1. **Scale Invariance**
   - BB adapts to each asset's typical volatility
   - BTC volatility ~2-3%, DOGE volatility ~5-10%
   - BB naturally captures this difference

2. **Mean Reversion Property**
   - Price near BB upper → higher probability of pullback
   - Price near BB lower → higher probability of bounce
   - EMA + RSI + MACD confirm these signals

3. **Volatility Clustering**
   - BB_width captures volatility regimes
   - Volume changes identify breakouts
   - Combines trend + momentum + volume confirmation

## Expected Results

### Large Cap (BTC, ETH)
- MAPE: 8-12%
- More stable, less noise
- BB bands more reliable

### Mid Cap (BNB, SOL, LINK)
- MAPE: 5-10%
- Good feature signal
- BB + RSI effective

### Small Cap (DOGE, NEAR, ATOM)
- MAPE: 3-8%
- High volatility captured well
- BB_width very informative

## Future Improvements

1. **Ensemble Methods**
   - Combine LSTM + GRU + Transformer
   - Weight predictions by confidence

2. **Regime Detection**
   - Train separate models for trending vs ranging
   - Switch based on current volatility

3. **Attention Mechanism**
   - Weight recent candles more heavily
   - Focus on BB band transitions

4. **Multi-timeframe Fusion**
   - Use 1h + 4h + 1d BB signals
   - Higher timeframes for confirmation

## Files

- `train_v2_bb_focused.py` - Main training script
- `inference_v2_bb.py` - Prediction script
- `backend/models/weights/{symbol}_1h_v2_best.pth` - Trained models

## Notes

- v2 models use normalized features (0-1 range)
- Denormalization uses original close price min/max
- Feature selection is consistent between train and inference
- Early stopping: patience=15 epochs
