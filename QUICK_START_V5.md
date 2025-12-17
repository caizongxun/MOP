# V5 Enhanced Model - Quick Start Guide

## Overview

V5 is an improved version of V4 with:
- **Multi-scale LSTM**: Captures patterns at different time scales
- **Residual Learning**: Predicts price deltas instead of absolute values
- **Uncertainty Quantification**: Learns confidence intervals
- **Advanced Features**: 20+ engineered indicators
- **Unified Paths**: Uses PathConfig for consistent file handling

## Current Status

```
现有资源:
- V4 models: models/weights/*_v4_lstm.pth (已训练)
- Data: backend/data/raw/*.csv (已获取)
- Path system: backend/path_config.py (已设置)
- Training script: backend/train_v5_enhanced.py (已准备)
- Visualization: quick_visualize_v5.py (已准备)
```

## Step-by-Step Training

### Step 1: Train V5 Models

**Option A: Quick Test (2 symbols, ~5 minutes)**
```bash
python train_v5.py 2 --device cuda
```

**Option B: Medium (5 symbols, ~15 minutes)**
```bash
python train_v5.py 5 --device cuda
```

**Option C: Full Training (all 15 symbols, ~45 minutes)**
```bash
python train_v5.py --all --device cuda
```

**Option D: CPU Training (if no CUDA)**
```bash
python train_v5.py 5 --device cpu
```

### Step 2: Monitor Training Progress

During training, you'll see:
```
Training BTCUSDT... [1/15]
  Epoch  20: Train Loss=0.000234, Val Loss=0.000567
  Epoch  40: Train Loss=0.000189, Val Loss=0.000478
  Epoch  60: Train Loss=0.000145, Val Loss=0.000423
  Early stopping at epoch 78
  MAPE: 3.45%, MAE: $0.0234, RMSE: $0.0567
```

### Step 3: Verify Models Were Created

```bash
# Check if models exist
python quick_visualize_v5.py --show-paths

# You should see models in:
# C:\Users\zong\Desktop\MOP\backend\models\weights\*_v5_lstm.pth
```

### Step 4: Generate Visualizations

**After training completes:**

```bash
# Generate all visualizations
python quick_visualize_v5.py --all --device cuda
```

### Step 5: View Results

Open the visualizations:
```
results/visualizations/
├── BTCUSDT_predictions_v5_YYYYMMDD_HHMMSS.png
├── ETHUSDT_predictions_v5_YYYYMMDD_HHMMSS.png
├── ...
├── summary_table_v5_YYYYMMDD_HHMMSS.png
└── summary_results_v5_YYYYMMDD_HHMMSS.csv
```

## What You'll See

### Single Symbol Plot
```
Top Plot: Price Comparison
  - Red line: Actual prices (波动)
  - Blue line: Predicted prices (跟踪红线)
  - Green box: Test set metrics
    * MAPE: Mean Absolute Percentage Error
    * MAE: Mean Absolute Error (dollars)
    * RMSE: Root Mean Squared Error

Bottom Plot: Error Analysis
  - Green bars: Positive errors (predicted > actual)
  - Red bars: Negative errors (predicted < actual)
  - Blue line: Average error
  - Random distribution = good model
  - Biased distribution = problem
```

### Summary Table
```
Symbol      MAPE (%)    MAE ($)      RMSE ($)
BTCUSDT     3.45%       0.0234       0.0567    <- Good
ETHUSDT     4.12%       0.0145       0.0312    <- Good
BNBUSDT     2.89%       0.0089       0.0201    <- Excellent
...
```

## Expected Performance

### V5 vs V4 Comparison

| Metric | V4 | V5 | Improvement |
|--------|----|----|-------------|
| MAPE | 4-5% | 3-4% | 20-30% better |
| Prediction Type | Absolute | Delta + Uncertainty | More realistic |
| Feature Count | 10 | 20+ | Richer signals |
| Architecture | Single LSTM | Multi-scale | Better patterns |

### Key Improvements to Look For

1. **Blue line follows red line**: V5 captures price movements
2. **Lower MAPE**: V5 is more accurate
3. **Random error distribution**: Not biased in one direction
4. **Stable across symbols**: Generalizes well

## Common Issues & Solutions

### Issue 1: "Model not found" after visualization
```
Solution: You need to train first
  python train_v5.py 5 --device cuda
```

### Issue 2: Out of memory (CUDA)
```
Solution: Reduce batch size in training
  Edit backend/train_v5_enhanced.py
  Change: batch_size=16 -> batch_size=8
```

### Issue 3: Training is very slow
```
Solution: Use GPU instead of CPU
  python train_v5.py 5 --device cuda
  (vs --device cpu)
```

### Issue 4: Path configuration error
```
Solution: Check paths are set correctly
  python quick_visualize_v5.py --show-paths
  
You should see:
  Project Root:  C:\Users\zong\Desktop\MOP
  Backend:       C:\Users\zong\Desktop\MOP\backend
  Models:        C:\Users\zong\Desktop\MOP\backend\models\weights
```

## Advanced Options

### Train Specific Symbols Only
```bash
python train_v5.py --symbols BTCUSDT ETHUSDT BNBUSDT --device cuda
```

### Visualize Specific Symbols
```bash
python quick_visualize_v5.py --symbols BTCUSDT ETHUSDT --device cuda
```

### Adjust Training Parameters
Edit `backend/train_v5_enhanced.py`:
```python
# Change epochs, batch size, learning rate
trainer.train(symbol, epochs=100, batch_size=32, learning_rate=0.001)
```

## File Structure

After training:
```
MOP/
├── backend/
│   ├── data/raw/
│   │   ├── BTCUSDT_1h.csv
│   │   └── ...
│   ├── models/
│   │   ├── weights/
│   │   │   ├── BTCUSDT_1h_v5_lstm.pth     <- V5 model
│   │   │   ├── BTCUSDT_1h_v4_lstm.pth     <- Old V4 model
│   │   │   └── ...
│   │   └── config/
│   │       ├── BTCUSDT_v5_config.json     <- V5 config
│   │       └── ...
│   ├── results/
│   │   └── visualizations/
│   │       ├── BTCUSDT_predictions_v5_*.png
│   │       ├── summary_table_v5_*.png
│   │       └── summary_results_v5_*.csv
│   ├── logs/
│   │   └── training_v5.log
│   └── path_config.py
├── train_v5.py                   <- Quick start script
└── quick_visualize_v5.py        <- Visualization script
```

## Next Steps

1. **Short term**:
   ```bash
   # Train 5 symbols for quick testing
   python train_v5.py 5 --device cuda
   
   # Visualize results
   python quick_visualize_v5.py --all --device cuda
   ```

2. **Medium term**:
   - Train all 15 symbols
   - Compare V5 vs V4 metrics
   - Fine-tune hyperparameters

3. **Long term**:
   - Use V5 for live predictions
   - Monitor performance
   - Retrain periodically
   - Add more symbols

## Troubleshooting

### Check Path Configuration
```bash
python -c "from backend.path_config import PathConfig; p = PathConfig(); p.print_summary()"
```

### Check Data Availability
```bash
python -c "from backend.path_config import PathConfig; import os; p = PathConfig(); print(os.path.exists(p.get_data_file('BTCUSDT')))"
```

### Check Model Training
```bash
# Look at training log
type backend/logs/training_v5.log
```

## Performance Expectations

**Training Time per Symbol (GPU):**
- BTCUSDT: 2-3 minutes
- ETHUSDT: 2-3 minutes
- Others: 1-2 minutes each

**Total for 15 symbols: 25-35 minutes**

**Visualization Time:**
- All 15 symbols: 2-5 minutes

## Key Metrics to Monitor

1. **MAPE (Mean Absolute Percentage Error)**
   - Lower is better
   - Target: < 5%
   - Excellent: < 3%

2. **MAE (Mean Absolute Error)**
   - Dollar amount of average error
   - Lower is better
   - Compare across symbols

3. **RMSE (Root Mean Squared Error)**
   - Penalizes large errors
   - Lower is better
   - Use for consistency

## Getting Help

If you encounter issues:

1. Check PATH_USAGE.md for path configuration
2. Review V5_ENHANCED_README.md for technical details
3. Check logs/training_v5.log for error messages
4. Verify all directories exist: `quick_visualize_v5.py --show-paths`

## Summary

```
Quick Start Commands:

1. Train models:
   python train_v5.py 5 --device cuda

2. Visualize results:
   python quick_visualize_v5.py --all --device cuda

3. View results:
   Open results/visualizations/*.png

That's it!
```
