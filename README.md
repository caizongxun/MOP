# MOP - Cryptocurrency Price Prediction System

A professional-grade cryptocurrency price prediction system with advanced LSTM neural networks. Features include local CUDA-accelerated model training with multi-timeframe analysis, incremental data management, and cloud-deployed Discord bot for 24/7 market analysis.

## System Architecture

```
Incremental Data Collection
    |
    |-- Binance API (1000 candles per API call)
    |-- Store locally in data/raw/
    |-- Support all 25 cryptocurrencies
    |-- Track metadata (last update, row count)
    |
    v
Local GPU Training (CUDA)
    |
    |-- Read from data/raw/ (or fallback to API)
    |-- Multi-timeframe: 15m + 1h
    |-- 35+ technical indicators
    |-- Dynamic feature selection
    |
    v
    Train Models
    - 15m timeframe
    - 1h timeframe
    - Unified multi-timeframe model
    |
    v
    Upload to Hugging Face Hub
    |
    v
GCP VM Deployment (CPU)
    |
    v
    Discord Bot (torch CPU version)
    |
    v
    24/7 Price Prediction & Alerts
```

## Key Features

- **Incremental Data Management**: Fetch once, reuse forever. No duplicate API calls
- **Local Data Storage**: CSV files for each crypto + timeframe in `data/raw/`
- **35+ Technical Indicators**: Comprehensive market analysis with auto-validation
- **Multi-Timeframe Training**: 15m (short-term) + 1h (medium-term) + unified models
- **Dynamic Feature Selection**: Auto-skips indicators without sufficient data
- **25 Cryptocurrency Support**: Full coverage across major, layer 1, and emerging coins
- **High-Accuracy Predictions**: Target MAPE < 0.001%
- **Production Ready**: CPU-optimized inference, Hugging Face deployment

## Data Management System

### Folder Structure

```
data/
└── raw/
    ├── BTCUSDT_15m.csv          # 1000+ BTC 15m candles
    ├── BTCUSDT_1h.csv           # 1000+ BTC 1h candles
    ├── ETHUSDT_15m.csv          # 1000+ ETH 15m candles
    ├── ETHUSDT_1h.csv           # 1000+ ETH 1h candles
    ├── ... (23 more coins)
    └── metadata.json            # Tracking info
```

### Workflow: Data Collection

```bash
# One-time setup: Fetch all data for all cryptocurrencies
python backend/data_fetcher.py

# Creates:
# data/raw/BTCUSDT_15m.csv    (1000 rows)
# data/raw/BTCUSDT_1h.csv     (1000 rows)
# data/raw/ETHUSDT_15m.csv    (1000 rows)
# ... 50 CSV files total (25 coins × 2 timeframes)
```

### Workflow: Training

```bash
# Training now reads from local data
python backend/train.py

# Flow:
# 1. For each coin and timeframe:
#    a. Try to load from data/raw/SYMBOL_TIMEFRAME.csv
#    b. If not found, fallback to Binance API
#    c. Save new data to data/raw/ automatically
# 2. Calculate 35+ indicators
# 3. Train models with dynamic feature count
```

### Key Benefits

1. **No Duplicate API Calls**: Fetch 1000 candles once, train multiple times
2. **Parallel Training**: Run multiple training sessions without re-fetching
3. **Data Versioning**: metadata.json tracks last update for each file
4. **Incremental Updates**: Can append new data without re-downloading history
5. **Offline Training**: Train on stored data even without internet

## Technical Indicators (35+ Dynamic)

### Auto-Calculated with Validation

**Trend Indicators (10)**
- SMA: 10, 20, 50, 100, 200 periods
- EMA: 10, 20, 50, 100, 200 periods

**Momentum Indicators (8)**
- RSI: 14, 21 periods
- MACD: + Signal + Histogram
- Stochastic K & D
- ROC 12

**Volatility Indicators (6)**
- Bollinger Bands: Upper, Middle, Lower
- BB Width & %B
- ATR 14, NATR

**Trend Direction (3)**
- ADX, DI+, DI-

**Keltner Channels (3)**
- Upper, Middle, Lower

**Volume Indicators (4)**
- OBV, CMF, MFI, VPT

**Other (7)**
- Momentum, Price Change, Volume Change, HL2

**Total: 40-45 features** (depends on data sufficiency)

## Data Configuration

```python
DATA_CONFIG = {
    'timeframes': ['15m', '1h'],    # Multi-timeframe
    'history_limit': 1000,          # Per API call
    'use_binance_api': True,        # Primary source
    'prediction_horizon': 1,        # 1 candle ahead
    'min_candles': 200,             # For all indicators
}
```

## Model Configuration (V1)

```python
MODEL_CONFIG = {
    'hidden_size': 64,          # LSTM hidden layer
    'num_layers': 2,            # 2-layer LSTM stack
    'dropout': 0.3,             # Regularization
    'learning_rate': 0.005,     # Adam optimizer
    'batch_size': 64,           # Batch size
    'epochs': 150,              # Total epochs
    'lookback': 60,             # 60 candle context
}
```

## Supported Cryptocurrencies (25 coins)

**Major (5)**: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT
**Layer 1 (5)**: ADAUSDT, AVAXUSDT, DOGEUSDT, LINKUSDT, MATICUSDT
**Other Layer 1 (5)**: LTCUSDT, UNIUSDT, ATOMUSDT, XMRUSDT, NEARUSDT
**Layer 2 & New (5)**: ARBUSDT, OPUSDT, APTUSDT, SUIUSDT, PEPEUSDT
**Meme & Utility (5)**: SHIBUSDT, ICPUSDT, ETCUSDT, FILUSDT, AAVEUSDT

## Project Structure

```
MOP/
├── backend/
│   ├── data/
│   │   ├── data_loader.py       # 35+ indicators with validation
│   │   ├── data_manager.py      # Local storage & incremental fetch
│   │   └── __init__.py
│   ├── models/
│   │   ├── lstm_model.py        # Dynamic input size LSTM
│   │   └── weights/             # Trained weights
│   ├── train.py                 # Training with local data
│   ├── data_fetcher.py          # Standalone fetch script
│   ├── inference.py             # CPU inference
│   └── requirements.txt
├── config/
│   └── model_config.py          # V1 optimized config
├── data/
│   └── raw/                     # Local storage (git-ignored)
│       ├── BTCUSDT_15m.csv
│       ├── BTCUSDT_1h.csv
│       └── ... (50 files)
├── logs/                        # Training logs
├── .env.example                 # API key template
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/caizongxun/MOP.git
cd MOP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your keys:
# BINANCE_API_KEY
# BINANCE_API_SECRET
# HF_TOKEN
# DISCORD_BOT_TOKEN
```

## Usage Guide

### Step 1: Fetch Data (One-time Setup)

```bash
# Fetch 1000 candles for all 25 cryptocurrencies
# Both 15m and 1h timeframes
python backend/data_fetcher.py

# Output:
# Creating 50 CSV files in data/raw/
# Metadata saved to data/raw/metadata.json
# Takes ~5-10 minutes depending on network
```

### Step 2: View Data Statistics

```python
from backend.data.data_manager import DataManager

manager = DataManager()
manager.print_statistics()

# Output:
# ==================================================
# Data Storage Statistics
# ==================================================
# Total files: 50
# Total rows: 50,000 (25 coins × 2 timeframes × 1000 rows)
#
# BTCUSDT:
#   15m: 1000 rows (2024-12-01 to 2024-12-31)
#   1h: 1000 rows (2024-12-01 to 2024-12-31)
# ...
```

### Step 3: Train Models

```bash
# Train all 25 cryptocurrencies
# Reads from data/raw/ automatically
python backend/train.py

# Flow:
# 1. For each coin (25 total)
# 2. For each timeframe (15m, 1h)
#    - Load from data/raw/SYMBOL_TIMEFRAME.csv
#    - Calculate 35+ indicators
#    - Train model
# 3. Train unified model (combines 15m + 1h)
# Total: 75 models trained

# Output:
# - Models saved to: backend/models/weights/
# - Logs saved to: logs/training.log
# - Data stats: data/raw/metadata.json updated
```

### Step 4: Monitor Training

```bash
# Watch training progress in real-time
tail -f logs/training.log

# Check data usage
python -c "from backend.data.data_manager import DataManager; \
          DataManager().print_statistics()"
```

## Advanced: Incremental Data Updates

```python
from backend.data.data_manager import DataManager

manager = DataManager()

# Fetch latest data for specific coin and timeframe
df = manager.fetch_and_store('BTCUSDT', '1h', append=True)
# This appends new data to existing, avoiding duplicates

# Get specific coin data
data = manager.get_stored_data('ETHUSDT', '15m')
print(f"Loaded {len(data)} rows")
```

## Model Training Output Example

```
Starting training for 25 cryptocurrencies...
Local data: True
Timeframes: ['15m', '1h']

============================================================
Training BTCUSDT - 15m
============================================================
Attempting to load BTCUSDT (15m) from local storage...
Loaded 1000 rows from local storage
Calculating indicators with 1000 candles

Successfully calculated 42 features
Available features: [open, high, low, close, ..., momentum]
Using 42 features for training

Data shape - X: (939, 60, 42), y: (939, 1)

Epoch [10/150], Loss: 0.000234, Best: 0.000234
Epoch [20/150], Loss: 0.000156, Best: 0.000156
...
Completed BTCUSDT (15m). Final loss: 0.000089
Model saved to backend/models/weights/BTCUSDT_15m_v1.pt

============================================================
Data Storage Statistics
============================================================
Total files: 50
Total rows: 50,000+

BTCUSDAT:
  15m: 1000 rows (2024-12-01 to 2024-12-31)
  1h: 1000 rows (2024-12-01 to 2024-12-31)
```

## Inference (CPU Optimized)

```python
from backend.inference import ModelInference
from backend.data.data_manager import DataManager

# Load stored data
manager = DataManager()
data = manager.get_stored_data('BTCUSDT', '1h')

# Predict
inference = ModelInference(device='cpu')
predictions = inference.predict('BTCUSDT', data, num_steps=7)
print(f"Next 7 candles: {predictions}")
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| MAPE | < 0.001% | In Development |
| RMSE | < 0.0005 | In Development |
| Inference Speed (CPU) | < 100ms | Optimized |
| Data Fetch Time | < 10min (all coins) | Optimized |
| 24/7 Uptime | 99.9% | TBD |

## Troubleshooting

### No data in data/raw/
```bash
# Run data fetcher
python backend/data_fetcher.py
```

### API rate limiting
```python
# data_loader has built-in rate limit handling
# CCXT auto-retries with backoff
```

### GPU memory issues
```python
# Reduce batch_size in config/model_config.py
'batch_size': 32,  # From 64
```

## Next Steps

1. Run data fetcher: `python backend/data_fetcher.py`
2. Check data: `python -c "from backend.data.data_manager import DataManager; DataManager().print_statistics()"`
3. Train models: `python backend/train.py`
4. Upload to Hugging Face
5. Deploy Discord bot (separate workspace)
6. GCP VM deployment

## Version History

**V1** (Current)
- 35+ dynamic technical indicators
- Incremental data management
- Local CSV storage for all 25 coins
- Metadata tracking
- Multi-timeframe support (15m, 1h, unified)
- Dynamic feature selection

## Dependencies

- torch 2.1.0 (CUDA for training, CPU for inference)
- pandas, numpy, scikit-learn
- ccxt (fallback data)
- requests (Binance API)
- huggingface-hub (model upload)
- python-dotenv (environment)

## License

MIT License

## Support

GitHub Issues: https://github.com/caizongxun/MOP/issues
