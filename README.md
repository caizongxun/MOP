# MOP - Cryptocurrency Price Prediction System

A professional-grade cryptocurrency price prediction system with advanced LSTM neural networks. Features include local CUDA-accelerated model training with multi-timeframe analysis and cloud-deployed Discord bot for 24/7 market analysis and signals.

## System Architecture

```
Local GPU Training (CUDA)
    |
    |-- Binance API (1000 candles per timeframe)
    |
    v
    Train 25+ Cryptocurrency Models
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

- **High-Accuracy Predictions**: Target MAPE < 0.001% for precise price curve forecasting
- **Multi-Coin Support**: Train and predict 25 cryptocurrencies simultaneously
- **Multi-Timeframe Analysis**: Train separate models for 15m and 1h, plus unified cross-timeframe model
- **Advanced Features**: 19 technical indicators for enhanced prediction accuracy
- **Optimized LSTM Architecture**: Carefully tuned hyperparameters (hidden=64, layers=2, dropout=0.3)
- **Robust Data Fetching**: Binance REST API with CCXT fallback
- **Discord Integration**: Real-time trading signals via Discord bot
- **Production Ready**: CPU-optimized inference for cloud deployment

## Technical Indicators (19 Total)

### Price & Volume (5 features)
- Open, High, Low, Close, Volume

### Trend Indicators (5 features)
- SMA 10, 20, 50 (Simple Moving Averages)
- EMA 10, 20 (Exponential Moving Averages)

### Momentum Indicators (7 features)
- RSI (Relative Strength Index)
- MACD + Signal Line
- Bollinger Bands (Upper, Middle, Lower)
- ATR (Average True Range)

### Price Action (1 feature)
- Momentum (4-period)

**Total: 19 features for comprehensive market analysis**

## Data Configuration

```python
DATA_CONFIG = {
    'timeframes': ['15m', '1h'],    # Multi-timeframe support
    'history_limit': 1000,          # 1000 candles per timeframe
    'use_binance_api': True,        # Binance API (with CCXT fallback)
    'prediction_horizon': 1,        # Predict 1 candle ahead
}
```

## Model Configuration (V1)

```python
MODEL_CONFIG = {
    'hidden_size': 64,          # LSTM hidden layer
    'num_layers': 2,            # 2-layer LSTM stack
    'dropout': 0.3,             # Regularization
    'learning_rate': 0.005,     # Adam optimizer
    'batch_size': 64,           # Training batch size
    'epochs': 150,              # Total epochs
    'lookback': 60,             # 60 candle context
}
```

## Supported Cryptocurrencies (25 coins)

**Major (5)**: BTC, ETH, BNB, SOL, XRP  
**Layer 1 (5)**: ADA, AVAX, DOGE, LINK, MATIC  
**Other Layer 1 (5)**: LTC, UNI, ATOM, XMR, NEAR  
**Layer 2 & New (5)**: ARB, OP, APT, SUI, PEPE  
**Meme & Utility (5)**: SHIB, ICP, ETC, FIL, AAVE

## Project Structure

```
MOP/
├── backend/
│   ├── data/
│   │   └── data_loader.py          # Multi-timeframe, 19 indicators, Binance API
│   ├── models/
│   │   ├── lstm_model.py           # LSTM with 19 input features
│   │   └── weights/                # Trained model weights
│   ├── train.py                 # Multi-timeframe training script
│   ├── inference.py             # CPU-optimized inference
│   └── requirements.txt
├── config/
│   └── model_config.py          # V1 optimized config
├── logs/                       # Training logs
├── .env.example               # API key template
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

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (GPU version with CUDA)
pip install -r backend/requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
# - BINANCE_API_KEY
# - BINANCE_API_SECRET
# - HF_TOKEN (for model upload)
# - DISCORD_BOT_TOKEN (for frontend)
```

## Usage

### Training Models

```bash
# Train all models (15m, 1h, unified) for all cryptocurrencies
python backend/train.py

# Output:
# - Trains 3 model types per coin (15m, 1h, unified)
# - Total: 75 models for 25 cryptocurrencies
# - Models saved to: backend/models/weights/
# - Logs: logs/training.log
```

### Example Training Output

```
Starting training for 25 cryptocurrencies...
Timeframes: ['15m', '1h']
History: 1000 candles
Features: 19 technical indicators

============================================================
Training BTCUSDT - 15m
============================================================
Fetching 1000 candles for BTCUSDT (15m)...
Loaded 900 candles (after indicator calculation)
Using 19 features: ['open', 'high', 'low', 'close', ..., 'momentum']
Data shape - X: (840, 60, 19), y: (840, 1)

Epoch [10/150], Loss: 0.000234
Epoch [20/150], Loss: 0.000156
...
Completed BTCUSDT (15m). Final loss: 0.000089
Model saved to backend/models/weights/BTCUSDT_15m_v1.pt
```

### Inference (CPU Optimized)

```python
from backend.inference import ModelInference
from backend.data.data_loader import CryptoDataLoader

# Initialize inference engine (CPU)
inference = ModelInference(device='cpu')
data_loader = CryptoDataLoader()

# Load data
data = data_loader.fetch_ohlcv('BTCUSDT', timeframe='1h', limit=100)
data = data_loader.calculate_technical_indicators(data)

# Make predictions
predictions = inference.predict('BTCUSDT', data, num_steps=7)
print(f"Predicted prices for next 7 candles: {predictions}")
```

## Model Hyperparameter Optimization Notes

### Why These Settings?

1. **Hidden Size (64)**: 
   - Balances model capacity and computational efficiency
   - Prevents overfitting while capturing complex patterns

2. **Num Layers (2)**:
   - Two LSTM layers capture hierarchical temporal patterns
   - Sufficient for cryptocurrency price movements
   - Avoids vanishing gradient issues

3. **Dropout (0.3)**:
   - Regularization to prevent overfitting
   - 30% is optimal sweet spot for time series

4. **Learning Rate (0.005)**:
   - Adam optimizer ensures stable convergence
   - Not too aggressive, not too conservative

5. **Batch Size (64)**:
   - GPU memory efficient
   - Gradient stability
   - Sufficient for stable updates

6. **Epochs (150)**:
   - With early stopping, achieves convergence
   - Prevents excessive overfitting

7. **Lookback (60)**:
   - 15m: 15 hours of context
   - 1h: 2.5 days of context
   - Captures both micro and macro movements

## Data Pipeline

```
Binance API (REST)
    |
    v
OHLCV Extraction
    |
    v
Technical Indicator Calculation (19 indicators)
    |
    v
Independent Feature Normalization (MinMax)
    |
    v
Sequence Creation (60 lookback)
    |
    v
Tensor Conversion & DataLoader
    |
    v
LSTM Model Training
```

## Binance API vs CCXT

- **Primary**: Binance REST API (direct, faster, more reliable)
- **Fallback**: CCXT (universal cryptocurrency API library)
- **Auto-failover**: If API fails, automatically tries CCXT

## Deployment

### Model Upload to Hugging Face

```python
# Automatic in next phase
from huggingface_hub import upload_folder

upload_folder(
    folder_path='backend/models/weights/',
    repo_id='your_username/trading-models',
    repo_type='model'
)
```

### GCP VM Deployment

```bash
# CPU version of torch (for GCP VM)
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Discord bot runs 24/7 with CPU inference
python frontend/discord_bot.py
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| MAPE | < 0.001% | In Development |
| RMSE | < 0.0005 | In Development |
| Inference Speed (CPU) | < 100ms | Optimized |
| 24/7 Uptime | 99.9% | TBD |

## Next Steps

1. Run initial training: `python backend/train.py`
2. Verify model accuracy on validation set
3. Optimize hyperparameters if needed
4. Upload models to Hugging Face
5. Implement Discord bot frontend (separate workspace)
6. Deploy to GCP VM

## Version History

**V1** (Current)
- Multi-timeframe support (15m, 1h, unified)
- 19 technical indicators
- Binance API integration
- LSTM model optimization
- Early stopping and learning rate scheduling
- 25 cryptocurrencies

## Dependencies

- torch 2.1.0 (CUDA for training, CPU for inference)
- pandas, numpy, scikit-learn
- ccxt (fallback data source)
- requests (Binance API)
- huggingface-hub (model upload)
- python-dotenv (environment variables)

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please open an issue on GitHub.
