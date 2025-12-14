# MOP - Cryptocurrency Price Prediction System

A professional-grade cryptocurrency price prediction system with LSTM neural networks. Features include local CUDA-accelerated model training and cloud-deployed Discord bot for 24/7 market analysis and signals.

## System Architecture

```
Local GPU Training (CUDA)
    ↓
    Train 20+ Cryptocurrency Models
    ↓
    Upload to Hugging Face Hub
    ↓
GCP VM Deployment (CPU)
    ↓
    Discord Bot (torch CPU version)
    ↓
    24/7 Price Prediction & Alerts
```

## Features

- **High-Accuracy Predictions**: Target MAPE < 0.001% for precise price curve forecasting
- **Multi-Coin Support**: Train and predict 20+ cryptocurrencies simultaneously
- **Optimized LSTM Architecture**: Carefully tuned hyperparameters for minimal overfitting
- **Discord Integration**: Real-time trading signals via Discord bot
- **Production Ready**: Separate CPU inference for cloud deployment

## Project Structure

```
MOP/
├── backend/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # LSTM model architecture
│   ├── train.py       # Training script (CUDA)
│   ├── inference.py   # Inference engine (CPU optimized)
│   └── requirements.txt
├── config/
│   └── model_config.py # Model hyperparameters & crypto list
├── .gitignore
└── README.md
```

## Model Configuration (V1)

```python
MODEL_CONFIG = {
    'hidden_size': 64,        # Optimized hidden layer size
    'num_layers': 2,          # 2-layer LSTM stack
    'dropout': 0.3,           # Dropout for regularization
    'learning_rate': 0.005,   # Adam optimizer learning rate
    'batch_size': 64,         # Training batch size
    'epochs': 150,            # Total training epochs
    'lookback': 60,           # 60 candle lookback period
}
```

## Supported Cryptocurrencies

BTC, ETH, BNB, SOL, XRP, ADA, AVAX, DOGE, LINK, MATIC, LTC, UNI, ATOM, XMR, NEAR, ARB, OP, APT, SUI, PEPE, SHIB, ICP, ETC, FIL, AAVE

## Installation

```bash
# Clone repository
git clone https://github.com/caizongxun/MOP.git
cd MOP

# Install dependencies (GPU version with CUDA)
pip install -r backend/requirements.txt
```

## Training Models

```bash
# Train all 20+ cryptocurrency models on GPU
python backend/train.py

# Models saved to: backend/models/weights/
```

## Inference

```python
from backend.inference import ModelInference

# Load inference engine (CPU optimized)
inference = ModelInference(device='cpu')

# Make predictions
predictions = inference.predict('BTC', data, num_steps=7)
```

## Deployment

Models are uploaded to Hugging Face Hub and deployed on GCP VM with CPU-optimized torch version for 24/7 Discord bot operation.

## Next Steps

- Configure `.env` file with API keys (HF, Discord, Binance, CoinGecko)
- Frontend Discord bot implementation (separate workspace)
- Model upload automation to Hugging Face
- GCP VM deployment configuration

## Version

V1 - Initial model optimization and backend implementation

## License

MIT
