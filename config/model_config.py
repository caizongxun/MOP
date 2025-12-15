# MOP Model Configuration - V1
# Optimized settings for cryptocurrency price prediction
# Multi-timeframe training support

# === Core Model Configuration ===
MODEL_CONFIG = {
    'hidden_size': 64,           # Optimized LSTM hidden layer
    'num_layers': 2,             # 2-layer LSTM stack for pattern depth
    'dropout': 0.3,              # Regularization to prevent overfitting
    'learning_rate': 0.005,      # Adam optimizer learning rate
    'batch_size': 64,            # Training batch size
    'epochs': 150,               # Total training epochs
    'lookback': 60,              # 60 candle lookback period
}

# === Data Configuration ===
DATA_CONFIG = {
    'timeframes': ['15m', '1h'],           # Multi-timeframe training
    'history_limit': 1000,                 # Fetch 1000 candles per timeframe
    'use_binance_api': True,               # Use Binance REST API (with CCXT fallback)
    'prediction_horizon': 1,               # Predict 1 candle ahead
}

# === Technical Indicators ===
# 15 indicators for enhanced feature set
TECHNICAL_INDICATORS = {
    'price_features': ['open', 'high', 'low', 'close', 'volume'],
    'sma': [10, 20, 50],                   # Simple Moving Averages
    'ema': [10, 20],                       # Exponential Moving Averages
    'rsi_period': 14,                      # Relative Strength Index
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD parameters
    'bollinger_bands': {'period': 20, 'std_dev': 2}, # Bollinger Bands
    'atr_period': 14,                      # Average True Range
    'momentum_period': 4,                  # Price momentum
}

# Features list for model input (19 total features)
MODEL_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',      # 5 OHLCV
    'sma_10', 'sma_20', 'sma_50',                   # 3 SMA
    'ema_10', 'ema_20',                             # 2 EMA
    'rsi',                                          # 1 RSI
    'macd', 'macd_signal',                          # 2 MACD
    'bb_upper', 'bb_middle', 'bb_lower',           # 3 Bollinger Bands
    'atr',                                          # 1 ATR
    'momentum',                                     # 1 Momentum
]  # Total: 19 features

# === Cryptocurrencies (20+ coins) ===
CRYPTOCURRENCIES = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'LINKUSDT', 'MATICUSDT',
    'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'XMRUSDT', 'NEARUSDT',
    'ARBUSDT', 'OPUSDT', 'APTUSDT', 'SUIUSDT', 'PEPEUSDT',
    'SHIBUSDT', 'ICPUSDT', 'ETCUSDT', 'FILUSDT', 'AAVEUSDT'
]

# === Target Metrics ===
TARGET_METRICS = {
    'mape': 0.001,              # 0.001% - Target Mean Absolute Percentage Error
    'rmse_threshold': 0.0005,   # Root Mean Square Error threshold
}

# === Multi-timeframe Training Strategy ===
MULTI_TIMEFRAME_CONFIG = {
    # Train unified model on both 15m and 1h data
    # This creates more robust predictions across different timeframes
    '15m': {
        'weight': 0.5,           # Weight for 15m data in combined training
        'model_name_suffix': '15m',
    },
    '1h': {
        'weight': 0.5,           # Weight for 1h data in combined training
        'model_name_suffix': '1h',
    },
    'unified': {
        'enabled': True,         # Train single model on both timeframes
        'model_name_suffix': 'unified',
    }
}

# === Model Hyperparameters Optimization Notes ===
"""
Optimization Strategy:

1. Hidden Size (64):
   - Provides good balance between model capacity and training speed
   - Not too small to miss complex patterns
   - Not too large to avoid overfitting

2. Num Layers (2):
   - Two layers capture hierarchical temporal patterns
   - Sufficient for time series without excessive computation
   - Reduces overfitting vs 3+ layers

3. Dropout (0.3):
   - Prevents overfitting while maintaining good generalization
   - Applied between LSTM layers

4. Learning Rate (0.005):
   - Adam optimizer ensures stable convergence
   - Not too high to cause oscillation
   - Not too low to cause premature convergence

5. Batch Size (64):
   - Balances memory usage and gradient stability
   - Allows efficient GPU/CPU processing

6. Epochs (150):
   - Sufficient for convergence with early stopping
   - Prevents excessive overfitting from too many epochs

7. Lookback (60):
   - Captures 60 candles of historical context
   - Balanced window for 15m (15 hours) and 1h (2.5 days)

8. Features (19 total):
   - Combines price action with momentum indicators
   - Technical indicators capture different market aspects
   - Normalized independently for fair model weighting

9. Multi-timeframe:
   - 15m: Captures short-term volatility and micro-movements
   - 1h: Captures medium-term trends and consolidation patterns
   - Unified training: Creates robust model for both timeframes
"""
