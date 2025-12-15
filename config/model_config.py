# MOP Model Configuration - V1
# Optimized settings for cryptocurrency price prediction
# Multi-timeframe training support with 35+ indicators

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
    'min_candles': 200,                    # Minimum candles for all indicators
}

# === Technical Indicators (35+) ===
# Dynamically calculated based on available data
# Auto-validates: if data insufficient, indicator is skipped
TECHNICAL_INDICATORS = {
    # Price & Volume (5 base)
    'price_features': ['open', 'high', 'low', 'close', 'volume'],
    
    # Trend Indicators (10)
    'sma': [10, 20, 50, 100, 200],         # 5 Simple Moving Averages
    'ema': [10, 20, 50, 100, 200],         # 5 Exponential Moving Averages
    
    # Momentum Indicators (8)
    'rsi': [14, 21],                       # RSI periods
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD parameters
    'stochastic': {'period': 14, 'k': 3, 'd': 3},  # Stochastic Oscillator
    'roc_period': 12,                      # Rate of Change
    'momentum_period': 5,                  # Price momentum
    
    # Volatility Indicators (6)
    'bollinger_bands': {'period': 20, 'std_dev': 2},
    'atr_period': 14,
    'natr_period': 14,                     # Normalized ATR
    'keltner_channels': {'period': 20, 'atr_period': 14},
    
    # Trend Direction (3)
    'adx': {'period': 14},                 # Average Directional Index
    'di_period': 14,                       # Directional Indicators (+/-)
    
    # Volume Analysis (4)
    'obv': {},                             # On-Balance Volume
    'cmf_period': 20,                      # Chaikin Money Flow
    'mfi_period': 14,                      # Money Flow Index
    'vpt': {},                             # Volume Price Trend
    
    # Other (4)
    'hl2': {},                             # High-Low 2
    'price_change': {},                    # Price change 1-period
    'volume_change': {},                   # Volume change 1-period
}

# Default features list (will be dynamically updated based on available data)
# Start with guaranteed features from 1000 candles
MODEL_FEATURES = [
    # Price & Volume (5)
    'open', 'high', 'low', 'close', 'volume',
    
    # Trend (10) - all should be available with 200+ candles
    'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    
    # Momentum (8) - all should be available with 200+ candles
    'rsi_14', 'rsi_21',
    'macd', 'macd_signal', 'macd_histogram',
    'stochastic_k', 'stochastic_d',
    'roc_12',
    
    # Volatility (6) - all should be available with 200+ candles
    'bb_upper', 'bb_middle', 'bb_lower',
    'bb_width', 'bb_percent_b',
    'atr_14',
    
    # Trend Direction (3) - available with 200+ candles
    'adx', 'di_plus', 'di_minus',
    
    # Keltner Channels (3) - available with 200+ candles
    'kc_upper', 'kc_middle', 'kc_lower',
    
    # Volume (4) - all available with 200+ candles
    'obv', 'cmf', 'mfi', 'vpt',
    
    # Other (7)
    'momentum', 'roc_12', 'natr',
    'price_change', 'volume_change', 'hl2',
]  # Expected: 40-45 features depending on data sufficiency

print(f"Total expected features: ~{len(MODEL_FEATURES)} (dynamic based on available data)")

# === Cryptocurrencies (25 coins) ===
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
    '15m': {
        'weight': 0.5,
        'model_name_suffix': '15m',
    },
    '1h': {
        'weight': 0.5,
        'model_name_suffix': '1h',
    },
    'unified': {
        'enabled': True,
        'model_name_suffix': 'unified',
    }
}

# === Indicator Descriptions ===
"""
TECHNICAL INDICATORS BREAKDOWN (35+ indicators):

TREND INDICATORS (10):
- SMA 10, 20, 50, 100, 200: Simple moving averages at different periods
- EMA 10, 20, 50, 100, 200: Exponential moving averages (faster response)
  * All require minimum period candles
  * 1000 candles supports up to 200-period MA

MOMENTUM INDICATORS (8):
- RSI 14, 21: Relative Strength Index at different periods
  * Requires 14 and 21 candles respectively
- MACD + Signal + Histogram: Moving Average Convergence Divergence
  * Requires 26 candles (slow EMA period)
- Stochastic K & D: Stochastic Oscillator
  * Requires 14 candles
- ROC 12: Rate of Change
  * Requires 12 candles

VOLATILITY INDICATORS (6):
- Bollinger Bands (Upper, Middle, Lower): Standard bands ± 2 std dev
- BB Width: Upper - Lower (volatility measure)
- BB %B: Relative position within bands
- ATR 14: Average True Range
  * All require 20 candles
- NATR: Normalized ATR (ATR / Close) × 100
  * Requires 14 candles

TREND DIRECTION (3):
- ADX: Average Directional Index
- DI+: Plus Directional Indicator
- DI-: Minus Directional Indicator
  * All require 14 candles

KELTNER CHANNELS (3):
- KC Upper, Middle, Lower: EMA ± 2 × ATR
  * Requires 20 candles

VOLUME INDICATORS (4):
- OBV: On-Balance Volume (cumulative)
  * Requires 1 candle
- CMF: Chaikin Money Flow
  * Requires 20 candles
- MFI: Money Flow Index
  * Requires 14 candles
- VPT: Volume Price Trend (cumulative)
  * Requires 1 candle

OTHER INDICATORS (7):
- Momentum: 5-period price change
- Price Change: 1-period price change
- Volume Change: 1-period volume change
- HL2: (High + Low) / 2
  * All require minimal candles

AUTOMATIC VALIDATION:
1. Data loaded: 1000 candles
2. Technical indicators calculated (with validation)
3. After cleaning: ~800-900 usable candles
4. Features with insufficient data are automatically skipped
5. Training uses only available features
6. No model errors due to insufficient data

EXPECTED RESULTS:
- With 1000 candles: ~40-45 features typically available
- After cleaning: ~800-900 sequences for training
- Rich feature set improves model accuracy
"""

# === Model Hyperparameters Optimization Notes ===
"""
WHY 64 HIDDEN SIZE + 35+ FEATURES?

1. Hidden Size (64):
   - Can process 35+ dimensional input effectively
   - Sufficient capacity for complex feature interactions
   - Prevents overfitting with 35+ features

2. Num Layers (2):
   - Two LSTM layers capture multi-level feature patterns
   - First layer: Raw feature relationships
   - Second layer: Higher-order temporal patterns

3. Dropout (0.3):
   - Critical with 35+ features to prevent overfitting
   - Ensures generalization across different markets

4. Features (35-45 dynamic):
   - Redundancy: Some indicators correlate (good for robustness)
   - Coverage: Multiple market aspects (trend, momentum, volume)
   - Validation: Auto-skip insufficient-data indicators
   - Result: Model always gets optimal feature set
"""
