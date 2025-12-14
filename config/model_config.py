# MOP Model Configuration - V1
# Optimized settings for cryptocurrency price prediction

MODEL_CONFIG = {
    'hidden_size': 64,           # Optimized
    'num_layers': 2,             # Optimized
    'dropout': 0.3,              # Optimized
    'learning_rate': 0.005,      # Optimized
    'batch_size': 64,            # Optimized
    'epochs': 150,
    'lookback': 60,
}

# Cryptocurrencies to train (20+ coins)
CRYPTOCURRENCIES = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP',
    'ADA', 'AVAX', 'DOGE', 'LINK', 'MATIC',
    'LTC', 'UNI', 'ATOM', 'XMR', 'NEAR',
    'ARB', 'OP', 'APT', 'SUI', 'PEPE',
    'SHIB', 'ICP', 'ETC', 'FIL', 'AAVE'
]

# Target metric
TARGET_MAPE = 0.001  # 0.001% - extremely accurate prediction

# Model hyperparameters optimization notes:
# hidden_size: 64 provides good balance between model capacity and training speed
# num_layers: 2 layers sufficient for time series patterns without overfitting
# dropout: 0.3 prevents overfitting while maintaining good generalization
# learning_rate: 0.005 ensures stable convergence without oscillation
# batch_size: 64 balances memory usage and gradient stability
# lookback: 60 periods (candles) captures sufficient historical context
