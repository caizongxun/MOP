import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class CryptoDataLoader:
    """
    Load cryptocurrency data from Binance API
    """
    
    def __init__(self, lookback_period=60):
        self.exchange = ccxt.binance()
        self.lookback_period = lookback_period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=500):
        """
        Fetch OHLCV data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (default: '1h')
            limit: Number of candles to fetch (default: 500)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def normalize_data(self, data):
        """
        Normalize data using MinMax scaling
        """
        return self.scaler.fit_transform(data)
    
    def create_sequences(self, data, seq_length):
        """
        Create sequences for time series prediction
        
        Args:
            data: Normalized data
            seq_length: Sequence length (lookback period)
        
        Returns:
            X, y arrays
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
