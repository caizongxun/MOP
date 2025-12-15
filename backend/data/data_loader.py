import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class CryptoDataLoader:
    """
    Load cryptocurrency data from Binance
    Supports:
    - Multiple timeframes (15m, 1h, etc.)
    - Extended historical data (up to 1000+ candles)
    - Multiple features for enhanced prediction accuracy
    - Multi-timeframe training for robust models
    """
    
    # Timeframe mapping to minutes
    TIMEFRAME_MINUTES = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    # Technical indicators to calculate
    FEATURES = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_20', 'sma_50',  # Simple Moving Averages
        'ema_10', 'ema_20',              # Exponential Moving Averages
        'rsi',                           # Relative Strength Index
        'macd', 'macd_signal',           # MACD
        'bb_upper', 'bb_middle', 'bb_lower',  # Bollinger Bands
        'atr',                           # Average True Range
        'momentum',                      # Price momentum
    ]
    
    def __init__(self, lookback_period=60, use_binance_api=True):
        """
        Initialize data loader
        
        Args:
            lookback_period: Number of candles for lookback (default: 60)
            use_binance_api: Use Binance direct API or CCXT (default: True)
        """
        self.lookback_period = lookback_period
        self.use_binance_api = use_binance_api
        
        # Initialize both methods
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Scalers for each feature
        self.scalers = {}
    
    def fetch_ohlcv_binance_api(self, symbol, timeframe='15m', limit=1000):
        """
        Fetch OHLCV data directly from Binance REST API
        More reliable for large historical data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candle timeframe (default: '15m')
            limit: Number of candles to fetch (max 1000 per request)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Binance API endpoint mapping
            timeframe_map = {
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            interval = timeframe_map.get(timeframe, '1h')
            url = f'https://api.binance.com/api/v3/klines'
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance max 1000 per request
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'num_trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            
            # Keep only OHLCV columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('timestamp', inplace=True)
            df = df.dropna()
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe}) from Binance API")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Binance API for {symbol}: {str(e)}")
            logger.info("Falling back to CCXT...")
            return self.fetch_ohlcv_ccxt(symbol, timeframe, limit)
        except Exception as e:
            logger.error(f"Error processing Binance API response: {str(e)}")
            return None
    
    def fetch_ohlcv_ccxt(self, symbol, timeframe='15m', limit=1000):
        """
        Fetch OHLCV data using CCXT
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # CCXT requires '/' in symbol
            if '/' not in symbol:
                symbol = symbol[:-4] + '/' + symbol[-4:]  # e.g., BTCUSDT -> BTC/USDT
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe}) from CCXT")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from CCXT for {symbol}: {str(e)}")
            return None
    
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=1000):
        """
        Fetch OHLCV data - tries Binance API first, falls back to CCXT
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe (15m, 1h, 4h, 1d)
            limit: Number of candles (default: 1000)
        
        Returns:
            DataFrame with OHLCV data
        """
        if self.use_binance_api:
            return self.fetch_ohlcv_binance_api(symbol, timeframe, limit)
        else:
            return self.fetch_ohlcv_ccxt(symbol, timeframe, limit)
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for enhanced feature set
        Adds 15 technical indicators for better prediction accuracy
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional technical indicators
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # Relative Strength Index (RSI)
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        macd_result = self._calculate_macd(df['close'])
        df['macd'] = macd_result['macd']
        df['macd_signal'] = macd_result['signal']
        
        # Bollinger Bands
        bb_result = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_result['upper']
        df['bb_middle'] = bb_result['middle']
        df['bb_lower'] = bb_result['lower']
        
        # Average True Range (ATR)
        df['atr'] = self._calculate_atr(df)
        
        # Momentum
        df['momentum'] = df['close'].diff(4)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _calculate_rsi(close, period=14):
        """
        Calculate Relative Strength Index
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(close, fast=12, slow=26, signal=9):
        """
        Calculate MACD
        """
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return {'macd': macd, 'signal': signal_line}
    
    @staticmethod
    def _calculate_bollinger_bands(close, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    @staticmethod
    def _calculate_atr(df, period=14):
        """
        Calculate Average True Range
        """
        df = df.copy()
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df['tr'].rolling(window=period).mean()
        return atr
    
    def normalize_features(self, df, features_to_use=None):
        """
        Normalize all features independently
        Each feature gets its own MinMax scaler
        
        Args:
            df: DataFrame with features
            features_to_use: List of features to normalize
        
        Returns:
            Normalized array
        """
        if features_to_use is None:
            features_to_use = [col for col in df.columns if col in self.FEATURES]
        
        df_normalized = pd.DataFrame(index=df.index)
        
        for feature in features_to_use:
            if feature in df.columns:
                if feature not in self.scalers:
                    self.scalers[feature] = MinMaxScaler(feature_range=(0, 1))
                
                scaled = self.scalers[feature].fit_transform(df[[feature]])
                df_normalized[feature] = scaled.flatten()
        
        return df_normalized.values
    
    def create_sequences(self, data, seq_length, prediction_horizon=1):
        """
        Create sequences for time series prediction
        
        Args:
            data: Normalized data (num_samples, num_features)
            seq_length: Sequence length (lookback period)
            prediction_horizon: Steps ahead to predict (default: 1)
        
        Returns:
            X, y arrays
        """
        X, y = [], []
        for i in range(len(data) - seq_length - prediction_horizon + 1):
            X.append(data[i:i+seq_length])
            # Predict the close price at prediction_horizon steps ahead (close is feature index 3)
            y.append(data[i+seq_length+prediction_horizon-1, 3])
        return np.array(X), np.array(y)
    
    def fetch_multi_timeframe_data(self, symbol, timeframes=['15m', '1h'], limit=1000):
        """
        Fetch data for multiple timeframes
        Creates unified training dataset from multiple timeframes
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframes: List of timeframes (default: ['15m', '1h'])
            limit: Number of candles per timeframe (default: 1000)
        
        Returns:
            Dictionary with data for each timeframe
        """
        data = {}
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit)
            if df is not None:
                df = self.calculate_technical_indicators(df)
                data[tf] = df
                logger.info(f"Loaded {len(df)} candles for {symbol} ({tf})")
            else:
                logger.warning(f"Failed to load data for {symbol} ({tf})")
        return data
