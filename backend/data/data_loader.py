import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import os
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)

class CryptoDataLoader:
    """
    Load cryptocurrency data from Binance
    Supports:
    - Multiple timeframes (15m, 1h, etc.)
    - Extended historical data (3000+, 5000+ candles using loop)
    - 35+ technical indicators with data validation
    - Multi-timeframe training for robust models
    - CORRECT timestamp handling (milliseconds to UTC datetime)
    """
    
    # Timeframe mapping to minutes
    TIMEFRAME_MINUTES = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    # Timeframe mapping to milliseconds (for loop calculation)
    TIMEFRAME_MS = {
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }
    
    # All available technical indicators
    INDICATORS_CONFIG = {
        # Price & Volume (5)
        'open': 1,
        'high': 1,
        'low': 1,
        'close': 1,
        'volume': 1,
        
        # Simple Moving Averages (5)
        'sma_10': 10,
        'sma_20': 20,
        'sma_50': 50,
        'sma_100': 100,
        'sma_200': 200,
        
        # Exponential Moving Averages (5)
        'ema_10': 10,
        'ema_20': 20,
        'ema_50': 50,
        'ema_100': 100,
        'ema_200': 200,
        
        # Momentum Indicators (8)
        'rsi_14': 14,
        'rsi_21': 21,
        'macd': 26,
        'macd_signal': 35,
        'macd_histogram': 35,
        'momentum': 5,
        'roc_12': 12,
        'stochastic_k': 14,
        'stochastic_d': 17,
        
        # Volatility Indicators (6)
        'bb_upper': 20,
        'bb_middle': 20,
        'bb_lower': 20,
        'bb_width': 20,
        'bb_percent_b': 20,
        'atr_14': 14,
        
        # Trend Indicators (5)
        'adx': 14,
        'di_plus': 14,
        'di_minus': 14,
        'natr': 14,
        'kc_upper': 20,
        'kc_middle': 20,
        'kc_lower': 20,
        
        # Volume Indicators (4)
        'obv': 1,
        'cmf': 20,
        'vpt': 1,
        'mfi': 14,
        
        # Other Indicators (3)
        'price_change': 1,
        'volume_change': 1,
        'hl2': 1,
    }
    
    def __init__(self, lookback_period=60, use_binance_api=True, min_candles=200):
        """
        Initialize data loader
        
        Args:
            lookback_period: Number of candles for lookback (default: 60)
            use_binance_api: Use Binance direct API or CCXT (default: True)
            min_candles: Minimum candles needed for all indicators (default: 200)
        """
        self.lookback_period = lookback_period
        self.use_binance_api = use_binance_api
        self.min_candles = min_candles
        
        # Initialize both methods
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Scalers for each feature
        self.scalers = {}
        self.available_features = []
    
    def _interval_to_milliseconds(self, interval):
        """
        Convert interval string to milliseconds
        """
        return self.TIMEFRAME_MS.get(interval, 60 * 60 * 1000)
    
    def fetch_ohlcv_binance_api_batch(self, symbol, timeframe='15m', limit=3000, batch_size=1000):
        """
        Fetch OHLCV data using Binance REST API with proper looping
        This is the CORRECT way to fetch large amounts of historical data
        
        Based on Binance official documentation:
        - Max 1000 candles per request
        - Use startTime/endTime + loop to get more
        - Timestamps are in milliseconds
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candle timeframe (default: '15m')
            limit: Total candles to fetch (default: 3000)
            batch_size: Candles per request (max 1000)
        
        Returns:
            DataFrame with OHLCV data (timestamp as datetime, NOT milliseconds)
        """
        try:
            timeframe_map = {
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            interval = timeframe_map.get(timeframe, '1h')
            url = 'https://api.binance.com/api/v3/klines'
            
            all_data = []
            batch_size = min(batch_size, 1000)  # Cap at 1000
            interval_ms = self._interval_to_milliseconds(timeframe)
            
            # Calculate how many batches needed
            num_batches = (limit + batch_size - 1) // batch_size
            logger.info(f"Fetching {symbol} ({timeframe}): {limit} candles in {num_batches} batches of {batch_size}")
            
            # Start from the oldest timestamp possible (go back far enough)
            # Calculate: now - (limit * interval_ms)
            end_time = int(time.time() * 1000)  # Now in milliseconds
            start_time = end_time - (limit * interval_ms)
            
            current_start = start_time
            batch_num = 0
            
            while batch_num < num_batches and len(all_data) < limit:
                batch_num += 1
                
                # Calculate end time for this batch
                current_end = current_start + (batch_size * interval_ms)
                
                logger.debug(f"Batch {batch_num}/{num_batches}: startTime={current_start}, endTime={current_end}")
                
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': int(current_start),
                    'endTime': int(current_end),
                    'limit': batch_size
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logger.info(f"Batch {batch_num}: No more data available")
                    break
                
                all_data.extend(data)
                logger.info(f"Batch {batch_num}: Fetched {len(data)} candles (Total: {len(all_data)})")
                
                # Move to next batch
                current_start = data[-1][0] + interval_ms  # Start from last candle timestamp + interval
                
                time.sleep(0.1)  # Small delay to avoid rate limiting
            
            # Trim to exact limit
            all_data = all_data[:limit]
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'num_trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore']
            )
            
            # CRITICAL FIX: Convert timestamp from milliseconds to datetime
            # This is the correct way: pd.to_datetime(value, unit='ms')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Keep only OHLCV columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol} ({timeframe})")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Binance API for {symbol}: {str(e)}")
            logger.info("Falling back to CCXT...")
            return self.fetch_ohlcv_ccxt(symbol, timeframe, limit)
        except Exception as e:
            logger.error(f"Error processing Binance API response: {str(e)}", exc_info=True)
            return None
    
    def fetch_ohlcv_binance_api(self, symbol, timeframe='15m', limit=1000):
        """
        Simple wrapper - delegates to batch fetcher
        """
        return self.fetch_ohlcv_binance_api_batch(symbol, timeframe, limit, batch_size=1000)
    
    def fetch_ohlcv_ccxt(self, symbol, timeframe='15m', limit=1000):
        """
        Fetch OHLCV data using CCXT (fallback method)
        CRITICAL: Convert timestamp from milliseconds to datetime
        """
        try:
            if '/' not in symbol:
                symbol = symbol[:-4] + '/' + symbol[-4:]
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # CRITICAL FIX: Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe}) from CCXT")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from CCXT for {symbol}: {str(e)}")
            return None
    
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=1000):
        """
        Fetch OHLCV data - tries Binance API first, falls back to CCXT
        
        Returns DataFrame with timestamp as datetime (UTC)
        """
        if self.use_binance_api:
            return self.fetch_ohlcv_binance_api_batch(symbol, timeframe, limit)
        else:
            return self.fetch_ohlcv_ccxt(symbol, timeframe, limit)
    
    def calculate_technical_indicators(self, df):
        """
        Calculate 35+ technical indicators
        Automatically skips indicators that don't have enough data
        
        Args:
            df: DataFrame with OHLCV data (timestamp as datetime)
        
        Returns:
            DataFrame with technical indicators, list of available features
        """
        df = df.copy()
        self.available_features = ['open', 'high', 'low', 'close', 'volume']
        
        num_candles = len(df)
        logger.info(f"Calculating indicators with {num_candles} candles")
        
        try:
            # Moving Averages
            for period in [10, 20, 50, 100, 200]:
                if num_candles >= period:
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                    self.available_features.extend([f'sma_{period}', f'ema_{period}'])
                else:
                    logger.warning(f"Skipping SMA/EMA {period}: need {period}, have {num_candles}")
            
            # RSI
            if num_candles >= 14:
                df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
                self.available_features.append('rsi_14')
            if num_candles >= 21:
                df['rsi_21'] = self._calculate_rsi(df['close'], period=21)
                self.available_features.append('rsi_21')
            
            # MACD
            if num_candles >= 26:
                macd_result = self._calculate_macd(df['close'])
                df['macd'] = macd_result['macd']
                df['macd_signal'] = macd_result['signal']
                df['macd_histogram'] = macd_result['histogram']
                self.available_features.extend(['macd', 'macd_signal', 'macd_histogram'])
            
            # Bollinger Bands
            if num_candles >= 20:
                bb_result = self._calculate_bollinger_bands(df['close'])
                df['bb_upper'] = bb_result['upper']
                df['bb_middle'] = bb_result['middle']
                df['bb_lower'] = bb_result['lower']
                df['bb_width'] = bb_result['upper'] - bb_result['lower']
                df['bb_percent_b'] = (df['close'] - bb_result['lower']) / (bb_result['upper'] - bb_result['lower'])
                self.available_features.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent_b'])
            
            # ATR
            if num_candles >= 14:
                df['atr_14'] = self._calculate_atr(df, period=14)
                self.available_features.append('atr_14')
            
            # ADX
            if num_candles >= 14:
                di_result = self._calculate_directional_indicators(df)
                df['di_plus'] = di_result['di_plus']
                df['di_minus'] = di_result['di_minus']
                df['adx'] = di_result['adx']
                self.available_features.extend(['di_plus', 'di_minus', 'adx'])
            
            # Keltner Channels
            if num_candles >= 20:
                kc_result = self._calculate_keltner_channels(df)
                df['kc_upper'] = kc_result['upper']
                df['kc_middle'] = kc_result['middle']
                df['kc_lower'] = kc_result['lower']
                self.available_features.extend(['kc_upper', 'kc_middle', 'kc_lower'])
            
            # Stochastic Oscillator
            if num_candles >= 14:
                stoch_result = self._calculate_stochastic(df['high'], df['low'], df['close'])
                df['stochastic_k'] = stoch_result['k']
                df['stochastic_d'] = stoch_result['d']
                self.available_features.extend(['stochastic_k', 'stochastic_d'])
            
            # Volume Indicators
            df['obv'] = self._calculate_obv(df['close'], df['volume'])
            self.available_features.append('obv')
            
            if num_candles >= 20:
                df['cmf'] = self._calculate_cmf(df, period=20)
                df['mfi'] = self._calculate_mfi(df, period=14)
                self.available_features.extend(['cmf', 'mfi'])
            
            df['vpt'] = self._calculate_vpt(df['close'], df['volume'])
            self.available_features.append('vpt')
            
            # Rate of Change
            if num_candles >= 12:
                df['roc_12'] = self._calculate_roc(df['close'], period=12)
                self.available_features.append('roc_12')
            
            # Other indicators
            df['momentum'] = df['close'].diff(5)
            df['price_change'] = df['close'].diff()
            df['volume_change'] = df['volume'].diff()
            df['hl2'] = (df['high'] + df['low']) / 2
            df['natr'] = (self._calculate_atr(df, period=14) / df['close']) * 100
            self.available_features.extend(['momentum', 'price_change', 'volume_change', 'hl2', 'natr'])
            
            # Remove duplicates and NaN
            self.available_features = list(set(self.available_features))
            df = df.dropna()
            
            logger.info(f"Successfully calculated {len(self.available_features)} features")
            logger.info(f"Available features: {sorted(self.available_features)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            return df.dropna()
    
    @staticmethod
    def _calculate_rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def _calculate_bollinger_bands(close, period=20, std_dev=2):
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    @staticmethod
    def _calculate_atr(df, period=14):
        df = df.copy()
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df['tr'].rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _calculate_directional_indicators(df, period=14):
        df = df.copy()
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['up'] = df['high'].diff()
        df['down'] = -df['low'].diff()
        df['di_plus'] = np.where(df['up'] > df['down'], df['up'], 0)
        df['di_minus'] = np.where(df['down'] > df['up'], df['down'], 0)
        di_plus_smooth = df['di_plus'].rolling(window=period).sum()
        di_minus_smooth = df['di_minus'].rolling(window=period).sum()
        tr_smooth = df['tr'].rolling(window=period).sum()
        di_plus = (di_plus_smooth / tr_smooth) * 100
        di_minus = (di_minus_smooth / tr_smooth) * 100
        di_sum = (di_plus + di_minus).replace(0, 1)
        adx = abs(di_plus - di_minus) / di_sum * 100
        adx = adx.rolling(window=period).mean()
        return {'di_plus': di_plus, 'di_minus': di_minus, 'adx': adx}
    
    @staticmethod
    def _calculate_keltner_channels(df, period=20, atr_period=14):
        atr = CryptoDataLoader._calculate_atr(df, atr_period)
        middle = df['close'].rolling(window=period).mean()
        upper = middle + (atr * 2)
        lower = middle - (atr * 2)
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    @staticmethod
    def _calculate_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3):
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        k_smooth = k.rolling(window=smooth_k).mean()
        d_smooth = k_smooth.rolling(window=smooth_d).mean()
        return {'k': k_smooth, 'd': d_smooth}
    
    @staticmethod
    def _calculate_obv(close, volume):
        obv = volume.copy()
        obv[close.diff() < 0] *= -1
        return obv.cumsum()
    
    @staticmethod
    def _calculate_cmf(df, period=20):
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf
    
    @staticmethod
    def _calculate_mfi(df, period=14):
        tp = (df['high'] + df['low'] + df['close']) / 3
        pmf = tp * df['volume']
        pmf[tp < tp.shift()] = -pmf
        positive_mf = pmf.copy()
        positive_mf[pmf < 0] = 0
        negative_mf = pmf.copy()
        negative_mf[pmf > 0] = 0
        pmf_ratio = positive_mf.rolling(window=period).sum() / negative_mf.rolling(window=period).sum().abs()
        mfi = 100 - (100 / (1 + pmf_ratio))
        return mfi
    
    @staticmethod
    def _calculate_vpt(close, volume):
        return (volume * close.pct_change()).cumsum()
    
    @staticmethod
    def _calculate_roc(close, period=12):
        return ((close - close.shift(period)) / close.shift(period)) * 100
    
    def normalize_features(self, df, features_to_use=None):
        if features_to_use is None:
            features_to_use = self.available_features
        df_normalized = pd.DataFrame(index=df.index)
        for feature in features_to_use:
            if feature in df.columns:
                if feature not in self.scalers:
                    self.scalers[feature] = MinMaxScaler(feature_range=(0, 1))
                scaled = self.scalers[feature].fit_transform(df[[feature]])
                df_normalized[feature] = scaled.flatten()
        return df_normalized.values, features_to_use
    
    def create_sequences(self, data, seq_length, prediction_horizon=1):
        X, y = [], []
        for i in range(len(data) - seq_length - prediction_horizon + 1):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length+prediction_horizon-1, 3])
        return np.array(X), np.array(y)
    
    def fetch_multi_timeframe_data(self, symbol, timeframes=['15m', '1h'], limit=1000):
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
