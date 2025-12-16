"""
V4 Adaptive LSTM+XGBoost Inference
Uses volatility-based models for accurate predictions
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import os

import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

logger = logging.getLogger(__name__)


class LSTMWithWarmup(torch.nn.Module):
    """LSTM model for feature extraction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class FeatureCalculator:
    """計算技術指標"""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標"""
        df = df.copy()
        
        # Price-based indicators
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # Volatility indicators
        df['atr_14'] = df[['high', 'low', 'close']].apply(
            lambda x: (x['high'] - x['low']).mean() if len(x) >= 14 else 0, axis=1
        )
        
        # Momentum indicators
        df['rsi_14'] = FeatureCalculator._calculate_rsi(df['close'], 14)
        df['macd'] = (df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean())
        df['momentum'] = df['close'] - df['close'].shift(12)
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Additional features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        return df.fillna(method='bfill').fillna(method='ffill')
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """計算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class V4AdaptiveInference:
    """V4 Adaptive 推理器"""
    
    def __init__(self, data_dir: str = 'backend/data/raw', device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.data_dir = data_dir
        self.feature_calc = FeatureCalculator()
        self.models_cache = {}
        logger.info(f"V4 Adaptive Inference initialized with device: {self.device}")
    
    def load_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """加載數據"""
        filepath = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data not found: {filepath}")
        return pd.read_csv(filepath)
    
    def predict_symbol(self, symbol: str) -> Dict:
        """預測單個幣種的下一個䮷格"""
        
        logger.info(f"Predicting {symbol} with V4 Adaptive...")
        
        try:
            # 1. 加載數據
            raw_data = self.load_data(symbol)
            close_prices = raw_data['close'].values
            current_price = close_prices[-1]
            
            # 2. 特徵工程
            features_df = self.feature_calc.calculate_features(raw_data)
            
            # 3. 數據標準化
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(features_df.dropna().values)
            
            # 4. 整理成序列
            seq_length = 60
            if len(X_scaled) >= seq_length:
                X_seq = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            else:
                X_seq = X_scaled.reshape(1, -1, X_scaled.shape[1])
            
            # 5. 加載模型
            lstm_model, xgb_model, scaler_y = self._load_models(symbol)
            
            # 6. LSTM 提取特徵
            X_t = torch.FloatTensor(X_seq).to(self.device)
            lstm_model.eval()
            
            with torch.no_grad():
                _, hidden = lstm_model.lstm(X_t)
                lstm_features = hidden[-1].cpu().numpy()
            
            # 7. XGBoost 預測
            pred_scaled = xgb_model.predict(lstm_features)[0]
            pred_price = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
            
            # 8. 計算變化
            price_change = pred_price - current_price
            pct_change = (price_change / current_price) * 100
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(pred_price),
                'price_change': float(price_change),
                'pct_change': float(pct_change),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {str(e)}")
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}
    
    def predict_batch(self, symbols: list = None) -> list:
        """批量預測"""
        
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f"\nBatch Prediction with V4 Adaptive: {len(symbols)} symbols\n")
        
        predictions = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Predicting {symbol}...")
            pred = self.predict_symbol(symbol)
            predictions.append(pred)
        
        # 保存預測結果
        results_path = f"backend/predictions/v4_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        Path("backend/predictions").mkdir(parents=True, exist_ok=True)
        
        df_results = pd.DataFrame(predictions)
        df_results.to_csv(results_path, index=False)
        
        logger.info(f"\nPredictions saved to {results_path}")
        logger.info(f"\nSummary:")
        for pred in predictions:
            if 'error' not in pred:
                logger.info(f"{pred['symbol']}: {pred['current_price']:.2f} -> {pred['predicted_price']:.2f} ({pred['pct_change']:+.2f}%)")
        
        return predictions
    
    def _load_models(self, symbol: str):
        """加載LSTM和XGBoost模型"""
        
        # LSTM模型
        lstm_path = f'backend/models/weights/{symbol}_1h_v4_lstm.pth'
        config_path = f'backend/models/config/{symbol}_v4_config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                lstm_config = config['lstm']
        else:
            lstm_config = {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}
        
        lstm_model = LSTMWithWarmup(
            input_size=18,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout']
        ).to(self.device)
        
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
        lstm_model.eval()
        
        # XGBoost模型
        xgb_path = f'backend/models/weights/{symbol}_1h_v4_xgb.json'
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)
        
        # 拘量變換
        scaler_y = StandardScaler()
        raw_data = self.load_data(symbol)
        scaler_y.fit(raw_data['close'].values.reshape(-1, 1))
        
        return lstm_model, xgb_model, scaler_y


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    inference = V4AdaptiveInference()
    inference.predict_batch()
