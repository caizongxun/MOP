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

import torch
import xgboost as xgb

from backend.data_manager import DataManager
from backend.feature_engineering import FeatureEngineer
from backend.train_v4_adaptive import LSTMWithWarmup, VolatilityAnalyzer

logger = logging.getLogger(__name__)

class V4AdaptiveInference:
    """V4 Adaptive 推理器"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        self.models_cache = {}
        logger.info(f"V4 Adaptive Inference initialized with device: {device}")
    
    def predict_symbol(self, symbol: str, use_adaptive: bool = True) -> Dict:
        """預測單個幣種的下一個䮷格"""
        
        logger.info(f"Predicting {symbol} with V4 Adaptive...")
        
        try:
            # 1. 加載數據
            raw_data = self.data_manager.load_data(symbol)
            close_prices = raw_data['close'].values
            current_price = close_prices[-1]
            
            # 2. 計算波動性
            if use_adaptive:
                volatility_metrics = VolatilityAnalyzer.calculate_volatility_metrics(close_prices)
                volatility_category = VolatilityAnalyzer.get_volatility_category(volatility_metrics)
                logger.info(f"{symbol} Volatility Category: {volatility_category.upper()}")
            
            # 3. 特徵工程
            features_df = self.feature_engineer.calculate_features(raw_data)
            
            # 4. 數據標準化
            from sklearn.preprocessing import StandardScaler
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(features_df.values)
            
            # 5. 整理成序列
            seq_length = 60
            if len(X_scaled) >= seq_length:
                X_seq = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            else:
                X_seq = X_scaled.reshape(1, -1, X_scaled.shape[1])
            
            # 6. 加載模型
            lstm_model, xgb_model, scaler_y = self._load_models(symbol, use_adaptive)
            
            # 7. LSTM 提取特徵
            X_t = torch.FloatTensor(X_seq).to(self.device)
            lstm_model.eval()
            
            with torch.no_grad():
                _, hidden = lstm_model.lstm(X_t)
                lstm_features = hidden[-1].cpu().numpy()
            
            # 8. XGBoost 預測
            pred_scaled = xgb_model.predict(lstm_features)[0]
            pred_price = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
            
            # 9. 計算變化
            price_change = pred_price - current_price
            pct_change = (price_change / current_price) * 100
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(pred_price),
                'price_change': float(price_change),
                'pct_change': float(pct_change),
                'volatility_category': volatility_category if use_adaptive else 'N/A',
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {str(e)}")
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}
    
    def predict_batch(self, symbols: list = None, use_adaptive: bool = True) -> list:
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
            pred = self.predict_symbol(symbol, use_adaptive)
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
    
    def _load_models(self, symbol: str, use_adaptive: bool = True):
        """加載LSTM和XGBoost模型"""
        
        # LSTM模型
        lstm_path = f'backend/models/weights/{symbol}_1h_v4_lstm.pth'
        
        # 了解LSTM一些基本姓恕
        config_path = f'backend/models/config/{symbol}_v4_config.json'
        
        if use_adaptive and Path(config_path).exists():
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
        from sklearn.preprocessing import StandardScaler
        scaler_y = StandardScaler()
        
        raw_data = self.data_manager.load_data(symbol)
        scaler_y.fit(raw_data['close'].values.reshape(-1, 1))
        
        return lstm_model, xgb_model, scaler_y


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    inference = V4AdaptiveInference()
    inference.predict_batch(use_adaptive=True)
