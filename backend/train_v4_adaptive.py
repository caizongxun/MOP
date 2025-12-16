"""
V4 Adaptive LSTM+XGBoost Training with Volatility-Based Hyperparameter Optimization
Automatically adjusts learning strategies based on asset volatility
Target: MAPE < 2% for all symbols
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import xgboost as xgb

from backend.data_manager import DataManager
from backend.feature_engineering import FeatureEngineer
from backend.models.lstm_model import LSTMModel
from backend.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """分析幣種波動性並決定訓練策略"""
    
    @staticmethod
    def calculate_volatility_metrics(data: np.ndarray) -> Dict[str, float]:
        """計算波動性指標"""
        returns = np.diff(data) / data[:-1]
        
        volatility = np.std(returns) * np.sqrt(252)
        cv = np.std(data) / np.mean(data)
        price_range = (np.max(data) - np.min(data)) / np.mean(data)
        
        return {
            'volatility': volatility,
            'cv': cv,
            'price_range': price_range,
            'mean_return': np.mean(returns)
        }
    
    @staticmethod
    def get_volatility_category(metrics: Dict[str, float]) -> str:
        """分類波動性等級"""
        volatility = metrics['volatility']
        cv = metrics['cv']
        
        if volatility > 1.5 or cv > 0.15:
            return 'high'
        elif volatility > 0.8 or cv > 0.08:
            return 'medium'
        else:
            return 'low'


class AdaptiveHyperparameterConfig:
    """根據波動性自適應超參數"""
    
    CONFIGS = {
        'low': {
            'lstm': {
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.15,
                'lr': 0.0005,
                'batch_size': 8,
                'weight_decay': 1e-5,
                'epochs': 100
            },
            'xgb': {
                'max_depth': 7,
                'learning_rate': 0.02,
                'n_estimators': 500,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5
            }
        },
        'medium': {
            'lstm': {
                'hidden_size': 192,
                'num_layers': 2,
                'dropout': 0.2,
                'lr': 0.0008,
                'batch_size': 12,
                'weight_decay': 2e-5,
                'epochs': 80
            },
            'xgb': {
                'max_depth': 6,
                'learning_rate': 0.035,
                'n_estimators': 400,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 1,
                'reg_lambda': 1
            }
        },
        'high': {
            'lstm': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'lr': 0.001,
                'batch_size': 16,
                'weight_decay': 5e-5,
                'epochs': 60
            },
            'xgb': {
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 2,
                'reg_lambda': 2
            }
        }
    }
    
    @classmethod
    def get_config(cls, category: str) -> Dict:
        """獲取對應波動性等級的配置"""
        return cls.CONFIGS.get(category, cls.CONFIGS['medium'])


class LSTMWithWarmup(nn.Module):
    """LSTM with Learning Rate Warmup"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class V4AdaptiveTrainer:
    """V4 自適應訓練器"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        logger.info(f"V4 Adaptive Trainer initialized with device: {device}")
    
    def train_symbol(self, symbol: str, target_mape: float = 0.02) -> Dict:
        """訓練單個幣種，自動調整超參數"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {symbol} with V4 Adaptive (Target MAPE: {target_mape*100:.2f}%)")
        logger.info(f"{'='*80}\n")
        
        try:
            # 1. 加載數據
            raw_data = self.data_manager.load_data(symbol)
            logger.info(f"Loaded {len(raw_data)} rows for {symbol}")
            
            # 2. 計算波動性
            close_prices = raw_data['close'].values
            volatility_metrics = VolatilityAnalyzer.calculate_volatility_metrics(close_prices)
            volatility_category = VolatilityAnalyzer.get_volatility_category(volatility_metrics)
            
            logger.info(f"{symbol} Volatility Analysis:")
            logger.info(f"  Category: {volatility_category.upper()}")
            logger.info(f"  Volatility (annualized): {volatility_metrics['volatility']:.4f}")
            logger.info(f"  Coefficient of Variation: {volatility_metrics['cv']:.4f}")
            logger.info(f"  Price Range: {volatility_metrics['price_range']:.4f}\n")
            
            # 3. 獲取自適應超參數
            config = AdaptiveHyperparameterConfig.get_config(volatility_category)
            logger.info(f"Using {volatility_category.upper()} volatility config")
            logger.info(f"LSTM Config: {config['lstm']}")
            logger.info(f"XGBoost Config: {config['xgb']}\n")
            
            # 4. 特徵工程
            features_df = self.feature_engineer.calculate_features(raw_data)
            logger.info(f"Generated {len(features_df.columns)} features")
            
            # 5. 數據預處理
            X, y, scaler_X, scaler_y = self._prepare_data(features_df, raw_data)
            logger.info(f"Data shape: X={X.shape}, y={y.shape}")
            
            # 6. 分割數據集
            train_idx = int(0.7 * len(X))
            val_idx = int(0.85 * len(X))
            
            X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
            y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
            
            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")
            
            # 7. LSTM訓練
            logger.info("="*80)
            logger.info("Stage 1: Training LSTM Feature Extractor")
            logger.info("="*80)
            
            lstm_model = self._train_lstm(
                X_train, y_train, X_val, y_val,
                config['lstm'], symbol, scaler_y
            )
            
            # 8. 提取LSTM特徵
            lstm_features_train = self._extract_lstm_features(lstm_model, X_train)
            lstm_features_val = self._extract_lstm_features(lstm_model, X_val)
            lstm_features_test = self._extract_lstm_features(lstm_model, X_test)
            
            logger.info(f"LSTM features: train={lstm_features_train.shape}, val={lstm_features_val.shape}, test={lstm_features_test.shape}\n")
            
            # 9. XGBoost訓練
            logger.info("="*80)
            logger.info("Stage 2: Training XGBoost Regressor")
            logger.info("="*80)
            
            xgb_metrics = self._train_xgboost(
                lstm_features_train, y_train,
                lstm_features_val, y_val,
                lstm_features_test, y_test,
                config['xgb'], symbol, scaler_y
            )
            
            # 10. 檢查是否達到目標
            test_mape = xgb_metrics['test_mape']
            logger.info(f"\nFinal Results for {symbol}:")
            logger.info(f"  Test MAPE: {test_mape*100:.4f}% (Target: {target_mape*100:.2f}%)")
            logger.info(f"  Test MAE: ${xgb_metrics['test_mae']:.2f}")
            logger.info(f"  Test RMSE: ${xgb_metrics['test_rmse']:.2f}")
            
            if test_mape <= target_mape:
                logger.info(f"SUCCESS: {symbol} achieved target MAPE!\n")
            else:
                logger.warning(f"WARNING: {symbol} MAPE {test_mape*100:.4f}% above target {target_mape*100:.2f}%\n")
            
            # 11. 保存模型
            self._save_models(lstm_model, symbol, volatility_category)
            
            return {
                'symbol': symbol,
                'volatility_category': volatility_category,
                'volatility_metrics': volatility_metrics,
                'config': config,
                'test_mape': test_mape,
                'test_mae': xgb_metrics['test_mae'],
                'test_rmse': xgb_metrics['test_rmse'],
                'status': 'success' if test_mape <= target_mape else 'warning'
            }
        
        except Exception as e:
            logger.error(f"Error training {symbol}: {str(e)}")
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}
    
    def _prepare_data(self, features_df: pd.DataFrame, raw_data: pd.DataFrame,
                     seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
        """準備LSTM訓練數據"""
        
        features = features_df.values
        target = raw_data['close'].values[len(raw_data) - len(features_df):]
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).ravel()
        
        X_seq = np.array([X_scaled[i:i+seq_length] for i in range(len(X_scaled) - seq_length)])
        y_seq = y_scaled[seq_length:]
        
        return X_seq, y_seq, scaler_X, scaler_y
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   config: Dict, symbol: str, scaler_y: StandardScaler) -> nn.Module:
        """訓練LSTM"""
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        model = LSTMWithWarmup(
            input_size=X_train.shape[2],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            steps_per_epoch=max(1, len(X_train) // config['batch_size']),
            epochs=config['epochs']
        )
        criterion = nn.MSELoss()
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}: Train={train_loss/len(train_loader):.6f}, Val={val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'backend/models/weights/{symbol}_lstm_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load(f'backend/models/weights/{symbol}_lstm_best.pth'))
                break
        
        return model
    
    def _extract_lstm_features(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """提取LSTM隱層特徵"""
        X_t = torch.FloatTensor(X).to(self.device)
        model.eval()
        
        with torch.no_grad():
            _, hidden = model.lstm(X_t)
            features = hidden[-1].cpu().numpy()
        
        return features
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      config: Dict, symbol: str, scaler_y: StandardScaler) -> Dict:
        """訓練XGBoost"""
        
        model = xgb.XGBRegressor(
            **config,
            random_state=42,
            n_jobs=-1,
            gpu_id=0 if self.device == 'cuda' else None
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        y_test_pred = model.predict(X_test)
        
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
        
        mape = mean_absolute_percentage_error(y_test_orig, y_test_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
        
        model.save_model(f'backend/models/weights/{symbol}_1h_v4_xgb.json')
        
        logger.info(f"XGBoost Training completed")
        logger.info(f"Test MAPE: {mape*100:.4f}%")
        logger.info(f"Test MAE: ${mae:.2f}")
        logger.info(f"Test RMSE: ${rmse:.2f}")
        
        return {'test_mape': mape, 'test_mae': mae, 'test_rmse': rmse}
    
    def _save_models(self, lstm_model: nn.Module, symbol: str, category: str):
        """保存模型和配置"""
        torch.save(lstm_model.state_dict(), f'backend/models/weights/{symbol}_1h_v4_lstm.pth')
        
        config = AdaptiveHyperparameterConfig.get_config(category)
        config['volatility_category'] = category
        
        with open(f'backend/models/config/{symbol}_v4_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_batch(self, symbols: list = None, target_mape: float = 0.02) -> Dict:
        """批量訓練"""
        
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f"\nV4 Adaptive Batch Training: {len(symbols)} symbols")
        logger.info(f"Target MAPE: {target_mape*100:.2f}%\n")
        
        results = []
        successful = 0
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...\n")
            result = self.train_symbol(symbol, target_mape)
            results.append(result)
            
            if result['status'] != 'error':
                if result.get('test_mape', 1) <= target_mape:
                    successful += 1
        
        # 生成總結報告
        summary = self._generate_summary(results, symbols, successful, target_mape)
        return summary
    
    def _generate_summary(self, results: list, symbols: list, successful: int, target_mape: float) -> Dict:
        """生成訓練總結"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'successful_symbols': successful,
            'target_mape': target_mape,
            'symbols_data': []
        }
        
        mape_list = []
        for result in results:
            if result['status'] != 'error':
                summary['symbols_data'].append({
                    'symbol': result['symbol'],
                    'volatility_category': result['volatility_category'],
                    'test_mape': result['test_mape'],
                    'test_mae': result['test_mae'],
                    'status': result['status']
                })
                mape_list.append(result['test_mape'])
        
        if mape_list:
            summary['avg_mape'] = np.mean(mape_list)
            summary['median_mape'] = np.median(mape_list)
            summary['max_mape'] = np.max(mape_list)
            summary['min_mape'] = np.min(mape_list)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"V4 Adaptive Training Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Successful: {successful}/{len(symbols)} symbols achieved MAPE < {target_mape*100:.2f}%")
        if mape_list:
            logger.info(f"Average MAPE: {summary['avg_mape']*100:.4f}%")
            logger.info(f"Median MAPE: {summary['median_mape']*100:.4f}%")
            logger.info(f"Range: {summary['min_mape']*100:.4f}% - {summary['max_mape']*100:.4f}%")
        
        # 保存總結
        summary_path = f"backend/results/v4_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("backend/results").mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}\n")
        
        return summary


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    trainer = V4AdaptiveTrainer()
    trainer.train_batch(target_mape=0.02)
