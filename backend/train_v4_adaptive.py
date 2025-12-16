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
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import xgboost as xgb

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """Analyze asset volatility and determine training strategy"""
    
    @staticmethod
    def calculate_volatility_metrics(data: np.ndarray) -> Dict[str, float]:
        """Calculate volatility metrics"""
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
        """Classify volatility level"""
        volatility = metrics['volatility']
        cv = metrics['cv']
        
        if volatility > 1.5 or cv > 0.15:
            return 'high'
        elif volatility > 0.8 or cv > 0.08:
            return 'medium'
        else:
            return 'low'

class AdaptiveHyperparameterConfig:
    """Adaptive hyperparameters based on volatility"""
    
    CONFIGS = {
        'low': {
            'lstm': {
                'hidden_size': 256, 'num_layers': 3, 'dropout': 0.15,
                'lr': 0.0005, 'batch_size': 8, 'weight_decay': 1e-5, 'epochs': 100
            },
            'xgb': {
                'max_depth': 7, 'learning_rate': 0.02, 'n_estimators': 500,
                'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
                'early_stopping_patience': 20
            }
        },
        'medium': {
            'lstm': {
                'hidden_size': 192, 'num_layers': 2, 'dropout': 0.2,
                'lr': 0.0008, 'batch_size': 12, 'weight_decay': 2e-5, 'epochs': 80
            },
            'xgb': {
                'max_depth': 6, 'learning_rate': 0.035, 'n_estimators': 400,
                'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 1, 'reg_lambda': 1,
                'early_stopping_patience': 15
            }
        },
        'high': {
            'lstm': {
                'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3,
                'lr': 0.001, 'batch_size': 16, 'weight_decay': 5e-5, 'epochs': 60
            },
            'xgb': {
                'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2, 'reg_lambda': 2,
                'early_stopping_patience': 10
            }
        }
    }
    
    @classmethod
    def get_config(cls, category: str) -> Dict:
        """Get config for volatility category"""
        return cls.CONFIGS.get(category, cls.CONFIGS['medium'])

class LSTMWithWarmup(nn.Module):
    """LSTM with Learning Rate Warmup"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class FeatureCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features"""
        df = df.copy()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi_14'] = FeatureCalculator._calculate_rsi(df['close'], 14)
        df['macd'] = (df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean())
        df['momentum'] = df['close'] - df['close'].shift(12)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        return df.fillna(method='bfill').fillna(method='ffill')
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class V4AdaptiveTrainer:
    """V4 Adaptive trainer"""
    
    def __init__(self, data_dir: str = 'backend/data/raw', device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.data_dir = data_dir
        self.feature_calc = FeatureCalculator()
        logger.info(f"V4 Adaptive Trainer initialized with device: {self.device}")
    
    def load_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load data"""
        filepath = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data not found: {filepath}")
        df = pd.read_csv(filepath)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in numeric_cols if col in df.columns]
        return df[available_cols]
    
    def train_symbol(self, symbol: str, target_mape: float = 0.02) -> Dict:
        """Train single symbol"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {symbol} with V4 Adaptive (Target MAPE: {target_mape*100:.2f}%)")
        logger.info(f"{'='*80}\n")
        
        try:
            raw_data = self.load_data(symbol)
            logger.info(f"Loaded {len(raw_data)} rows for {symbol}")
            
            close_prices = raw_data['close'].values
            volatility_metrics = VolatilityAnalyzer.calculate_volatility_metrics(close_prices)
            volatility_category = VolatilityAnalyzer.get_volatility_category(volatility_metrics)
            
            logger.info(f"{symbol} Volatility Analysis:")
            logger.info(f"  Category: {volatility_category.upper()}")
            logger.info(f"  Volatility (annualized): {volatility_metrics['volatility']:.4f}")
            logger.info(f"  Coefficient of Variation: {volatility_metrics['cv']:.4f}")
            logger.info(f"  Price Range: {volatility_metrics['price_range']:.4f}\n")
            
            config = AdaptiveHyperparameterConfig.get_config(volatility_category)
            logger.info(f"Using {volatility_category.upper()} volatility config")
            logger.info(f"LSTM Config: {config['lstm']}")
            logger.info(f"XGBoost Config: {config['xgb']}\n")
            
            features_df = self.feature_calc.calculate_features(raw_data)
            logger.info(f"Generated {len(features_df.columns)} features")
            
            X, y, scaler_X, scaler_y = self._prepare_data(features_df, raw_data)
            logger.info(f"Data shape: X={X.shape}, y={y.shape}")
            
            train_idx = int(0.7 * len(X))
            val_idx = int(0.85 * len(X))
            X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
            y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")
            
            logger.info("="*80)
            logger.info("Stage 1: Training LSTM Feature Extractor")
            logger.info("="*80)
            lstm_model = self._train_lstm(X_train, y_train, X_val, y_val, config['lstm'], symbol)
            
            lstm_features_train = self._extract_lstm_features(lstm_model, X_train)
            lstm_features_val = self._extract_lstm_features(lstm_model, X_val)
            lstm_features_test = self._extract_lstm_features(lstm_model, X_test)
            logger.info(f"LSTM features: train={lstm_features_train.shape}, val={lstm_features_val.shape}, test={lstm_features_test.shape}\n")
            
            logger.info("="*80)
            logger.info("Stage 2: Training XGBoost Regressor with Early Stopping")
            logger.info("="*80)
            xgb_metrics = self._train_xgboost_with_early_stopping(
                lstm_features_train, y_train, lstm_features_val, y_val,
                lstm_features_test, y_test, config['xgb'], symbol, scaler_y
            )
            
            test_mape = xgb_metrics['test_mape']
            logger.info(f"\nFinal Results for {symbol}:")
            logger.info(f"  Test MAPE: {test_mape*100:.4f}% (Target: {target_mape*100:.2f}%)")
            logger.info(f"  Test MAE: ${xgb_metrics['test_mae']:.2f}")
            logger.info(f"  Test RMSE: ${xgb_metrics['test_rmse']:.2f}")
            logger.info(f"  Estimators used: {xgb_metrics.get('n_estimators_used', 'N/A')}")
            
            if test_mape <= target_mape:
                logger.info(f"SUCCESS: {symbol} achieved target MAPE!\n")
            else:
                logger.warning(f"WARNING: {symbol} MAPE {test_mape*100:.4f}% above target\n")
            
            self._save_models(lstm_model, symbol, volatility_category)
            
            return {
                'symbol': symbol, 'volatility_category': volatility_category,
                'test_mape': test_mape, 'test_mae': xgb_metrics['test_mae'],
                'test_rmse': xgb_metrics['test_rmse'],
                'status': 'success' if test_mape <= target_mape else 'warning'
            }
        
        except Exception as e:
            logger.error(f"Error training {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}
    
    def _prepare_data(self, features_df: pd.DataFrame, raw_data: pd.DataFrame, seq_length: int = 60):
        """Prepare LSTM data"""
        numeric_features = features_df.select_dtypes(include=[np.number]).dropna()
        features = numeric_features.values
        target = raw_data['close'].values[-len(numeric_features):]
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).ravel()
        
        X_seq = np.array([X_scaled[i:i+seq_length] for i in range(len(X_scaled) - seq_length)])
        y_seq = y_scaled[seq_length:]
        return X_seq, y_seq, scaler_X, scaler_y
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, config, symbol):
        """Train LSTM"""
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        model = LSTMWithWarmup(
            input_size=X_train.shape[2], hidden_size=config['hidden_size'],
            num_layers=config['num_layers'], dropout=config['dropout']
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['lr'],
            steps_per_epoch=max(1, len(X_train) // config['batch_size']),
            epochs=config['epochs']
        )
        criterion = nn.MSELoss()
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        best_val_loss = float('inf')
        patience, patience_counter = 15, 0
        
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
                best_val_loss, patience_counter = val_loss, 0
                Path('backend/models/weights').mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), f'backend/models/weights/{symbol}_lstm_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load(f'backend/models/weights/{symbol}_lstm_best.pth'))
                break
        
        return model
    
    def _extract_lstm_features(self, model, X):
        """Extract LSTM features and reshape to 2D for XGBoost"""
        X_t = torch.FloatTensor(X).to(self.device)
        model.eval()
        with torch.no_grad():
            _, hidden = model.lstm(X_t)
            features = hidden[-1].cpu().numpy()
        return features.reshape(features.shape[0], -1)
    
    def _train_xgboost_with_early_stopping(self, X_train, y_train, X_val, y_val, X_test, y_test, config, symbol, scaler_y):
        """Train XGBoost with custom early stopping based on validation loss"""
        early_stopping_patience = config.pop('early_stopping_patience', 15)
        n_est_total = config.pop('n_estimators', 500)
        model = xgb.XGBRegressor(n_estimators=n_est_total, **config, random_state=42, n_jobs=-1, verbosity=0)
        
        best_val_score = float('inf')
        patience_counter = 0
        n_rounds_used = 0
        
        chunk_size = max(10, n_est_total // 20)
        
        for round_idx in range(0, n_est_total, chunk_size):
            n_est = min(chunk_size, n_est_total - round_idx)
            if round_idx == 0:
                model.set_params(n_estimators=n_est)
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
            
            y_val_pred = model.predict(X_val)
            val_score = mean_squared_error(y_val, y_val_pred)
            n_rounds_used += n_est
            
            if val_score < best_val_score:
                best_val_score = val_score
                patience_counter = 0
                logger.info(f"XGBoost Epoch {n_rounds_used:3d}: Val_MSE={val_score:.6f} (improved)")
            else:
                patience_counter += 1
                logger.info(f"XGBoost Epoch {n_rounds_used:3d}: Val_MSE={val_score:.6f} (patience: {patience_counter}/{early_stopping_patience})")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at estimator {n_rounds_used}")
                break
        
        y_test_pred = model.predict(X_test)
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
        
        mape = mean_absolute_percentage_error(y_test_orig, y_test_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
        
        Path('backend/models/weights').mkdir(parents=True, exist_ok=True)
        model.save_model(f'backend/models/weights/{symbol}_1h_v4_xgb.json')
        
        logger.info(f"XGBoost Training completed")
        logger.info(f"Test MAPE: {mape*100:.4f}%")
        logger.info(f"Test MAE: ${mae:.2f}")
        logger.info(f"Test RMSE: ${rmse:.2f}")
        
        return {'test_mape': mape, 'test_mae': mae, 'test_rmse': rmse, 'n_estimators_used': n_rounds_used}
    
    def _save_models(self, lstm_model, symbol, category):
        """Save models and config"""
        Path('backend/models/weights').mkdir(parents=True, exist_ok=True)
        Path('backend/models/config').mkdir(parents=True, exist_ok=True)
        torch.save(lstm_model.state_dict(), f'backend/models/weights/{symbol}_1h_v4_lstm.pth')
        
        config = AdaptiveHyperparameterConfig.get_config(category)
        config['volatility_category'] = category
        with open(f'backend/models/config/{symbol}_v4_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_batch(self, symbols=None, target_mape=0.02):
        """Batch training"""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                      'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                      'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT']
        
        logger.info(f"\nV4 Adaptive Batch Training: {len(symbols)} symbols")
        logger.info(f"Target MAPE: {target_mape*100:.2f}%\n")
        
        results, successful = [], 0
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...\n")
            result = self.train_symbol(symbol, target_mape)
            results.append(result)
            if result['status'] != 'error' and result.get('test_mape', 1) <= target_mape:
                successful += 1
        
        self._generate_summary(results, symbols, successful, target_mape)
        return results
    
    def _generate_summary(self, results, symbols, successful, target_mape):
        """Generate summary"""
        mape_list = [r['test_mape'] for r in results if r['status'] != 'error']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"V4 Adaptive Training Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Successful: {successful}/{len(symbols)} symbols achieved MAPE < {target_mape*100:.2f}%")
        
        if mape_list:
            logger.info(f"Average MAPE: {np.mean(mape_list)*100:.4f}%")
            logger.info(f"Median MAPE: {np.median(mape_list)*100:.4f}%")
            logger.info(f"Range: {np.min(mape_list)*100:.4f}% - {np.max(mape_list)*100:.4f}%")
        
        Path("backend/results").mkdir(parents=True, exist_ok=True)
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'successful': successful,
            'avg_mape': float(np.mean(mape_list)) if mape_list else None
        }
        
        with open(f"backend/results/v4_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    trainer = V4AdaptiveTrainer()
    trainer.train_batch(target_mape=0.02)
