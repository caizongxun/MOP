"""
V5 Enhanced Model Training

Key improvements over V4:
1. Residual Learning - Predict price deltas instead of absolute values
2. Multi-Scale LSTM - Capture patterns at different time scales
3. Uncertainty Quantification - Learn confidence intervals
4. Better Feature Engineering - Add momentum, volatility, micro-structure
5. Advanced Regularization - Prevent mode collapse to mean value
6. Ensemble Loss - Combine multiple objectives (regression + directional + volatility)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_v5')

class FeatureEngineerV5:
    """V5 Feature Engineering - Multi-scale and advanced indicators"""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """Calculate comprehensive features"""
        features = pd.DataFrame(index=df.index)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Price features
        features['returns'] = pd.Series(close).pct_change() * 100
        features['price_normalized'] = (close - close.min()) / (close.max() - close.min())
        
        # Multi-scale momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = pd.Series(close).diff(period)
            features[f'roc_{period}'] = pd.Series(close).pct_change(period) * 100
        
        # Volatility (critical for residual prediction)
        for period in [5, 10, 20]:
            returns = pd.Series(close).pct_change()
            features[f'volatility_{period}'] = returns.rolling(period).std() * 100
            features[f'vol_change_{period}'] = features[f'volatility_{period}'].diff()
        
        # Micro-structure
        features['hl_ratio'] = (high - low) / close * 100
        features['close_position'] = (close - low) / (high - low)
        features['range_expansion'] = pd.Series(high - low).rolling(5).mean()
        
        # Volume
        features['volume_ma_ratio'] = volume / pd.Series(volume).rolling(20).mean()
        features['volume_trend'] = pd.Series(volume).pct_change() * 100
        
        # Price acceleration
        for period in [5, 10]:
            price_diff1 = pd.Series(close).diff(period)
            price_diff2 = price_diff1.diff(period)
            features[f'acceleration_{period}'] = price_diff2
        
        # Mean reversion signal
        for period in [10, 20, 50]:
            sma = pd.Series(close).rolling(period).mean()
            features[f'deviation_from_sma_{period}'] = (close - sma) / sma * 100
        
        return features.fillna(0)
    
    @staticmethod
    def calculate_targets(df: pd.DataFrame, steps_ahead: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dual targets: price level and volatility"""
        close = df['close'].values
        
        # Target 1: Residual (delta from current)
        target_residual = close[steps_ahead:] - close[:-steps_ahead]
        
        # Target 2: Volatility (uncertainty)
        returns = np.diff(np.log(close[steps_ahead:]))
        target_volatility = np.abs(returns) * 100
        
        return target_residual, target_volatility

class MultiScaleLSTMV5(nn.Module):
    """V5 Multi-Scale LSTM with Uncertainty"""
    
    def __init__(self, input_size: int, hidden_size: int = 192, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Multi-scale LSTMs
        self.lstm_short = nn.LSTM(input_size, hidden_size // 2, num_layers, 
                                   batch_first=True, dropout=0.2)
        self.lstm_long = nn.LSTM(input_size, hidden_size // 2, num_layers, 
                                  batch_first=True, dropout=0.2)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Output heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        # Short-term pattern (first 30% of sequence)
        seq_len = x.size(1)
        short_len = max(seq_len // 3, 1)
        
        lstm_out_short, _ = self.lstm_short(x[:, :short_len, :])
        short_feat = lstm_out_short[:, -1, :]
        
        # Long-term pattern (full sequence)
        lstm_out_long, _ = self.lstm_long(x)
        long_feat = lstm_out_long[:, -1, :]
        
        # Fuse both scales
        combined = torch.cat([short_feat, long_feat], dim=1)
        features = self.fusion(combined)
        
        # Predictions
        price_pred = self.price_head(features)
        uncertainty = self.uncertainty_head(features)
        
        return price_pred, uncertainty

class V5EnhancedTrainer:
    """V5 Training Pipeline"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.feature_calc = FeatureEngineerV5()
    
    def load_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load data"""
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(backend_dir)
        data_path = os.path.join(root_dir, 'backend', 'data', 'raw', f'{symbol}_{timeframe}.csv')
        
        if not os.path.exists(data_path):
            data_path = os.path.join(backend_dir, 'data', 'raw', f'{symbol}_{timeframe}.csv')
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df
    
    def _prepare_data(self, features_df: pd.DataFrame, close_prices: np.ndarray,
                      lookback: int = 60) -> Tuple:
        """Prepare sequences"""
        feature_values = features_df.values
        target_delta, target_volatility = self.feature_calc.calculate_targets(close_prices)
        
        X, y_delta, y_vol = [], [], []
        
        for i in range(len(feature_values) - lookback - 1):
            X.append(feature_values[i:i+lookback])
            y_delta.append(target_delta[i+lookback])
            y_vol.append(target_volatility[i+lookback])
        
        X = np.array(X, dtype=np.float32)
        y_delta = np.array(y_delta, dtype=np.float32).reshape(-1, 1)
        y_vol = np.array(y_vol, dtype=np.float32).reshape(-1, 1)
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_delta = StandardScaler()
        scaler_vol = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_delta_scaled = scaler_delta.fit_transform(y_delta)
        y_vol_scaled = scaler_vol.fit_transform(y_vol)
        
        logger.info(f"Data shape: X={X_scaled.shape}, y_delta={y_delta_scaled.shape}, y_vol={y_vol_scaled.shape}")
        
        return X_scaled, y_delta_scaled, y_vol_scaled, scaler_X, scaler_delta, scaler_vol
    
    def train(self, symbol: str, num_symbols: int = 1, symbol_idx: int = 1,
              epochs: int = 150, batch_size: int = 16, learning_rate: float = 0.0005):
        """Train V5 model"""
        logger.info(f"\n[{symbol_idx}/{num_symbols}] Training {symbol} with V5 Enhanced")
        
        # Load and prepare data
        df = self.load_data(symbol)
        close_prices = df['close'].values
        
        features_df = self.feature_calc.calculate_features(df)
        X, y_delta, y_vol, scaler_X, scaler_delta, scaler_vol = self._prepare_data(
            features_df, close_prices
        )
        
        # Split: 70% train, 15% val, 15% test
        train_idx = int(0.70 * len(X))
        val_idx = int(0.85 * len(X))
        
        X_train, y_delta_train, y_vol_train = X[:train_idx], y_delta[:train_idx], y_vol[:train_idx]
        X_val, y_delta_val, y_vol_val = X[train_idx:val_idx], y_delta[train_idx:val_idx], y_vol[train_idx:val_idx]
        X_test, y_delta_test, y_vol_test = X[val_idx:], y_delta[val_idx:], y_vol[val_idx:]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_delta_train),
            torch.FloatTensor(y_vol_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_delta_val),
            torch.FloatTensor(y_vol_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Model
        model = MultiScaleLSTMV5(input_size=X.shape[2]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                          patience=15, verbose=False)
        
        # Loss with weights
        def combined_loss(pred_delta, pred_vol, true_delta, true_vol, uncertainty):
            # Regression loss on delta
            delta_loss = torch.mean((pred_delta - true_delta) ** 2)
            
            # Uncertainty-weighted loss
            weighted_loss = torch.mean(((pred_delta - true_delta) ** 2) / (2 * uncertainty ** 2 + 1e-4))
            
            # Volatility loss
            vol_loss = torch.mean((pred_vol - true_vol) ** 2)
            
            # Total
            return 0.6 * weighted_loss + 0.3 * delta_loss + 0.1 * vol_loss
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Training...")
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0
            
            for X_batch, y_delta_batch, y_vol_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_delta_batch = y_delta_batch.to(self.device)
                y_vol_batch = y_vol_batch.to(self.device)
                
                pred_delta, pred_vol = model(X_batch)
                loss = combined_loss(pred_delta, pred_vol, y_delta_batch, y_vol_batch, pred_vol)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_delta_batch, y_vol_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_delta_batch = y_delta_batch.to(self.device)
                    y_vol_batch = y_vol_batch.to(self.device)
                    
                    pred_delta, pred_vol = model(X_batch)
                    loss = combined_loss(pred_delta, pred_vol, y_delta_batch, y_vol_batch, pred_vol)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'weights',
                                      f'{symbol}_1h_v5_lstm.pth'))
            else:
                patience_counter += 1
                if patience_counter >= 25:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            y_pred_delta, y_pred_vol = model(X_test_t)
            y_pred_delta = y_pred_delta.cpu().numpy()
            y_pred_vol = y_pred_vol.cpu().numpy()
        
        # Inverse transform
        y_delta_test_orig = scaler_delta.inverse_transform(y_delta_test)
        y_pred_delta_orig = scaler_delta.inverse_transform(y_pred_delta)
        
        # Calculate metrics on original scale
        mape = mean_absolute_percentage_error(y_delta_test_orig, y_pred_delta_orig)
        mae = mean_absolute_error(y_delta_test_orig, y_pred_delta_orig)
        rmse = np.sqrt(mean_squared_error(y_delta_test_orig, y_pred_delta_orig))
        
        logger.info(f"\nTest Results for {symbol}:")
        logger.info(f"  MAPE (Delta): {mape*100:.2f}%")
        logger.info(f"  MAE: ${mae:.4f}")
        logger.info(f"  RMSE: ${rmse:.4f}")
        
        # Save config
        config = {
            'lstm': {'hidden_size': 192, 'num_layers': 2},
            'training': {'epochs': epoch, 'batch_size': batch_size, 'lr': learning_rate},
            'metrics': {'mape': float(mape), 'mae': float(mae), 'rmse': float(rmse)}
        }
        
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'config',
                                   f'{symbol}_v5_config.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return mape

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('count', type=int, help='Number of symbols to train')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    # Default symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 
               'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT',
               'ATOMUSDT', 'UNIUSDT', 'APTUSDT'][:args.count]
    
    trainer = V5EnhancedTrainer(device=args.device)
    mapes = []
    
    for idx, symbol in enumerate(symbols, 1):
        mape = trainer.train(symbol, num_symbols=len(symbols), symbol_idx=idx)
        mapes.append(mape)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"V5 Training Complete")
    logger.info(f"Average MAPE: {np.mean(mapes)*100:.2f}%")
    logger.info(f"Best: {np.min(mapes)*100:.2f}% | Worst: {np.max(mapes)*100:.2f}%")
