import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import pickle
import argparse

backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm_model import CryptoLSTM
from data.data_manager import DataManager
from data.data_loader import CryptoDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMFeatureExtractor(nn.Module):
    """LSTM for extracting temporal features"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x):
        """Extract LSTM features (hidden state)"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state as features
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        return last_hidden, lstm_out

class LSTMXGBoostTrainer:
    """Train hybrid LSTM+XGBoost model for price prediction"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = DataManager()
        self.data_loader = CryptoDataLoader()
        logger.info(f'Device: {self.device}')
        
        # LSTM config
        self.lstm_config = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-5,
            'epochs': 50,
        }
        
        # XGBoost config
        self.xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'random_state': 42,
        }
        
        logger.info(f'LSTM Config: {self.lstm_config}')
        logger.info(f'XGBoost Config: {self.xgb_params}')
    
    def select_bb_focused_features(self, df_indicators, top_k=18):
        """Select BB-focused features"""
        core_features = [
            'bb_upper', 'bb_middle', 'bb_lower', 
            'bb_percent_b', 'bb_width'
        ]
        
        support_features = [
            'ema_10', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi_14', 'rsi_21',
            'volume', 'volume_change',
            'atr_14', 'di_plus', 'di_minus',
            'mfi', 'obv'
        ]
        
        available_core = [f for f in core_features if f in df_indicators.columns]
        available_support = [f for f in support_features if f in df_indicators.columns]
        
        if len(available_support) > (top_k - len(available_core)):
            needed = top_k - len(available_core)
            X_support = df_indicators[available_support].fillna(0)
            y = df_indicators['close'].values
            
            selector = SelectKBest(f_regression, k=needed)
            selector.fit(X_support, y)
            
            selected_support = [
                available_support[i] for i in selector.get_support(indices=True)
            ]
        else:
            selected_support = available_support
        
        final_features = available_core + selected_support
        return final_features[:top_k]
    
    def prepare_data(self, symbol, timeframe='1h'):
        """Prepare training data"""
        df = self.dm.get_stored_data(symbol, timeframe)
        if df is None:
            logger.error(f'Failed to load data for {symbol}')
            return None, None, None, None, None
        
        df_ind = self.data_loader.calculate_technical_indicators(df)
        if df_ind is None:
            logger.error('Failed to calculate indicators')
            return None, None, None, None, None
        
        # Select features
        selected_features = self.select_bb_focused_features(df_ind, top_k=18)
        logger.info(f'Using {len(selected_features)} features')
        
        # Normalize features
        scaler = MinMaxScaler()
        df_normalized = df_ind.copy()
        df_normalized[selected_features] = scaler.fit_transform(
            df_ind[selected_features]
        )
        
        close_min = df_ind['close'].min()
        close_max = df_ind['close'].max()
        close_range = close_max - close_min
        
        # Normalize target
        df_normalized['close_norm'] = (df_ind['close'] - close_min) / close_range
        
        # Create sequences
        lookback_period = 60
        X_list = []
        y_list = []
        xgb_features_list = []  # Store technical indicators for XGBoost
        
        for i in range(len(df_normalized) - lookback_period):
            X_list.append(df_normalized[selected_features].iloc[i:i+lookback_period].values)
            y_list.append(df_normalized['close_norm'].iloc[i+lookback_period])
            # Store the last technical indicator values for XGBoost
            xgb_features_list.append(df_normalized[selected_features].iloc[i+lookback_period].values)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        xgb_features = np.array(xgb_features_list, dtype=np.float32)
        
        logger.info(f'{symbol}: Data shape X={X.shape}, xgb_features={xgb_features.shape}, y={y.shape}')
        logger.info(f'Close price range: {close_min:.2f} - {close_max:.2f}')
        
        return X, y, xgb_features, (close_min, close_max), selected_features
    
    def train_lstm_extractor(self, symbol, X_train, X_val, y_train, y_val):
        """Stage 1: Train LSTM to extract features"""
        logger.info(f'\n{"="*80}')
        logger.info(f'Stage 1: Training LSTM Feature Extractor for {symbol}')
        logger.info(f'{"="*80}')
        
        # Create model
        model = LSTMFeatureExtractor(
            input_size=X_train.shape[2],
            hidden_size=self.lstm_config['hidden_size'],
            num_layers=self.lstm_config['num_layers'],
            dropout=self.lstm_config['dropout']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lstm_config['lr'],
            weight_decay=self.lstm_config['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # Convert data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        batch_size = self.lstm_config['batch_size']
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.lstm_config['epochs']):
            model.train()
            train_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                hidden, _ = model(batch_X)
                # Use hidden state as features, train auxiliary regression head
                aux_out = hidden.mean(dim=1, keepdim=True)  # Predict from features
                loss = criterion(aux_out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_hidden, _ = model(X_val_tensor)
                val_aux = val_hidden.mean(dim=1, keepdim=True)
                val_loss = criterion(val_aux, y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, P={patience_counter}/{patience}')
            
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        return model
    
    def extract_lstm_features(self, model, X_data):
        """Extract LSTM hidden states as features for XGBoost"""
        X_tensor = torch.FloatTensor(X_data).to(self.device)
        model.eval()
        
        with torch.no_grad():
            hidden_states, _ = model(X_tensor)
        
        return hidden_states.cpu().numpy()
    
    def train_xgboost(self, symbol, lstm_features_train, y_train, lstm_features_val, y_val):
        """Stage 2: Train XGBoost on LSTM features"""
        logger.info(f'\n{"="*80}')
        logger.info(f'Stage 2: Training XGBoost Regressor for {symbol}')
        logger.info(f'{"="*80}')
        
        # Create XGBoost model - simple training without early stopping
        xgb_model = xgb.XGBRegressor(
            **self.xgb_params,
            tree_method='auto',
            verbosity=1
        )
        
        # Train
        logger.info(f'Training on {len(lstm_features_train)} samples')
        xgb_model.fit(
            lstm_features_train, y_train,
            verbose=True
        )
        
        logger.info(f'XGBoost training completed')
        return xgb_model
    
    def evaluate_model(self, symbol, lstm_model, xgb_model, X_test, y_test, close_range):
        """Evaluate combined model"""
        # Extract LSTM features
        lstm_features = self.extract_lstm_features(lstm_model, X_test)
        
        # XGBoost prediction
        y_pred_norm = xgb_model.predict(lstm_features)
        
        # Denormalize
        close_min, close_max = close_range
        close_range_val = close_max - close_min
        
        y_test_denorm = y_test * close_range_val + close_min
        y_pred_denorm = y_pred_norm * close_range_val + close_min
        
        # Metrics
        mae = np.mean(np.abs(y_test_denorm - y_pred_denorm))
        rmse = np.sqrt(np.mean((y_test_denorm - y_pred_denorm) ** 2))
        mape = np.mean(np.abs((y_test_denorm - y_pred_denorm) / np.abs(y_test_denorm))) * 100
        
        logger.info(f'\n{symbol} V3 LSTM+XGBoost Final Results:')
        logger.info(f'  MAPE: {mape:.2f}%')
        logger.info(f'  MAE: ${mae:.2f}')
        logger.info(f'  RMSE: ${rmse:.2f}')
        
        return {'mape': mape, 'mae': mae, 'rmse': rmse}
    
    def train_model(self, symbol, timeframe='1h'):
        """Train complete LSTM+XGBoost model"""
        logger.info(f'\n{"#"*80}')
        logger.info(f'Training {symbol} with LSTM+XGBoost (V3)')
        logger.info(f'{"#"*80}')
        
        # Prepare data
        result = self.prepare_data(symbol, timeframe)
        if result[0] is None:
            return False
        X, y, xgb_features, close_range, feature_names = result
        
        # Split: 70% train, 15% val, 15% test (NO SHUFFLE for time series)
        total_len = len(X)
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.15)
        
        X_train = X[:train_len]
        y_train = y[:train_len]
        xgb_train = xgb_features[:train_len]
        
        X_val = X[train_len:train_len+val_len]
        y_val = y[train_len:train_len+val_len]
        xgb_val = xgb_features[train_len:train_len+val_len]
        
        X_test = X[train_len+val_len:]
        y_test = y[train_len+val_len:]
        xgb_test = xgb_features[train_len+val_len:]
        
        logger.info(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
        logger.info(f'XGB Train: {xgb_train.shape}, XGB Val: {xgb_val.shape}, XGB Test: {xgb_test.shape}')
        
        try:
            # Stage 1: Train LSTM
            lstm_model = self.train_lstm_extractor(symbol, X_train, X_val, y_train, y_val)
            
            # Extract features
            lstm_train_features = self.extract_lstm_features(lstm_model, X_train)
            lstm_val_features = self.extract_lstm_features(lstm_model, X_val)
            lstm_test_features = self.extract_lstm_features(lstm_model, X_test)
            
            logger.info(f'LSTM features extracted: train={lstm_train_features.shape}, val={lstm_val_features.shape}, test={lstm_test_features.shape}')
            
            # Stage 2: Train XGBoost
            xgb_model = self.train_xgboost(
                symbol,
                lstm_train_features, y_train,
                lstm_val_features, y_val
            )
            
            # Evaluate
            metrics = self.evaluate_model(symbol, lstm_model, xgb_model, X_test, y_test, close_range)
            
            # Save models
            lstm_path = f'backend/models/weights/{symbol}_1h_v3_lstm.pth'
            xgb_path = f'backend/models/weights/{symbol}_1h_v3_xgb.pkl'
            
            os.makedirs(os.path.dirname(lstm_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': lstm_model.state_dict(),
                'config': {
                    'input_size': X_train.shape[2],
                    'hidden_size': self.lstm_config['hidden_size'],
                    'num_layers': self.lstm_config['num_layers'],
                },
                'version': 'v3_lstm_xgboost',
            }, lstm_path)
            
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            
            logger.info(f'Models saved: {lstm_path}, {xgb_path}')
            
            return True
        
        except Exception as e:
            logger.error(f'Error training {symbol}: {str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def train_all_symbols(self, symbols=None, timeframe='1h'):
        """Train all symbols"""
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f'\nV3 Training: {len(symbols)} symbols with LSTM+XGBoost')
        
        success_count = 0
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f'\n[{idx}/{len(symbols)}] Processing {symbol}...')
            try:
                if self.train_model(symbol, timeframe):
                    success_count += 1
            except Exception as e:
                logger.error(f'Error training {symbol}: {str(e)}')
        
        logger.info(f'\nTraining Summary: {success_count}/{len(symbols)} symbols trained successfully')
        return success_count

def main():
    parser = argparse.ArgumentParser(description='Train LSTM+XGBoost hybrid model')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--device', default=None, help='Device')
    
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    
    try:
        trainer = LSTMXGBoostTrainer(device=device)
        success = trainer.train_all_symbols(
            symbols=args.symbols,
            timeframe=args.timeframe
        )
        return 0 if success > 0 else 1
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
