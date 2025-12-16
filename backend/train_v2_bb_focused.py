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

class BBFocusedTrainer:
    """Train with Bollinger Bands as core feature for improved MAPE"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = DataManager()
        self.data_loader = CryptoDataLoader()
        logger.info(f'Device: {self.device}')
        
        # Optimized hyperparameters for v2
        self.hyperparams = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-5,
        }
        logger.info(f'V2 Hyperparameters (BB-focused): {self.hyperparams}')
    
    def select_bb_focused_features(self, df_indicators, top_k=18):
        """
        Select features with Bollinger Bands as core + supporting indicators
        Core BB features: bb_upper, bb_middle, bb_lower, bb_percent_b, bb_width
        Supporting: EMA, MACD, RSI, Volume, ATR
        """
        # Core BB features - always include
        core_features = [
            'bb_upper', 'bb_middle', 'bb_lower', 
            'bb_percent_b', 'bb_width'
        ]
        
        # Supporting features
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
        
        logger.info(f'Core BB features: {available_core}')
        logger.info(f'Available support features: {len(available_support)}')
        
        # If we have enough support features, use statistical selection
        if len(available_support) > (top_k - len(available_core)):
            needed = top_k - len(available_core)
            X_support = df_indicators[available_support].fillna(0)
            y = df_indicators['close'].values
            
            selector = SelectKBest(f_regression, k=needed)
            selector.fit(X_support, y)
            
            selected_support = [
                available_support[i] for i in selector.get_support(indices=True)
            ]
            logger.info(f'Selected {len(selected_support)} support features by mutual information')
        else:
            selected_support = available_support
        
        final_features = available_core + selected_support
        return final_features[:top_k]
    
    def prepare_data(self, symbol, timeframe='1h'):
        """Prepare training data with BB-focused features"""
        df = self.dm.get_stored_data(symbol, timeframe)
        if df is None:
            logger.error(f'Failed to load data for {symbol}')
            return None, None, None
        
        df_ind = self.data_loader.calculate_technical_indicators(df)
        if df_ind is None:
            logger.error('Failed to calculate indicators')
            return None, None, None
        
        # Select BB-focused features
        selected_features = self.select_bb_focused_features(df_ind, top_k=18)
        logger.info(f'Using {len(selected_features)} features: {selected_features}')
        
        # Normalize features
        scaler = MinMaxScaler()
        df_normalized = df_ind.copy()
        df_normalized[selected_features] = scaler.fit_transform(
            df_ind[selected_features]
        )
        
        close_min = df_ind['close'].min()
        close_max = df_ind['close'].max()
        close_range = close_max - close_min
        
        # Normalize close price separately for denormalization
        df_normalized['close_norm'] = (df_ind['close'] - close_min) / close_range
        
        # Create sequences
        lookback_period = 60
        X_list = []
        y_list = []
        
        for i in range(len(df_normalized) - lookback_period):
            X_list.append(df_normalized[selected_features].iloc[i:i+lookback_period].values)
            y_list.append(df_normalized['close_norm'].iloc[i+lookback_period])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        logger.info(f'{symbol}: Data shape X={X.shape}, y={y.shape}')
        logger.info(f'Close price range: {close_min:.2f} - {close_max:.2f}')
        
        return X, y, (close_min, close_max)
    
    def train_model(self, symbol, timeframe='1h', epochs=100):
        """Train model with BB-focused features"""
        logger.info(f'\n{"="*80}')
        logger.info(f'[V2 BB-Focused] Training {symbol}')
        logger.info(f'{"="*80}')
        
        # Prepare data
        X, y, close_range = self.prepare_data(symbol, timeframe)
        if X is None:
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        
        logger.info(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
        
        # Create model
        model = CryptoLSTM(
            input_size=X_train.shape[2],
            hidden_size=self.hyperparams['hidden_size'],
            num_layers=self.hyperparams['num_layers'],
            dropout=self.hyperparams['dropout'],
            output_size=1
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hyperparams['lr'],
            weight_decay=self.hyperparams['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        
        batch_size = self.hyperparams['batch_size']
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_epoch = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            num_batches = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                model_path = f'backend/models/weights/{symbol}_1h_v2_best.pth'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'input_size': X_train.shape[2],
                        'hidden_size': self.hyperparams['hidden_size'],
                        'num_layers': self.hyperparams['num_layers'],
                        'output_size': 1,
                    },
                    'epoch': epoch,
                    'version': 'v2_bb_focused',
                }, model_path)
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or patience_counter >= patience:
                logger.info(f'Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, Test={test_loss:.6f}, Patience={patience_counter}/{patience}')
            
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1} (best: {best_epoch+1})')
                break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor).cpu().numpy()
        
        close_min, close_max = close_range
        close_range_val = close_max - close_min
        
        y_test_denorm = y_test * close_range_val + close_min
        pred_denorm = test_outputs.squeeze() * close_range_val + close_min
        
        mae = np.mean(np.abs(y_test_denorm - pred_denorm))
        rmse = np.sqrt(np.mean((y_test_denorm - pred_denorm) ** 2))
        mape = np.mean(np.abs((y_test_denorm - pred_denorm) / np.abs(y_test_denorm))) * 100
        
        logger.info(f'\n{symbol} V2 Final Results:')
        logger.info(f'  MAPE: {mape:.2f}%')
        logger.info(f'  MAE: ${mae:.2f}')
        logger.info(f'  RMSE: ${rmse:.2f}')
        
        return True
    
    def train_all_symbols(self, symbols=None, timeframe='1h', epochs=100):
        """Train all symbols with v2 BB-focused approach"""
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f'\n{"#"*80}')
        logger.info(f'V2 BB-Focused Training: {len(symbols)} symbols')
        logger.info(f'{"#"*80}')
        
        success_count = 0
        results = []
        
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f'\n[{idx}/{len(symbols)}] Processing {symbol}...')
            try:
                if self.train_model(symbol, timeframe, epochs):
                    success_count += 1
                    results.append({'symbol': symbol, 'status': 'success'})
                else:
                    results.append({'symbol': symbol, 'status': 'failed'})
            except Exception as e:
                logger.error(f'Error training {symbol}: {str(e)}')
                results.append({'symbol': symbol, 'status': 'error', 'error': str(e)})
                import traceback
                traceback.print_exc()
        
        logger.info(f'\n{"#"*80}')
        logger.info(f'Training Summary: {success_count}/{len(symbols)} symbols trained successfully')
        logger.info(f'{"#"*80}')
        
        return success_count, results

def main():
    parser = argparse.ArgumentParser(description='Train models with BB-focused features (v2)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    try:
        trainer = BBFocusedTrainer(device=device)
        success, results = trainer.train_all_symbols(
            symbols=args.symbols,
            timeframe=args.timeframe,
            epochs=args.epochs
        )
        
        return 0 if success > 0 else 1
    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
