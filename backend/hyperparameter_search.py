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
import argparse
import json
from itertools import product

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

class HyperparameterSearch:
    def __init__(self, symbol='BTCUSDT', device=None):
        self.symbol = symbol
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = DataManager()
        self.data_loader = CryptoDataLoader()
        logger.info(f'Device: {self.device}')
    
    def prepare_data(self, timeframe='1h'):
        """準備訓練數據"""
        df = self.dm.get_stored_data(self.symbol, timeframe)
        if df is None:
            logger.error(f'Failed to load data for {self.symbol}')
            return None, None, None
        
        df_ind = self.data_loader.calculate_technical_indicators(df)
        if df_ind is None:
            logger.error('Failed to calculate indicators')
            return None, None, None
        
        feature_cols = [col for col in df_ind.columns if col != 'timestamp']
        scaler = MinMaxScaler()
        df_normalized = df_ind.copy()
        df_normalized[feature_cols] = scaler.fit_transform(df_ind[feature_cols])
        
        close_min = df_ind['close'].min()
        close_max = df_ind['close'].max()
        
        # Create sequences
        lookback_period = 60
        X_list = []
        y_list = []
        
        for i in range(len(df_normalized) - lookback_period):
            X_list.append(df_normalized[feature_cols].iloc[i:i+lookback_period].values)
            y_list.append(df_ind['close'].iloc[i+lookback_period])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        logger.info(f'Data shape: X={X.shape}, y={y.shape}')
        
        return X, y, (close_min, close_max)
    
    def train_model(self, X_train, y_train, X_val, y_val, hyperparams, epochs=50):
        """訓練單個模型"""
        model = CryptoLSTM(
            input_size=X_train.shape[2],
            hidden_size=hyperparams['hidden_size'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout'],
            output_size=1
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        batch_size = hyperparams['batch_size']
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
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
            
            train_loss /= (len(X_train) // batch_size + 1)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f'  Early stopping at epoch {epoch}')
                break
        
        return model, train_loss, val_loss
    
    def evaluate_model(self, model, X_test, y_test, close_range):
        """評估模型 MAPE"""
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()
        
        # Denormalize
        close_min, close_max = close_range
        y_test_denorm = y_test * (close_max - close_min) + close_min
        pred_denorm = predictions.squeeze() * (close_max - close_min) + close_min
        
        mape = np.mean(np.abs((y_test_denorm - pred_denorm) / np.abs(y_test_denorm))) * 100
        mae = np.mean(np.abs(y_test_denorm - pred_denorm))
        rmse = np.sqrt(np.mean((y_test_denorm - pred_denorm) ** 2))
        
        return mape, mae, rmse
    
    def grid_search(self, param_grid=None, timeframe='1h', epochs=50, test_size=0.2):
        """執行網格搜索"""
        if param_grid is None:
            param_grid = {
                'hidden_size': [64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.2, 0.3, 0.4],
                'lr': [0.001, 0.0005, 0.0001],
                'batch_size': [16, 32, 64],
                'weight_decay': [0, 1e-5, 1e-4]
            }
        
        # Prepare data
        X, y, close_range = self.prepare_data(timeframe)
        if X is None:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        
        logger.info(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
        
        # Generate parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        logger.info(f'Total combinations: {len(combinations)}')
        
        results = []
        
        for idx, combo in enumerate(combinations, 1):
            hyperparams = dict(zip(keys, combo))
            
            logger.info(f'\n[{idx}/{len(combinations)}] Testing: {hyperparams}')
            
            try:
                model, train_loss, val_loss = self.train_model(
                    X_train, y_train, X_val, y_val, hyperparams, epochs
                )
                
                mape, mae, rmse = self.evaluate_model(model, X_test, y_test, close_range)
                
                result = {
                    'hyperparams': hyperparams,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'test_mape': float(mape),
                    'test_mae': float(mae),
                    'test_rmse': float(rmse),
                }
                results.append(result)
                
                logger.info(f'  MAPE: {mape:.2f}% | MAE: ${mae:.2f} | RMSE: ${rmse:.2f}')
            
            except Exception as e:
                logger.error(f'  Error: {str(e)}')
        
        return results
    
    def save_results(self, results, output_dir='results'):
        """保存搜索結果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(output_dir, f'hyperparameter_search_{self.symbol}_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f'Results saved to {results_path}')
        
        # Sort by MAPE and save top 10
        df = pd.DataFrame(results)
        df = df.sort_values('test_mape')
        
        csv_path = os.path.join(output_dir, f'hyperparameter_search_{self.symbol}_{timestamp}_top.csv')
        df.head(10).to_csv(csv_path, index=False)
        logger.info(f'Top 10 saved to {csv_path}')
        
        return results_path, csv_path, df

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter grid search')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per model')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick search (smaller grid)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    try:
        searcher = HyperparameterSearch(symbol=args.symbol, device=device)
        
        # Quick search with fewer options
        if args.quick:
            param_grid = {
                'hidden_size': [64, 128],
                'num_layers': [2],
                'dropout': [0.2, 0.3],
                'lr': [0.001, 0.0005],
                'batch_size': [32, 64],
                'weight_decay': [0, 1e-5]
            }
            logger.info('Running QUICK hyperparameter search...')
        else:
            param_grid = None
            logger.info('Running FULL hyperparameter search...')
        
        results = searcher.grid_search(
            param_grid=param_grid,
            timeframe=args.timeframe,
            epochs=args.epochs
        )
        
        if results:
            json_path, csv_path, df = searcher.save_results(results, args.output_dir)
            
            logger.info('\n' + '='*100)
            logger.info('TOP 10 CONFIGURATIONS')
            logger.info('='*100)
            for idx, row in df.head(10).iterrows():
                logger.info(f"\nMAP: {row['test_mape']:.2f}%")
                logger.info(f"Config: {row['hyperparams']}")
            logger.info('\n' + '='*100)
        
        return 0
    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
