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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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

class DirectionalLSTM(nn.Module):
    """LSTM for directional classification (Up/Neutral/Down)"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(DirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes: Down, Neutral, Up
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last output
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        return logits

class DirectionalTrainer:
    """Train directional prediction model for sharp, actionable predictions"""
    
    def __init__(self, device=None, threshold_pct=2.0, use_weighted_loss=True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = DataManager()
        self.data_loader = CryptoDataLoader()
        self.threshold_pct = threshold_pct  # Classification threshold
        self.use_weighted_loss = use_weighted_loss
        logger.info(f'Device: {self.device}')
        logger.info(f'Directional Threshold: {threshold_pct}%')
        
        self.hyperparams = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-5,
        }
        logger.info(f'Hyperparameters: {self.hyperparams}')
    
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
    
    def create_directional_labels(self, prices):
        """
        Create labels based on price direction
        0: Down (< -threshold_pct%)
        1: Neutral (-threshold_pct% to +threshold_pct%)
        2: Up (> +threshold_pct%)
        """
        labels = []
        for i in range(len(prices) - 1):
            pct_change = ((prices[i + 1] - prices[i]) / prices[i]) * 100
            
            if pct_change > self.threshold_pct:
                labels.append(2)  # Up
            elif pct_change < -self.threshold_pct:
                labels.append(0)  # Down
            else:
                labels.append(1)  # Neutral
        
        return np.array(labels)
    
    def prepare_data(self, symbol, timeframe='1h'):
        """Prepare data with directional labels"""
        df = self.dm.get_stored_data(symbol, timeframe)
        if df is None:
            logger.error(f'Failed to load data for {symbol}')
            return None, None, None, None
        
        df_ind = self.data_loader.calculate_technical_indicators(df)
        if df_ind is None:
            logger.error('Failed to calculate indicators')
            return None, None, None, None
        
        # Select features
        selected_features = self.select_bb_focused_features(df_ind, top_k=18)
        logger.info(f'Using {len(selected_features)} features')
        
        # Normalize features
        scaler = MinMaxScaler()
        df_normalized = df_ind.copy()
        df_normalized[selected_features] = scaler.fit_transform(
            df_ind[selected_features]
        )
        
        # Create directional labels
        prices = df_ind['close'].values
        labels = self.create_directional_labels(prices)
        
        # Log class distribution
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f'Label distribution: Down={counts[0] if 0 in unique else 0}, Neutral={counts[1] if 1 in unique else 0}, Up={counts[2] if 2 in unique else 0}')
        
        # Create sequences (remove last label since we predict next direction)
        lookback_period = 60
        X_list = []
        y_list = []
        
        for i in range(len(df_normalized) - lookback_period - 1):
            X_list.append(df_normalized[selected_features].iloc[i:i+lookback_period].values)
            y_list.append(labels[i + lookback_period])  # Label for next period
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        
        logger.info(f'{symbol}: Data shape X={X.shape}, y={y.shape}')
        
        return X, y, selected_features, prices
    
    def compute_class_weights(self, labels):
        """Compute weights for imbalanced classes"""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = {}
        for u, c in zip(unique, counts):
            weights[u] = total / (3 * c)  # Normalize to 3 classes
        
        # Convert to tensor
        class_weights = torch.tensor(
            [weights.get(i, 1.0) for i in range(3)],
            dtype=torch.float32
        ).to(self.device)
        logger.info(f'Class weights: {class_weights}')
        return class_weights
    
    def train_model(self, symbol, timeframe='1h', epochs=100):
        """Train directional model"""
        logger.info(f'\n{"="*80}')
        logger.info(f'[V2 Directional] Training {symbol}')
        logger.info(f'{"="*80}')
        
        # Prepare data
        result = self.prepare_data(symbol, timeframe)
        if result[0] is None:
            return False
        X, y, selected_features, prices = result
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        
        logger.info(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
        
        # Create model
        model = DirectionalLSTM(
            input_size=X_train.shape[2],
            hidden_size=self.hyperparams['hidden_size'],
            num_layers=self.hyperparams['num_layers'],
            dropout=self.hyperparams['dropout']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hyperparams['lr'],
            weight_decay=self.hyperparams['weight_decay']
        )
        
        # Loss with class weights
        class_weights = self.compute_class_weights(y_train) if self.use_weighted_loss else None
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Convert data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        batch_size = self.hyperparams['batch_size']
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        best_epoch = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                train_correct += (pred == batch_y).sum().item()
                num_batches += 1
            
            train_loss /= num_batches
            train_acc = train_correct / len(X_train)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_pred = torch.argmax(val_logits, dim=1)
                val_acc = (val_pred == y_val_tensor).float().mean().item()
                
                test_logits = model(X_test_tensor)
                test_pred = torch.argmax(test_logits, dim=1)
                test_acc = (test_pred == y_test_tensor).float().mean().item()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                model_path = f'backend/models/weights/{symbol}_1h_v2_directional_best.pth'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'input_size': X_train.shape[2],
                        'hidden_size': self.hyperparams['hidden_size'],
                        'num_layers': self.hyperparams['num_layers'],
                    },
                    'epoch': epoch,
                    'version': 'v2_directional',
                }, model_path)
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or patience_counter >= patience:
                logger.info(f'Epoch {epoch+1:3d}: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}, Loss={train_loss:.6f}, P={patience_counter}/{patience}')
            
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1} (best: {best_epoch+1})')
                break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
            test_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
        
        # Metrics
        test_acc = accuracy_score(y_test, test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, test_pred)
        
        logger.info(f'\n{symbol} V2 Directional Final Results:')
        logger.info(f'  Overall Accuracy: {test_acc:.4f}')
        logger.info(f'  Precision (weighted): {precision:.4f}')
        logger.info(f'  Recall (weighted): {recall:.4f}')
        logger.info(f'  F1 Score (weighted): {f1:.4f}')
        logger.info(f'  Confusion Matrix:\n{cm}')
        
        return True
    
    def train_all_symbols(self, symbols=None, timeframe='1h', epochs=100):
        """Train all symbols"""
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f'\n{"#"*80}')
        logger.info(f'V2 Directional Training: {len(symbols)} symbols')
        logger.info(f'{"#"*80}')
        
        success_count = 0
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f'\n[{idx}/{len(symbols)}] Processing {symbol}...')
            try:
                if self.train_model(symbol, timeframe, epochs):
                    success_count += 1
            except Exception as e:
                logger.error(f'Error training {symbol}: {str(e)}')
                import traceback
                traceback.print_exc()
        
        logger.info(f'\n{"#"*80}')
        logger.info(f'Training Summary: {success_count}/{len(symbols)} symbols trained')
        logger.info(f'{"#"*80}')
        return success_count

def main():
    parser = argparse.ArgumentParser(description='Train directional LSTM (Up/Down/Neutral)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--threshold', type=float, default=2.0, help='Classification threshold (%)')
    parser.add_argument('--device', default=None, help='Device')
    
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    
    try:
        trainer = DirectionalTrainer(device=device, threshold_pct=args.threshold)
        success = trainer.train_all_symbols(
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
