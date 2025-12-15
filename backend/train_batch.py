#!/usr/bin/env python
r"""
Batch Training Script for 1h + 15m Timeframes

Train 20+ cryptocurrencies across multiple timeframes simultaneously.

Usage:
    # Train all 20+ pairs in 1h
    python backend/train_batch.py --all --timeframe 1h
    
    # Train all 20+ pairs in 15m
    python backend/train_batch.py --all --timeframe 15m
    
    # Train both 1h and 15m (one after another)
    python backend/train_batch.py --all --timeframe 1h 15m
    
    # Custom symbols
    python backend/train_batch.py --symbols BTCUSDT ETHUSDT --timeframe 1h --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
import sys
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import MODEL_CONFIG, DATA_CONFIG
from backend.data.data_loader import CryptoDataLoader
from backend.data.data_manager import DataManager
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedCryptoGRU(nn.Module):
    """GRU-based model with attention mechanism"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, output_size=1):
        super(ImprovedCryptoGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout / 2)
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.attention = nn.Linear(hidden_size, 1)
        self.attention_softmax = nn.Softmax(dim=1)
        
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense1_bn = nn.BatchNorm1d(hidden_size // 2)
        self.dense1_dropout = nn.Dropout(dropout)
        
        self.dense2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dense2_bn = nn.BatchNorm1d(hidden_size // 4)
        self.dense2_dropout = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() == 1:
                    nn.init.constant_(param, 0)
                elif 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
    
    def forward(self, x):
        x = self.relu(self.input_proj(x))
        x = self.input_dropout(x)
        
        gru_out, hidden = self.gru(x)
        
        attention_weights = self.attention(gru_out)
        attention_weights = self.attention_softmax(attention_weights)
        
        context = torch.sum(gru_out * attention_weights, dim=1)
        
        out = self.relu(self.dense1_bn(self.dense1(context)))
        out = self.dense1_dropout(out)
        
        out = self.relu(self.dense2_bn(self.dense2(out)))
        out = self.dense2_dropout(out)
        
        out = self.output_layer(out)
        
        return out

class BatchTrainer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        self.data_manager = DataManager()
        self.data_loader = CryptoDataLoader()
        self.results = defaultdict(lambda: defaultdict(dict))
    
    def train_single(self, symbol, timeframe='1h', epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train single cryptocurrency on specific timeframe
        Returns: success (bool), best_loss (float), message (str)
        """
        try:
            # Load data
            data = self.data_manager.get_stored_data(symbol, timeframe)
            if data is None or len(data) < 100:
                rows = len(data) if data is not None else 0
                return False, None, f"Insufficient data ({rows} rows)"
            
            # Calculate indicators
            data_with_indicators = self.data_loader.calculate_technical_indicators(data)
            if data_with_indicators is None or data_with_indicators.empty:
                return False, None, "Failed to calculate indicators"
            
            # Normalize
            feature_cols = list(data_with_indicators.columns)
            scaler = MinMaxScaler()
            data_normalized = data_with_indicators.copy()
            data_scaled = scaler.fit_transform(data_with_indicators[feature_cols])
            data_normalized[feature_cols] = data_scaled
            
            # Create sequences
            lookback = MODEL_CONFIG['lookback']
            X_list, y_list = [], []
            
            for i in range(len(data_normalized) - lookback):
                X_list.append(data_normalized[feature_cols].iloc[i:i+lookback].values)
                y_list.append(data_with_indicators['close'].iloc[i+lookback])
            
            if len(X_list) == 0:
                return False, None, "No sequences created"
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Normalize y
            close_min = data_with_indicators['close'].min()
            close_max = data_with_indicators['close'].max()
            y_normalized = (y - close_min) / (close_max - close_min)
            
            # Create tensors and split
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y_normalized).unsqueeze(1)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            val_size = int(len(dataset) * 0.1)
            train_size = len(dataset) - val_size
            
            if train_size < 1:
                return False, None, "Dataset too small after split"
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Model
            num_features = X.shape[2]
            model = ImprovedCryptoGRU(
                input_size=num_features,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                output_size=1
            ).to(self.device)
            
            # Training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=10, verbose=False, min_lr=1e-6
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_epoch = 0
            
            for epoch in range(epochs):
                # Train
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        pred = model(X_batch)
                        loss = criterion(pred, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best
                    model_dir = Path('backend/models/weights')
                    model_dir.mkdir(parents=True, exist_ok=True)
                    best_path = model_dir / f"{symbol}_{timeframe}_best.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': {
                            'input_size': num_features,
                            'hidden_size': 128,
                            'num_layers': 2,
                            'output_size': 1,
                        },
                        'scalers': {
                            'close_min': float(close_min),
                            'close_max': float(close_max),
                        },
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, best_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:
                    break
            
            # Save final
            final_path = model_dir / f"{symbol}_{timeframe}_final.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': num_features,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'output_size': 1,
                },
                'scalers': {
                    'close_min': float(close_min),
                    'close_max': float(close_max),
                },
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, final_path)
            
            return True, best_val_loss, f"Loss={best_val_loss:.6f} (epoch {best_epoch})"
            
        except Exception as e:
            return False, None, str(e)
    
    def train_batch(self, symbols, timeframes, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train multiple cryptocurrencies across multiple timeframes
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Batch Training Configuration")
        logger.info(f"{'='*80}")
        logger.info(f"Cryptocurrencies: {len(symbols)}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
        logger.info(f"Total models to train: {len(symbols) * len(timeframes)}")
        logger.info(f"{'='*80}\n")
        
        success_count = 0
        failed_count = 0
        total = len(symbols) * len(timeframes)
        current = 0
        
        for timeframe in timeframes:
            logger.info(f"\n[TIMEFRAME: {timeframe}]\n")
            
            for idx, symbol in enumerate(symbols, 1):
                current += 1
                logger.info(f"[{current}/{total}] {symbol} ({timeframe})", end=' ')
                
                success, best_loss, message = self.train_single(
                    symbol, timeframe, epochs, batch_size, learning_rate
                )
                
                if success:
                    logger.info(f"OK - {message}")
                    success_count += 1
                    self.results[timeframe][symbol]['status'] = 'Success'
                    self.results[timeframe][symbol]['best_loss'] = best_loss
                else:
                    logger.warning(f"FAIL - {message}")
                    failed_count += 1
                    self.results[timeframe][symbol]['status'] = 'Failed'
                    self.results[timeframe][symbol]['error'] = message
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total: {total} | Success: {success_count} | Failed: {failed_count}\n")
        
        for timeframe in timeframes:
            logger.info(f"[{timeframe}]")
            for symbol in symbols:
                result = self.results[timeframe][symbol]
                if result['status'] == 'Success':
                    loss = result['best_loss']
                    logger.info(f"  {symbol:12s} OK   (loss={loss:.6f})")
                else:
                    error = result.get('error', 'Unknown error')
                    logger.info(f"  {symbol:12s} FAIL ({error[:40]})")
        
        logger.info(f"\nModels saved to: backend/models/weights/")
        logger.info(f"{'='*80}")

def get_default_symbols():
    """20+ default cryptocurrency symbols"""
    return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
        'XRPUSDT', 'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'LINKUSDT',
        'SUSHIUSDT', 'AVAXUSDT', 'SOLUSDT', 'FTMUSDT', 'ATOMUSDT',
        'NEARUSDT', 'APTUSDT', 'ARBITUSDT', 'OPTIMUSDT', 'MKRUSDT',
    ]

def main():
    parser = ArgumentParser(description='Batch train crypto models')
    parser.add_argument('--symbols', nargs='+', help='Symbols (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--all', action='store_true', help='Train all 20+ default symbols')
    parser.add_argument('--timeframe', nargs='+', default=['1h'], help='Timeframes (e.g., 1h 15m)')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.all:
        symbols = get_default_symbols()
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Train
    trainer = BatchTrainer()
    trainer.train_batch(
        symbols,
        args.timeframe,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

if __name__ == '__main__':
    main()
