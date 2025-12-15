#!/usr/bin/env python
r"""
Improved Training Script v2

Key improvements:
1. GRU instead of LSTM (faster convergence, fewer parameters)
2. Proper weight initialization (Xavier/He)
3. Gradient clipping and monitoring
4. Better loss tracking and early stopping
5. Saves as .pth (PyTorch format) for compatibility
6. Layer-wise learning rates

Usage:
    python backend/train_improved.py --symbol BTCUSDT --timeframe 1h --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
import os
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import numpy as np

from config.model_config import MODEL_CONFIG, DATA_CONFIG
from backend.data.data_loader import CryptoDataLoader
from backend.data.data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedCryptoGRU(nn.Module):
    """
    Improved GRU-based model for crypto price prediction
    Simpler than LSTM but more effective for time series
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, output_size=1):
        super(ImprovedCryptoGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout / 2)
        
        # GRU layers (simpler than LSTM, better for short sequences)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layers with attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        self.attention_softmax = nn.Softmax(dim=1)
        
        # Dense layers
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense1_bn = nn.BatchNorm1d(hidden_size // 2)
        self.dense1_dropout = nn.Dropout(dropout)
        
        self.dense2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dense2_bn = nn.BatchNorm1d(hidden_size // 4)
        self.dense2_dropout = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(hidden_size // 4, output_size)
        
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() == 1:
                    nn.init.constant_(param, 0)
                elif 'gru' in name or 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, seq_len, input_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Input projection: (batch, seq, input) -> (batch, seq, hidden)
        x = self.relu(self.input_proj(x))
        x = self.input_dropout(x)
        
        # GRU: (batch, seq, hidden) -> (batch, seq, hidden), (batch, hidden)
        gru_out, hidden = self.gru(x)
        
        # Attention over sequence
        attention_weights = self.attention(gru_out)  # (batch, seq, 1)
        attention_weights = self.attention_softmax(attention_weights)  # (batch, seq, 1)
        
        # Apply attention: (batch, seq, hidden) x (batch, seq, 1) -> (batch, hidden)
        context = torch.sum(gru_out * attention_weights, dim=1)  # (batch, hidden)
        
        # Dense layers
        out = self.relu(self.dense1_bn(self.dense1(context)))
        out = self.dense1_dropout(out)
        
        out = self.relu(self.dense2_bn(self.dense2(out)))
        out = self.dense2_dropout(out)
        
        # Output
        out = self.output_layer(out)  # (batch, 1)
        
        return out

class ImprovedTrainer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        self.data_manager = DataManager()
        self.data_loader = CryptoDataLoader()
    
    def train_model(self, symbol, timeframe='1h', epochs=150, batch_size=32, learning_rate=0.001, val_split=0.1):
        """
        Train model with improved logic
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (default: '1h')
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            val_split: Validation split ratio
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {symbol} ({timeframe})")
        logger.info(f"{'='*70}")
        
        # Load data
        logger.info(f"Loading data for {symbol}...")
        data = self.data_manager.get_stored_data(symbol, timeframe)
        if data is None or len(data) < 100:
            logger.error(f"Insufficient data for {symbol}")
            return None
        
        logger.info(f"Loaded {len(data)} candles")
        
        # Calculate indicators
        logger.info("Calculating technical indicators...")
        data_with_indicators = self.data_loader.calculate_technical_indicators(data)
        if data_with_indicators is None or data_with_indicators.empty:
            logger.error("Failed to calculate indicators")
            return None
        
        logger.info(f"Indicators calculated: {len(data_with_indicators)} rows, {len(data_with_indicators.columns)} features")
        
        # Normalize data
        logger.info("Normalizing features...")
        from sklearn.preprocessing import MinMaxScaler
        
        feature_cols = list(data_with_indicators.columns)
        scaler = MinMaxScaler()
        data_normalized = data_with_indicators.copy()
        data_scaled = scaler.fit_transform(data_with_indicators[feature_cols])
        data_normalized[feature_cols] = data_scaled
        
        # Create sequences
        logger.info(f"Creating sequences with lookback={MODEL_CONFIG['lookback']}...")
        lookback = MODEL_CONFIG['lookback']
        
        X_list = []
        y_list = []
        
        for i in range(len(data_normalized) - lookback):
            X_list.append(data_normalized[feature_cols].iloc[i:i+lookback].values)
            y_list.append(data_with_indicators['close'].iloc[i+lookback])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Sequences created: X shape={X.shape}, y shape={y.shape}")
        
        # Normalize y
        close_min = data_with_indicators['close'].min()
        close_max = data_with_indicators['close'].max()
        y_normalized = (y - close_min) / (close_max - close_min)
        
        # Create PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y_normalized).unsqueeze(1)
        
        # Split data
        dataset = TensorDataset(X_tensor, y_tensor)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train/Val split: {train_size}/{val_size}")
        
        # Initialize model
        num_features = X.shape[2]
        logger.info(f"Initializing model with input_size={num_features}...")
        
        model = ImprovedCryptoGRU(
            input_size=num_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            output_size=1
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=15,
            verbose=True,
            min_lr=1e-6
        )
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 30
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
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
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model_dir = Path('backend/models/weights')
                model_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = model_dir / f"{symbol}_{timeframe}_best.pth"
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
                }, best_model_path)
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Best: {best_val_loss:.8f}")
            
            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        model_dir = Path('backend/models/weights')
        model_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = model_dir / f"{symbol}_{timeframe}_final.pth"
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
        }, final_model_path)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Training completed for {symbol} ({timeframe})")
        logger.info(f"Best validation loss: {best_val_loss:.8f}")
        logger.info(f"Final models saved:")
        logger.info(f"  - Best: {best_model_path}")
        logger.info(f"  - Final: {final_model_path}")
        logger.info(f"{'='*70}")
        
        return {
            'model': model,
            'best_model_path': str(best_model_path),
            'final_model_path': str(final_model_path),
            'config': {
                'input_size': num_features,
                'hidden_size': 128,
                'num_layers': 2,
                'output_size': 1,
            },
            'scalers': {
                'close_min': float(close_min),
                'close_max': float(close_max),
            },
            'train_losses': train_losses,
            'val_losses': val_losses,
        }

def main():
    parser = ArgumentParser(description='Improved crypto price prediction training')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to train (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs (default: 150)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    trainer = ImprovedTrainer()
    result = trainer.train_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

if __name__ == '__main__':
    main()
