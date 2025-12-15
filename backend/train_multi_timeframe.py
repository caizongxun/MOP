import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_multi_timeframe import MultiTimeframeFusion
from data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiTimeframeTrainer:
    def __init__(self, symbol='BTCUSDT', device=None, model_dir='models'):
        self.symbol = symbol
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiTimeframeFusion().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.dm = DataManager()
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f'Using device: {self.device}')
        logger.info(f'Model: MultiTimeframeFusion')
    
    def prepare_data(self, seq_len_1h=60):
        """Prepare 1h and 15m data"""
        logger.info(f'Loading {self.symbol} data for 1h and 15m...')
        
        try:
            # Load 1h data
            data_1h = self.dm.load_data(self.symbol, '1h')
            if data_1h is None or len(data_1h) < seq_len_1h:
                logger.error(f'Insufficient 1h data for {self.symbol}')
                return None, None, None
            
            # Load 15m data
            data_15m = self.dm.load_data(self.symbol, '15m')
            if data_15m is None or len(data_15m) < seq_len_1h * 4:
                logger.error(f'Insufficient 15m data for {self.symbol}')
                return None, None, None
            
            logger.info(f'Loaded {len(data_1h)} rows for 1h, {len(data_15m)} rows for 15m')
            
            # Create sequences
            X_1h, y_1h = self._create_sequences(data_1h, seq_len_1h)
            X_15m, y_15m = self._create_sequences(data_15m, seq_len_1h * 4)
            
            if len(X_1h) == 0 or len(X_15m) == 0:
                logger.error(f'Failed to create sequences for {self.symbol}')
                return None, None, None
            
            # Align labels
            min_len = min(len(y_1h), len(y_15m))
            X_1h = torch.FloatTensor(X_1h[:min_len]).to(self.device)
            X_15m = torch.FloatTensor(X_15m[:min_len]).to(self.device)
            y = torch.FloatTensor(y_1h[:min_len]).unsqueeze(1).to(self.device)
            
            logger.info(f'Data shapes: X_1h={X_1h.shape}, X_15m={X_15m.shape}, y={y.shape}')
            return X_1h, X_15m, y
            
        except Exception as e:
            logger.error(f'Error preparing data: {str(e)}')
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _create_sequences(self, data, seq_len=60):
        """Create input sequences from time series data"""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len, 4])  # Take close price (column 4)
        return np.array(X), np.array(y)
    
    def train(self, epochs=100, batch_size=32):
        """Train multi-timeframe model"""
        X_1h, X_15m, y = self.prepare_data()
        
        if X_1h is None:
            logger.error(f'Cannot train {self.symbol}: data preparation failed')
            return False
        
        dataset = TensorDataset(X_1h, X_15m, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f'Training {self.symbol} for {epochs} epochs')
        logger.info(f'Batch size: {batch_size}, Dataset size: {len(dataset)}')
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for x_1h_batch, x_15m_batch, y_batch in loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                pred = self.model(x_1h_batch, x_15m_batch)
                loss = self.criterion(pred, y_batch)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                self._save_model()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} (Best: {best_loss:.6f})')
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        logger.info(f'Training complete for {self.symbol}. Best loss: {best_loss:.6f}')
        return True
    
    def _save_model(self):
        """Save model weights"""
        model_path = os.path.join(self.model_dir, f'{self.symbol}_multi_timeframe.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.debug(f'Model saved to {model_path}')

def main():
    parser = argparse.ArgumentParser(description='Train multi-timeframe models')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'], help='Symbols to train')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    logger.info('='*80)
    logger.info('Multi-Timeframe Training (1H + 15M Fusion)')
    logger.info('='*80)
    logger.info(f'Symbols: {args.symbols}')
    logger.info(f'Epochs: {args.epochs}, Batch Size: {args.batch_size}')
    logger.info('='*80)
    
    device = torch.device(args.device) if args.device else None
    
    for symbol in args.symbols:
        logger.info(f'\nTraining {symbol}...')
        trainer = MultiTimeframeTrainer(symbol=symbol, device=device)
        trainer.train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
