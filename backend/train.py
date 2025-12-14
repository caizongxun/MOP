import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from pathlib import Path
from datetime import datetime

from config.model_config import MODEL_CONFIG, CRYPTOCURRENCIES
from backend.models.lstm_model import CryptoLSTM
from backend.data.data_loader import CryptoDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trainer for cryptocurrency price prediction models
    """
    
    def __init__(self, config=MODEL_CONFIG):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.data_loader = CryptoDataLoader(lookback_period=config['lookback'])
        self.models = {}
        self.histories = {}
        
    def prepare_data(self, symbol, data):
        """
        Prepare data for training
        """
        import numpy as np
        
        # Use close price for prediction
        close_prices = data[['close']].values
        
        # Normalize data
        normalized = self.data_loader.normalize_data(close_prices)
        
        # Create sequences
        X, y = self.data_loader.create_sequences(normalized, self.config['lookback'])
        
        # Convert to torch tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        return X, y
    
    def train_model(self, symbol):
        """
        Train model for a specific cryptocurrency
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*50}")
        
        try:
            # Fetch data
            logger.info(f"Fetching data for {symbol}...")
            data = self.data_loader.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=500)
            
            if data is None or data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Prepare data
            X, y = self.prepare_data(symbol, data)
            
            # Create data loader
            dataset = TensorDataset(X, y)
            train_loader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            # Initialize model
            model = CryptoLSTM(
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            
            # Training loop
            history = {'loss': []}
            
            for epoch in range(self.config['epochs']):
                total_loss = 0
                model.train()
                
                for X_batch, y_batch in train_loader:
                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                history['loss'].append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config['epochs']}], Loss: {avg_loss:.6f}")
            
            # Save model
            model_dir = Path('backend/models/weights')
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{symbol}_model_v1.pt"
            model.save_model(str(model_path))
            
            self.models[symbol] = model
            self.histories[symbol] = history
            
            logger.info(f"Training completed for {symbol}. Model saved to {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return None
    
    def train_all_models(self):
        """
        Train models for all cryptocurrencies
        """
        logger.info(f"Starting training for {len(CRYPTOCURRENCIES)} cryptocurrencies...")
        logger.info(f"Config: {self.config}")
        
        results = {}
        for symbol in CRYPTOCURRENCIES:
            model = self.train_model(symbol)
            results[symbol] = 'Success' if model is not None else 'Failed'
        
        logger.info(f"\n{'='*50}")
        logger.info("Training Summary")
        logger.info(f"{'='*50}")
        for symbol, status in results.items():
            logger.info(f"{symbol}: {status}")

if __name__ == "__main__":
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    trainer = ModelTrainer()
    trainer.train_all_models()
