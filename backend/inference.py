import torch
import numpy as np
import logging
from pathlib import Path

from config.model_config import MODEL_CONFIG
from backend.models.lstm_model import CryptoLSTM
from backend.data.data_loader import CryptoDataLoader

logger = logging.getLogger(__name__)

class ModelInference:
    """
    Inference engine for trained cryptocurrency price prediction models
    Optimized for CPU usage (for Discord Bot deployment)
    """
    
    def __init__(self, model_dir='backend/models/weights', device='cpu'):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.data_loader = CryptoDataLoader(lookback_period=MODEL_CONFIG['lookback'])
        self.models = {}
        self.scalers = {}
    
    def load_model(self, symbol):
        """
        Load trained model for a cryptocurrency
        """
        try:
            model_path = self.model_dir / f"{symbol}_model_v1.pt"
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            model = CryptoLSTM(
                hidden_size=MODEL_CONFIG['hidden_size'],
                num_layers=MODEL_CONFIG['num_layers'],
                dropout=MODEL_CONFIG['dropout']
            ).to(self.device)
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            self.models[symbol] = model
            logger.info(f"Model loaded for {symbol}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            return None
    
    def predict(self, symbol, data, num_steps=7):
        """
        Predict future price movements
        
        Args:
            symbol: Cryptocurrency symbol
            data: Historical OHLCV data
            num_steps: Number of future candles to predict (5-10)
        
        Returns:
            Array of predicted prices
        """
        try:
            if symbol not in self.models:
                self.load_model(symbol)
            
            model = self.models.get(symbol)
            if model is None:
                logger.error(f"Model not available for {symbol}")
                return None
            
            # Prepare data
            close_prices = data[['close']].values
            normalized = self.data_loader.normalize_data(close_prices)
            
            # Use last lookback period
            current_seq = normalized[-MODEL_CONFIG['lookback']:].reshape(1, -1, 1)
            current_seq = torch.FloatTensor(current_seq).to(self.device)
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(num_steps):
                    output = model(current_seq)
                    predictions.append(output.cpu().numpy()[0, 0])
                    
                    # Update sequence
                    new_seq = np.append(current_seq.cpu().numpy()[0, 1:, 0], output.cpu().numpy()[0, 0])
                    current_seq = torch.FloatTensor(new_seq.reshape(1, -1, 1)).to(self.device)
            
            # Denormalize predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.data_loader.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {str(e)}")
            return None
