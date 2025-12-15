#!/usr/bin/env python
r"""
Inference script - Use trained models for predictions
Loads model architecture from models_architecture.json

Usage:
    python backend/inference.py --symbol BTCUSDT --timeframe 1h
    python backend/inference.py --symbol ETHUSDT --timeframe 15m --num-samples 10
"""

import logging
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.models.lstm_model import CryptoLSTM
from backend.data.data_manager import DataManager
from backend.data.data_loader import CryptoDataLoader
from config.model_config import MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, architecture_file='models_architecture.json', device='cpu'):
        """
        Initialize inference engine
        """
        self.device = device
        self.model_dir = Path('backend/models/weights')
        self.data_manager = DataManager()
        self.data_loader = CryptoDataLoader()
        
        # Load model architecture
        self.architectures = self._load_architectures(architecture_file)
        logger.info(f"Loaded architectures for {len(self.architectures)} model groups")
    
    def _load_architectures(self, filename):
        """
        Load model architectures from JSON file
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Architecture file not found: {filename}")
            return {}
    
    def get_model_architecture(self, symbol, timeframe):
        """
        Get model architecture for symbol and timeframe
        """
        key = f"{symbol}_{timeframe}"
        
        if key not in self.architectures:
            logger.error(f"No architecture found for {key}")
            return None
        
        model_info = self.architectures[key][0]
        return model_info['model_config']
    
    def load_model(self, symbol, timeframe):
        """
        Load trained model
        """
        model_path = self.model_dir / f"{symbol}_{timeframe}_v1.pt"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            state = torch.load(model_path, map_location=self.device)
            
            config = state['model_config']
            
            model = CryptoLSTM(
                input_size=config['input_size'],
                hidden_size=MODEL_CONFIG['hidden_size'],
                num_layers=MODEL_CONFIG['num_layers'],
                output_size=config['output_size'],
                dropout=MODEL_CONFIG['dropout']
            ).to(self.device)
            
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            
            logger.info(f"Loaded model: {symbol} ({timeframe})")
            logger.info(f"  Input size: {config['input_size']}")
            logger.info(f"  Output size: {config['output_size']}")
            
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def predict(self, symbol, timeframe, num_predictions=1):
        """
        Make predictions for a symbol
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '15m')
            num_predictions: Number of recent samples to predict
        
        Returns:
            Dictionary with predictions and analysis
        """
        logger.info(f"\nPredicting for {symbol} ({timeframe})...")
        
        data = self.data_manager.get_stored_data(symbol, timeframe)
        if data is None or len(data) < 30:
            logger.error(f"Insufficient data for {symbol} ({timeframe})")
            return None
        
        data_with_indicators = self.data_loader.calculate_technical_indicators(data)
        if data_with_indicators is None or data_with_indicators.empty:
            logger.error(f"Failed to calculate indicators for {symbol}")
            return None
        
        all_feature_cols = [col for col in data_with_indicators.columns]
        
        logger.info(f"Total available features: {len(all_feature_cols)}")
        logger.info(f"Data shape: {data_with_indicators.shape}")
        
        scaler = MinMaxScaler()
        data_normalized = data_with_indicators.copy()
        data_normalized[all_feature_cols] = scaler.fit_transform(data_with_indicators[all_feature_cols])
        
        lookback = MODEL_CONFIG['lookback']
        
        if len(data_normalized) < lookback:
            logger.error(f"Not enough data. Have {len(data_normalized)}, need {lookback}")
            return None
        
        model = self.load_model(symbol, timeframe)
        if model is None:
            return None
        
        logger.info(f"\nGenerating {num_predictions} predictions...")
        logger.info(f"Available data points: {len(data_normalized)}")
        logger.info(f"Lookback window: {lookback}")
        logger.info(f"Max sequences possible: {len(data_normalized) - lookback + 1}")
        
        predictions = []
        timestamps = []
        actual_prices = []
        
        with torch.no_grad():
            for i in range(num_predictions):
                # Use the last available data for all predictions
                # Last lookback samples end at index len(data_normalized)-1
                start_idx = len(data_normalized) - lookback
                end_idx = len(data_normalized)
                
                # Adjust if we need historical predictions
                if i > 0:
                    # For historical predictions, shift back
                    offset = i
                    if len(data_normalized) - lookback - offset < 0:
                        logger.warning(f"Not enough historical data for prediction {i+1}")
                        continue
                    start_idx = len(data_normalized) - lookback - offset
                    end_idx = len(data_normalized) - offset
                
                if start_idx < 0:
                    logger.warning(f"Not enough historical data for prediction {i+1}")
                    continue
                
                logger.info(f"\nPrediction {i+1}: using data from index {start_idx} to {end_idx-1}")
                
                x = data_normalized[all_feature_cols].iloc[start_idx:end_idx].values
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                
                logger.info(f"  x shape: {x.shape}, x_tensor shape: {x_tensor.shape}")
                
                try:
                    pred = model(x_tensor).cpu().numpy()[0, 0]
                    predictions.append(float(pred))
                    
                    # Timestamp is at the end_idx-1 position
                    timestamp_idx = end_idx - 1
                    if timestamp_idx < len(data_with_indicators.index):
                        timestamp = data_with_indicators.index[timestamp_idx]
                        timestamps.append(str(timestamp))
                        actual_price = float(data_with_indicators['close'].iloc[timestamp_idx])
                        actual_prices.append(actual_price)
                        logger.info(f"  Predicted (normalized): {pred:.6f}")
                        logger.info(f"  Actual price: {actual_price:.6f}")
                        logger.info(f"  Timestamp: {timestamp}")
                    else:
                        logger.warning(f"  Timestamp index {timestamp_idx} out of bounds")
                        continue
                
                except Exception as e:
                    logger.error(f"Error on prediction {i+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not predictions:
            logger.error("No successful predictions")
            return None
        
        # Build results only with successful predictions
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'predictions': [
                {
                    'timestamp': timestamps[i],
                    'predicted_price': predictions[i],
                    'actual_price': actual_prices[i],
                    'difference': actual_prices[i] - predictions[i],
                    'error_percent': (abs(actual_prices[i] - predictions[i]) / actual_prices[i]) * 100
                }
                for i in range(len(predictions))
            ],
            'summary': {
                'total_predictions': len(predictions),
                'avg_error': float(np.mean([
                    abs(actual_prices[i] - predictions[i]) / actual_prices[i] * 100
                    for i in range(len(predictions))
                ])),
                'current_price': float(data_with_indicators['close'].iloc[-1])
            }
        }
        
        return results


def main():
    parser = ArgumentParser(description='Make predictions using trained models')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to predict (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--num-predictions', type=int, default=1, help='Number of predictions (default: 1)')
    parser.add_argument('--arch-file', default='models_architecture.json', help='Architecture JSON file')
    
    args = parser.parse_args()
    
    inference = ModelInference(architecture_file=args.arch_file)
    result = inference.predict(args.symbol, args.timeframe, args.num_predictions)
    
    if result:
        logger.info("\n" + "="*70)
        logger.info("PREDICTION RESULTS")
        logger.info("="*70)
        logger.info(json.dumps(result, indent=2, default=str))
    else:
        logger.error("Prediction failed")


if __name__ == "__main__":
    main()
