#!/usr/bin/env python
r"""
Improved Inference Script

Supports:
- Loading .pth models from train_improved.py
- Proper denormalization with saved scaler values
- Batch prediction
- Model visualization

Usage:
    python backend/inference_improved.py --symbol BTCUSDT --timeframe 1h --model-type best
"""

import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.data_manager import DataManager
from backend.data.data_loader import CryptoDataLoader
from config.model_config import MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedCryptoGRU(nn.Module):
    """Same as in train_improved.py"""
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
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
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

class ImprovedInference:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        self.data_manager = DataManager()
        self.data_loader = CryptoDataLoader()
    
    def load_model(self, symbol, timeframe, model_type='best'):
        """
        Load model from .pth file
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            model_type: 'best' or 'final'
        
        Returns:
            model, config, scalers
        """
        model_dir = Path('backend/models/weights')
        model_path = model_dir / f"{symbol}_{timeframe}_{model_type}.pth"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None, None, None
        
        logger.info(f"Loading model from {model_path}...")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            config = checkpoint['model_config']
            scalers = checkpoint['scalers']
            
            logger.info(f"Model config: {config}")
            logger.info(f"Scalers: close_min={scalers['close_min']:.2f}, close_max={scalers['close_max']:.2f}")
            
            # Create and load model
            model = ImprovedCryptoGRU(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=0.3,
                output_size=config['output_size']
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"Model loaded successfully")
            
            return model, config, scalers
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, None, None
    
    def predict_batch(self, model, scalers, symbol, timeframe, num_predictions=50):
        """
        Generate batch predictions
        
        Args:
            model: Trained model
            scalers: Min/max values for denormalization
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            num_predictions: Number of predictions to generate
        
        Returns:
            predictions dict with actual and predicted prices
        """
        logger.info(f"\nGenerating {num_predictions} predictions...")
        
        # Load data
        data = self.data_manager.get_stored_data(symbol, timeframe)
        if data is None:
            logger.error(f"No data found for {symbol}")
            return None
        
        # Calculate indicators
        data_with_indicators = self.data_loader.calculate_technical_indicators(data)
        if data_with_indicators is None or data_with_indicators.empty:
            logger.error("Failed to calculate indicators")
            return None
        
        # Normalize
        from sklearn.preprocessing import MinMaxScaler
        feature_cols = list(data_with_indicators.columns)
        scaler_obj = MinMaxScaler()
        data_normalized = data_with_indicators.copy()
        data_scaled = scaler_obj.fit_transform(data_with_indicators[feature_cols])
        data_normalized[feature_cols] = data_scaled
        
        # Prepare sequences
        lookback = MODEL_CONFIG['lookback']
        predictions = []
        actuals = []
        timestamps = []
        
        max_idx = len(data_normalized) - lookback
        start_idx = max(0, max_idx - num_predictions)
        
        logger.info(f"Data prepared: {len(data_normalized)} rows, {len(feature_cols)} features")
        logger.info(f"Generating predictions from index {start_idx} to {max_idx}...")
        
        with torch.no_grad():
            for i in range(start_idx, max_idx):
                # Get sequence
                seq = data_normalized[feature_cols].iloc[i:i+lookback].values
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                # Predict
                pred_norm = model(seq_tensor).cpu().numpy()[0, 0]
                
                # Denormalize
                close_min = scalers['close_min']
                close_max = scalers['close_max']
                pred_actual = pred_norm * (close_max - close_min) + close_min
                
                # Get actual
                actual_idx = i + lookback
                if actual_idx < len(data_with_indicators):
                    actual = data_with_indicators['close'].iloc[actual_idx]
                    timestamp = data_with_indicators.index[actual_idx]
                    
                    predictions.append(float(pred_actual))
                    actuals.append(float(actual))
                    timestamps.append(timestamp)
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        errors = actuals - predictions
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(np.abs(errors / actuals)) * 100
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PREDICTION RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Predictions: {len(predictions)}")
        logger.info(f"MAE: ${mae:.2f}")
        logger.info(f"RMSE: ${rmse:.2f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"Avg Actual: ${np.mean(actuals):.2f}")
        logger.info(f"Avg Predicted: ${np.mean(predictions):.2f}")
        logger.info(f"Actual Range: ${actuals.min():.2f} - ${actuals.max():.2f}")
        logger.info(f"Predicted Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        logger.info(f"{'='*70}")
        
        return {
            'timestamps': timestamps,
            'predictions': predictions,
            'actuals': actuals,
            'errors': errors,
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'avg_actual': float(np.mean(actuals)),
                'avg_predicted': float(np.mean(predictions)),
            }
        }

def main():
    parser = ArgumentParser(description='Improved inference for crypto predictions')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--model-type', default='best', help='best or final (default: best)')
    parser.add_argument('--num-predictions', type=int, default=50, help='Number of predictions')
    
    args = parser.parse_args()
    
    inferencer = ImprovedInference()
    
    # Load model
    model, config, scalers = inferencer.load_model(
        args.symbol,
        args.timeframe,
        args.model_type
    )
    
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Generate predictions
    results = inferencer.predict_batch(
        model,
        scalers,
        args.symbol,
        args.timeframe,
        args.num_predictions
    )
    
    if results:
        logger.info("Inference completed successfully!")

if __name__ == '__main__':
    main()
