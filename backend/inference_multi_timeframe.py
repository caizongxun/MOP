import os
import sys
import logging
import torch
import numpy as np
import argparse
from datetime import datetime

from model_multi_timeframe import MultiTimeframeFusion
from data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiTimeframePredictor:
    def __init__(self, symbol='BTCUSDT', device=None, model_dir='models'):
        self.symbol = symbol
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiTimeframeFusion().to(self.device)
        self.dm = DataManager()
        self.model_dir = model_dir
        
        # Load model
        model_path = os.path.join(model_dir, f'{symbol}_multi_timeframe.pth')
        if not os.path.exists(model_path):
            logger.error(f'Model not found: {model_path}')
            raise FileNotFoundError(f'Model not found: {model_path}')
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f'Model loaded from {model_path}')
    
    def predict(self, num_predictions=50):
        """Generate predictions using multi-timeframe fusion"""
        logger.info(f'Generating {num_predictions} predictions for {self.symbol}')
        
        # Load latest data
        data_1h = self.dm.load_data(self.symbol, '1h')
        data_15m = self.dm.load_data(self.symbol, '15m')
        
        if data_1h is None or data_15m is None:
            logger.error('Failed to load data')
            return None
        
        # Get last sequences
        seq_len_1h = 60
        seq_len_15m = 240
        
        x_1h = data_1h[-seq_len_1h:].copy()
        x_15m = data_15m[-seq_len_15m:].copy()
        
        # Store actual prices for reference
        last_price_1h = data_1h[-1, 4]  # Close price
        last_price_15m = data_15m[-1, 4]
        
        predictions = []
        actual_future = []
        
        logger.info(f'Starting from 1h price: {last_price_1h:.2f}, 15m price: {last_price_15m:.2f}')
        
        with torch.no_grad():
            for i in range(num_predictions):
                # Prepare batch
                x_1h_tensor = torch.FloatTensor(x_1h).unsqueeze(0).to(self.device)
                x_15m_tensor = torch.FloatTensor(x_15m).unsqueeze(0).to(self.device)
                
                # Predict
                pred = self.model(x_1h_tensor, x_15m_tensor).item()
                predictions.append(pred)
                
                # Update sequences (shift and append)
                dummy_candle_1h = np.zeros(x_1h.shape[1])
                dummy_candle_1h[4] = pred  # Set close price
                x_1h = np.vstack([x_1h[1:], dummy_candle_1h])
                
                # 15m gets 4 candles for each 1h
                for j in range(4):
                    dummy_candle_15m = np.zeros(x_15m.shape[1])
                    dummy_candle_15m[4] = pred
                    x_15m = np.vstack([x_15m[1:], dummy_candle_15m])
                
                if (i + 1) % 10 == 0:
                    logger.info(f'Prediction {i+1}/{num_predictions}: {pred:.2f}')
        
        return predictions
    
    def evaluate_predictions(self, predictions, test_lookback=50):
        """Evaluate prediction accuracy against actual data"""
        data_1h = self.dm.load_data(self.symbol, '1h')
        
        if data_1h is None or len(data_1h) < test_lookback:
            logger.warning('Insufficient data for evaluation')
            return None
        
        # Get actual future prices (if available)
        # Note: This is for testing against already-occurred data
        actual = data_1h[-test_lookback:, 4]  # Close prices
        
        if len(actual) < len(predictions):
            predictions = predictions[:len(actual)]
        
        predictions = np.array(predictions[:len(actual)])
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'avg_actual': np.mean(actual),
            'avg_predicted': np.mean(predictions),
            'actual_range': (np.min(actual), np.max(actual)),
            'predicted_range': (np.min(predictions), np.max(predictions))
        }
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Multi-timeframe prediction')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to predict')
    parser.add_argument('--num-predictions', type=int, default=50, help='Number of predictions')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate against recent data')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    try:
        predictor = MultiTimeframePredictor(symbol=args.symbol, device=device)
        predictions = predictor.predict(num_predictions=args.num_predictions)
        
        if predictions:
            logger.info('\n' + '='*80)
            logger.info('PREDICTION RESULTS')
            logger.info('='*80)
            logger.info(f'Predictions: {len(predictions)}')
            logger.info(f'Range: {np.min(predictions):.2f} - {np.max(predictions):.2f}')
            logger.info(f'Mean: {np.mean(predictions):.2f}')
            
            if args.evaluate:
                metrics = predictor.evaluate_predictions(predictions, test_lookback=50)
                if metrics:
                    logger.info('\nEvaluation Metrics (vs recent actual data):')
                    logger.info(f'MAE: ${metrics["mae"]:.2f}')
                    logger.info(f'RMSE: ${metrics["rmse"]:.2f}')
                    logger.info(f'MAPE: {metrics["mape"]:.2f}%')
                    logger.info(f'Avg Actual: ${metrics["avg_actual"]:.2f}')
                    logger.info(f'Avg Predicted: ${metrics["avg_predicted"]:.2f}')
                    logger.info(f'Actual Range: ${metrics["actual_range"][0]:.2f} - ${metrics["actual_range"][1]:.2f}')
                    logger.info(f'Predicted Range: ${metrics["predicted_range"][0]:.2f} - ${metrics["predicted_range"][1]:.2f}')
            
            logger.info('='*80)
    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
