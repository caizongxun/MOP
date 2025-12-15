import os
import sys
import logging
import torch
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Fix path for imports
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_multi_timeframe import MultiTimeframeFusion
from data.data_manager import DataManager
from data.data_loader import CryptoDataLoader

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
        self.data_loader = CryptoDataLoader()
        self.model_dir = model_dir
        
        # Load model
        model_path = os.path.join(model_dir, f'{symbol}_multi_timeframe.pth')
        if not os.path.exists(model_path):
            logger.error(f'Model not found: {model_path}')
            raise FileNotFoundError(f'Model not found: {model_path}')
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f'Model loaded from {model_path}')
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            return self.data_loader.calculate_technical_indicators(df)
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def predict(self, num_predictions=50):
        """Generate predictions using multi-timeframe fusion"""
        logger.info(f'Generating {num_predictions} predictions for {self.symbol}')
        
        try:
            # Load latest data
            df_1h = self.dm.get_stored_data(self.symbol, '1h')
            df_15m = self.dm.get_stored_data(self.symbol, '15m')
            
            if df_1h is None or df_15m is None:
                logger.error('Failed to load data')
                return None, None, None, None
            
            # Calculate indicators
            logger.info('Calculating indicators for 1h...')
            df_1h_ind = self._calculate_indicators(df_1h)
            if df_1h_ind is None or df_1h_ind.empty:
                logger.error('Failed to calculate 1h indicators')
                return None, None, None, None
            
            logger.info('Calculating indicators for 15m...')
            df_15m_ind = self._calculate_indicators(df_15m)
            if df_15m_ind is None or df_15m_ind.empty:
                logger.error('Failed to calculate 15m indicators')
                return None, None, None, None
            
            # Normalize 1h
            scaler_1h = MinMaxScaler()
            feature_cols_1h = [col for col in df_1h_ind.columns if col != 'timestamp']
            data_1h_normalized = df_1h_ind.copy()
            data_1h_scaled = scaler_1h.fit_transform(df_1h_ind[feature_cols_1h])
            data_1h_normalized[feature_cols_1h] = data_1h_scaled
            data_1h = data_1h_normalized[feature_cols_1h].values.astype(np.float32)
            
            # Normalize 15m
            scaler_15m = MinMaxScaler()
            feature_cols_15m = [col for col in df_15m_ind.columns if col != 'timestamp']
            data_15m_normalized = df_15m_ind.copy()
            data_15m_scaled = scaler_15m.fit_transform(df_15m_ind[feature_cols_15m])
            data_15m_normalized[feature_cols_15m] = data_15m_scaled
            data_15m = data_15m_normalized[feature_cols_15m].values.astype(np.float32)
            
            # Get last sequences
            seq_len_1h = 60
            seq_len_15m = 240
            
            x_1h = data_1h[-seq_len_1h:].copy()
            x_15m = data_15m[-seq_len_15m:].copy()
            
            # Store close price info for denormalization
            close_min = df_1h_ind['close'].min()
            close_max = df_1h_ind['close'].max()
            last_price_1h = df_1h_ind['close'].iloc[-1]
            
            predictions_normalized = []
            
            logger.info(f'Starting from 1h close: {last_price_1h:.2f}')
            logger.info(f'Close price range for denormalization: {close_min:.2f} - {close_max:.2f}')
            logger.info(f'Data shapes: 1h={x_1h.shape}, 15m={x_15m.shape}')
            
            with torch.no_grad():
                for i in range(num_predictions):
                    # Prepare batch
                    x_1h_tensor = torch.FloatTensor(x_1h).unsqueeze(0).to(self.device)
                    x_15m_tensor = torch.FloatTensor(x_15m).unsqueeze(0).to(self.device)
                    
                    # Predict (output is normalized between 0-1)
                    pred_normalized = self.model(x_1h_tensor, x_15m_tensor).item()
                    predictions_normalized.append(pred_normalized)
                    
                    # Create dummy candle with predicted normalized value
                    dummy_candle_1h = np.zeros(x_1h.shape[1])
                    dummy_candle_1h[0] = pred_normalized  # Set first feature
                    x_1h = np.vstack([x_1h[1:], dummy_candle_1h])
                    
                    # 15m gets 4 candles for each 1h
                    for j in range(4):
                        dummy_candle_15m = np.zeros(x_15m.shape[1])
                        dummy_candle_15m[0] = pred_normalized
                        x_15m = np.vstack([x_15m[1:], dummy_candle_15m])
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f'Prediction {i+1}/{num_predictions}: {pred_normalized:.6f}')
            
            # Denormalize predictions to actual prices
            predictions_normalized = np.array(predictions_normalized)
            predictions_actual = predictions_normalized * (close_max - close_min) + close_min
            
            logger.info(f'Generated {len(predictions_actual)} predictions')
            logger.info(f'Actual price predictions range: {predictions_actual.min():.2f} - {predictions_actual.max():.2f}')
            
            return predictions_actual, last_price_1h, close_min, close_max
        
        except Exception as e:
            logger.error(f'Error in predict: {str(e)}')
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def evaluate_predictions(self, predictions, last_price, close_min, close_max, test_lookback=50):
        """Evaluate prediction accuracy against actual data"""
        try:
            df_1h = self.dm.get_stored_data(self.symbol, '1h')
            
            if df_1h is None or len(df_1h) < test_lookback:
                logger.warning('Insufficient data for evaluation')
                return None
            
            # Get actual close prices
            actual_prices = df_1h['close'].iloc[-test_lookback:].values
            
            if len(actual_prices) < len(predictions):
                predictions = predictions[:len(actual_prices)]
            
            predictions = np.array(predictions[:len(actual_prices)])
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - actual_prices))
            rmse = np.sqrt(np.mean((predictions - actual_prices) ** 2))
            mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'avg_actual': np.mean(actual_prices),
                'avg_predicted': np.mean(predictions),
                'actual_range': (np.min(actual_prices), np.max(actual_prices)),
                'predicted_range': (np.min(predictions), np.max(predictions))
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f'Error in evaluate: {str(e)}')
            import traceback
            traceback.print_exc()
            return None

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
        result = predictor.predict(num_predictions=args.num_predictions)
        
        if result[0] is not None:
            predictions, last_price, close_min, close_max = result
            logger.info('\n' + '='*80)
            logger.info('PREDICTION RESULTS')
            logger.info('='*80)
            logger.info(f'Predictions: {len(predictions)}')
            logger.info(f'Range: {np.min(predictions):.2f} - {np.max(predictions):.2f}')
            logger.info(f'Mean: {np.mean(predictions):.2f}')
            logger.info(f'Last actual price: {last_price:.2f}')
            
            if args.evaluate:
                metrics = predictor.evaluate_predictions(predictions, last_price, close_min, close_max, test_lookback=50)
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
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
