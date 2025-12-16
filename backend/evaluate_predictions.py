import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.model_multi_timeframe import MultiTimeframeFusion
from backend.data.data_manager import DataManager
from backend.data.data_loader import CryptoDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionEvaluator:
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
    
    def generate_predictions(self, num_predictions=50):
        """Generate predictions"""
        logger.info(f'Generating {num_predictions} predictions for {self.symbol}')
        
        try:
            # Load data
            df_1h = self.dm.get_stored_data(self.symbol, '1h')
            df_15m = self.dm.get_stored_data(self.symbol, '15m')
            
            if df_1h is None or df_15m is None:
                logger.error('Failed to load data')
                return None
            
            # Calculate indicators
            df_1h_ind = self._calculate_indicators(df_1h)
            df_15m_ind = self._calculate_indicators(df_15m)
            
            if df_1h_ind is None or df_15m_ind is None:
                logger.error('Failed to calculate indicators')
                return None
            
            # Normalize
            scaler_1h = MinMaxScaler()
            feature_cols_1h = [col for col in df_1h_ind.columns if col != 'timestamp']
            data_1h_normalized = df_1h_ind.copy()
            data_1h_normalized[feature_cols_1h] = scaler_1h.fit_transform(df_1h_ind[feature_cols_1h])
            data_1h = data_1h_normalized[feature_cols_1h].values.astype(np.float32)
            
            scaler_15m = MinMaxScaler()
            feature_cols_15m = [col for col in df_15m_ind.columns if col != 'timestamp']
            data_15m_normalized = df_15m_ind.copy()
            data_15m_normalized[feature_cols_15m] = scaler_15m.fit_transform(df_15m_ind[feature_cols_15m])
            data_15m = data_15m_normalized[feature_cols_15m].values.astype(np.float32)
            
            # Get sequences
            seq_len_1h = 60
            seq_len_15m = 240
            
            x_1h = data_1h[-seq_len_1h:].copy()
            x_15m = data_15m[-seq_len_15m:].copy()
            
            # Get denormalization params
            close_min = df_1h_ind['close'].min()
            close_max = df_1h_ind['close'].max()
            
            predictions_normalized = []
            
            with torch.no_grad():
                for i in range(num_predictions):
                    x_1h_tensor = torch.FloatTensor(x_1h).unsqueeze(0).to(self.device)
                    x_15m_tensor = torch.FloatTensor(x_15m).unsqueeze(0).to(self.device)
                    
                    pred_normalized = self.model(x_1h_tensor, x_15m_tensor).item()
                    predictions_normalized.append(pred_normalized)
                    
                    dummy_candle_1h = np.zeros(x_1h.shape[1])
                    dummy_candle_1h[0] = pred_normalized
                    x_1h = np.vstack([x_1h[1:], dummy_candle_1h])
                    
                    for j in range(4):
                        dummy_candle_15m = np.zeros(x_15m.shape[1])
                        dummy_candle_15m[0] = pred_normalized
                        x_15m = np.vstack([x_15m[1:], dummy_candle_15m])
            
            # Denormalize
            predictions_normalized = np.array(predictions_normalized)
            predictions_actual = predictions_normalized * (close_max - close_min) + close_min
            
            return {
                'predictions': predictions_actual,
                'predictions_normalized': predictions_normalized,
                'close_min': close_min,
                'close_max': close_max
            }
        
        except Exception as e:
            logger.error(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_metrics(self, predictions, lookback=50):
        """Calculate evaluation metrics"""
        try:
            df_1h = self.dm.get_stored_data(self.symbol, '1h')
            
            if df_1h is None:
                logger.error('Failed to load data for evaluation')
                return None
            
            # Get actual prices
            actual_prices = df_1h['close'].iloc[-lookback:].values
            
            # Align lengths
            if len(actual_prices) < len(predictions):
                predictions = predictions[:len(actual_prices)]
            
            predictions = np.array(predictions[:len(actual_prices)])
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - actual_prices))
            rmse = np.sqrt(np.mean((predictions - actual_prices) ** 2))
            mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
            
            # Direction accuracy
            pred_direction = np.diff(predictions)
            actual_direction = np.diff(actual_prices)
            direction_correct = np.sum((pred_direction > 0) == (actual_direction > 0))
            direction_accuracy = direction_correct / len(pred_direction) * 100
            
            # Additional metrics
            r_squared = 1 - (np.sum((actual_prices - predictions) ** 2) / np.sum((actual_prices - np.mean(actual_prices)) ** 2))
            mean_actual = np.mean(actual_prices)
            mean_pred = np.mean(predictions)
            std_actual = np.std(actual_prices)
            std_pred = np.std(predictions)
            
            metrics = {
                'symbol': self.symbol,
                'timestamp': datetime.now().isoformat(),
                'num_predictions': len(predictions),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy),
                'r_squared': float(r_squared),
                'mean_actual': float(mean_actual),
                'mean_predicted': float(mean_pred),
                'std_actual': float(std_actual),
                'std_predicted': float(std_pred),
                'min_actual': float(np.min(actual_prices)),
                'max_actual': float(np.max(actual_prices)),
                'min_predicted': float(np.min(predictions)),
                'max_predicted': float(np.max(predictions)),
                'actual_range': float(np.max(actual_prices) - np.min(actual_prices)),
                'predicted_range': float(np.max(predictions) - np.min(predictions)),
            }
            
            return metrics, predictions, actual_prices
        
        except Exception as e:
            logger.error(f'Error calculating metrics: {str(e)}')
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def save_results(self, metrics, predictions, actual_prices, output_dir='results'):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, f'{self.symbol}_metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f'Metrics saved to {metrics_path}')
        
        # Save predictions vs actual as CSV
        csv_path = os.path.join(output_dir, f'{self.symbol}_predictions_{timestamp}.csv')
        df = pd.DataFrame({
            'Index': range(1, len(predictions) + 1),
            'Actual': actual_prices,
            'Predicted': predictions,
            'Error': np.abs(predictions - actual_prices),
            'Error_Pct': (np.abs(predictions - actual_prices) / actual_prices) * 100
        })
        df.to_csv(csv_path, index=False)
        logger.info(f'Predictions saved to {csv_path}')
        
        return metrics_path, csv_path
    
    def print_metrics(self, metrics):
        """Print metrics nicely"""
        logger.info('\n' + '='*80)
        logger.info('PREDICTION EVALUATION METRICS')
        logger.info('='*80)
        logger.info(f'Symbol: {metrics["symbol"]}')
        logger.info(f'Timestamp: {metrics["timestamp"]}')
        logger.info(f'Number of Predictions: {metrics["num_predictions"]}')
        
        logger.info('\n--- Error Metrics ---')
        logger.info(f'MAE (Mean Absolute Error): ${metrics["mae"]:.2f}')
        logger.info(f'RMSE (Root Mean Squared Error): ${metrics["rmse"]:.2f}')
        logger.info(f'MAPE (Mean Absolute Percentage Error): {metrics["mape"]:.2f}%')
        
        logger.info('\n--- Accuracy Metrics ---')
        logger.info(f'Direction Accuracy: {metrics["direction_accuracy"]:.2f}%')
        logger.info(f'R-squared: {metrics["r_squared"]:.4f}')
        
        logger.info('\n--- Statistical Comparison ---')
        logger.info(f'Mean Actual Price: ${metrics["mean_actual"]:.2f}')
        logger.info(f'Mean Predicted Price: ${metrics["mean_predicted"]:.2f}')
        logger.info(f'Std Dev Actual: ${metrics["std_actual"]:.2f}')
        logger.info(f'Std Dev Predicted: ${metrics["std_predicted"]:.2f}')
        
        logger.info('\n--- Price Ranges ---')
        logger.info(f'Actual Range: ${metrics["min_actual"]:.2f} - ${metrics["max_actual"]:.2f} (Width: ${metrics["actual_range"]:.2f})')
        logger.info(f'Predicted Range: ${metrics["min_predicted"]:.2f} - ${metrics["max_predicted"]:.2f} (Width: ${metrics["predicted_range"]:.2f})')
        
        logger.info('='*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to evaluate')
    parser.add_argument('--num-predictions', type=int, default=50, help='Number of predictions')
    parser.add_argument('--lookback', type=int, default=50, help='Lookback period for evaluation')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    try:
        evaluator = PredictionEvaluator(symbol=args.symbol, device=device)
        
        # Generate predictions
        logger.info(f'\nGenerating {args.num_predictions} predictions...')
        pred_result = evaluator.generate_predictions(num_predictions=args.num_predictions)
        
        if pred_result is None:
            logger.error('Failed to generate predictions')
            return 1
        
        predictions = pred_result['predictions']
        
        # Calculate metrics
        logger.info(f'\nCalculating metrics (lookback={args.lookback})...')
        metrics, pred_aligned, actual_prices = evaluator.calculate_metrics(predictions, lookback=args.lookback)
        
        if metrics is None:
            logger.error('Failed to calculate metrics')
            return 1
        
        # Print metrics
        evaluator.print_metrics(metrics)
        
        # Save results
        if not args.no_save:
            metrics_path, csv_path = evaluator.save_results(metrics, pred_aligned, actual_prices, args.output_dir)
            logger.info(f'\nResults saved:')
            logger.info(f'  Metrics: {metrics_path}')
            logger.info(f'  Predictions: {csv_path}')
        
        return 0
    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
