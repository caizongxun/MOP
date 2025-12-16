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
from collections import defaultdict

# Fix imports
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm_model import CryptoLSTM
from data.data_manager import DataManager
from data.data_loader import CryptoDataLoader
from config.model_config import MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchModelEvaluator:
    def __init__(self, model_dir='backend/models/weights', device=None):
        self.model_dir = model_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = DataManager()
        self.data_loader = CryptoDataLoader()
        logger.info(f'Using device: {self.device}')
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            return self.data_loader.calculate_technical_indicators(df)
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def find_models(self, pattern='*_1h_final.pth'):
        """Find all model files matching pattern"""
        model_dir = Path(self.model_dir)
        models = sorted(model_dir.glob(pattern))
        logger.info(f'Found {len(models)} models matching pattern: {pattern}')
        for model in models:
            logger.info(f'  - {model.name}')
        return models
    
    def extract_symbol(self, filename):
        """Extract symbol from filename (e.g., BTCUSDT_1h_final.pth -> BTCUSDT)"""
        return filename.replace('_1h_final.pth', '').replace('_1h_best.pth', '')
    
    def load_model(self, model_path):
        """Load a single model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            model_config = checkpoint.get('model_config', {})
            model = CryptoLSTM(
                input_size=model_config.get('input_size', 19),
                hidden_size=model_config.get('hidden_size', 64),
                num_layers=model_config.get('num_layers', 2),
                output_size=model_config.get('output_size', 1)
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model, checkpoint
        
        except Exception as e:
            logger.error(f'Error loading model {model_path}: {str(e)}')
            return None, None
    
    def evaluate_model(self, symbol, model_path, timeframe='1h', lookback=50):
        """Evaluate a single model"""
        try:
            # Load model
            model, checkpoint = self.load_model(model_path)
            if model is None:
                return None
            
            # Load data
            df = self.dm.get_stored_data(symbol, timeframe)
            if df is None or len(df) < 100:
                logger.warning(f'{symbol}: Insufficient data')
                return None
            
            # Calculate indicators
            df_ind = self._calculate_indicators(df)
            if df_ind is None or df_ind.empty:
                logger.warning(f'{symbol}: Failed to calculate indicators')
                return None
            
            # Prepare data
            feature_cols = [col for col in df_ind.columns if col != 'timestamp']
            scaler = MinMaxScaler()
            df_normalized = df_ind.copy()
            df_normalized[feature_cols] = scaler.fit_transform(df_ind[feature_cols])
            
            # Get close price scaler for denormalization
            close_min = df_ind['close'].min()
            close_max = df_ind['close'].max()
            
            # Create sequences
            lookback_period = MODEL_CONFIG.get('lookback', 60)
            X_list = []
            y_list = []
            
            for i in range(len(df_normalized) - lookback_period):
                X_list.append(df_normalized[feature_cols].iloc[i:i+lookback_period].values)
                y_list.append(df_ind['close'].iloc[i+lookback_period])
            
            if len(X_list) < 10:
                logger.warning(f'{symbol}: Not enough sequences')
                return None
            
            X = np.array(X_list)
            y_actual = np.array(y_list)
            
            # Make predictions
            predictions = []
            with torch.no_grad():
                for x_seq in X[-lookback:]:
                    x_tensor = torch.FloatTensor(x_seq).unsqueeze(0).to(self.device)
                    pred_normalized = model(x_tensor).item()
                    # Denormalize
                    pred_actual = pred_normalized * (close_max - close_min) + close_min
                    predictions.append(pred_actual)
            
            # Get actual values
            actual = y_actual[-lookback:]
            predictions = np.array(predictions)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            # Direction accuracy
            pred_dir = np.diff(predictions)
            actual_dir = np.diff(actual)
            dir_correct = np.sum((pred_dir > 0) == (actual_dir > 0))
            dir_accuracy = dir_correct / len(pred_dir) * 100 if len(pred_dir) > 0 else 0
            
            # R-squared
            ss_res = np.sum((actual - predictions) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                'symbol': symbol,
                'timeframe': timeframe,
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'direction_accuracy': float(dir_accuracy),
                'r_squared': float(r_squared),
                'mean_actual': float(np.mean(actual)),
                'mean_predicted': float(np.mean(predictions)),
                'num_predictions': len(predictions),
                'model_epoch': checkpoint.get('epoch', 'N/A'),
                'model_train_loss': checkpoint.get('train_loss', 'N/A'),
                'model_val_loss': checkpoint.get('val_loss', 'N/A'),
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f'Error evaluating {symbol}: {str(e)}')
            import traceback
            traceback.print_exc()
            return None
    
    def batch_evaluate(self, symbols=None, timeframe='1h', lookback=50):
        """Evaluate all models"""
        # Find models
        model_files = self.find_models(f'*_{timeframe}_final.pth')
        
        if not model_files:
            logger.warning(f'No models found with pattern *_{timeframe}_final.pth')
            return None
        
        results = []
        
        for idx, model_path in enumerate(model_files, 1):
            symbol = self.extract_symbol(model_path.name)
            
            # Skip if symbols filter provided
            if symbols and symbol not in symbols:
                continue
            
            logger.info(f'\n[{idx}/{len(model_files)}] Evaluating {symbol}...')
            
            metrics = self.evaluate_model(symbol, str(model_path), timeframe, lookback)
            if metrics:
                results.append(metrics)
                logger.info(f'  MAPE: {metrics["mape"]:.2f}% | MAE: ${metrics["mae"]:.2f} | R²: {metrics["r_squared"]:.4f}')
            else:
                logger.warning(f'  Failed to evaluate {symbol}')
        
        return results
    
    def save_results(self, results, output_dir='results'):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = os.path.join(output_dir, f'batch_evaluation_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Results saved to {json_path}')
        
        # Save as CSV
        csv_path = os.path.join(output_dir, f'batch_evaluation_{timestamp}.csv')
        df = pd.DataFrame(results)
        df = df.sort_values('mape')
        df.to_csv(csv_path, index=False)
        logger.info(f'CSV saved to {csv_path}')
        
        return json_path, csv_path
    
    def print_summary(self, results):
        """Print summary statistics"""
        if not results:
            logger.warning('No results to summarize')
            return
        
        df = pd.DataFrame(results)
        
        logger.info('\n' + '='*100)
        logger.info('BATCH EVALUATION SUMMARY')
        logger.info('='*100)
        logger.info(f'\nTotal Models Evaluated: {len(results)}')
        
        logger.info('\n--- MAPE (Mean Absolute Percentage Error) ---')
        logger.info(f'Best: {df["mape"].min():.2f}% ({df.loc[df["mape"].idxmin(), "symbol"]})')
        logger.info(f'Worst: {df["mape"].max():.2f}% ({df.loc[df["mape"].idxmax(), "symbol"]})')
        logger.info(f'Average: {df["mape"].mean():.2f}%')
        logger.info(f'Median: {df["mape"].median():.2f}%')
        
        logger.info('\n--- MAE (Mean Absolute Error) ---')
        logger.info(f'Best: ${df["mae"].min():.2f} ({df.loc[df["mae"].idxmin(), "symbol"]})')
        logger.info(f'Worst: ${df["mae"].max():.2f} ({df.loc[df["mae"].idxmax(), "symbol"]})')
        logger.info(f'Average: ${df["mae"].mean():.2f}')
        
        logger.info('\n--- R-squared ---')
        logger.info(f'Best: {df["r_squared"].max():.4f} ({df.loc[df["r_squared"].idxmax(), "symbol"]})')
        logger.info(f'Worst: {df["r_squared"].min():.4f} ({df.loc[df["r_squared"].idxmin(), "symbol"]})')
        logger.info(f'Average: {df["r_squared"].mean():.4f}')
        
        logger.info('\n--- Direction Accuracy ---')
        logger.info(f'Best: {df["direction_accuracy"].max():.2f}% ({df.loc[df["direction_accuracy"].idxmax(), "symbol"]})')
        logger.info(f'Worst: {df["direction_accuracy"].min():.2f}% ({df.loc[df["direction_accuracy"].idxmin(), "symbol"]})')
        logger.info(f'Average: {df["direction_accuracy"].mean():.2f}%')
        
        logger.info('\n--- Top 5 Models by MAPE ---')
        top_5 = df.nsmallest(5, 'mape')[['symbol', 'mape', 'mae', 'r_squared', 'direction_accuracy']]
        for idx, row in top_5.iterrows():
            logger.info(f'{row["symbol"]:12s} MAPE: {row["mape"]:6.2f}% | MAE: ${row["mae"]:8.2f} | R²: {row["r_squared"]:7.4f} | Dir Acc: {row["direction_accuracy"]:6.2f}%')
        
        logger.info('\n' + '='*100)

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate all trained models')
    parser.add_argument('--model-dir', default='backend/models/weights', help='Model directory')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (1h, 15m)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to evaluate')
    parser.add_argument('--lookback', type=int, default=50, help='Lookback period')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    try:
        evaluator = BatchModelEvaluator(model_dir=args.model_dir, device=device)
        
        logger.info(f'\nEvaluating all {args.timeframe} models...')
        results = evaluator.batch_evaluate(
            symbols=args.symbols,
            timeframe=args.timeframe,
            lookback=args.lookback
        )
        
        if results:
            evaluator.print_summary(results)
            
            if not args.no_save:
                json_path, csv_path = evaluator.save_results(results, args.output_dir)
                logger.info(f'\nResults saved:')
                logger.info(f'  JSON: {json_path}')
                logger.info(f'  CSV: {csv_path}')
        else:
            logger.error('No results to process')
            return 1
        
        return 0
    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
