import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from datetime import datetime
import json

from backend.data.data_manager import DataManager
from backend.models.lstm_model import CryptoLSTM
from backend.data.data_loader import CryptoDataLoader
from config.model_config import MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluate trained LSTM models on test data
    Generate visualizations and metrics
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize evaluator
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.data_manager = DataManager()
        self.data_loader = CryptoDataLoader()
        self.model_dir = Path('backend/models/weights')
        
        logger.info(f"ModelEvaluator initialized on {device}")
    
    def load_model(self, symbol, timeframe, model_type='v1'):
        """
        Load trained model
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '15m')
            model_type: Model version (default: 'v1')
        
        Returns:
            Loaded model on specified device
        """
        model_path = self.model_dir / f"{symbol}_{timeframe}_{model_type}.pt"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        
        try:
            # Load model state
            state = torch.load(model_path, map_location=self.device)
            
            # Create model with same config
            model = CryptoLSTM(
                input_size=state['model_config']['input_size'],
                hidden_size=MODEL_CONFIG['hidden_size'],
                num_layers=MODEL_CONFIG['num_layers'],
                output_size=state['model_config']['output_size'],
                dropout=MODEL_CONFIG['dropout']
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            
            logger.info(f"Loaded model: {symbol} ({timeframe})")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def evaluate_symbol(self, symbol, timeframe='1h', test_split=0.2):
        """
        Evaluate model on a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            test_split: Fraction of data to use for testing (0.2 = last 20%)
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating {symbol} ({timeframe})")
        logger.info(f"{'='*70}")
        
        # Load data
        data = self.data_manager.get_stored_data(symbol, timeframe)
        if data is None or len(data) < 100:
            logger.warning(f"Insufficient data for {symbol} ({timeframe})")
            return None
        
        logger.info(f"Loaded {len(data)} rows of data")
        
        # Calculate indicators
        data_with_indicators = self.data_loader.calculate_technical_indicators(data)
        
        if data_with_indicators is None or data_with_indicators.empty:
            logger.warning(f"Failed to calculate indicators for {symbol}")
            return None
        
        # Get feature columns (exclude OHLCV)
        feature_cols = [col for col in data_with_indicators.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Using {len(feature_cols)} features for prediction")
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_normalized = data_with_indicators.copy()
        data_normalized[feature_cols] = scaler.fit_transform(data_with_indicators[feature_cols])
        
        # Prepare sequences
        lookback = MODEL_CONFIG['lookback']
        X = []
        y = []
        close_prices = data_with_indicators['close'].values
        
        for i in range(len(data_normalized) - lookback):
            X.append(data_normalized[feature_cols].iloc[i:i+lookback].values)
            y.append(close_prices[i+lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared {len(X)} sequences with lookback={lookback}")
        
        # Split into train and test
        split_idx = int(len(X) * (1 - test_split))
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        test_timestamps = data_with_indicators.index[lookback+split_idx:]
        
        logger.info(f"Test set: {len(X_test)} samples (from {test_timestamps[0]} to {test_timestamps[-1]})")
        
        # Load model
        model = self.load_model(symbol, timeframe)
        if model is None:
            return None
        
        # Make predictions
        logger.info("\nGenerating predictions...")
        y_pred = []
        
        with torch.no_grad():
            for i, x in enumerate(X_test):
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                pred = model(x_tensor).cpu().numpy()[0, 0]
                y_pred.append(pred)
                
                if (i + 1) % max(1, len(X_test) // 5) == 0:
                    logger.info(f"  Predicted {i+1}/{len(X_test)} samples")
        
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Calculate directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'test_samples': len(X_test),
            'metrics': {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MAPE': float(mape),
                'Directional_Accuracy': float(directional_accuracy),
                'Mean_Actual_Price': float(np.mean(y_test)),
                'Mean_Predicted_Price': float(np.mean(y_pred)),
                'Price_Range_Actual': f"{np.min(y_test):.2f} - {np.max(y_test):.2f}",
                'Price_Range_Predicted': f"{np.min(y_pred):.2f} - {np.max(y_pred):.2f}",
            },
            'data': {
                'timestamps': test_timestamps.tolist(),
                'actual': y_test.tolist(),
                'predicted': y_pred.tolist(),
            }
        }
        
        # Log results
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATION RESULTS: {symbol} ({timeframe})")
        logger.info(f"{'='*70}")
        logger.info(f"Test Samples: {len(X_test)}")
        logger.info(f"\nMetrics:")
        logger.info(f"  MAE (Mean Absolute Error): {mae:.6f}")
        logger.info(f"  RMSE (Root Mean Squared Error): {rmse:.6f}")
        logger.info(f"  MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        logger.info(f"\nPrice Statistics:")
        logger.info(f"  Actual Mean: {np.mean(y_test):.2f}")
        logger.info(f"  Predicted Mean: {np.mean(y_pred):.2f}")
        logger.info(f"  Actual Range: {np.min(y_test):.2f} - {np.max(y_test):.2f}")
        logger.info(f"  Predicted Range: {np.min(y_pred):.2f} - {np.max(y_pred):.2f}")
        logger.info(f"{'='*70}")
        
        return results
    
    def evaluate_all_symbols(self, symbols=None, timeframe='1h'):
        """
        Evaluate all models
        
        Args:
            symbols: List of symbols to evaluate (default: all)
            timeframe: Timeframe to evaluate
        
        Returns:
            Dictionary with all results
        """
        if symbols is None:
            # Get list of all model files
            symbols = set()
            for model_file in self.model_dir.glob(f"*_{timeframe}_v1.pt"):
                symbol = model_file.name.replace(f"_{timeframe}_v1.pt", "")
                symbols.add(symbol)
            symbols = sorted(list(symbols))
        
        logger.info(f"\n\nEvaluating {len(symbols)} symbols on {timeframe} timeframe...")
        
        all_results = {}
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] {symbol}")
            result = self.evaluate_symbol(symbol, timeframe)
            
            if result:
                all_results[symbol] = result
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info(f"\n\n{'='*70}")
        logger.info(f"EVALUATION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Successful: {successful}/{len(symbols)}")
        logger.info(f"Failed: {failed}/{len(symbols)}")
        
        if all_results:
            # Find best and worst performers
            mape_scores = {sym: res['metrics']['MAPE'] for sym, res in all_results.items()}
            best_symbol = min(mape_scores, key=mape_scores.get)
            worst_symbol = max(mape_scores, key=mape_scores.get)
            
            logger.info(f"\nBest Performer: {best_symbol} (MAPE: {mape_scores[best_symbol]:.4f}%)")
            logger.info(f"Worst Performer: {worst_symbol} (MAPE: {mape_scores[worst_symbol]:.4f}%)")
            logger.info(f"Average MAPE: {np.mean(list(mape_scores.values())):.4f}%")
        
        logger.info(f"{'='*70}")
        
        return all_results
    
    def save_results(self, results, filename='evaluation_results.json'):
        """
        Save evaluation results to JSON
        """
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


if __name__ == "__main__":
    """
    Evaluate trained models
    """
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device='cpu')
    
    # Evaluate single symbol
    logger.info("\nEvaluating single symbol: BTCUSDT (1h)")
    result = evaluator.evaluate_symbol('BTCUSDT', timeframe='1h')
    
    if result:
        # Save results
        evaluator.save_results(result, 'BTCUSDT_1h_evaluation.json')
    
    # Optionally evaluate all symbols
    # logger.info("\n\nEvaluating all symbols...")
    # all_results = evaluator.evaluate_all_symbols(timeframe='1h')
    # evaluator.save_results(all_results, 'all_evaluations.json')
