#!/usr/bin/env python
r"""
Improved visualization - Understands model training target

The model was trained to predict:
y.append(data[i+seq_length+prediction_horizon-1, 3])  # Close price at next step

Usage:
    python backend/visualize_predictions_v2.py --symbol BTCUSDT --timeframe 1h --num-predictions 50
"""

import logging
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from backend.models.lstm_model import CryptoLSTM
from backend.data.data_manager import DataManager
from backend.data.data_loader import CryptoDataLoader
from config.model_config import MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionVisualizerV2:
    def __init__(self, device='cpu'):
        self.device = device
        self.model_dir = Path('backend/models/weights')
        self.data_manager = DataManager()
        self.data_loader = CryptoDataLoader()
    
    def load_model(self, symbol, timeframe):
        """Load trained model"""
        for ext in ['.pt', '.pth']:
            model_path = self.model_dir / f"{symbol}_{timeframe}_v1{ext}"
            if model_path.exists():
                logger.info(f"Found model: {model_path.name}")
                break
        else:
            logger.error(f"Model file not found for {symbol}_{timeframe}")
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
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def generate_predictions_debug(self, symbol, timeframe, num_predictions=50):
        """Generate predictions with detailed debug info"""
        logger.info(f"\nGenerating predictions for {symbol} ({timeframe})...")
        
        data = self.data_manager.get_stored_data(symbol, timeframe)
        if data is None or len(data) < 30:
            logger.error(f"Insufficient data for {symbol}")
            return None
        
        data_with_indicators = self.data_loader.calculate_technical_indicators(data)
        if data_with_indicators is None or data_with_indicators.empty:
            logger.error(f"Failed to calculate indicators")
            return None
        
        all_feature_cols = [col for col in data_with_indicators.columns]
        logger.info(f"Feature columns: {len(all_feature_cols)} - {all_feature_cols}")
        
        # Normalize ALL features
        scaler = MinMaxScaler()
        data_normalized = data_with_indicators.copy()
        data_scaled = scaler.fit_transform(data_with_indicators[all_feature_cols])
        data_normalized[all_feature_cols] = data_scaled
        
        # Get min/max of close price for denormalization
        close_min = data_with_indicators['close'].min()
        close_max = data_with_indicators['close'].max()
        logger.info(f"Close price range: {close_min:.2f} - {close_max:.2f}")
        
        lookback = MODEL_CONFIG['lookback']
        max_predictions = len(data_normalized) - lookback
        actual_num_predictions = min(num_predictions, max_predictions)
        
        logger.info(f"Data points: {len(data_normalized)}, Max predictions: {max_predictions}")
        logger.info(f"Generating {actual_num_predictions} predictions...")
        
        model = self.load_model(symbol, timeframe)
        if model is None:
            return None
        
        predictions_normalized = []
        predictions_denorm = []
        actual_prices = []
        actual_prices_norm = []
        timestamps = []
        
        with torch.no_grad():
            for i in range(actual_num_predictions):
                offset = actual_num_predictions - 1 - i
                end_idx = len(data_normalized) - offset
                start_idx = end_idx - lookback
                
                if start_idx < 0:
                    continue
                
                # Input: sequence of 60 candles with 44 features
                x = data_normalized[all_feature_cols].iloc[start_idx:end_idx].values
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                
                logger.info(f"\nPrediction {i+1}:")
                logger.info(f"  Input index range: [{start_idx}, {end_idx})")
                logger.info(f"  x shape: {x.shape}")
                
                try:
                    # Model output: normalized close price at position end_idx
                    pred_normalized = model(x_tensor).cpu().numpy()[0, 0]
                    
                    # Denormalize: pred_norm * (max - min) + min
                    pred_denorm = pred_normalized * (close_max - close_min) + close_min
                    
                    predictions_normalized.append(float(pred_normalized))
                    predictions_denorm.append(float(pred_denorm))
                    
                    # Get actual price at position end_idx-1 (the target index)
                    actual_idx = end_idx - 1
                    if actual_idx < len(data_with_indicators):
                        actual_price = float(data_with_indicators['close'].iloc[actual_idx])
                        actual_prices.append(actual_price)
                        
                        actual_norm = (actual_price - close_min) / (close_max - close_min)
                        actual_prices_norm.append(actual_norm)
                        
                        timestamp = data_with_indicators.index[actual_idx]
                        timestamps.append(timestamp)
                        
                        logger.info(f"  Actual index: {actual_idx}")
                        logger.info(f"  Predicted (norm): {pred_normalized:.6f}")
                        logger.info(f"  Predicted (denorm): {pred_denorm:.2f}")
                        logger.info(f"  Actual (denorm): {actual_price:.2f}")
                        logger.info(f"  Error: {abs(actual_price - pred_denorm):.2f} ({abs(actual_price - pred_denorm)/actual_price*100:.2f}%)")
                    
                except Exception as e:
                    logger.error(f"Error on prediction {i+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not predictions_denorm:
            logger.error("No successful predictions")
            return None
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamps': timestamps,
            'actual_prices': actual_prices,
            'predicted_prices': predictions_denorm,
            'statistics': {
                'num_predictions': len(predictions_denorm),
                'avg_actual': np.mean(actual_prices),
                'avg_predicted': np.mean(predictions_denorm),
                'rmse': np.sqrt(np.mean([(a - p)**2 for a, p in zip(actual_prices, predictions_denorm)])),
                'mae': np.mean([abs(a - p) for a, p in zip(actual_prices, predictions_denorm)]),
                'mape': np.mean([abs(a - p) / a * 100 for a, p in zip(actual_prices, predictions_denorm)])
            }
        }
        
        return results
    
    def plot_predictions(self, results, output_file=None):
        """Plot actual vs predicted prices"""
        if not results:
            logger.error("No results to plot")
            return
        
        symbol = results['symbol']
        timeframe = results['timeframe']
        timestamps = results['timestamps']
        actual_prices = results['actual_prices']
        predicted_prices = results['predicted_prices']
        stats = results['statistics']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Actual vs Predicted
        ax1.plot(timestamps, actual_prices, label='Actual Price', linewidth=2.5, color='#00a6fb', marker='o', markersize=4, markeredgewidth=0.5)
        ax1.plot(timestamps, predicted_prices, label='Predicted Price', linewidth=2.5, color='#ff006e', marker='s', markersize=4, markeredgewidth=0.5, alpha=0.9)
        ax1.set_title(f'{symbol} ({timeframe}) - Actual vs Predicted Prices (Denormalized)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Timestamp', fontsize=12)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(fontsize=12, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Error/Residuals
        errors = [a - p for a, p in zip(actual_prices, predicted_prices)]
        colors = ['green' if e >= 0 else 'red' for e in errors]
        ax2.bar(range(len(errors)), errors, color=colors, alpha=0.7, label='Residuals (Actual - Predicted)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Prediction Errors (Residuals)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Prediction Index', fontsize=12)
        ax2.set_ylabel('Error (USD)', fontsize=12)
        ax2.legend(fontsize=12, loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"""Model Performance:
RMSE: ${stats['rmse']:.2f}
MAE: ${stats['mae']:.2f}
MAPE: {stats['mape']:.2f}%
Avg Actual: ${stats['avg_actual']:.2f}
Avg Predicted: ${stats['avg_predicted']:.2f}
Num Predictions: {stats['num_predictions']}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=11, family='monospace', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        
        if output_file is None:
            output_file = f"prediction_plot_v2_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        logger.info(f"\nPlot saved to: {output_file}")
        
        plt.show()
    
    def visualize(self, symbol, timeframe, num_predictions=50, output_file=None):
        """Generate and visualize predictions"""
        logger.info(f"\nVisualizing predictions for {symbol} ({timeframe})")
        logger.info(f"{'='*70}")
        
        results = self.generate_predictions_debug(symbol, timeframe, num_predictions)
        
        if results:
            stats = results['statistics']
            logger.info(f"\n{'='*70}")
            logger.info(f"PREDICTION STATISTICS")
            logger.info(f"{'='*70}")
            logger.info(f"Predictions: {stats['num_predictions']}")
            logger.info(f"RMSE: ${stats['rmse']:.2f}")
            logger.info(f"MAE: ${stats['mae']:.2f}")
            logger.info(f"MAPE: {stats['mape']:.2f}%")
            logger.info(f"Avg Actual Price: ${stats['avg_actual']:.2f}")
            logger.info(f"Avg Predicted Price: ${stats['avg_predicted']:.2f}")
            logger.info(f"Price Difference: ${abs(stats['avg_actual'] - stats['avg_predicted']):.2f}")
            
            self.plot_predictions(results, output_file)
        else:
            logger.error("Failed to generate predictions")


def main():
    parser = ArgumentParser(description='Visualize predictions vs actual prices (v2)')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--num-predictions', type=int, default=50, help='Number of predictions (default: 50)')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    visualizer = PredictionVisualizerV2()
    visualizer.visualize(args.symbol, args.timeframe, args.num_predictions, args.output)


if __name__ == "__main__":
    main()
