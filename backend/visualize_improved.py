#!/usr/bin/env python
r"""
Visualization script for improved model

Shows:
1. Actual vs Predicted prices
2. Prediction errors/residuals
3. Training curves (loss history)

Usage:
    python backend/visualize_improved.py --symbol BTCUSDT --timeframe 1h --num-predictions 50
"""

import torch
import logging
import sys
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.inference_improved import ImprovedInference, ImprovedCryptoGRU
from backend.data.data_manager import DataManager
from backend.data.data_loader import CryptoDataLoader
from config.model_config import MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedVisualizer:
    def __init__(self):
        self.inferencer = ImprovedInference()
        self.data_manager = DataManager()
    
    def visualize_predictions(self, symbol, timeframe, model_type='best', num_predictions=50, output_file=None):
        """
        Visualize predictions vs actual prices
        """
        logger.info(f"\nVisualizing predictions for {symbol} ({timeframe})...")
        
        # Load model
        model, config, scalers = self.inferencer.load_model(symbol, timeframe, model_type)
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Generate predictions
        results = self.inferencer.predict_batch(model, scalers, symbol, timeframe, num_predictions)
        if results is None:
            logger.error("Failed to generate predictions")
            return
        
        timestamps = results['timestamps']
        actuals = results['actuals']
        predictions = results['predictions']
        errors = results['errors']
        metrics = results['metrics']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # Plot 1: Actual vs Predicted
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(timestamps, actuals, label='Actual Price', linewidth=2.5, color='#00a6fb', marker='o', markersize=5, markeredgewidth=0.5)
        ax1.plot(timestamps, predictions, label='Predicted Price', linewidth=2.5, color='#ff006e', marker='s', markersize=5, markeredgewidth=0.5, alpha=0.85)
        ax1.fill_between(timestamps, actuals, predictions, alpha=0.1, color='gray')
        ax1.set_title(f'{symbol} ({timeframe}) - Actual vs Predicted Prices (Improved GRU)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Errors
        ax2 = fig.add_subplot(gs[1])
        colors = ['green' if e >= 0 else 'red' for e in errors]
        ax2.bar(range(len(errors)), errors, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Prediction Errors (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Error (USD)', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cumulative metrics
        ax3 = fig.add_subplot(gs[2])
        abs_errors = np.abs(errors)
        cumulative_error = np.cumsum(abs_errors)
        ax3.plot(range(len(cumulative_error)), cumulative_error, linewidth=2, color='#ff006e')
        ax3.fill_between(range(len(cumulative_error)), cumulative_error, alpha=0.3, color='#ff006e')
        ax3.set_title('Cumulative Absolute Error', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Prediction Index', fontsize=11)
        ax3.set_ylabel('Cumulative Error (USD)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics_text = f"""Model Performance (Improved GRU)
MAE: ${metrics['mae']:.2f}
RMSE: ${metrics['rmse']:.2f}
MAPE: {metrics['mape']:.2f}%
Avg Actual: ${metrics['avg_actual']:.2f}
Avg Predicted: ${metrics['avg_predicted']:.2f}
Difference: ${abs(metrics['avg_actual'] - metrics['avg_predicted']):.2f}
Predictions: {len(predictions)}"""
        
        fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        if output_file is None:
            from datetime import datetime
            output_file = f"prediction_improved_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_file}")
        plt.show()
    
    def visualize_training_history(self, symbol, timeframe, output_file=None):
        """
        Load and visualize training history from checkpoint
        """
        logger.info(f"\nVisualizing training history for {symbol} ({timeframe})...")
        
        model_dir = Path('backend/models/weights')
        model_path = model_dir / f"{symbol}_{timeframe}_best.pth"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 'unknown')
            train_loss = checkpoint.get('train_loss', 'unknown')
            val_loss = checkpoint.get('val_loss', 'unknown')
            
            logger.info(f"Best model checkpoint: epoch={epoch}, train_loss={train_loss}, val_loss={val_loss}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")

def main():
    parser = ArgumentParser(description='Visualize improved model predictions')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--model-type', default='best', help='best or final (default: best)')
    parser.add_argument('--num-predictions', type=int, default=50, help='Number of predictions')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    visualizer = ImprovedVisualizer()
    visualizer.visualize_predictions(
        args.symbol,
        args.timeframe,
        args.model_type,
        args.num_predictions,
        args.output
    )

if __name__ == '__main__':
    main()
