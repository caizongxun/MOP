import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionVisualizer:
    """
    Visualize model predictions vs actual prices
    Generate comparison charts and statistics
    """
    
    def __init__(self):
        """
        Initialize visualizer
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available. Install with: pip install matplotlib")
    
    def plot_predictions(self, evaluation_result, symbol, timeframe='1h', figsize=(16, 8), save_path=None):
        """
        Plot actual vs predicted prices
        
        Args:
            evaluation_result: Result dictionary from ModelEvaluator
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            figsize: Figure size (width, height)
            save_path: Path to save the figure (default: None, display only)
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib not installed")
            return
        
        # Extract data
        timestamps = pd.to_datetime(evaluation_result['data']['timestamps'])
        actual = np.array(evaluation_result['data']['actual'])
        predicted = np.array(evaluation_result['data']['predicted'])
        metrics = evaluation_result['metrics']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f'{symbol} ({timeframe}) - Model Evaluation', fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs Predicted
        ax1 = axes[0]
        ax1.plot(timestamps, actual, label='Actual Price', linewidth=2, color='#1f77b4', marker='o', markersize=3, alpha=0.8)
        ax1.plot(timestamps, predicted, label='Predicted Price', linewidth=2, color='#ff7f0e', marker='s', markersize=3, alpha=0.8)
        ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Actual vs Predicted Prices (Test Set: {len(actual)} samples)', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Error and Directional Accuracy
        ax2 = axes[1]
        error = np.abs(actual - predicted)
        percentage_error = np.abs((actual - predicted) / actual) * 100
        
        ax2.bar(range(len(error)), error, alpha=0.7, color='#d62728', label='Absolute Error')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(len(percentage_error)), percentage_error, color='#2ca02c', linewidth=2, marker='o', markersize=3, label='Percentage Error (%)')
        
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Absolute Error (USDT)', fontsize=12, fontweight='bold', color='#d62728')
        ax2_twin.set_ylabel('Percentage Error (%)', fontsize=12, fontweight='bold', color='#2ca02c')
        ax2.set_title('Prediction Errors', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        ax2_twin.tick_params(axis='y', labelcolor='#2ca02c')
        
        # Add metrics text box
        metrics_text = f"""METRICS:
MAE: {metrics['MAE']:.6f} USDT
RMSE: {metrics['RMSE']:.6f} USDT
MAPE: {metrics['MAPE']:.4f}%
Dir. Accuracy: {metrics['Directional_Accuracy']:.2f}%

PRICE STATS:
Actual Mean: {metrics['Mean_Actual_Price']:.2f} USDT
Pred Mean: {metrics['Mean_Predicted_Price']:.2f} USDT
Actual Range: {metrics['Price_Range_Actual']}
Pred Range: {metrics['Price_Range_Predicted']}"""
        
        fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_comparison_report(self, evaluation_result, symbol, timeframe='1h', save_path='evaluation_report.txt'):
        """
        Create a text report of evaluation results
        
        Args:
            evaluation_result: Result dictionary from ModelEvaluator
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            save_path: Path to save report
        """
        actual = np.array(evaluation_result['data']['actual'])
        predicted = np.array(evaluation_result['data']['predicted'])
        metrics = evaluation_result['metrics']
        
        report = f"""
{'='*70}
MODEL EVALUATION REPORT
{'='*70}

Symbol: {symbol}
Timeframe: {timeframe}
Test Samples: {evaluation_result['test_samples']}

{'='*70}
PREDICTION METRICS
{'='*70}

MAE (Mean Absolute Error): {metrics['MAE']:.8f} USDT
RMSE (Root Mean Squared Error): {metrics['RMSE']:.8f} USDT
MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.6f}%
Directional Accuracy: {metrics['Directional_Accuracy']:.4f}%

Interpretation:
- MAPE < 1%: Excellent prediction
- MAPE < 5%: Very good prediction
- MAPE < 10%: Good prediction
- MAPE >= 10%: Poor prediction

{'='*70}
PRICE STATISTICS
{'='*70}

Actual Prices:
  Mean: {metrics['Mean_Actual_Price']:.2f} USDT
  Min: {np.min(actual):.2f} USDT
  Max: {np.max(actual):.2f} USDT
  Std Dev: {np.std(actual):.2f} USDT
  Range: {metrics['Price_Range_Actual']}

Predicted Prices:
  Mean: {metrics['Mean_Predicted_Price']:.2f} USDT
  Min: {np.min(predicted):.2f} USDT
  Max: {np.max(predicted):.2f} USDT
  Std Dev: {np.std(predicted):.2f} USDT
  Range: {metrics['Price_Range_Predicted']}

Mean Difference: {abs(metrics['Mean_Actual_Price'] - metrics['Mean_Predicted_Price']):.2f} USDT

{'='*70}
ERROR ANALYSIS
{'='*70}

Absolute Error:
  Mean: {np.mean(np.abs(actual - predicted)):.6f} USDT
  Min: {np.min(np.abs(actual - predicted)):.6f} USDT
  Max: {np.max(np.abs(actual - predicted)):.6f} USDT
  Std Dev: {np.std(np.abs(actual - predicted)):.6f} USDT

Percentage Error:
  Mean: {np.mean(np.abs((actual - predicted) / actual) * 100):.6f}%
  Min: {np.min(np.abs((actual - predicted) / actual) * 100):.6f}%
  Max: {np.max(np.abs((actual - predicted) / actual) * 100):.6f}%
  Std Dev: {np.std(np.abs((actual - predicted) / actual) * 100):.6f}%

{'='*70}
DIRECTIONAL ACCURACY
{'='*70}

Accuracy: {metrics['Directional_Accuracy']:.2f}%

This metric shows how often the model correctly predicts
whether the price will go up or down.

{'='*70}
CONCLUSION
{'='*70}

"""
        
        # Add conclusion based on MAPE
        mape = metrics['MAPE']
        if mape < 1:
            report += "Model Performance: EXCELLENT - MAPE < 1%\n"
            report += "The model provides highly accurate price predictions.\n"
        elif mape < 5:
            report += "Model Performance: VERY GOOD - MAPE < 5%\n"
            report += "The model provides strong price predictions.\n"
        elif mape < 10:
            report += "Model Performance: GOOD - MAPE < 10%\n"
            report += "The model provides reasonable price predictions.\n"
        else:
            report += "Model Performance: NEEDS IMPROVEMENT - MAPE >= 10%\n"
            report += "Consider improving the model with more data or tuning.\n"
        
        report += f"\n{'='*70}\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*70}\n"
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {save_path}")
        print(report)
        
        return report


if __name__ == "__main__":
    """
    Visualize evaluation results
    """
    Path('logs').mkdir(exist_ok=True)
    
    # Load evaluation results
    result_file = 'BTCUSDT_1h_evaluation.json'
    
    if not Path(result_file).exists():
        logger.error(f"Evaluation result file not found: {result_file}")
        logger.info("Please run backend/evaluate_model.py first")
        exit(1)
    
    with open(result_file, 'r') as f:
        evaluation_result = json.load(f)
    
    symbol = evaluation_result['symbol']
    timeframe = evaluation_result['timeframe']
    
    # Create visualizer
    visualizer = PredictionVisualizer()
    
    # Create report
    logger.info(f"Creating evaluation report for {symbol} ({timeframe})...")
    visualizer.create_comparison_report(
        evaluation_result,
        symbol,
        timeframe,
        save_path=f"{symbol}_{timeframe}_report.txt"
    )
    
    # Plot predictions
    if HAS_MATPLOTLIB:
        logger.info(f"Creating visualization for {symbol} ({timeframe})...")
        visualizer.plot_predictions(
            evaluation_result,
            symbol,
            timeframe,
            save_path=f"{symbol}_{timeframe}_predictions.png"
        )
