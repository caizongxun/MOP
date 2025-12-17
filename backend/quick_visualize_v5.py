#!/usr/bin/env python3
"""
Quick Visualizer for V5 Adaptive Predictions
Generates plots of actual vs predicted prices with error analysis
Uses unified PathConfig for consistent path handling
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_config import PathConfig
from train_v5_enhanced import FeatureEngineerV5, MultiScaleLSTMV5

class V5Visualizer:
    """Visualizer for V5 predictions with residual reconstruction"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.feature_calc = FeatureEngineerV5()
        self.paths = PathConfig()
        print(f"Using unified path configuration")
    
    def load_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load data using unified path config"""
        data_path = self.paths.get_data_file(symbol, timeframe)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        return pd.read_csv(data_path)
    
    def generate_predictions(self, symbol: str, timeframe: str = '1h', lookback: int = 60):
        """Generate predictions for a symbol"""
        print(f"\nGenerating predictions for {symbol}...")
        
        try:
            # Load data
            df = self.load_data(symbol, timeframe)
            close_prices = df['close'].values
            
            # Calculate features
            features_df = self.feature_calc.calculate_features(df)
            feature_values = features_df.values
            
            # Prepare sequences
            X = []
            for i in range(len(feature_values) - lookback):
                X.append(feature_values[i:i+lookback])
            
            if len(X) == 0:
                print(f"  Not enough data for {symbol}")
                return None
            
            X = np.array(X, dtype=np.float32)
            
            # Normalize features
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Load model using unified path config
            model_path = self.paths.get_model_weights_file(symbol, timeframe, version='v5')
            
            if not os.path.exists(model_path):
                print(f"  Model not found: {model_path}")
                print(f"  Path config: {self.paths.models_weights_dir}")
                return None
            
            model = MultiScaleLSTMV5(input_size=X.shape[2]).to(self.device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"  Error loading model: {e}")
                return None
            
            model.eval()
            
            # Generate predictions
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                pred_delta, pred_uncertainty = model(X_t)
                pred_delta = pred_delta.cpu().numpy().ravel()
                pred_uncertainty = pred_uncertainty.cpu().numpy().ravel()
            
            # Reconstruct prices from deltas
            pred_prices = np.zeros(len(close_prices))
            pred_prices[:lookback] = close_prices[:lookback]
            
            for i in range(lookback, len(close_prices)):
                idx = i - lookback
                if idx < len(pred_delta):
                    pred_prices[i] = pred_prices[i-1] + pred_delta[idx]
                else:
                    pred_prices[i] = pred_prices[i-1]
            
            # Use test set for metrics
            train_idx = int(0.70 * len(close_prices))
            val_idx = int(0.85 * len(close_prices))
            test_idx = val_idx
            
            actual_test = close_prices[test_idx:]
            pred_test = pred_prices[test_idx:]
            
            # Calculate metrics on test set
            if len(actual_test) > 0:
                mape = mean_absolute_percentage_error(actual_test, pred_test)
                mae = mean_absolute_error(actual_test, pred_test)
                rmse = np.sqrt(mean_squared_error(actual_test, pred_test))
            else:
                mape = mae = rmse = 0
            
            return {
                'symbol': symbol,
                'actual': close_prices,
                'predicted': pred_prices,
                'uncertainty': pred_uncertainty,
                'mape': mape,
                'mae': mae,
                'rmse': rmse,
                'test_idx': test_idx
            }
        
        except Exception as e:
            print(f"  Error: {str(e)[:100]}")
            return None
    
    def plot_predictions(self, results):
        """Plot actual vs predicted prices using unified path config"""
        save_dir = self.paths.results_visualizations_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        symbol = results['symbol']
        actual = results['actual']
        predicted = results['predicted']
        uncertainty = results['uncertainty']
        mape = results['mape']
        mae = results['mae']
        rmse = results['rmse']
        test_idx = results['test_idx']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'{symbol} - Actual vs Predicted Prices (V5 Adaptive)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Price comparison
        ax1 = axes[0]
        x_range = np.arange(len(actual))
        
        ax1.plot(x_range, actual, 'r-', linewidth=2.5, label='Actual Price', alpha=0.8)
        ax1.plot(x_range, predicted, 'b-', linewidth=2, label='Predicted Price', alpha=0.7)
        ax1.fill_between(x_range, actual, predicted, alpha=0.2, color='gray')
        
        # Highlight test set
        ax1.axvline(x=test_idx, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Test Split')
        
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Price (USD)', fontsize=11)
        ax1.set_title('Price Comparison', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f'MAPE: {mape*100:.2f}% | MAE: ${mae:.4f} | RMSE: ${rmse:.4f}'
        ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Error analysis
        ax2 = axes[1]
        errors = actual - predicted
        
        colors = ['green' if e >= 0 else 'red' for e in errors]
        ax2.bar(x_range, errors, color=colors, alpha=0.6, label='Prediction Error')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=np.mean(errors), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean Error: ${np.mean(errors):.4f}')
        ax2.axvline(x=test_idx, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Error (USD)', fontsize=11)
        ax2.set_title('Prediction Error Analysis', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'{symbol}_predictions_v5_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_multiple_symbols(self, symbols):
        """Plot predictions for multiple symbols"""
        results_list = []
        for symbol in symbols:
            results = self.generate_predictions(symbol)
            if results:
                self.plot_predictions(results)
                results_list.append({
                    'symbol': symbol,
                    'mape': results['mape'],
                    'mae': results['mae'],
                    'rmse': results['rmse']
                })
        
        # Create summary table
        if results_list:
            self.plot_summary_table(results_list)
        
        return results_list
    
    def plot_summary_table(self, results_list):
        """Create summary table of all predictions"""
        save_dir = self.paths.results_visualizations_dir
        
        df = pd.DataFrame(results_list)
        df['mape'] = df['mape'] * 100
        
        # Sort by MAPE
        df = df.sort_values('mape')
        
        # Create figure for table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = [['Symbol', 'MAPE (%)', 'MAE ($)', 'RMSE ($)']]
        for _, row in df.iterrows():
            table_data.append([
                row['symbol'],
                f"{row['mape']:.2f}%",
                f"${row['mae']:.4f}",
                f"${row['rmse']:.4f}"
            ])
        
        # Color code by performance
        colors = []
        header_color = ['#40466e'] * 4
        colors.append(header_color)
        
        for _, row in df.iterrows():
            if row['mape'] <= 3:
                row_color = ['#90EE90']
            elif row['mape'] <= 5:
                row_color = ['#FFD700']
            elif row['mape'] <= 8:
                row_color = ['#FFA500']
            else:
                row_color = ['#FF6B6B']
            
            row_color = row_color * 4
            colors.append(row_color)
        
        table = ax.table(cellText=table_data, cellColours=colors,
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(4):
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('V5 Adaptive Model - Performance Summary', 
                 fontsize=14, fontweight='bold', pad=20)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'summary_table_v5_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Summary table saved: {save_path}")
        plt.close()
        
        # Also save as CSV
        csv_path = os.path.join(save_dir, f'summary_results_v5_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results CSV saved: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='V5 Adaptive Predictions Visualizer')
    parser.add_argument('--all', action='store_true', help='Visualize all symbols')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to visualize')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("V5 ADAPTIVE PREDICTIONS VISUALIZER")
    print("="*70)
    print(f"Device: {args.device}")
    
    # Create visualizer (this initializes PathConfig)
    visualizer = V5Visualizer(device=args.device)
    
    # Print path configuration
    visualizer.paths.print_summary()
    
    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT',
                  'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT',
                  'ATOMUSDT', 'UNIUSDT', 'APTUSDT']
    else:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    print(f"Processing {len(symbols)} symbols: {', '.join(symbols[:5])}...")
    print("="*70)
    
    # Generate visualizations
    results_list = visualizer.plot_multiple_symbols(symbols)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    
    if results_list:
        print(f"\nSuccessfully visualized {len(results_list)} symbols")
        for result in results_list:
            print(f"  {result['symbol']}: MAPE={result['mape']*100:.2f}%")
        print(f"\nResults saved to: {visualizer.paths.results_visualizations_dir}")
    else:
        print("\nNo visualizations were created.")
        print(f"Please ensure V5 models exist in: {visualizer.paths.models_weights_dir}")
        print("Run training first: python train_v5.py 5 --device cuda")

if __name__ == '__main__':
    main()
