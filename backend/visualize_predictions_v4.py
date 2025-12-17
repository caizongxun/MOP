"""
Visualization Script for V4 Adaptive Model Predictions
Plots actual vs predicted prices with error analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Import the trainer
from train_v4_adaptive import V4AdaptiveTrainer, LSTMWithWarmup

class V4Visualizer:
    """Visualizer for V4 Adaptive predictions"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.trainer = V4AdaptiveTrainer(device=device)
    
    def generate_predictions(self, symbol: str, timeframe: str = '1h'):
        """Generate predictions for a symbol"""
        print(f"\nGenerating predictions for {symbol}...")
        
        try:
            # Load data
            raw_data = self.trainer.load_data(symbol, timeframe)
            close_prices = raw_data['close'].values
            
            # Calculate features
            features_df = self.trainer.feature_calc.calculate_features(raw_data)
            X, y, scaler_X, scaler_y = self.trainer._prepare_data(features_df, raw_data)
            
            # Split data (same as training)
            train_idx = int(0.7 * len(X))
            val_idx = int(0.85 * len(X))
            X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
            y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
            
            # Load LSTM model
            lstm_path = f'backend/models/weights/{symbol}_1h_v4_lstm.pth'
            if not os.path.exists(lstm_path):
                print(f"LSTM model not found: {lstm_path}")
                return None
            
            # Create and load LSTM
            model = LSTMWithWarmup(
                input_size=X_train.shape[2],
                hidden_size=192,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            
            model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            model.eval()
            
            # Extract LSTM features
            lstm_features_test = self._extract_lstm_features(model, X_test)
            
            # Load XGBoost model
            import xgboost as xgb
            xgb_path = f'backend/models/weights/{symbol}_1h_v4_xgb.json'
            if not os.path.exists(xgb_path):
                print(f"XGBoost model not found: {xgb_path}")
                return None
            
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(xgb_path)
            
            # Predictions
            y_pred = xgb_model.predict(lstm_features_test)
            
            # Inverse transform
            y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig)
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            
            return {
                'symbol': symbol,
                'actual': y_test_orig,
                'predicted': y_pred_orig,
                'mape': mape,
                'mae': mae,
                'rmse': rmse,
                'scaler_y': scaler_y
            }
        
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            return None
    
    def _extract_lstm_features(self, model, X):
        """Extract LSTM features"""
        X_t = torch.FloatTensor(X).to(self.device)
        model.eval()
        with torch.no_grad():
            lstm_out, _ = model.lstm(X_t)
            features = lstm_out[:, -1, :].cpu().numpy()
        return np.ascontiguousarray(features.astype(np.float32))
    
    def plot_predictions(self, results, save_dir='backend/results/visualizations'):
        """Plot actual vs predicted prices"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        symbol = results['symbol']
        actual = results['actual']
        predicted = results['predicted']
        mape = results['mape']
        mae = results['mae']
        rmse = results['rmse']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'{symbol} - Actual vs Predicted Prices (V4 Adaptive)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Price comparison
        ax1 = axes[0]
        x_range = np.arange(len(actual))
        
        ax1.plot(x_range, actual, 'r-', linewidth=2.5, label='Actual Price', alpha=0.8)
        ax1.plot(x_range, predicted, 'b-', linewidth=2, label='Predicted Price', alpha=0.7)
        ax1.fill_between(x_range, actual, predicted, alpha=0.2, color='gray')
        
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
        error_pct = (errors / actual) * 100
        
        colors = ['green' if e >= 0 else 'red' for e in errors]
        ax2.bar(x_range, errors, color=colors, alpha=0.6, label='Prediction Error')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=np.mean(errors), color='blue', linestyle='--', linewidth=2, label=f'Mean Error: ${np.mean(errors):.4f}')
        
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Error (USD)', fontsize=11)
        ax2.set_title('Prediction Error Analysis', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'{symbol}_predictions_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
        
        return save_path
    
    def plot_multiple_symbols(self, symbols, save_dir='backend/results/visualizations'):
        """Plot predictions for multiple symbols"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        results_list = []
        for symbol in symbols:
            results = self.generate_predictions(symbol)
            if results:
                self.plot_predictions(results, save_dir)
                results_list.append({
                    'symbol': symbol,
                    'mape': results['mape'],
                    'mae': results['mae'],
                    'rmse': results['rmse']
                })
        
        # Create summary table
        if results_list:
            self.plot_summary_table(results_list, save_dir)
        
        return results_list
    
    def plot_summary_table(self, results_list, save_dir):
        """Create summary table of all predictions"""
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
                row_color = ['#90EE90']  # Light green
            elif row['mape'] <= 5:
                row_color = ['#FFD700']  # Gold
            elif row['mape'] <= 8:
                row_color = ['#FFA500']  # Orange
            else:
                row_color = ['#FF6B6B']  # Light red
            
            row_color = row_color * 4
            colors.append(row_color)
        
        table = ax.table(cellText=table_data, cellColours=colors,
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('V4 Adaptive Model - Performance Summary', 
                 fontsize=14, fontweight='bold', pad=20)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'summary_table_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Summary table saved: {save_path}")
        
        # Also save as CSV
        csv_path = os.path.join(save_dir, f'summary_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results CSV saved: {csv_path}")

if __name__ == '__main__':
    # Initialize visualizer
    visualizer = V4Visualizer(device='cpu')
    
    # Single symbol visualization
    # results = visualizer.generate_predictions('BTCUSDT')
    # if results:
    #     visualizer.plot_predictions(results)
    
    # Multiple symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
    visualizer.plot_multiple_symbols(symbols)
    
    print("\nVisualization complete!")
