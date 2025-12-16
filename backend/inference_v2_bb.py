import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime, timedelta

backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm_model import CryptoLSTM
from data.data_manager import DataManager
from data.data_loader import CryptoDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BBFocusedInference:
    """Inference with v2 BB-focused models"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = DataManager()
        self.data_loader = CryptoDataLoader()
        logger.info(f'Inference Device: {self.device}')
    
    def select_bb_focused_features(self, df_indicators, top_k=18):
        """Select BB-focused features (same as training)"""
        core_features = [
            'bb_upper', 'bb_middle', 'bb_lower', 
            'bb_percent_b', 'bb_width'
        ]
        
        support_features = [
            'ema_10', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi_14', 'rsi_21',
            'volume', 'volume_change',
            'atr_14', 'di_plus', 'di_minus',
            'mfi', 'obv'
        ]
        
        available_core = [f for f in core_features if f in df_indicators.columns]
        available_support = [f for f in support_features if f in df_indicators.columns]
        
        if len(available_support) > (top_k - len(available_core)):
            needed = top_k - len(available_core)
            X_support = df_indicators[available_support].fillna(0)
            y = df_indicators['close'].values
            
            selector = SelectKBest(f_regression, k=needed)
            selector.fit(X_support, y)
            
            selected_support = [
                available_support[i] for i in selector.get_support(indices=True)
            ]
        else:
            selected_support = available_support
        
        final_features = available_core + selected_support
        return final_features[:top_k]
    
    def predict_next_price(self, symbol, timeframe='1h', lookback_hours=1):
        """
        Predict next price for symbol
        """
        logger.info(f'\nPredicting {symbol} ({timeframe})...')
        
        # Load latest data
        df = self.dm.get_stored_data(symbol, timeframe)
        if df is None or len(df) < 61:
            logger.error(f'Insufficient data for {symbol}')
            return None
        
        df_ind = self.data_loader.calculate_technical_indicators(df)
        if df_ind is None:
            logger.error('Failed to calculate indicators')
            return None
        
        # Select features
        selected_features = self.select_bb_focused_features(df_ind, top_k=18)
        
        # Normalize
        scaler = MinMaxScaler()
        df_normalized = df_ind.copy()
        df_normalized[selected_features] = scaler.fit_transform(
            df_ind[selected_features]
        )
        
        close_min = df_ind['close'].min()
        close_max = df_ind['close'].max()
        close_range = close_max - close_min
        
        # Load model
        model_path = f'backend/models/weights/{symbol}_1h_v2_best.pth'
        if not os.path.exists(model_path):
            logger.error(f'Model not found: {model_path}')
            return None
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        model = CryptoLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout=0.2
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Use last 60 candles for prediction
        lookback_period = 60
        X_latest = df_normalized[selected_features].tail(lookback_period).values
        X_tensor = torch.FloatTensor(X_latest).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_normalized = model(X_tensor).cpu().numpy()[0][0]
        
        # Denormalize
        pred_price = pred_normalized * close_range + close_min
        
        # Get current price
        current_price = df_ind['close'].iloc[-1]
        
        # Calculate statistics
        bb_upper = df_ind['bb_upper'].iloc[-1]
        bb_lower = df_ind['bb_lower'].iloc[-1]
        bb_middle = df_ind['bb_middle'].iloc[-1]
        
        price_change = pred_price - current_price
        pct_change = (price_change / current_price) * 100 if current_price != 0 else 0
        
        logger.info(f'Current Price: ${current_price:.2f}')
        logger.info(f'Predicted Price: ${pred_price:.2f}')
        logger.info(f'Change: ${price_change:.2f} ({pct_change:+.2f}%)')
        logger.info(f'BB Range: ${bb_lower:.2f} - ${bb_upper:.2f} (Mid: ${bb_middle:.2f})')
        
        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(pred_price),
            'change_usd': float(price_change),
            'change_pct': float(pct_change),
            'bb_upper': float(bb_upper),
            'bb_middle': float(bb_middle),
            'bb_lower': float(bb_lower),
            'timestamp': df_ind.index[-1] if hasattr(df_ind.index[-1], 'isoformat') else str(df_ind.index[-1])
        }
    
    def batch_predict(self, symbols=None, timeframe='1h'):
        """Batch prediction for all symbols"""
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f'\nBatch Prediction (V2 BB-Focused): {len(symbols)} symbols')
        
        results = []
        for idx, symbol in enumerate(symbols, 1):
            try:
                result = self.predict_next_price(symbol, timeframe)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f'Error predicting {symbol}: {str(e)}')
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'backend/predictions/batch_predictions_v2_{timestamp}.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_results.to_csv(output_file, index=False)
            logger.info(f'\nSaved {len(results)} predictions to {output_file}')
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference with v2 BB-focused models')
    parser.add_argument('--symbol', help='Single symbol to predict')
    parser.add_argument('--batch', action='store_true', help='Batch predict all symbols')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    
    try:
        inferencer = BBFocusedInference(device=device)
        
        if args.batch:
            inferencer.batch_predict(timeframe=args.timeframe)
        elif args.symbol:
            inferencer.predict_next_price(args.symbol, args.timeframe)
        else:
            logger.info('Use --symbol <SYM> for single prediction or --batch for all')
            return 1
        
        return 0
    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
