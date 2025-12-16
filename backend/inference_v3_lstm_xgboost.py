import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime

backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from data.data_loader import CryptoDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMFeatureExtractor(nn.Module):
    """LSTM for extracting temporal features"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x):
        """Extract LSTM features (hidden state)"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return last_hidden, lstm_out

class LSTMXGBoostInference:
    """Inference with LSTM+XGBoost hybrid model"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = DataManager()
        self.data_loader = CryptoDataLoader()
        logger.info(f'Inference Device: {self.device}')
    
    def select_bb_focused_features(self, df_indicators, top_k=18):
        """Select BB-focused features"""
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
    
    def predict_price(self, symbol, timeframe='1h'):
        """
        Predict next price using LSTM+XGBoost
        """
        logger.info(f'\nPredicting {symbol} ({timeframe})...')
        
        # Load data
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
        df_normalized['close_norm'] = (df_ind['close'] - close_min) / close_range
        
        # Load LSTM model
        lstm_path = f'backend/models/weights/{symbol}_1h_v3_lstm.pth'
        xgb_path = f'backend/models/weights/{symbol}_1h_v3_xgb.pkl'
        
        if not os.path.exists(lstm_path) or not os.path.exists(xgb_path):
            logger.error(f'Models not found for {symbol}')
            return None
        
        # Load LSTM
        checkpoint = torch.load(lstm_path, map_location=self.device)
        lstm_model = LSTMFeatureExtractor(
            input_size=18,
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers'],
            dropout=0.2
        ).to(self.device)
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.eval()
        
        # Load XGBoost
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
        
        # Prepare input
        lookback_period = 60
        X_latest = df_normalized[selected_features].tail(lookback_period).values
        X_tensor = torch.FloatTensor(X_latest).unsqueeze(0).to(self.device)
        
        # Stage 1: Extract LSTM features
        with torch.no_grad():
            lstm_features = lstm_model(X_tensor)[0].cpu().numpy()
        
        # Stage 2: XGBoost prediction
        y_pred_norm = xgb_model.predict(lstm_features)[0]
        
        # Denormalize
        pred_price = y_pred_norm * close_range + close_min
        current_price = df_ind['close'].iloc[-1]
        
        price_change = pred_price - current_price
        pct_change = (price_change / current_price) * 100 if current_price != 0 else 0
        
        # Get technical indicators
        bb_upper = df_ind['bb_upper'].iloc[-1]
        bb_lower = df_ind['bb_lower'].iloc[-1]
        bb_middle = df_ind['bb_middle'].iloc[-1]
        bb_percent_b = df_ind['bb_percent_b'].iloc[-1]
        rsi_14 = df_ind['rsi_14'].iloc[-1]
        macd = df_ind['macd'].iloc[-1]
        volume = df_ind['volume'].iloc[-1]
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'predicted_price': float(pred_price),
            'change_usd': float(price_change),
            'change_pct': float(pct_change),
            'bb_upper': float(bb_upper),
            'bb_middle': float(bb_middle),
            'bb_lower': float(bb_lower),
            'bb_position': float(bb_percent_b),
            'rsi_14': float(rsi_14),
            'macd': float(macd),
            'volume': float(volume),
        }
        
        logger.info(f'Current Price: ${current_price:.2f}')
        logger.info(f'Predicted Price: ${pred_price:.2f}')
        logger.info(f'Change: ${price_change:.2f} ({pct_change:+.2f}%)')
        logger.info(f'BB Range: ${bb_lower:.2f} - ${bb_upper:.2f} (Mid: ${bb_middle:.2f})')
        
        return result
    
    def batch_predict(self, symbols=None, timeframe='1h'):
        """Batch prediction"""
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f'\nBatch Prediction (V3 LSTM+XGBoost): {len(symbols)} symbols')
        
        results = []
        for idx, symbol in enumerate(symbols, 1):
            try:
                result = self.predict_price(symbol, timeframe)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f'Error predicting {symbol}: {str(e)}')
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'backend/predictions/batch_v3_lstm_xgboost_{timestamp}.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_results.to_csv(output_file, index=False)
            logger.info(f'\nSaved {len(results)} predictions to {output_file}')
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM+XGBoost inference')
    parser.add_argument('--symbol', help='Single symbol')
    parser.add_argument('--batch', action='store_true', help='Batch predict')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--device', default=None, help='Device')
    
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    
    try:
        inferencer = LSTMXGBoostInference(device=device)
        
        if args.batch:
            inferencer.batch_predict(timeframe=args.timeframe)
        elif args.symbol:
            inferencer.predict_price(args.symbol, args.timeframe)
        else:
            logger.info('Use --symbol <SYM> or --batch')
            return 1
        
        return 0
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
