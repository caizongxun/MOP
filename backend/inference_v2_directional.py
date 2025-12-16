import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
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

class DirectionalLSTM(nn.Module):
    """LSTM for directional classification"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(DirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes: Down, Neutral, Up
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        return logits

class DirectionalInference:
    """Directional prediction inference"""
    
    DIRECTION_MAP = {
        0: 'DOWN',
        1: 'NEUTRAL',
        2: 'UP'
    }
    
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
    
    def predict_direction(self, symbol, timeframe='1h'):
        """
        Predict direction for next candle
        Returns: {'direction': 'UP'/'DOWN'/'NEUTRAL', 'probability': 0.85, ...}
        """
        logger.info(f'\nPredicting direction for {symbol} ({timeframe})...')
        
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
        
        # Load model
        model_path = f'backend/models/weights/{symbol}_1h_v2_directional_best.pth'
        if not os.path.exists(model_path):
            logger.error(f'Model not found: {model_path}')
            return None
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        model = DirectionalLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=0.2
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Prepare input
        lookback_period = 60
        X_latest = df_normalized[selected_features].tail(lookback_period).values
        X_tensor = torch.FloatTensor(X_latest).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = model(X_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = torch.argmax(logits, dim=1).item()
        
        # Get current price and technical indicators
        current_price = df_ind['close'].iloc[-1]
        bb_upper = df_ind['bb_upper'].iloc[-1]
        bb_middle = df_ind['bb_middle'].iloc[-1]
        bb_lower = df_ind['bb_lower'].iloc[-1]
        bb_percent_b = df_ind['bb_percent_b'].iloc[-1]
        rsi_14 = df_ind['rsi_14'].iloc[-1]
        macd = df_ind['macd'].iloc[-1]
        volume = df_ind['volume'].iloc[-1]
        
        direction = self.DIRECTION_MAP[prediction]
        max_confidence = probabilities[prediction]
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'direction': direction,
            'probability_down': float(probabilities[0]),
            'probability_neutral': float(probabilities[1]),
            'probability_up': float(probabilities[2]),
            'confidence': float(max_confidence),
            'action_signal': 'BUY' if probabilities[2] > 0.6 else ('SELL' if probabilities[0] > 0.6 else 'HOLD'),
            'bb_upper': float(bb_upper),
            'bb_middle': float(bb_middle),
            'bb_lower': float(bb_lower),
            'bb_position': float(bb_percent_b),
            'rsi_14': float(rsi_14),
            'macd': float(macd),
            'volume': float(volume),
        }
        
        logger.info(f'Direction: {direction} (confidence: {max_confidence:.2%})')
        logger.info(f'Probabilities - Down: {probabilities[0]:.2%}, Neutral: {probabilities[1]:.2%}, Up: {probabilities[2]:.2%}')
        logger.info(f'Action Signal: {result["action_signal"]}')
        logger.info(f'Price: ${current_price:.2f}, BB Position: {bb_percent_b:.2%}')
        logger.info(f'RSI: {rsi_14:.2f}, MACD: {macd:.6f}')
        
        return result
    
    def batch_predict(self, symbols=None, timeframe='1h'):
        """Batch prediction for all symbols"""
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
            ]
        
        logger.info(f'\nBatch Directional Prediction: {len(symbols)} symbols')
        
        results = []
        buy_signals = []
        sell_signals = []
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                result = self.predict_direction(symbol, timeframe)
                if result:
                    results.append(result)
                    if result['action_signal'] == 'BUY':
                        buy_signals.append(symbol)
                    elif result['action_signal'] == 'SELL':
                        sell_signals.append(symbol)
            except Exception as e:
                logger.error(f'Error predicting {symbol}: {str(e)}')
        
        # Summary
        logger.info(f'\n{"="*80}')
        logger.info(f'Batch Prediction Summary:')
        logger.info(f'BUY signals: {buy_signals}')
        logger.info(f'SELL signals: {sell_signals}')
        logger.info(f'HOLD count: {len(symbols) - len(buy_signals) - len(sell_signals)}')
        logger.info(f'{"="*80}')
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'backend/predictions/batch_directional_{timestamp}.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_results.to_csv(output_file, index=False)
            logger.info(f'Saved {len(results)} predictions to {output_file}')
        
        return results, buy_signals, sell_signals

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Directional prediction inference')
    parser.add_argument('--symbol', help='Single symbol')
    parser.add_argument('--batch', action='store_true', help='Batch predict')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--device', default=None, help='Device')
    
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    
    try:
        inferencer = DirectionalInference(device=device)
        
        if args.batch:
            inferencer.batch_predict(timeframe=args.timeframe)
        elif args.symbol:
            inferencer.predict_direction(args.symbol, args.timeframe)
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
