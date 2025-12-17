"""
V5 Enhanced Inference - Residual to Price Reconstruction
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from train_v5_enhanced import FeatureEngineerV5, MultiScaleLSTMV5

class V5InferenceEngine:
    """V5 Inference with residual reconstruction"""
    
    def __init__(self, symbol: str, device: str = 'cpu'):
        self.symbol = symbol
        self.device = torch.device(device)
        self.feature_calc = FeatureEngineerV5()
        
        # Find model path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(backend_dir)
        self.model_path = os.path.join(root_dir, 'models', 'weights', f'{symbol}_1h_v5_lstm.pth')
        
    def predict_batch(self, X: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for batch"""
        X_t = torch.FloatTensor(X).to(self.device)
        model.eval()
        
        with torch.no_grad():
            pred_delta, pred_vol = model(X_t)
            pred_delta = pred_delta.cpu().numpy()
            pred_vol = pred_vol.cpu().numpy()
        
        return pred_delta, pred_vol
    
    def infer(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full inference pipeline"""
        close_prices = df['close'].values
        
        # Features
        features_df = self.feature_calc.calculate_features(df)
        feature_values = features_df.values
        
        # Prepare sequences
        X = []
        for i in range(len(feature_values) - lookback):
            X.append(feature_values[i:i+lookback])
        X = np.array(X, dtype=np.float32)
        
        # Normalize
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Load model
        model = MultiScaleLSTMV5(input_size=X.shape[2]).to(self.device)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # Predict deltas
        pred_delta, pred_vol = self.predict_batch(X_scaled, model)
        pred_delta = pred_delta.ravel()
        pred_vol = pred_vol.ravel()
        
        # Reconstruct prices from residuals
        # Start from known price, add predicted deltas
        pred_prices = np.zeros(len(close_prices))
        pred_prices[:lookback] = close_prices[:lookback]
        
        for i in range(lookback, len(close_prices)):
            idx = i - lookback
            if idx < len(pred_delta):
                pred_prices[i] = pred_prices[i-1] + pred_delta[idx]
            else:
                pred_prices[i] = pred_prices[i-1]
        
        return close_prices, pred_prices, pred_vol
