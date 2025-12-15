#!/usr/bin/env python
r"""
Diagnostic script to analyze training data quality
Checks if target values have variation

Usage:
    python backend/diagnose_training_data.py --symbol BTCUSDT --timeframe 1h
"""

import logging
import sys
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.data.data_manager import DataManager
from backend.data.data_loader import CryptoDataLoader
from config.model_config import MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingDataDiagnostic:
    def __init__(self):
        self.data_manager = DataManager()
        self.data_loader = CryptoDataLoader()
    
    def analyze_data(self, symbol, timeframe):
        """Analyze training data quality"""
        logger.info(f"\nAnalyzing data for {symbol} ({timeframe})")
        logger.info(f"{'='*70}")
        
        # Load raw data
        data = self.data_manager.get_stored_data(symbol, timeframe)
        if data is None:
            logger.error(f"No data found for {symbol}")
            return
        
        logger.info(f"\n1. RAW DATA")
        logger.info(f"   Total candles: {len(data)}")
        logger.info(f"   Close price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        logger.info(f"   Close std dev: {data['close'].std():.2f}")
        logger.info(f"   Close variation: {(data['close'].std() / data['close'].mean()) * 100:.2f}%")
        
        # Calculate indicators
        data_with_indicators = self.data_loader.calculate_technical_indicators(data)
        
        logger.info(f"\n2. AFTER INDICATORS")
        logger.info(f"   Data points: {len(data_with_indicators)}")
        logger.info(f"   Rows dropped: {len(data) - len(data_with_indicators)}")
        logger.info(f"   Close price range: {data_with_indicators['close'].min():.2f} - {data_with_indicators['close'].max():.2f}")
        
        # Normalize
        all_feature_cols = [col for col in data_with_indicators.columns]
        scaler = MinMaxScaler()
        data_normalized = data_with_indicators.copy()
        data_scaled = scaler.fit_transform(data_with_indicators[all_feature_cols])
        data_normalized[all_feature_cols] = data_scaled
        
        logger.info(f"\n3. NORMALIZED DATA")
        logger.info(f"   Features: {len(all_feature_cols)}")
        logger.info(f"   Normalized close range: {data_normalized['close'].min():.6f} - {data_normalized['close'].max():.6f}")
        logger.info(f"   Normalized close std: {data_normalized['close'].std():.6f}")
        
        # Create sequences
        lookback = MODEL_CONFIG['lookback']
        close_min = data_with_indicators['close'].min()
        close_max = data_with_indicators['close'].max()
        
        logger.info(f"\n4. SEQUENCE CREATION")
        logger.info(f"   Lookback: {lookback}")
        logger.info(f"   Total data points: {len(data_normalized)}")
        logger.info(f"   Max sequences: {len(data_normalized) - lookback}")
        
        # Analyze target values
        targets_normalized = []
        targets_actual = []
        
        for i in range(len(data_normalized) - lookback):
            # Target is the close price at position i+lookback
            target_idx = i + lookback
            target_actual = data_with_indicators['close'].iloc[target_idx]
            target_norm = (target_actual - close_min) / (close_max - close_min)
            
            targets_normalized.append(target_norm)
            targets_actual.append(target_actual)
        
        targets_normalized = np.array(targets_normalized)
        targets_actual = np.array(targets_actual)
        
        logger.info(f"\n5. TARGET VALUES ANALYSIS")
        logger.info(f"   Total targets: {len(targets_normalized)}")
        logger.info(f"   Target (normalized) range: {targets_normalized.min():.6f} - {targets_normalized.max():.6f}")
        logger.info(f"   Target (normalized) mean: {targets_normalized.mean():.6f}")
        logger.info(f"   Target (normalized) std: {targets_normalized.std():.6f}")
        logger.info(f"   Target (actual) range: ${targets_actual.min():.2f} - ${targets_actual.max():.2f}")
        logger.info(f"   Target (actual) std: ${targets_actual.std():.2f}")
        logger.info(f"   Target variation: {(targets_actual.std() / targets_actual.mean()) * 100:.2f}%")
        
        # Check for constant targets
        unique_targets = len(np.unique(targets_normalized))
        logger.info(f"\n6. TARGET QUALITY CHECK")
        logger.info(f"   Unique normalized values: {unique_targets}")
        logger.info(f"   Is target constant? {'YES - PROBLEM!' if unique_targets < 10 else 'NO - Good'}")
        
        if targets_normalized.std() < 0.01:
            logger.error(f"   WARNING: Target std is too low ({targets_normalized.std():.6f})")
            logger.error(f"   The model cannot learn from constant targets!")
        
        # Show sample targets
        logger.info(f"\n7. SAMPLE TARGETS (first 20)")
        for i in range(min(20, len(targets_normalized))):
            logger.info(f"   [{i:3d}] Norm: {targets_normalized[i]:.6f} | Actual: ${targets_actual[i]:.2f}")
        
        # Feature analysis
        logger.info(f"\n8. FEATURE VARIATION")
        feature_stats = {}
        for col in all_feature_cols:
            feature_stats[col] = {
                'mean': data_normalized[col].mean(),
                'std': data_normalized[col].std(),
                'min': data_normalized[col].min(),
                'max': data_normalized[col].max()
            }
        
        # Show features with low variation
        low_variation_features = [f for f, s in feature_stats.items() if s['std'] < 0.01]
        if low_variation_features:
            logger.warning(f"   Features with low variation (std < 0.01):")
            for f in low_variation_features:
                logger.warning(f"     - {f}: std={feature_stats[f]['std']:.6f}")
        
        # Correlation between close and target
        logger.info(f"\n9. INPUT-TARGET CORRELATION")
        corr_with_target = []
        for i in range(min(5, len(all_feature_cols))):
            col = all_feature_cols[i]
            if data_normalized[col].std() > 0:  # Avoid division by zero
                corr = np.corrcoef(data_normalized[col].iloc[-len(targets_normalized):], targets_normalized)[0, 1]
                corr_with_target.append((col, corr))
                logger.info(f"   {col:20s}: {corr:7.4f}")
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"DIAGNOSIS SUMMARY")
        logger.info(f"{'='*70}")
        
        if targets_normalized.std() < 0.01:
            logger.error(f"CRITICAL: Target values have no variation!")
            logger.error(f"The model learned a constant output because targets don't vary.")
            logger.error(f"Solution: Check if the target calculation is correct.")
        else:
            logger.info(f"Target values have good variation: std={targets_normalized.std():.6f}")
            logger.info(f"The model should be able to learn.")


def main():
    parser = ArgumentParser(description='Diagnose training data quality')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    
    args = parser.parse_args()
    
    diagnostic = TrainingDataDiagnostic()
    diagnostic.analyze_data(args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
