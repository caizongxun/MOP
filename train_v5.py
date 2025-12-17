#!/usr/bin/env python3
"""
Quick start for V5 Enhanced Training

Usage:
    python train_v5.py 5 --device cuda
    python train_v5.py 15 --device cuda
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from train_v5_enhanced import V5EnhancedTrainer
import argparse
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V5 Enhanced Training')
    parser.add_argument('count', type=int, help='Number of symbols to train')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT',
               'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT',
               'ATOMUSDT', 'UNIUSDT', 'APTUSDT'][:args.count]
    
    print("\n" + "="*70)
    print("V5 ENHANCED MODEL TRAINING")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Symbols: {len(symbols)} - {', '.join(symbols[:5])}...")
    print("="*70 + "\n")
    
    trainer = V5EnhancedTrainer(device=args.device)
    mapes = []
    
    for idx, symbol in enumerate(symbols, 1):
        try:
            mape = trainer.train(symbol, num_symbols=len(symbols), symbol_idx=idx)
            mapes.append(mape)
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
            mapes.append(None)
    
    print("\n" + "="*70)
    print("V5 TRAINING COMPLETE")
    print("="*70)
    
    valid_mapes = [m for m in mapes if m is not None]
    if valid_mapes:
        print(f"Successful: {len(valid_mapes)}/{len(symbols)}")
        print(f"Average MAPE: {np.mean(valid_mapes)*100:.2f}%")
        print(f"Best: {np.min(valid_mapes)*100:.2f}%")
        print(f"Worst: {np.max(valid_mapes)*100:.2f}%")
    print("\nModels saved to: ./models/weights/")
    print("Configs saved to: ./models/config/")
    print("="*70 + "\n")
