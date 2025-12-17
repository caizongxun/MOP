#!/usr/bin/env python3
"""
V5 Training - Quick Start from Project Root

Usage:
    python train_v5.py 5 --device cuda       # Train 5 symbols with CUDA
    python train_v5.py 2 --device cpu        # Train 2 symbols with CPU
    python train_v5.py --all --device cuda   # Train all 15 symbols

This script uses unified PathConfig for all paths.
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
    parser = argparse.ArgumentParser(description='V5 Enhanced Model Training')
    parser.add_argument('count', type=int, nargs='?', default=2, help='Number of symbols to train')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols')
    parser.add_argument('--all', action='store_true', help='Train all symbols')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("V5 ENHANCED MODEL TRAINING")
    print("="*70)
    print(f"Device: {args.device}")
    
    # Initialize trainer
    trainer = V5EnhancedTrainer(device=args.device)
    trainer.paths.print_summary()
    
    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 
                  'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT',
                  'ATOMUSDT', 'UNIUSDT', 'APTUSDT']
    else:
        all_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 
                       'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT',
                       'ATOMUSDT', 'UNIUSDT', 'APTUSDT']
        symbols = all_symbols[:args.count]
    
    print(f"Training {len(symbols)} symbols: {', '.join(symbols)}")
    print("="*70 + "\n")
    
    # Train
    mapes = []
    for idx, symbol in enumerate(symbols, 1):
        try:
            mape = trainer.train(symbol, num_symbols=len(symbols), symbol_idx=idx)
            if mape is not None:
                mapes.append(mape)
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("V5 TRAINING COMPLETE")
    print("="*70)
    if mapes:
        print(f"Successfully trained: {len(mapes)} symbols")
        print(f"Average MAPE: {np.mean(mapes)*100:.2f}%")
        print(f"Best: {np.min(mapes)*100:.2f}% | Worst: {np.max(mapes)*100:.2f}%")
    print(f"\nModels saved to: {trainer.paths.models_weights_dir}")
    print(f"Configs saved to: {trainer.paths.models_config_dir}")
    print("\nNext, run visualizations:")
    print("  python quick_visualize_v5.py --all --device cuda")
    print("="*70 + "\n")
