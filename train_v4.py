#!/usr/bin/env python3
"""
V4 Adaptive Training Launcher
Run from project root: python train_v4.py

Usage:
  python train_v4.py                 # Default 5% MAPE target
  python train_v4.py 2                # 2% MAPE target
  python train_v4.py 5 --device cuda  # 5% with GPU
"""

import sys
import os
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from train_v4_adaptive import V4AdaptiveTrainer

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    target_mape = 0.05  # Default 5%
    device = 'cpu'
    
    if len(sys.argv) > 1:
        try:
            target_mape = float(sys.argv[1]) / 100
        except ValueError:
            print(f"Invalid MAPE value: {sys.argv[1]}")
            sys.exit(1)
    
    if '--device' in sys.argv:
        idx = sys.argv.index('--device')
        if idx + 1 < len(sys.argv):
            device = sys.argv[idx + 1]
    
    print(f"\nV4 Adaptive Training Launcher")
    print(f"Target MAPE: {target_mape*100:.1f}%")
    print(f"Device: {device}\n")
    
    trainer = V4AdaptiveTrainer(device=device)
    trainer.train_batch(target_mape=target_mape)
