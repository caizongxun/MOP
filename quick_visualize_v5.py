#!/usr/bin/env python3
"""
Quick Visualizer for V5 - Run from project root

Usage:
    python quick_visualize_v5.py --all --device cuda
    python quick_visualize_v5.py --symbols BTCUSDT ETHUSDT --device cuda
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from quick_visualize_v5 import V5Visualizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V5 Adaptive Predictions Visualizer')
    parser.add_argument('--all', action='store_true', help='Visualize all symbols')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to visualize')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("V5 ADAPTIVE PREDICTIONS VISUALIZER")
    print("="*70)
    print(f"Device: {args.device}")
    
    # Create visualizer
    visualizer = V5Visualizer(device=args.device)
    
    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT',
                  'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT',
                  'ATOMUSDT', 'UNIUSDT', 'APTUSDT']
    else:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    print(f"Output: {visualizer.results_dir}")
    print(f"Processing {len(symbols)} symbols...")
    print("="*70 + "\n")
    
    # Generate visualizations
    results_list = visualizer.plot_multiple_symbols(symbols)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    
    if results_list:
        print(f"\nSuccessfully visualized {len(results_list)} symbols")
        valid_mapes = [r['mape']*100 for r in results_list]
        print(f"Average MAPE: {sum(valid_mapes)/len(valid_mapes):.2f}%")
        print(f"Best: {min(valid_mapes):.2f}%")
        print(f"Worst: {max(valid_mapes):.2f}%")
        print(f"\nResults saved to: {visualizer.results_dir}")
    else:
        print("\nNo visualizations were created.")
        print(f"Please ensure V5 models exist and training is complete.")
    
    print("="*70 + "\n")
