#!/usr/bin/env python3
"""
Quick Visualizer for V5 - Run from project root

Usage:
    python quick_visualize_v5.py --all --device cuda
    python quick_visualize_v5.py --symbols BTCUSDT ETHUSDT --device cuda
    python quick_visualize_v5.py --all --device cpu

This script uses unified PathConfig for consistent directory handling.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from quick_visualize_v5 import V5Visualizer
from path_config import PathConfig
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V5 Adaptive Predictions Visualizer')
    parser.add_argument('--all', action='store_true', help='Visualize all symbols')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to visualize')
    parser.add_argument('--show-paths', action='store_true', help='Show path configuration and exit')
    
    args = parser.parse_args()
    
    # Initialize path config
    paths = PathConfig()
    
    if args.show_paths:
        print("\n" + "="*70)
        print("PATH CONFIGURATION")
        print("="*70)
        paths.print_summary()
        sys.exit(0)
    
    print("\n" + "="*70)
    print("V5 ADAPTIVE PREDICTIONS VISUALIZER")
    print("="*70)
    print(f"Device: {args.device}")
    
    # Create visualizer
    visualizer = V5Visualizer(device=args.device)
    
    # Print path configuration
    visualizer.paths.print_summary()
    
    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT',
                  'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT',
                  'ATOMUSDT', 'UNIUSDT', 'APTUSDT']
    else:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
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
        print(f"\nResults saved to: {visualizer.paths.results_visualizations_dir}")
    else:
        print("\nNo visualizations were created.")
        print(f"Please ensure V5 models exist in:")
        print(f"  {visualizer.paths.models_weights_dir}")
        print("\nTo train V5 models, run:")
        print("  python train_v5.py 5 --device cuda")
    
    print("="*70 + "\n")
