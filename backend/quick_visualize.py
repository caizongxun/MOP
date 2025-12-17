#!/usr/bin/env python3
"""
Quick Visualization Launcher for V4 Adaptive Predictions

Usage:
  python quick_visualize.py                           # 視覺化前5個幣種
  python quick_visualize.py BTCUSDT ETHUSDT          # 視覺化指定幣種
  python quick_visualize.py --all                     # 視覺化所有幣種
"""

import sys
import argparse
from pathlib import Path

from visualize_predictions_v4 import V4Visualizer

DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']

ALL_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
    'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
    'LTCUSDT', 'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
]

def main():
    parser = argparse.ArgumentParser(
        description='Quick visualization launcher for V4 predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python quick_visualize.py                    # Default 5 symbols
  python quick_visualize.py BTCUSDT ETHUSDT   # Custom symbols
  python quick_visualize.py --all              # All 15 symbols
  python quick_visualize.py --device cuda      # Use GPU
        """
    )
    
    parser.add_argument('symbols', nargs='*', help='Symbols to visualize')
    parser.add_argument('--all', action='store_true', help='Visualize all symbols')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--output', default='backend/results/visualizations',
                       help='Output directory')
    parser.add_argument('--single', action='store_true', help='Process one symbol and exit')
    
    args = parser.parse_args()
    
    # Determine symbols to process
    if args.all:
        symbols = ALL_SYMBOLS
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_SYMBOLS
    
    print(f"\n{'='*60}")
    print("V4 Adaptive Predictions Visualizer")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"{'='*60}\n")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    try:
        visualizer = V4Visualizer(device=args.device)
        
        if args.single and symbols:
            # Process only first symbol
            symbol = symbols[0]
            print(f"Processing: {symbol}")
            results = visualizer.generate_predictions(symbol)
            if results:
                visualizer.plot_predictions(results, save_dir=args.output)
                print(f"\nSuccessfully created visualization for {symbol}")
                print(f"Output saved to: {args.output}/{symbol}_predictions_*.png")
            else:
                print(f"Failed to generate predictions for {symbol}")
        else:
            # Process multiple symbols
            results_list = visualizer.plot_multiple_symbols(symbols, save_dir=args.output)
            
            print(f"\n{'='*60}")
            print("Visualization Complete!")
            print(f"{'='*60}")
            
            if results_list:
                print(f"\nProcessed {len(results_list)} symbols successfully")
                print(f"Output files saved to: {args.output}/")
                print(f"\nPerformance Summary:")
                print(f"-" * 50)
                
                # Show summary
                for r in sorted(results_list, key=lambda x: x['mape']):
                    status = "✓" if r['mape'] <= 0.05 else "○" if r['mape'] <= 0.08 else "✗"
                    print(f"{status} {r['symbol']:10} | MAPE: {r['mape']*100:6.2f}% | MAE: ${r['mae']:8.4f}")
                
                print(f"-" * 50)
                avg_mape = sum(r['mape'] for r in results_list) / len(results_list)
                print(f"Average MAPE: {avg_mape*100:.2f}%")
            else:
                print("No visualizations were created.")
                print("Please ensure models have been trained first.")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure models are trained: python -m backend.train_v4_adaptive")
        print("2. Check that data files exist: backend/data/raw/")
        print("3. Verify model weights: backend/models/weights/")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
