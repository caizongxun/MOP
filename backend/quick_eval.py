#!/usr/bin/env python
"""
Quick evaluation script - One command to evaluate and visualize model

Usage:
    python backend/quick_eval.py
    python backend/quick_eval.py --symbol ETHUSDT --timeframe 1h
    python backend/quick_eval.py --all-symbols
"""

import logging
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

from backend.evaluate_model import ModelEvaluator
from backend.visualize_predictions import PredictionVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser(description='Quick model evaluation and visualization')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to evaluate (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--all-symbols', action='store_true', help='Evaluate all symbols')
    parser.add_argument('--no-plot', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device='cpu')
    visualizer = PredictionVisualizer()
    
    if args.all_symbols:
        logger.info("\nEvaluating all symbols...")
        all_results = evaluator.evaluate_all_symbols(timeframe=args.timeframe)
        evaluator.save_results(all_results, 'all_evaluations.json')
        
        # Visualize top 5 performers
        if not args.no_plot and all_results:
            mape_scores = {sym: res['metrics']['MAPE'] for sym, res in all_results.items()}
            top_symbols = sorted(mape_scores.items(), key=lambda x: x[1])[:5]
            
            logger.info("\nVisualizing top 5 performers...")
            for symbol, mape in top_symbols:
                logger.info(f"  {symbol}: MAPE = {mape:.4f}%")
                visualizer.create_comparison_report(
                    all_results[symbol],
                    symbol,
                    args.timeframe,
                    save_path=f"{symbol}_{args.timeframe}_report.txt"
                )
    else:
        symbol = args.symbol
        timeframe = args.timeframe
        
        logger.info(f"\nEvaluating {symbol} ({timeframe})...")
        result = evaluator.evaluate_symbol(symbol, timeframe)
        
        if result:
            evaluator.save_results(result, f'{symbol}_{timeframe}_evaluation.json')
            
            # Create text report
            logger.info(f"\nCreating report...")
            visualizer.create_comparison_report(
                result,
                symbol,
                timeframe,
                save_path=f"{symbol}_{timeframe}_report.txt"
            )
            
            # Create visualization
            if not args.no_plot:
                try:
                    import matplotlib.pyplot as plt
                    logger.info(f"\nCreating visualization...")
                    visualizer.plot_predictions(
                        result,
                        symbol,
                        timeframe,
                        save_path=f"{symbol}_{timeframe}_predictions.png"
                    )
                except ImportError:
                    logger.warning("matplotlib not available. Install with: pip install matplotlib")
        else:
            logger.error(f"Failed to evaluate {symbol}")
            sys.exit(1)

if __name__ == "__main__":
    main()
