#!/usr/bin/env python
r"""
Quick evaluation script - One command to evaluate and visualize model

Usage:
    python backend/quick_eval.py
    python backend/quick_eval.py --symbol ETHUSDT --timeframe 1h
    python backend/quick_eval.py --all-symbols

Run from project root directory
"""

import logging
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.evaluate_model import ModelEvaluator
from backend.data.data_manager import DataManager

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
    parser.add_argument('--check-data', action='store_true', help='Check available data without evaluating')
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Check data first
    if args.check_data:
        logger.info("\nChecking available data...")
        data_manager = DataManager()
        data_manager.print_statistics()
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device='cpu')
    
    if args.all_symbols:
        logger.info("\nEvaluating all symbols...")
        all_results = evaluator.evaluate_all_symbols(timeframe=args.timeframe)
        if all_results:
            evaluator.save_results(all_results, 'all_evaluations.json')
    else:
        symbol = args.symbol
        timeframe = args.timeframe
        
        logger.info(f"\nEvaluating {symbol} ({timeframe})...")
        
        # Check if data exists
        data_manager = DataManager()
        data = data_manager.get_stored_data(symbol, timeframe)
        if data is None:
            logger.error(f"No data found for {symbol} ({timeframe})")
            logger.info(f"\nAvailable data:")
            data_manager.print_statistics()
            sys.exit(1)
        
        logger.info(f"Data found: {len(data)} rows")
        
        result = evaluator.evaluate_symbol(symbol, timeframe)
        
        if result:
            evaluator.save_results(result, f'{symbol}_{timeframe}_evaluation.json')
            logger.info(f"\nEvaluation completed! Results saved to {symbol}_{timeframe}_evaluation.json")
        else:
            logger.error(f"Failed to evaluate {symbol}")
            sys.exit(1)

if __name__ == "__main__":
    main()
