#!/usr/bin/env python3
"""
Quick Visualizer for V4 Adaptive Models
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_models_dir():
    """
    Find models directory. Check multiple locations:
    1. ../models/weights (root level)
    2. ./models/weights (from backend)
    3. ./backend/models/weights (nested)
    """
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Option 1: Root level (most common after training)
    root_models = os.path.join(os.path.dirname(backend_dir), 'models', 'weights')
    if os.path.exists(root_models):
        print(f"Found models at: {root_models}")
        return root_models
    
    # Option 2: Backend level
    backend_models = os.path.join(backend_dir, 'models', 'weights')
    if os.path.exists(backend_models):
        print(f"Found models at: {backend_models}")
        return backend_models
    
    # Option 3: Nested backend/backend (shouldn't happen but check)
    nested_models = os.path.join(backend_dir, 'backend', 'models', 'weights')
    if os.path.exists(nested_models):
        print(f"Found models at: {nested_models}")
        return nested_models
    
    # Return default (will show error message)
    return root_models

def find_results_dir():
    """
    Find results directory
    """
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Option 1: Root level results
    root_results = os.path.join(os.path.dirname(backend_dir), 'results', 'visualizations')
    if not os.path.exists(root_results):
        os.makedirs(root_results, exist_ok=True)
    return root_results

def main():
    parser = argparse.ArgumentParser(description='V4 Adaptive Predictions Visualizer')
    parser.add_argument('--all', action='store_true', help='Visualize all symbols')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to visualize')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("V4 Adaptive Predictions Visualizer")
    print("="*60)
    print(f"Device: {args.device}")
    
    # Find directories
    models_dir = find_models_dir()
    results_dir = find_results_dir()
    
    print(f"Models: {models_dir}")
    print(f"Output: {results_dir}")
    
    # Check if models exist
    if not os.path.exists(models_dir):
        print(f"\nERROR: Models directory not found: {models_dir}")
        print("Please run: python train_v4.py 5 --device cuda")
        return
    
    # List available models
    lstm_models = [f for f in os.listdir(models_dir) if f.endswith('_1h_v4_lstm.pth')]
    
    if not lstm_models:
        print(f"\nERROR: No trained models found in {models_dir}")
        print("Available files:")
        for f in os.listdir(models_dir):
            print(f"  - {f}")
        return
    
    print(f"\nFound {len(lstm_models)} trained models")
    
    # Extract symbols from model files
    symbols = [f.replace('_1h_v4_lstm.pth', '') for f in lstm_models]
    
    if args.symbols:
        symbols = args.symbols
    
    print(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
    print("="*60)
    
    # Import visualizer
    try:
        from visualize_predictions_v4 import V4Visualizer
    except ImportError as e:
        print(f"ERROR: Could not import V4Visualizer: {e}")
        return
    
    # Create visualizer with correct paths
    visualizer = V4Visualizer(device=args.device)
    # Override paths
    visualizer.models_dir = models_dir
    visualizer.results_dir = results_dir
    
    # Generate visualizations
    results_list = visualizer.plot_multiple_symbols(symbols, results_dir)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    
    if results_list:
        print(f"\nSuccessfully visualized {len(results_list)} symbols")
        for result in results_list:
            print(f"  {result['symbol']}: MAPE={result['mape']*100:.2f}%")
        print(f"\nResults saved to: {results_dir}")
    else:
        print("\nNo visualizations were created.")
        print(f"Please ensure models exist in: {models_dir}")

if __name__ == '__main__':
    main()
