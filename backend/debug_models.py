#!/usr/bin/env python
r"""
Debug script to inspect all trained models and export their architectures to JSON

Usage:
    python backend/debug_models.py
    python backend/debug_models.py --symbol BTCUSDT
    python backend/debug_models.py --export
"""

import logging
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from backend.models.lstm_model import CryptoLSTM
from backend.data.data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDebugger:
    def __init__(self):
        self.model_dir = Path('backend/models/weights')
        self.data_manager = DataManager()
    
    def get_all_model_files(self):
        """Get all .pt model files"""
        return sorted(list(self.model_dir.glob('*.pt')))
    
    def inspect_model(self, model_path):
        """
        Inspect a single model and return architecture info
        """
        try:
            state = torch.load(model_path, map_location='cpu')
            
            model_config = state['model_config']
            
            # Extract model file info
            filename = model_path.stem  # Remove .pt
            parts = filename.split('_')
            
            if len(parts) >= 2:
                symbol = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
                timeframe_type = parts[-1]  # e.g., '1h', '15m', 'unified'
            else:
                symbol = filename
                timeframe_type = 'unknown'
            
            info = {
                'filename': model_path.name,
                'symbol': symbol,
                'timeframe': timeframe_type,
                'model_config': {
                    'input_size': model_config.get('input_size'),
                    'hidden_size': model_config.get('hidden_size'),
                    'num_layers': model_config.get('num_layers'),
                    'output_size': model_config.get('output_size'),
                    'dropout': model_config.get('dropout'),
                },
                'training_info': {
                    'trained_at': state.get('trained_at'),
                    'training_loss': state.get('training_loss'),
                    'final_loss': state.get('final_loss'),
                },
                'data_info': {
                    'lookback': model_config.get('lookback', 60),
                    'feature_count': model_config.get('input_size'),
                }
            }
            
            return info
        
        except Exception as e:
            logger.error(f"Error inspecting {model_path.name}: {str(e)}")
            return None
    
    def inspect_all_models(self):
        """
        Inspect all models and organize by symbol
        """
        model_files = self.get_all_model_files()
        logger.info(f"Found {len(model_files)} model files")
        
        all_models = {}
        
        for model_path in model_files:
            info = self.inspect_model(model_path)
            if info:
                symbol = info['symbol']
                if symbol not in all_models:
                    all_models[symbol] = []
                all_models[symbol].append(info)
        
        return all_models
    
    def get_data_features(self, symbol, timeframe):
        """
        Get actual features from data for a symbol
        """
        from backend.data.data_loader import CryptoDataLoader
        
        try:
            data = self.data_manager.get_stored_data(symbol, timeframe)
            if data is None:
                return None, 0
            
            data_loader = CryptoDataLoader()
            data_with_indicators = data_loader.calculate_technical_indicators(data)
            
            if data_with_indicators is None:
                return None, 0
            
            feature_cols = [col for col in data_with_indicators.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            return feature_cols, len(feature_cols)
        
        except Exception as e:
            logger.error(f"Error getting features for {symbol} ({timeframe}): {str(e)}")
            return None, 0
    
    def validate_model(self, symbol, timeframe):
        """
        Validate if model input matches actual data features
        """
        model_path = self.model_dir / f"{symbol}_{timeframe}_v1.pt"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            state = torch.load(model_path, map_location='cpu')
            model_input_size = state['model_config']['input_size']
            
            features, feature_count = self.get_data_features(symbol, timeframe)
            
            validation = {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_input_size': model_input_size,
                'actual_features': feature_count,
                'matches': model_input_size == feature_count,
                'features': features if features else []
            }
            
            return validation
        
        except Exception as e:
            logger.error(f"Error validating {symbol} ({timeframe}): {str(e)}")
            return None
    
    def print_models_info(self, all_models):
        """
        Print formatted model information
        """
        logger.info(f"\n{'='*80}")
        logger.info("MODEL ARCHITECTURE SUMMARY")
        logger.info(f"{'='*80}")
        
        for symbol in sorted(all_models.keys()):
            models = all_models[symbol]
            logger.info(f"\n{symbol}:")
            
            for model_info in models:
                logger.info(f"  File: {model_info['filename']}")
                logger.info(f"  Timeframe: {model_info['timeframe']}")
                config = model_info['model_config']
                logger.info(f"  Architecture:")
                logger.info(f"    Input Size: {config['input_size']}")
                logger.info(f"    Hidden Size: {config['hidden_size']}")
                logger.info(f"    Num Layers: {config['num_layers']}")
                logger.info(f"    Output Size: {config['output_size']}")
                logger.info(f"    Dropout: {config['dropout']}")
                logger.info(f"    Lookback: {model_info['data_info']['lookback']}")
                logger.info("-" * 40)
    
    def export_to_json(self, all_models, output_file='models_architecture.json'):
        """
        Export all model architectures to JSON file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(all_models, f, indent=2, default=str)
            logger.info(f"\nModels architecture exported to: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return False
    
    def validate_all_models(self):
        """
        Validate all models against actual data
        """
        model_files = self.get_all_model_files()
        
        logger.info(f"\n{'='*80}")
        logger.info("MODEL VALIDATION REPORT")
        logger.info(f"{'='*80}")
        
        validations = []
        mismatches = []
        
        for model_path in model_files:
            filename = model_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 2:
                symbol = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
                timeframe_type = parts[-1]
                
                if timeframe_type in ['1h', '15m']:
                    validation = self.validate_model(symbol, timeframe_type)
                    if validation:
                        validations.append(validation)
                        
                        if not validation['matches']:
                            mismatches.append(validation)
        
        # Print results
        for val in validations:
            status = "OK" if val['matches'] else "MISMATCH"
            logger.info(f"{val['symbol']:12} ({val['timeframe']:3}): Model={val['model_input_size']:2} Data={val['actual_features']:2} [{status}]")
        
        if mismatches:
            logger.info(f"\n{len(mismatches)} MISMATCHES FOUND:")
            for mismatch in mismatches:
                logger.error(f"  {mismatch['symbol']} ({mismatch['timeframe']}): Model expects {mismatch['model_input_size']} features, but data has {mismatch['actual_features']}")
        else:
            logger.info(f"\nAll {len(validations)} models validated successfully!")
        
        return validations, mismatches


def main():
    parser = ArgumentParser(description='Debug and validate model architectures')
    parser.add_argument('--symbol', help='Validate specific symbol')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--export', action='store_true', help='Export all models to JSON')
    parser.add_argument('--validate', action='store_true', help='Validate all models')
    
    args = parser.parse_args()
    
    debugger = ModelDebugger()
    
    if args.symbol:
        logger.info(f"\nValidating {args.symbol} ({args.timeframe})...")
        validation = debugger.validate_model(args.symbol, args.timeframe)
        if validation:
            logger.info(json.dumps(validation, indent=2))
    else:
        all_models = debugger.inspect_all_models()
        debugger.print_models_info(all_models)
        
        if args.export or args.validate:
            if args.export:
                debugger.export_to_json(all_models)
            
            if args.validate:
                validations, mismatches = debugger.validate_all_models()


if __name__ == "__main__":
    main()
