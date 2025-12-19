#!/usr/bin/env python3
"""
Batch Push V5 Models to Hugging Face Hub - Optimized Version

Pushes entire models folder at once to avoid API rate limiting.
Much faster and more reliable than uploading individual files.

Usage:
    python push_to_huggingface_batch.py --token YOUR_HF_TOKEN --repo zongowo111/mop-v5-models
    python push_to_huggingface_batch.py --token YOUR_HF_TOKEN --all --repo zongowo111/mop-v5-models

Features:
    - Batch upload entire model directories
    - No API rate limits
    - Automatic retry on network errors  
    - Progress tracking
    - Creates model cards automatically
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from huggingface_hub import HfApi, upload_folder
except ImportError:
    print("Error: huggingface-hub not installed")
    print("Install with: pip install huggingface-hub>=0.20.0")
    sys.exit(1)

from path_config import PathConfig

class BatchHFPusher:
    """Batch push models to Hugging Face - optimized for folder uploads"""
    
    def __init__(self, hf_token, repo_id):
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.api = HfApi(token=hf_token)
        self.paths = PathConfig()
        
        print(f"\nConnecting to Hugging Face...")
        print(f"Repository: {repo_id}")
        print(f"Token: {hf_token[:20]}...")
        
        # Verify repo exists
        try:
            repo_info = self.api.repo_info(self.repo_id, repo_type="model")
            print(f"Repository found: {self.repo_id}")
            print(f"Repository URL: https://huggingface.co/{self.repo_id}")
        except Exception as e:
            print(f"ERROR: Cannot access repository: {e}")
            print(f"Make sure repository exists: https://huggingface.co/new")
            sys.exit(1)
    
    def create_model_readme(self, symbol, config_data):
        """Create README.md for symbol"""
        readme = f"""---
license: mit
tags:
  - cryptocurrency
  - lstm
  - prediction
  - pytorch
---

# {symbol} V5 Price Prediction Model

Multi-scale LSTM model for predicting {symbol} price movements.

## Model Details

- **Symbol**: {symbol}
- **Timeframe**: 1 hour
- **Type**: PyTorch LSTM with multi-scale temporal patterns
- **Input**: 60-step sequence of 20+ engineered features
- **Output**: Price delta and uncertainty estimate

## Architecture

- Multi-scale LSTM (3 temporal scales)
- Uncertainty quantification
- Residual learning
- Feature engineering pipeline

## Training Data

- **Training**: 70% of historical data
- **Validation**: 15%
- **Test**: 15%
- **Date**: {datetime.now().strftime('%Y-%m-%d')}

## Performance

"""
        
        if 'metrics' in config_data:
            metrics = config_data['metrics']
            readme += f"""- **MAPE**: {metrics.get('mape', 0)*100:.2f}%
- **MAE**: ${metrics.get('mae', 0):.6f}
- **RMSE**: ${metrics.get('rmse', 0):.6f}

"""
        
        readme += f"""## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="{self.repo_id}",
    filename="{symbol}/pytorch_model.bin"
)

# Load
model = torch.load(model_path)
model.eval()

# Predict
with torch.no_grad():
    delta, uncertainty = model(features)
```

## Features Used

20+ engineered indicators:
- Price movements and returns
- Multi-scale momentum (5, 10, 20 periods)
- Volatility measures
- Micro-structure (high-low ratio, close position)
- Volume analysis
- Price acceleration
- Mean reversion signals

## Limitations

- Historical data only - past performance does not guarantee future results
- Crypto markets are highly volatile
- Use as part of broader trading strategy
- Not financial advice

## Project

**MOP**: Multi-scale LSTM Option Prediction
- GitHub: https://github.com/zongowo111/MOP
- Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## License

MIT License - See LICENSE for details
"""
        return readme
    
    def prepare_symbol_folder(self, symbol, temp_dir):
        """Prepare folder for single symbol"""
        model_file = self.paths.get_model_weights_file(symbol, version='v5')
        config_file = self.paths.get_model_config_file(symbol, version='v5')
        
        if not os.path.exists(model_file):
            print(f"  ERROR: Model not found: {model_file}")
            return False
        
        if not os.path.exists(config_file):
            print(f"  ERROR: Config not found: {config_file}")
            return False
        
        try:
            # Copy files
            symbol_dir = temp_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Copy model as pytorch_model.bin
            shutil.copy(model_file, symbol_dir / "pytorch_model.bin")
            
            # Copy config
            shutil.copy(config_file, symbol_dir / "config.json")
            
            # Load config and create README
            with open(config_file) as f:
                config_data = json.load(f)
            
            readme = self.create_model_readme(symbol, config_data)
            with open(symbol_dir / "README.md", "w") as f:
                f.write(readme)
            
            # Git LFS config
            with open(symbol_dir / ".gitattributes", "w") as f:
                f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
            
            return True
        
        except Exception as e:
            print(f"  ERROR preparing {symbol}: {e}")
            return False
    
    def push_batch(self, symbols):
        """Push all symbols as one batch"""
        print(f"\n{'='*70}")
        print(f"BATCH PUSH TO HUGGING FACE")
        print(f"{'='*70}")
        print(f"Repository: {self.repo_id}")
        print(f"Symbols to push: {len(symbols)}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"{'='*70}\n")
        
        # Create temp directory
        temp_dir = Path("./temp_hf_batch")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        try:
            # Prepare all symbol folders
            print("Preparing model folders...")
            success_count = 0
            for symbol in symbols:
                print(f"  Preparing {symbol}...", end=" ")
                if self.prepare_symbol_folder(symbol, temp_dir):
                    print("OK")
                    success_count += 1
                else:
                    print("FAILED")
            
            if success_count == 0:
                print("\nERROR: No models prepared successfully")
                return False
            
            print(f"\nSuccessfully prepared: {success_count}/{len(symbols)}")
            
            # Upload entire folder at once
            print(f"\nUploading {success_count} models to Hugging Face...")
            print(f"Folder size: {self._get_folder_size(temp_dir) / (1024**2):.1f} MB")
            print(f"This may take a few minutes...\n")
            
            start_time = time.time()
            
            # Retry logic for network errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Upload folder - single commit
                    api_url = self.api.upload_folder(
                        folder_path=str(temp_dir),
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"Batch upload {success_count} V5 models - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        create_pr=False,
                        multi_commit=False
                    )
                    
                    elapsed = time.time() - start_time
                    print(f"Upload completed successfully!")
                    print(f"Time taken: {elapsed:.1f} seconds")
                    print(f"URL: {api_url}")
                    
                    return True
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"Upload failed (attempt {attempt+1}/{max_retries}): {str(e)[:100]}")
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"\nFinal upload attempt failed: {e}")
                        return False
        
        except Exception as e:
            print(f"\nERROR: {e}")
            return False
        
        finally:
            # Cleanup
            if temp_dir.exists():
                print(f"\nCleaning up temporary files...")
                shutil.rmtree(temp_dir)
                print("Done.")
    
    @staticmethod
    def _get_folder_size(folder_path):
        """Calculate total folder size in bytes"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total

def main():
    parser = argparse.ArgumentParser(
        description='Batch push V5 models to Hugging Face Hub'
    )
    parser.add_argument(
        '--token',
        required=True,
        help='Hugging Face API token'
    )
    parser.add_argument(
        '--repo',
        required=True,
        help='Repository ID (e.g., zongowo111/mop-v5-models)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to push'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Push all symbols'
    )
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT',
            'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT',
            'NEARUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
        ]
    else:
        print("ERROR: Specify --symbols or --all")
        sys.exit(1)
    
    # Create pusher and push
    pusher = BatchHFPusher(args.token, args.repo)
    success = pusher.push_batch(symbols)
    
    if success:
        print(f"\n{'='*70}")
        print(f"SUCCESS: Models pushed to {args.repo}")
        print(f"View at: https://huggingface.co/{args.repo}")
        print(f"{'='*70}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print(f"FAILED: Could not push models")
        print(f"{'='*70}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
