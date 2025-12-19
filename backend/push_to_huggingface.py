#!/usr/bin/env python3
"""
Push V5 Models to Hugging Face Hub

Usage:
    python push_to_huggingface.py --token YOUR_HF_TOKEN --all
    python push_to_huggingface.py --token YOUR_HF_TOKEN --symbols BTCUSDT ETHUSDT
    python push_to_huggingface.py --token YOUR_HF_TOKEN --symbols BTCUSDT

Before running:
1. Create a Hugging Face account: https://huggingface.co/join
2. Get your token: https://huggingface.co/settings/tokens
3. Create a repository on HF: https://huggingface.co/new
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from huggingface_hub import HfApi, hf_hub_download, upload_folder
    from huggingface_hub import Repository
except ImportError:
    print("Error: huggingface-hub not installed")
    print("Install it with: pip install huggingface-hub")
    sys.exit(1)

from path_config import PathConfig

class HFPusher:
    """Push models to Hugging Face Hub"""
    
    def __init__(self, hf_token, repo_id):
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.api = HfApi()
        self.paths = PathConfig()
        
        print(f"Connecting to Hugging Face...")
        print(f"Repository: {repo_id}")
    
    def prepare_model_files(self, symbol):
        """Prepare model files for upload"""
        model_file = self.paths.get_model_weights_file(symbol, version='v5')
        config_file = self.paths.get_model_config_file(symbol, version='v5')
        
        if not os.path.exists(model_file):
            print(f"  ERROR: Model file not found: {model_file}")
            return None
        
        if not os.path.exists(config_file):
            print(f"  ERROR: Config file not found: {config_file}")
            return None
        
        return {
            'model': model_file,
            'config': config_file,
            'symbol': symbol
        }
    
    def create_model_card(self, symbol, config_data):
        """Create README.md for model on HF"""
        readme = f"""---
license: mit
tags:
  - cryptocurrency
  - lstm
  - prediction
  - trading
  - pytorch
---

# {symbol} V5 Prediction Model

Multi-scale LSTM model for {symbol} price prediction (1h timeframe).

## Model Details

**Model Type:** PyTorch LSTM

**Architecture:** Multi-Scale LSTM V5
- 3-scale temporal patterns (short, medium, long)
- Uncertainty quantification
- Residual learning

**Input:** 60-step sequence of 20+ engineered features

**Output:** 
- Price delta (predicted next price change)
- Uncertainty estimate (confidence level)

## Training Data

- **Symbol:** {symbol}
- **Timeframe:** 1 hour
- **Training Set:** 70% of available data
- **Validation Set:** 15%
- **Test Set:** 15%

## Performance Metrics

"""
        
        # Add metrics from config
        if 'metrics' in config_data:
            metrics = config_data['metrics']
            readme += f"""- **MAPE:** {metrics.get('mape', 'N/A')*100:.2f}%
- **MAE:** ${metrics.get('mae', 'N/A'):.4f}
- **RMSE:** ${metrics.get('rmse', 'N/A'):.4f}
"""
        
        readme += f"""## Features

The model uses 20+ engineered features:
- Price movements (returns, normalized price)
- Multi-scale momentum (5, 10, 20 periods)
- Volatility indicators
- Micro-structure (high-low ratios, price position)
- Volume analysis
- Price acceleration
- Mean reversion signals

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="{self.repo_id}",
    filename="{symbol}/pytorch_model.bin"
)

# Load model
model = torch.load(model_path)
model.eval()

# Make predictions
with torch.no_grad():
    output = model(features)
```

## Limitations

- Trained on historical data only
- Past performance does not guarantee future results
- Cryptocurrency markets are highly volatile
- Use as part of broader trading strategy

## References

- Project: MOP (Multi-scale LSTM for Option Prediction)
- Repository: https://github.com/caizongxun/MOP
- Training Date: {datetime.now().strftime('%Y-%m-%d')}

## License

MIT License
"""
        
        return readme
    
    def push_symbol(self, symbol):
        """Push single symbol model to HF"""
        print(f"\nPreparing {symbol}...")
        
        files = self.prepare_model_files(symbol)
        if not files:
            print(f"  FAILED: Could not prepare files")
            return False
        
        # Create temp directory
        temp_dir = Path(f"./temp_hf_{symbol}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy model file as pytorch_model.bin
            dest_model = temp_dir / "pytorch_model.bin"
            shutil.copy(files['model'], dest_model)
            print(f"  Model: {files['model']} -> {dest_model}")
            
            # Copy config
            dest_config = temp_dir / "config.json"
            shutil.copy(files['config'], dest_config)
            print(f"  Config: {files['config']} -> {dest_config}")
            
            # Create README.md
            with open(dest_config) as f:
                config_data = json.load(f)
            
            readme = self.create_model_card(symbol, config_data)
            with open(temp_dir / "README.md", "w") as f:
                f.write(readme)
            print(f"  README: Created")
            
            # Create .gitattributes for LFS
            with open(temp_dir / ".gitattributes", "w") as f:
                f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
            
            # Upload to HF
            print(f"  Uploading to Hugging Face...")
            repo_path = f"{self.repo_id}/{symbol}"
            
            api = HfApi(token=self.hf_token)
            api.upload_folder(
                folder_path=str(temp_dir),
                repo_id=self.repo_id,
                path_in_repo=symbol,
                repo_type="model",
                commit_message=f"Upload {symbol} V5 model",
                create_pr=False
            )
            
            print(f"  SUCCESS: Model uploaded to {repo_path}")
            return True
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return False
        
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"  Cleaned up temporary files")
    
    def push_all(self, symbols):
        """Push all models"""
        results = {}
        
        print(f"\n{'='*70}")
        print(f"PUSHING TO HUGGING FACE")
        print(f"{'='*70}")
        print(f"Repository: {self.repo_id}")
        print(f"Symbols: {len(symbols)}")
        print(f"{'='*70}")
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[{idx}/{len(symbols)}] Pushing {symbol}...")
            success = self.push_symbol(symbol)
            results[symbol] = 'success' if success else 'failed'
        
        # Summary
        print(f"\n{'='*70}")
        print(f"PUSH COMPLETE")
        print(f"{'='*70}")
        
        success_count = sum(1 for v in results.values() if v == 'success')
        failed_count = sum(1 for v in results.values() if v == 'failed')
        
        print(f"\nSuccessful: {success_count}/{len(symbols)}")
        print(f"Failed: {failed_count}/{len(symbols)}")
        
        if success_count > 0:
            print(f"\nView models at: https://huggingface.co/{self.repo_id}")
        
        print(f"{'='*70}\n")
        
        return results

def main():
    parser = argparse.ArgumentParser(
        description='Push V5 models to Hugging Face Hub'
    )
    parser.add_argument(
        '--token',
        required=True,
        help='Hugging Face API token (get from https://huggingface.co/settings/tokens)'
    )
    parser.add_argument(
        '--repo',
        default='mop-crypto-models',
        help='Repository ID (default: mop-crypto-models)'
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
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Create pusher
    pusher = HFPusher(args.token, args.repo)
    
    # Push models
    results = pusher.push_all(symbols)
    
    # Print results
    print("\nDetailed Results:")
    for symbol, status in results.items():
        print(f"  {symbol}: {status}")

if __name__ == '__main__':
    main()
