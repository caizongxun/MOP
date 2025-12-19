#!/usr/bin/env python3
"""
Push Models to Hugging Face Hub - Quick Start

Usage:
    python push_to_huggingface.py --token YOUR_HF_TOKEN --symbols BTCUSDT ETHUSDT
    python push_to_huggingface.py --token YOUR_HF_TOKEN --all

Before running:
1. Get Hugging Face token: https://huggingface.co/settings/tokens
2. Create a repo on HF: https://huggingface.co/new
3. Install huggingface-hub: pip install huggingface-hub
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from push_to_huggingface import HFPusher
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Push V5 models to Hugging Face')
    parser.add_argument('--token', required=True, help='Hugging Face API token')
    parser.add_argument('--repo', default='mop-crypto-models', help='Repository ID')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols')
    parser.add_argument('--all', action='store_true', help='Push all symbols')
    
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
        print("Error: Specify --symbols or --all")
        sys.exit(1)
    
    # Create pusher and push
    pusher = HFPusher(args.token, args.repo)
    results = pusher.push_all(symbols)
