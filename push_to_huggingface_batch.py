#!/usr/bin/env python3
"""
Batch Push Models to Hugging Face - Quick Start

Pushes entire models folder at once (NO API rate limits!)

Usage:
    # Batch push to your repo (RECOMMENDED - 1 operation for all models)
    python push_to_huggingface_batch.py --token YOUR_HF_TOKEN --repo zongowo111/mop-v5-models --all
    
    # Or specific symbols
    python push_to_huggingface_batch.py --token YOUR_HF_TOKEN --repo zongowo111/mop-v5-models --symbols BTCUSDT ETHUSDT

Why batch push?
    - NO API rate limiting
    - Much faster (1 commit for all models)
    - More reliable
    - Automatic retry on network errors
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from push_to_huggingface_batch import BatchHFPusher
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch push V5 models to Hugging Face (fast & reliable)'
    )
    parser.add_argument('--token', required=True, help='Hugging Face API token')
    parser.add_argument('--repo', required=True, help='Repository ID (e.g., zongowo111/mop-v5-models)')
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
        print("ERROR: Specify --symbols or --all")
        sys.exit(1)
    
    # Push
    pusher = BatchHFPusher(args.token, args.repo)
    success = pusher.push_batch(symbols)
    sys.exit(0 if success else 1)
