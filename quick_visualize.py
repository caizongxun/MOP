#!/usr/bin/env python3
"""
Quick Visualization Launcher for V4 Adaptive Predictions
Run from project root: python quick_visualize.py

Usage:
  python quick_visualize.py                           # Default 5 symbols
  python quick_visualize.py BTCUSDT ETHUSDT          # Custom symbols
  python quick_visualize.py --all                     # All 15 symbols
  python quick_visualize.py --device cuda             # Use GPU
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.quick_visualize import main

if __name__ == '__main__':
    sys.exit(main())
