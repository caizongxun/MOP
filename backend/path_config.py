#!/usr/bin/env python3
"""
Unified Path Configuration
Centralized path management for all scripts

Usage:
    from path_config import PathConfig
    paths = PathConfig()
    print(paths.models_dir)
    print(paths.data_dir)
    print(paths.results_dir)
"""

import os
import sys
from pathlib import Path

class PathConfig:
    """
    Centralized path configuration.
    Automatically detects project root and sets up all directories.
    Works from any location (root level, backend level, etc.)
    """
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.backend_dir = os.path.join(self.project_root, 'backend')
        self._ensure_dirs_exist()
    
    @staticmethod
    def _find_project_root():
        """
        Find project root by looking for key markers:
        - backend/ directory
        - models/ directory
        - results/ directory
        - .env file
        - README.md in root
        """
        # Start from current file location
        current = os.path.dirname(os.path.abspath(__file__))
        
        # Check if we're already in backend
        if os.path.basename(current) == 'backend':
            return os.path.dirname(current)
        
        # Check if backend exists here
        if os.path.exists(os.path.join(current, 'backend')):
            return current
        
        # Go up one level
        parent = os.path.dirname(current)
        if os.path.exists(os.path.join(parent, 'backend')):
            return parent
        
        # Try common patterns
        candidates = [
            os.path.expanduser('~/Desktop/MOP'),
            os.path.expanduser('~/MOP'),
            'C:\\Users\\zong\\Desktop\\MOP',
            '/home/user/MOP',
        ]
        
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, 'backend')):
                return candidate
        
        # Fallback to current directory
        return current
    
    def _ensure_dirs_exist(self):
        """Create all necessary directories"""
        dirs = [
            self.data_raw_dir,
            self.models_weights_dir,
            self.models_config_dir,
            self.results_visualizations_dir,
            self.logs_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    # === ROOT & MAIN DIRECTORIES ===
    @property
    def root(self):
        """Project root directory"""
        return self.project_root
    
    @property
    def backend(self):
        """Backend directory"""
        return self.backend_dir
    
    # === DATA DIRECTORIES ===
    @property
    def data_dir(self):
        """Data directory (backend/data)"""
        return os.path.join(self.backend_dir, 'data')
    
    @property
    def data_raw_dir(self):
        """Raw data directory (backend/data/raw)"""
        return os.path.join(self.backend_dir, 'data', 'raw')
    
    @property
    def data_processed_dir(self):
        """Processed data directory (backend/data/processed)"""
        return os.path.join(self.backend_dir, 'data', 'processed')
    
    # === MODELS DIRECTORIES ===
    @property
    def models_dir(self):
        """Models directory (backend/models)"""
        return os.path.join(self.backend_dir, 'models')
    
    @property
    def models_weights_dir(self):
        """Model weights directory (backend/models/weights)"""
        return os.path.join(self.backend_dir, 'models', 'weights')
    
    @property
    def models_config_dir(self):
        """Model config directory (backend/models/config)"""
        return os.path.join(self.backend_dir, 'models', 'config')
    
    # === RESULTS DIRECTORIES ===
    @property
    def results_dir(self):
        """Results directory (backend/results)"""
        return os.path.join(self.backend_dir, 'results')
    
    @property
    def results_visualizations_dir(self):
        """Results visualizations directory (backend/results/visualizations)"""
        return os.path.join(self.backend_dir, 'results', 'visualizations')
    
    # === LOGS DIRECTORIES ===
    @property
    def logs_dir(self):
        """Logs directory (backend/logs)"""
        return os.path.join(self.backend_dir, 'logs')
    
    # === FILE PATHS ===
    def get_data_file(self, symbol: str, timeframe: str = '1h') -> str:
        """Get path to data file"""
        return os.path.join(self.data_raw_dir, f'{symbol}_{timeframe}.csv')
    
    def get_model_weights_file(self, symbol: str, timeframe: str = '1h', version: str = 'v4') -> str:
        """Get path to model weights file"""
        return os.path.join(self.models_weights_dir, f'{symbol}_{timeframe}_{version}_lstm.pth')
    
    def get_model_config_file(self, symbol: str, version: str = 'v4') -> str:
        """Get path to model config file"""
        return os.path.join(self.models_config_dir, f'{symbol}_{version}_config.json')
    
    def get_xgb_model_file(self, symbol: str, timeframe: str = '1h', version: str = 'v4') -> str:
        """Get path to XGBoost model file"""
        return os.path.join(self.models_weights_dir, f'{symbol}_{timeframe}_{version}_xgb.json')
    
    def get_log_file(self, name: str = 'training') -> str:
        """Get path to log file"""
        return os.path.join(self.logs_dir, f'{name}.log')
    
    # === UTILITY METHODS ===
    def verify(self) -> dict:
        """Verify all paths exist and return status"""
        status = {
            'root': os.path.exists(self.root),
            'backend': os.path.exists(self.backend),
            'data_raw': os.path.exists(self.data_raw_dir),
            'models': os.path.exists(self.models_dir),
            'models_weights': os.path.exists(self.models_weights_dir),
            'models_config': os.path.exists(self.models_config_dir),
            'results': os.path.exists(self.results_dir),
            'results_viz': os.path.exists(self.results_visualizations_dir),
            'logs': os.path.exists(self.logs_dir),
        }
        return status
    
    def print_summary(self):
        """Print path configuration summary"""
        print("\n" + "="*70)
        print("PATH CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Project Root:  {self.root}")
        print(f"Backend:       {self.backend}")
        print(f"Data (Raw):    {self.data_raw_dir}")
        print(f"Models:        {self.models_dir}")
        print(f"  Weights:     {self.models_weights_dir}")
        print(f"  Config:      {self.models_config_dir}")
        print(f"Results:       {self.results_dir}")
        print(f"  Visualizations: {self.results_visualizations_dir}")
        print(f"Logs:          {self.logs_dir}")
        print("="*70 + "\n")
        
        status = self.verify()
        print("Directory Status:")
        for name, exists in status.items():
            symbol = "OK" if exists else "MISSING"
            print(f"  {name:20s}: {symbol}")
        print("="*70 + "\n")


# Quick test
if __name__ == '__main__':
    paths = PathConfig()
    paths.print_summary()
    
    print("Example file paths:")
    print(f"  BTCUSDT data:  {paths.get_data_file('BTCUSDT')}")
    print(f"  BTCUSDT model: {paths.get_model_weights_file('BTCUSDT')}")
    print(f"  Training log:  {paths.get_log_file('training')}")
