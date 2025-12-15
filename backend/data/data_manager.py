import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

from backend.data.data_loader import CryptoDataLoader
from config.model_config import CRYPTOCURRENCIES, DATA_CONFIG

load_dotenv()
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manage cryptocurrency data collection and storage
    Features:
    - Incremental data fetching (avoids re-downloading)
    - Local storage for all cryptocurrencies
    - Support for multiple timeframes (15m, 1h)
    - Metadata tracking (last update, row count, etc.)
    """
    
    def __init__(self, data_dir='data/raw', timeframes=['15m', '1h']):
        """
        Initialize data manager
        
        Args:
            data_dir: Directory to store raw data
            timeframes: List of timeframes to manage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.timeframes = timeframes
        self.data_loader = CryptoDataLoader(
            use_binance_api=DATA_CONFIG['use_binance_api']
        )
        
        # Metadata file for tracking data collection
        self.metadata_file = self.data_dir / 'metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"DataManager initialized with data directory: {self.data_dir}")
    
    def _load_metadata(self):
        """
        Load metadata about existing data
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {str(e)}")
                return {}
        return {}
    
    def _save_metadata(self):
        """
        Save metadata to file
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
    
    def _get_file_path(self, symbol, timeframe):
        """
        Get file path for a cryptocurrency and timeframe
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h')
        
        Returns:
            Path object
        """
        return self.data_dir / f"{symbol}_{timeframe}.csv"
    
    def get_stored_data(self, symbol, timeframe):
        """
        Load stored data for a symbol and timeframe
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
        
        Returns:
            DataFrame or None if not found
        """
        file_path = self._get_file_path(symbol, timeframe)
        
        if not file_path.exists():
            logger.debug(f"No stored data for {symbol} ({timeframe})")
            return None
        
        try:
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            logger.info(f"Loaded {len(df)} rows for {symbol} ({timeframe}) from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol} ({timeframe}): {str(e)}")
            return None
    
    def _append_new_data(self, symbol, timeframe, new_data, existing_data):
        """
        Append new data to existing data, avoiding duplicates
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            new_data: New DataFrame from API
            existing_data: Existing DataFrame from storage
        
        Returns:
            Combined DataFrame
        """
        if existing_data is None or existing_data.empty:
            return new_data
        
        # Combine and remove duplicates (keep first occurrence)
        combined = pd.concat([existing_data, new_data], sort=False)
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # Sort by timestamp
        combined = combined.sort_index()
        
        logger.info(f"Appended {len(new_data)} rows to {symbol} ({timeframe}). Total: {len(combined)} rows")
        return combined
    
    def save_data(self, symbol, timeframe, df):
        """
        Save data to CSV file
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            df: DataFrame to save
        """
        file_path = self._get_file_path(symbol, timeframe)
        
        try:
            df.to_csv(file_path)
            logger.info(f"Saved {len(df)} rows for {symbol} ({timeframe}) to {file_path}")
            
            # Update metadata
            key = f"{symbol}_{timeframe}"
            self.metadata[key] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'rows': len(df),
                'last_timestamp': str(df.index[-1]),
                'first_timestamp': str(df.index[0]),
                'last_updated': datetime.now().isoformat(),
            }
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol} ({timeframe}): {str(e)}")
    
    def fetch_and_store(self, symbol, timeframe='1h', limit=1000, append=True):
        """
        Fetch data from API and store locally
        Incremental: only fetches and appends new data
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (default: '1h')
            limit: Number of candles to fetch from API (max 1000)
            append: If True, append to existing data; if False, replace
        
        Returns:
            Combined DataFrame
        """
        logger.info(f"Fetching data for {symbol} ({timeframe})...")
        
        # Fetch from API
        new_data = self.data_loader.fetch_ohlcv(symbol, timeframe, limit)
        
        if new_data is None or new_data.empty:
            logger.warning(f"Failed to fetch data for {symbol} ({timeframe})")
            # Return existing data if available
            return self.get_stored_data(symbol, timeframe)
        
        # Get existing data
        if append:
            existing_data = self.get_stored_data(symbol, timeframe)
            combined_data = self._append_new_data(symbol, timeframe, new_data, existing_data)
        else:
            combined_data = new_data
        
        # Save combined data
        self.save_data(symbol, timeframe, combined_data)
        
        return combined_data
    
    def fetch_and_store_all(self, timeframes=None, limit=1000):
        """
        Fetch and store data for all cryptocurrencies
        
        Args:
            timeframes: List of timeframes (default: ['15m', '1h'])
            limit: Number of candles to fetch per API call
        """
        if timeframes is None:
            timeframes = self.timeframes
        
        logger.info(f"Starting data collection for {len(CRYPTOCURRENCIES)} cryptocurrencies")
        logger.info(f"Timeframes: {timeframes}")
        
        results = {}
        
        for symbol in CRYPTOCURRENCIES:
            symbol_results = {}
            
            for timeframe in timeframes:
                try:
                    df = self.fetch_and_store(symbol, timeframe, limit)
                    if df is not None and not df.empty:
                        symbol_results[timeframe] = {
                            'status': 'Success',
                            'rows': len(df),
                            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}"
                        }
                    else:
                        symbol_results[timeframe] = {
                            'status': 'Failed',
                            'error': 'No data returned'
                        }
                except Exception as e:
                    symbol_results[timeframe] = {
                        'status': 'Error',
                        'error': str(e)
                    }
            
            results[symbol] = symbol_results
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Data Collection Summary")
        logger.info(f"{'='*60}")
        
        total_success = 0
        for symbol, timeframe_results in results.items():
            logger.info(f"\n{symbol}:")
            for timeframe, result in timeframe_results.items():
                status = result['status']
                if status == 'Success':
                    logger.info(f"  {timeframe}: {status} ({result['rows']} rows) - {result['date_range']}")
                    total_success += 1
                else:
                    error = result.get('error', 'Unknown error')
                    logger.info(f"  {timeframe}: {status} - {error}")
        
        logger.info(f"\nTotal: {total_success}/{len(CRYPTOCURRENCIES) * len(timeframes)} successful")
        
        return results
    
    def get_data_statistics(self):
        """
        Get statistics about stored data
        
        Returns:
            Dictionary with data statistics
        """
        stats = {
            'total_files': 0,
            'total_rows': 0,
            'symbols': {},
            'date_created': datetime.now().isoformat(),
        }
        
        # Iterate through all CSV files
        for csv_file in self.data_dir.glob('*.csv'):
            if csv_file.name == 'metadata.json':
                continue
            
            try:
                df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
                parts = csv_file.stem.split('_')
                
                if len(parts) >= 2:
                    symbol = '_'.join(parts[:-1])
                    timeframe = parts[-1]
                    
                    if symbol not in stats['symbols']:
                        stats['symbols'][symbol] = {}
                    
                    stats['symbols'][symbol][timeframe] = {
                        'rows': len(df),
                        'date_range': {
                            'start': str(df.index[0].date()),
                            'end': str(df.index[-1].date())
                        }
                    }
                    
                    stats['total_files'] += 1
                    stats['total_rows'] += len(df)
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {str(e)}")
        
        return stats
    
    def print_statistics(self):
        """
        Print data statistics
        """
        stats = self.get_data_statistics()
        
        logger.info(f"\n{'='*60}")
        logger.info("Data Storage Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Total rows: {stats['total_rows']:,}")
        logger.info(f"\nData by Symbol:")
        
        for symbol in sorted(stats['symbols'].keys()):
            timeframes = stats['symbols'][symbol]
            logger.info(f"\n{symbol}:")
            for tf in sorted(timeframes.keys()):
                info = timeframes[tf]
                logger.info(f"  {tf}: {info['rows']:,} rows ({info['date_range']['start']} to {info['date_range']['end']})")
    
    def cleanup_old_files(self, days=30):
        """
        Remove CSV files older than specified days (optional)
        
        Args:
            days: Number of days to keep (default: 30)
        """
        logger.warning(f"Cleanup function not implemented (would remove files older than {days} days)")
