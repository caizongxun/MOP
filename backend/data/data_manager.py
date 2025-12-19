import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import sys
from dotenv import load_dotenv
import time

# Add parent to path to access config
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from data.data_loader import CryptoDataLoader
from config.model_config import CRYPTOCURRENCIES, DATA_CONFIG

load_dotenv()
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manage cryptocurrency data collection and storage
    Features:
    - Incremental data fetching (avoids re-downloading)
    - Multi-batch fetching (2000+, 3000+, 5000+ candles)
    - Local storage for all cryptocurrencies
    - Support for multiple timeframes (15m, 1h)
    - Metadata tracking (last update, row count, etc.)
    """
    
    # API limit: max 1000 per request, but can loop multiple times
    MAX_PER_REQUEST = 1000
    
    @staticmethod
    def _find_data_directory(base_path='backend/data/raw'):
        """
        Try to find data directory in multiple locations
        """
        possible_paths = [
            Path(base_path),
            Path(os.getcwd()) / base_path,
            Path(__file__).parent / 'raw',  # Current dir / raw
            Path(__file__).parent.parent.parent / base_path,
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found data directory at: {path.absolute()}")
                return path
        
        # If not found, create default at backend/data/raw
        default_path = Path(__file__).parent / 'raw'
        default_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory at: {default_path.absolute()}")
        return default_path
    
    def __init__(self, data_dir='backend/data/raw', timeframes=['15m', '1h']):
        """
        Initialize data manager
        
        Args:
            data_dir: Directory to store raw data
            timeframes: List of timeframes to manage
        """
        # Auto-detect data directory
        self.data_dir = self._find_data_directory(data_dir)
        
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
        """
        return self.data_dir / f"{symbol}_{timeframe}.csv"
    
    def load_data(self, symbol, timeframe):
        """
        Load stored data for a symbol and timeframe as numpy array
        Returns numpy array for compatibility with model training
        """
        file_path = self._get_file_path(symbol, timeframe)
        
        if not file_path.exists():
            logger.warning(f"No stored data for {symbol} ({timeframe}) at {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            # Convert to numpy array (skip timestamp column if present)
            if 'timestamp' in df.columns:
                data = df.drop('timestamp', axis=1).values.astype(np.float32)
            else:
                data = df.values.astype(np.float32)
            
            logger.info(f"Loaded {len(data)} rows for {symbol} ({timeframe})")
            return data
        except Exception as e:
            logger.error(f"Error loading data for {symbol} ({timeframe}): {str(e)}")
            return None
    
    def get_stored_data(self, symbol, timeframe):
        """
        Load stored data for a symbol and timeframe (returns DataFrame)
        FIXED: Now properly handles timestamps from Binance API (milliseconds)
        """
        file_path = self._get_file_path(symbol, timeframe)
        
        logger.debug(f"Looking for data file: {file_path}")
        
        if not file_path.exists():
            logger.warning(f"No stored data for {symbol} ({timeframe}) at {file_path}")
            # List available files for debugging
            logger.debug(f"Available files in {self.data_dir}:")
            if self.data_dir.exists():
                for f in self.data_dir.glob("*.csv"):
                    logger.debug(f"  - {f.name}")
            return None
        
        try:
            # Try reading with timestamp index
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            logger.info(f"Loaded {len(df)} rows for {symbol} ({timeframe}) from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol} ({timeframe}): {str(e)}")
            # Try reading without index and convert timestamp properly
            try:
                df = pd.read_csv(file_path)
                if 'timestamp' in df.columns:
                    # CRITICAL FIX: Binance API returns timestamps in milliseconds (unit='ms')
                    # Do NOT use unit='s' as that would divide by 1000 and result in 1970 dates
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    logger.info(f"Loaded {len(df)} rows (with ms conversion) for {symbol} ({timeframe})")
                else:
                    logger.warning(f"No timestamp column found in {file_path}")
                return df
            except Exception as e2:
                logger.error(f"Failed to load data even with ms conversion: {str(e2)}")
                return None
    
    def _append_new_data(self, symbol, timeframe, new_data, existing_data):
        """
        Append new data to existing data, avoiding duplicates
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
        Fetch data from API and store locally (single batch, max 1000 candles)
        """
        logger.info(f"Fetching data for {symbol} ({timeframe})...")
        
        new_data = self.data_loader.fetch_ohlcv(symbol, timeframe, limit)
        
        if new_data is None or new_data.empty:
            logger.warning(f"Failed to fetch data for {symbol} ({timeframe})")
            return self.get_stored_data(symbol, timeframe)
        
        if append:
            existing_data = self.get_stored_data(symbol, timeframe)
            combined_data = self._append_new_data(symbol, timeframe, new_data, existing_data)
        else:
            combined_data = new_data
        
        self.save_data(symbol, timeframe, combined_data)
        return combined_data
    
    def fetch_and_store_batch(self, symbol, timeframe='1h', total_limit=3000, batch_delay=1.0):
        """
        Fetch multiple batches to accumulate more candles
        Makes multiple API calls to get more historical data
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe (15m, 1h)
            total_limit: Total candles to fetch (2000, 3000, 5000, etc.)
            batch_delay: Delay between API calls (seconds) to avoid rate limit
        
        Returns:
            Combined DataFrame with all accumulated data
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch fetching {symbol} ({timeframe})")
        logger.info(f"Target: {total_limit} candles (batches of {self.MAX_PER_REQUEST})")
        logger.info(f"{'='*60}")
        
        all_data = self.get_stored_data(symbol, timeframe)
        
        # Calculate how many batches needed
        num_batches = (total_limit + self.MAX_PER_REQUEST - 1) // self.MAX_PER_REQUEST
        
        for batch_num in range(num_batches):
            logger.info(f"\nBatch {batch_num + 1}/{num_batches}")
            
            try:
                # Fetch this batch
                batch_data = self.data_loader.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=self.MAX_PER_REQUEST
                )
                
                if batch_data is None or batch_data.empty:
                    logger.warning(f"Batch {batch_num + 1} returned no data")
                    break
                
                logger.info(f"Fetched {len(batch_data)} rows in batch {batch_num + 1}")
                
                # Append to accumulated data
                if all_data is None or all_data.empty:
                    all_data = batch_data
                else:
                    all_data = self._append_new_data(symbol, timeframe, batch_data, all_data)
                
                # Delay before next batch to avoid rate limiting
                if batch_num < num_batches - 1:
                    logger.info(f"Waiting {batch_delay}s before next batch...")
                    time.sleep(batch_delay)
            
            except Exception as e:
                logger.error(f"Error in batch {batch_num + 1}: {str(e)}")
                break
        
        # Save final accumulated data
        if all_data is not None and not all_data.empty:
            self.save_data(symbol, timeframe, all_data)
            logger.info(f"\nCompleted: {len(all_data)} total rows saved")
            return all_data
        else:
            logger.warning(f"No data accumulated for {symbol} ({timeframe})")
            return None
    
    def fetch_and_store_all(self, timeframes=None, limit=1000):
        """
        Fetch and store data for all cryptocurrencies (single batch per coin)
        """
        if timeframes is None:
            timeframes = self.timeframes
        
        logger.info(f"Starting data collection for {len(CRYPTOCURRENCIES)} cryptocurrencies")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Limit per coin: {limit} candles")
        
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
    
    def fetch_and_store_all_batch(self, timeframes=None, total_limit=3000, batch_delay=1.0):
        """
        Batch fetch for all cryptocurrencies to accumulate more candles
        
        Args:
            timeframes: List of timeframes
            total_limit: Total candles per coin per timeframe (2000, 3000, 5000, etc.)
            batch_delay: Delay between batches (seconds)
        """
        if timeframes is None:
            timeframes = self.timeframes
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH DATA COLLECTION FOR ALL CRYPTOCURRENCIES")
        logger.info(f"{'='*70}")
        logger.info(f"Cryptocurrencies: {len(CRYPTOCURRENCIES)}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Target per coin: {total_limit} candles")
        logger.info(f"Batch size: {self.MAX_PER_REQUEST} candles per API call")
        logger.info(f"API call delay: {batch_delay}s")
        logger.info(f"{'='*70}")
        
        results = {}
        total_batches = len(CRYPTOCURRENCIES) * len(timeframes)
        current_batch = 0
        
        for symbol in CRYPTOCURRENCIES:
            symbol_results = {}
            logger.info(f"\n\nProcessing {symbol}...")
            
            for timeframe in timeframes:
                current_batch += 1
                logger.info(f"\n[{current_batch}/{total_batches}] {symbol} {timeframe}")
                
                try:
                    df = self.fetch_and_store_batch(
                        symbol,
                        timeframe=timeframe,
                        total_limit=total_limit,
                        batch_delay=batch_delay
                    )
                    
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
        
        # Print final summary
        logger.info(f"\n\n{'='*70}")
        logger.info("BATCH DATA COLLECTION COMPLETED")
        logger.info(f"{'='*70}")
        
        total_success = 0
        for symbol, timeframe_results in sorted(results.items()):
            logger.info(f"\n{symbol}:")
            for timeframe, result in timeframe_results.items():
                status = result['status']
                if status == 'Success':
                    logger.info(f"  {timeframe}: {status} ({result['rows']} rows) - {result['date_range']}")
                    total_success += 1
                else:
                    error = result.get('error', 'Unknown error')
                    logger.info(f"  {timeframe}: {status} - {error}")
        
        logger.info(f"\nTotal: {total_success}/{total_batches} successful")
        
        return results
    
    def get_data_statistics(self):
        """
        Get statistics about stored data
        """
        stats = {
            'total_files': 0,
            'total_rows': 0,
            'symbols': {},
            'data_dir': str(self.data_dir.absolute()),
            'date_created': datetime.now().isoformat(),
        }
        
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
        logger.info(f"Data Directory: {stats['data_dir']}")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Total rows: {stats['total_rows']:,}")
        logger.info(f"\nData by Symbol:")
        
        for symbol in sorted(stats['symbols'].keys()):
            timeframes = stats['symbols'][symbol]
            logger.info(f"\n{symbol}:")
            for tf in sorted(timeframes.keys()):
                info = timeframes[tf]
                logger.info(f"  {tf}: {info['rows']:,} rows ({info['date_range']['start']} to {info['date_range']['end']})")
