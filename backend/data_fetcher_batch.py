import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.data.data_manager import DataManager
from config.model_config import CRYPTOCURRENCIES, DATA_CONFIG

if __name__ == "__main__":
    """
    Batch data fetching script
    Fetches more than 1000 candles per cryptocurrency by making multiple API calls
    
    Usage:
        python backend/data_fetcher_batch.py
    
    This script will:
    1. Fetch 3000 candles per coin per timeframe (3 API calls per timeframe)
    2. Combine all data avoiding duplicates
    3. Store locally in data/raw/
    4. Works for all 25 cryptocurrencies and both timeframes (15m, 1h)
    
    For different amounts:
    - 2000 candles: 2 API calls
    - 3000 candles: 3 API calls (recommended)
    - 4000 candles: 4 API calls
    - 5000 candles: 5 API calls
    """
    
    # Create logs directory FIRST
    Path('logs').mkdir(exist_ok=True)
    
    # Now setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_fetcher_batch.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting batch data collection for {len(CRYPTOCURRENCIES)} cryptocurrencies")
    logger.info(f"Data directory: data/raw/")
    logger.info(f"Timeframes: {DATA_CONFIG['timeframes']}")
    
    # Initialize manager
    manager = DataManager(
        data_dir='data/raw',
        timeframes=DATA_CONFIG['timeframes']
    )
    
    # Fetch and store data with batching
    # Change total_limit to 2000, 3000, 4000, 5000 as needed
    results = manager.fetch_and_store_all_batch(
        timeframes=DATA_CONFIG['timeframes'],
        total_limit=3000,  # 3000 candles per coin per timeframe
        batch_delay=1.0    # 1 second between API calls
    )
    
    # Print statistics
    manager.print_statistics()
    
    logger.info("\nBatch data fetching completed!")
    logger.info(f"Check data/raw/ for stored CSV files")
    logger.info(f"Check logs/data_fetcher_batch.log for detailed logs")
