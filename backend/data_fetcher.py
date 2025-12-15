import logging
from pathlib import Path
from backend.data.data_manager import DataManager
from config.model_config import CRYPTOCURRENCIES, DATA_CONFIG

if __name__ == "__main__":
    """
    Standalone data fetching script
    Run this periodically to fetch and store data for all cryptocurrencies
    
    Usage:
        python backend/data_fetcher.py
    
    Features:
    - Incremental fetching (only adds new data)
    - Stores data locally in data/raw/
    - Supports multiple timeframes (15m, 1h)
    - Auto-creates metadata.json for tracking
    """
    
    # Create logs directory FIRST
    Path('logs').mkdir(exist_ok=True)
    
    # Now setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_fetcher.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting data collection for {len(CRYPTOCURRENCIES)} cryptocurrencies")
    logger.info(f"Data directory: data/raw/")
    logger.info(f"Timeframes: {DATA_CONFIG['timeframes']}")
    
    # Initialize manager
    manager = DataManager(
        data_dir='data/raw',
        timeframes=DATA_CONFIG['timeframes']
    )
    
    # Fetch and store data for all cryptocurrencies
    results = manager.fetch_and_store_all(
        timeframes=DATA_CONFIG['timeframes'],
        limit=DATA_CONFIG['history_limit']
    )
    
    # Print statistics
    manager.print_statistics()
    
    logger.info("\nData fetching completed!")
    logger.info(f"Check data/raw/ for stored CSV files")
