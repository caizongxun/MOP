import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from pathlib import Path
from datetime import datetime
import numpy as np

from config.model_config import MODEL_CONFIG, CRYPTOCURRENCIES, MODEL_FEATURES, DATA_CONFIG
from backend.models.lstm_model import CryptoLSTM
from backend.data.data_loader import CryptoDataLoader
from backend.data.data_manager import DataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trainer for cryptocurrency price prediction models
    Supports:
    - Multi-timeframe training (15m, 1h)
    - 35+ technical indicator features
    - Unified models across timeframes
    - Reading from local storage with API fallback
    """
    
    def __init__(self, config=MODEL_CONFIG, data_config=DATA_CONFIG, use_local_data=True):
        self.config = config
        self.data_config = data_config
        self.use_local_data = use_local_data
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.data_loader = CryptoDataLoader(
            lookback_period=config['lookback'],
            use_binance_api=data_config['use_binance_api']
        )
        
        # Initialize data manager for local storage
        if self.use_local_data:
            self.data_manager = DataManager(
                data_dir='data/raw',
                timeframes=data_config['timeframes']
            )
        else:
            self.data_manager = None
        
        self.models = {}
        self.histories = {}
    
    def get_data(self, symbol, timeframe):
        """
        Get data from local storage or API
        Priority: Local storage -> API
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe (15m, 1h)
        
        Returns:
            DataFrame with OHLCV data
        """
        # Try local storage first
        if self.use_local_data and self.data_manager:
            logger.info(f"Attempting to load {symbol} ({timeframe}) from local storage...")
            data = self.data_manager.get_stored_data(symbol, timeframe)
            if data is not None and not data.empty:
                logger.info(f"Loaded {len(data)} rows from local storage")
                return data
            else:
                logger.info(f"No local data found, falling back to API...")
        
        # Fallback to API
        logger.info(f"Fetching {symbol} ({timeframe}) from API...")
        data = self.data_loader.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=self.data_config['history_limit']
        )
        
        if data is not None and not data.empty:
            # Save to local storage for future use
            if self.use_local_data and self.data_manager:
                self.data_manager.save_data(symbol, timeframe, data)
        
        return data
    
    def prepare_data_multi_features(self, symbol, data):
        """
        Prepare data with technical indicators
        
        Args:
            symbol: Cryptocurrency symbol
            data: Raw OHLCV data
        
        Returns:
            X, y torch tensors with multi-feature input
        """
        # Calculate technical indicators
        data_with_indicators = self.data_loader.calculate_technical_indicators(data)
        
        # Get available features
        available_features = self.data_loader.available_features
        logger.info(f"Available features: {len(available_features)}")
        
        # Normalize all features independently
        normalized, features_used = self.data_loader.normalize_features(
            data_with_indicators,
            features_to_use=available_features
        )
        
        logger.info(f"Using {len(features_used)} features for training")
        
        # Create sequences
        X, y = self.data_loader.create_sequences(
            normalized,
            self.config['lookback'],
            prediction_horizon=self.data_config['prediction_horizon']
        )
        
        # Convert to torch tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
        
        # Return dynamic feature count
        return X, y, len(features_used)
    
    def train_model_single_timeframe(self, symbol, timeframe='15m'):
        """
        Train model for specific cryptocurrency and timeframe
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
            timeframe: Candle timeframe (default: '15m')
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol} - {timeframe}")
        logger.info(f"{'='*60}")
        
        try:
            # Get data from local or API
            data = self.get_data(symbol, timeframe)
            
            if data is None or data.empty:
                logger.warning(f"No data available for {symbol} ({timeframe})")
                return None
            
            logger.info(f"Loaded {len(data)} candles")
            
            # Prepare data with technical indicators
            X, y, num_features = self.prepare_data_multi_features(symbol, data)
            
            if len(X) < 2:
                logger.warning(f"Insufficient data sequences for {symbol} ({timeframe})")
                return None
            
            # Create data loader
            dataset = TensorDataset(X, y)
            train_loader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            # Initialize model with dynamic input size
            model = CryptoLSTM(
                input_size=num_features,  # Dynamic feature count
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                output_size=1
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=False
            )
            
            # Training loop
            history = {'loss': []}
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 20
            
            for epoch in range(self.config['epochs']):
                total_loss = 0
                model.train()
                
                for X_batch, y_batch in train_loader:
                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                history['loss'].append(avg_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config['epochs']}], Loss: {avg_loss:.8f}, Best: {best_loss:.8f}")
                
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save model
            model_dir = Path('backend/models/weights')
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{symbol}_{timeframe}_v1.pt"
            model.save_model(str(model_path))
            
            self.models[f"{symbol}_{timeframe}"] = model
            self.histories[f"{symbol}_{timeframe}"] = history
            
            logger.info(f"Completed {symbol} ({timeframe}). Final loss: {best_loss:.8f}")
            logger.info(f"Model saved to {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error training {symbol} ({timeframe}): {str(e)}", exc_info=True)
            return None
    
    def train_unified_model(self, symbol):
        """
        Train unified model on combined 15m + 1h data
        
        Args:
            symbol: Cryptocurrency symbol
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training unified model for {symbol}")
        logger.info(f"Combining data from: {self.data_config['timeframes']}")
        logger.info(f"{'='*60}")
        
        try:
            # Fetch multi-timeframe data
            all_data = {}
            for timeframe in self.data_config['timeframes']:
                data = self.get_data(symbol, timeframe)
                if data is not None and not data.empty:
                    all_data[timeframe] = data
            
            if not all_data:
                logger.warning(f"No data available for unified training of {symbol}")
                return None
            
            # Combine data from all timeframes
            combined_X = []
            combined_y = []
            
            for timeframe in self.data_config['timeframes']:
                if timeframe in all_data:
                    data = all_data[timeframe]
                    data_with_indicators = self.data_loader.calculate_technical_indicators(data)
                    available_features = self.data_loader.available_features
                    
                    normalized, features_used = self.data_loader.normalize_features(
                        data_with_indicators,
                        features_to_use=available_features
                    )
                    X, y = self.data_loader.create_sequences(
                        normalized,
                        self.config['lookback'],
                        prediction_horizon=self.data_config['prediction_horizon']
                    )
                    combined_X.append(X)
                    combined_y.append(y)
            
            # Concatenate all data
            X = torch.FloatTensor(np.vstack(combined_X)).to(self.device)
            y = torch.FloatTensor(np.hstack(combined_y)).unsqueeze(1).to(self.device)
            
            logger.info(f"Combined data shape - X: {X.shape}, y: {y.shape}")
            
            # Create data loader
            dataset = TensorDataset(X, y)
            train_loader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            # Initialize unified model with dynamic input size
            num_features = X.shape[2]
            model = CryptoLSTM(
                input_size=num_features,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                output_size=1
            ).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=False
            )
            
            # Training
            history = {'loss': []}
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config['epochs']):
                total_loss = 0
                model.train()
                
                for X_batch, y_batch in train_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                history['loss'].append(avg_loss)
                scheduler.step(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config['epochs']}], Loss: {avg_loss:.8f}")
                
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save model
            model_dir = Path('backend/models/weights')
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{symbol}_unified_v1.pt"
            model.save_model(str(model_path))
            
            logger.info(f"Unified model completed for {symbol}")
            logger.info(f"Model saved to {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error training unified model for {symbol}: {str(e)}", exc_info=True)
            return None
    
    def train_all_models(self):
        """
        Train models for all cryptocurrencies
        """
        logger.info(f"Starting training for {len(CRYPTOCURRENCIES)} cryptocurrencies...")
        logger.info(f"Timeframes: {self.data_config['timeframes']}")
        logger.info(f"History: {self.data_config['history_limit']} candles per API call")
        logger.info(f"Features: 35+ dynamic technical indicators")
        logger.info(f"Model config: {self.config}")
        logger.info(f"Local data: {self.use_local_data}")
        
        results = {}
        
        for symbol in CRYPTOCURRENCIES:
            symbol_results = {}
            
            # Train individual timeframe models
            for timeframe in self.data_config['timeframes']:
                model = self.train_model_single_timeframe(symbol, timeframe)
                symbol_results[timeframe] = 'Success' if model is not None else 'Failed'
            
            # Train unified model
            unified_model = self.train_unified_model(symbol)
            symbol_results['unified'] = 'Success' if unified_model is not None else 'Failed'
            
            results[symbol] = symbol_results
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Training Summary")
        logger.info(f"{'='*60}")
        for symbol, status_dict in results.items():
            logger.info(f"\n{symbol}:")
            for timeframe, status in status_dict.items():
                logger.info(f"  {timeframe}: {status}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total models trained: {sum(1 for s in results.values() for st in s.values() if st == 'Success')}")
        
        # Print data statistics
        if self.data_manager:
            self.data_manager.print_statistics()

if __name__ == "__main__":
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Use local data if available, fallback to API
    trainer = ModelTrainer(use_local_data=True)
    trainer.train_all_models()
