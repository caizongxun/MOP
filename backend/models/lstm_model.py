import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class CryptoLSTM(nn.Module):
    """
    LSTM model for cryptocurrency price prediction
    Optimized for high accuracy with minimal overfitting
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3, output_size=1):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            Output predictions
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last output from LSTM
        last_out = lstm_out[:, -1, :]
        
        # Batch normalization
        last_out = self.batch_norm(last_out)
        
        # Fully connected layers with activation
        fc_out = self.relu(self.fc1(last_out))
        fc_out = self.dropout(fc_out)
        output = self.fc2(fc_out)
        
        return output
    
    def save_model(self, filepath):
        """
        Save model weights
        """
        torch.save(self.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model weights
        """
        self.load_state_dict(torch.load(filepath))
        logger.info(f"Model loaded from {filepath}")
