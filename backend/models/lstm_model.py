import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class CryptoLSTM(nn.Module):
    """
    LSTM model for cryptocurrency price prediction
    Optimized for high accuracy with minimal overfitting
    
    Features:
    - Multi-feature input (19 technical indicators)
    - 2-layer LSTM with residual connections
    - Batch normalization and dropout for regularization
    - Configurable input/output sizes
    """
    
    def __init__(self, input_size=19, hidden_size=64, num_layers=2, dropout=0.3, output_size=1, bidirectional=False):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # Feature input processing
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Determine LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, output_size)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"CryptoLSTM initialized: input_size={input_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, dropout={dropout}, bidirectional={bidirectional}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
               where input_size is number of features (19)
        
        Returns:
            Output predictions of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Process input features: (batch_size, seq_length, input_size) -> (batch_size, seq_length, hidden_size)
        # Reshape for batch processing
        x_reshaped = x.reshape(-1, self.input_size)
        x_processed = self.relu(self.input_bn(self.input_linear(x_reshaped)))
        x_processed = x_processed.reshape(batch_size, seq_length, self.hidden_size)
        
        # LSTM forward: (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, lstm_output_size)
        lstm_out, (h_n, c_n) = self.lstm(x_processed)
        
        # Take the last output from LSTM sequence
        last_out = lstm_out[:, -1, :]  # (batch_size, lstm_output_size)
        
        # Batch normalization
        last_out = self.batch_norm(last_out)
        
        # Fully connected layers with residual connection opportunity
        fc_out = self.relu(self.fc1_bn(self.fc1(last_out)))
        fc_out = self.dropout(fc_out)
        
        fc_out = self.relu(self.fc2_bn(self.fc2(fc_out)))
        fc_out = self.dropout(fc_out)
        
        output = self.fc3(fc_out)
        
        return output
    
    def save_model(self, filepath):
        """
        Save model weights and architecture info
        
        Args:
            filepath: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_size': self.output_size,
                'bidirectional': self.bidirectional
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath, map_location='cpu'):
        """
        Load model weights and verify architecture
        
        Args:
            filepath: Path to load model from
            map_location: Device to load to (cpu, cuda, etc.)
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model config: {checkpoint['model_config']}")
    
    def get_model_info(self):
        """
        Get model architecture information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
Model Architecture:
- Input Size: {self.input_size} features
- Hidden Size: {self.hidden_size}
- Num Layers: {self.num_layers}
- Output Size: {self.output_size}
- Bidirectional: {self.bidirectional}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
        """
        return info
