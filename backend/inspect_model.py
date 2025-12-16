import os
import sys
import torch
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.model_multi_timeframe import MultiTimeframeFusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelInspector:
    def __init__(self, symbol='BTCUSDT', model_dir='models'):
        self.symbol = symbol
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # Load model
        self.model = MultiTimeframeFusion().to(self.device)
        model_path = os.path.join(model_dir, f'{symbol}_multi_timeframe.pth')
        
        if not os.path.exists(model_path):
            logger.error(f'Model not found: {model_path}')
            raise FileNotFoundError(f'Model not found: {model_path}')
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f'Model loaded from {model_path}')
    
    def print_architecture(self):
        """Print model architecture"""
        logger.info('\n' + '='*80)
        logger.info('MODEL ARCHITECTURE')
        logger.info('='*80)
        logger.info(self.model)
    
    def print_parameters(self):
        """Print model parameters and their shapes"""
        logger.info('\n' + '='*80)
        logger.info('MODEL PARAMETERS')
        logger.info('='*80)
        
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            
            logger.info(f'{name:50s} | Shape: {str(param.shape):20s} | Params: {num_params:>10,d} | Trainable: {param.requires_grad}')
        
        logger.info('\n' + '-'*80)
        logger.info(f'Total Parameters: {total_params:,d}')
        logger.info(f'Trainable Parameters: {trainable_params:,d}')
        logger.info(f'Non-trainable Parameters: {total_params - trainable_params:,d}')
        logger.info('='*80)
    
    def print_layer_details(self):
        """Print detailed information about each layer"""
        logger.info('\n' + '='*80)
        logger.info('LAYER DETAILS')
        logger.info('='*80)
        
        for name, module in self.model.named_modules():
            if name == '':  # Skip root module
                continue
            
            if isinstance(module, torch.nn.Sequential):
                logger.info(f'\n{name}: Sequential')
                for idx, layer in enumerate(module):
                    logger.info(f'  [{idx}] {layer.__class__.__name__}: {layer}')
            
            elif isinstance(module, (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN)):
                logger.info(f'\n{name}: {module.__class__.__name__}')
                logger.info(f'  Input size: {module.input_size}')
                logger.info(f'  Hidden size: {module.hidden_size}')
                logger.info(f'  Num layers: {module.num_layers}')
                logger.info(f'  Batch first: {module.batch_first}')
                logger.info(f'  Dropout: {module.dropout}')
                logger.info(f'  Bidirectional: {module.bidirectional}')
            
            elif isinstance(module, torch.nn.Linear):
                logger.info(f'\n{name}: Linear')
                logger.info(f'  Input: {module.in_features}, Output: {module.out_features}')
                if module.bias is not None:
                    logger.info(f'  Bias: Yes')
            
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                logger.info(f'\n{name}: {module.__class__.__name__}')
                logger.info(f'  Num features: {module.num_features}')
                logger.info(f'  Momentum: {module.momentum}')
                logger.info(f'  Eps: {module.eps}')
            
            elif isinstance(module, torch.nn.Dropout):
                logger.info(f'\n{name}: Dropout')
                logger.info(f'  Dropout rate: {module.p}')
    
    def print_weight_statistics(self):
        """Print statistics about weights"""
        logger.info('\n' + '='*80)
        logger.info('WEIGHT STATISTICS')
        logger.info('='*80)
        
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'bias' in name:
                data = param.data.cpu().numpy()
                logger.info(f'\n{name}')
                logger.info(f'  Mean: {data.mean():.6f}')
                logger.info(f'  Std: {data.std():.6f}')
                logger.info(f'  Min: {data.min():.6f}')
                logger.info(f'  Max: {data.max():.6f}')
                logger.info(f'  Shape: {data.shape}')
    
    def test_forward_pass(self, input_1h_shape=(1, 60, 44), input_15m_shape=(1, 240, 44)):
        """Test forward pass with dummy data"""
        logger.info('\n' + '='*80)
        logger.info('FORWARD PASS TEST')
        logger.info('='*80)
        
        try:
            # Create dummy inputs
            x_1h = torch.randn(input_1h_shape).to(self.device)
            x_15m = torch.randn(input_15m_shape).to(self.device)
            
            logger.info(f'\nInput shapes:')
            logger.info(f'  1h: {x_1h.shape}')
            logger.info(f'  15m: {x_15m.shape}')
            
            # Forward pass
            with torch.no_grad():
                output = self.model(x_1h, x_15m)
            
            logger.info(f'\nOutput shape: {output.shape}')
            logger.info(f'Output values (first 5): {output.squeeze()[:5].cpu().numpy()}')
            logger.info(f'Output range: [{output.min():.6f}, {output.max():.6f}]')
            logger.info(f'Output mean: {output.mean():.6f}')
            logger.info(f'\nForward pass successful!')
            
            return True
        
        except Exception as e:
            logger.error(f'Forward pass failed: {str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """Print complete model summary"""
        logger.info('\n' + '='*80)
        logger.info('MODEL SUMMARY')
        logger.info('='*80)
        logger.info(f'Symbol: {self.symbol}')
        logger.info(f'Device: {self.device}')
        logger.info(f'Model Type: MultiTimeframeFusion')
        logger.info(f'Training Status: Evaluation mode')
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f'\nTotal Parameters: {total_params:,d}')
        logger.info(f'Trainable Parameters: {trainable_params:,d}')
        logger.info(f'Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)')
        
        # Model info
        logger.info(f'\nInput Specifications:')
        logger.info(f'  1h Timeframe: (batch, 60 candles, 44 features)')
        logger.info(f'  15m Timeframe: (batch, 240 candles, 44 features)')
        logger.info(f'Output: Single price prediction value (normalized)')
        logger.info('='*80)
    
    def inspect_all(self):
        """Run all inspection functions"""
        self.print_summary()
        self.print_architecture()
        self.print_parameters()
        self.print_layer_details()
        self.print_weight_statistics()
        self.test_forward_pass()

def main():
    parser = argparse.ArgumentParser(description='Inspect trained model')
    parser.add_argument('--symbol', default='BTCUSDT', help='Model symbol')
    parser.add_argument('--model-dir', default='models', help='Model directory')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--architecture', action='store_true', help='Show architecture only')
    parser.add_argument('--parameters', action='store_true', help='Show parameters only')
    parser.add_argument('--weights', action='store_true', help='Show weight statistics only')
    parser.add_argument('--test', action='store_true', help='Test forward pass only')
    parser.add_argument('--all', action='store_true', help='Show all information (default)')
    
    args = parser.parse_args()
    
    try:
        inspector = ModelInspector(symbol=args.symbol, model_dir=args.model_dir)
        
        # If specific option selected, run only that
        if args.summary:
            inspector.print_summary()
        elif args.architecture:
            inspector.print_architecture()
        elif args.parameters:
            inspector.print_parameters()
        elif args.weights:
            inspector.print_weight_statistics()
        elif args.test:
            inspector.test_forward_pass()
        else:
            # Default: show all
            inspector.inspect_all()
    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
