# Unified Path Configuration Guide

## Overview

All scripts in the MOP project now use a unified `PathConfig` module to manage directory paths. This eliminates path-related errors regardless of where scripts are run from.

## Problem Solved

Before:
- Scripts failed when run from different locations
- Relative paths broke depending on working directory
- Different scripts used different path logic
- Hard-coded absolute paths on specific machines

After:
- Single source of truth for all paths
- Works from any location (root, backend, anywhere)
- Consistent across all scripts
- Automatic directory detection

## Usage

### Basic Usage in Any Script

```python
from backend.path_config import PathConfig

# Initialize paths
paths = PathConfig()

# Access directories
print(paths.root)                          # Project root
print(paths.backend)                       # backend/ directory
print(paths.data_raw_dir)                  # backend/data/raw
print(paths.models_weights_dir)            # backend/models/weights
print(paths.results_visualizations_dir)    # backend/results/visualizations

# Get specific file paths
data_file = paths.get_data_file('BTCUSDT', '1h')
model_file = paths.get_model_weights_file('BTCUSDT', '1h', version='v5')
config_file = paths.get_model_config_file('BTCUSDT', version='v4')
log_file = paths.get_log_file('training')
```

### Directory Structure

```
MOP/
├── backend/
│   ├── data/
│   │   └── raw/              # paths.data_raw_dir
│   ├── models/
│   │   ├── weights/          # paths.models_weights_dir
│   │   └── config/           # paths.models_config_dir
│   ├── results/
│   │   └── visualizations/   # paths.results_visualizations_dir
│   ├── logs/                 # paths.logs_dir
│   └── path_config.py        # THE MODULE
├── quick_visualize_v5.py     # Uses PathConfig
├── train_v5.py               # Uses PathConfig
└── ...
```

## PathConfig Properties

### Root & Backend

```python
paths.root                      # Project root directory
paths.backend                   # backend/ directory
```

### Data

```python
paths.data_dir                  # backend/data
paths.data_raw_dir              # backend/data/raw
paths.data_processed_dir        # backend/data/processed
```

### Models

```python
paths.models_dir                # backend/models
paths.models_weights_dir        # backend/models/weights
paths.models_config_dir         # backend/models/config
```

### Results & Logs

```python
paths.results_dir               # backend/results
paths.results_visualizations_dir # backend/results/visualizations
paths.logs_dir                  # backend/logs
```

## File Path Methods

### Get Data File

```python
# Get path to cryptocurrency data
path = paths.get_data_file('BTCUSDT', timeframe='1h')
# Returns: backend/data/raw/BTCUSDT_1h.csv
```

### Get Model Weights File

```python
# V5 model
path = paths.get_model_weights_file('BTCUSDT', '1h', version='v5')
# Returns: backend/models/weights/BTCUSDT_1h_v5_lstm.pth

# V4 model (default)
path = paths.get_model_weights_file('BTCUSDT', '1h')
# Returns: backend/models/weights/BTCUSDT_1h_v4_lstm.pth
```

### Get Model Config File

```python
path = paths.get_model_config_file('BTCUSDT', version='v4')
# Returns: backend/models/config/BTCUSDT_v4_config.json
```

### Get XGBoost Model File

```python
path = paths.get_xgb_model_file('BTCUSDT', '1h', version='v4')
# Returns: backend/models/weights/BTCUSDT_1h_v4_xgb.json
```

### Get Log File

```python
path = paths.get_log_file('training')
# Returns: backend/logs/training.log

path = paths.get_log_file('evaluation')
# Returns: backend/logs/evaluation.log
```

## Utility Methods

### Verify Paths

```python
status = paths.verify()
print(status)
# Output:
# {
#     'root': True,
#     'backend': True,
#     'data_raw': True,
#     'models': True,
#     'models_weights': True,
#     'models_config': True,
#     'results': True,
#     'results_viz': True,
#     'logs': True
# }
```

### Print Summary

```python
paths.print_summary()
# Prints formatted path configuration and directory status
```

## Examples by Script Type

### Training Script

```python
from backend.path_config import PathConfig
import torch

paths = PathConfig()

# Load data
data_file = paths.get_data_file(symbol='BTCUSDT')
df = pd.read_csv(data_file)

# Save model
model_path = paths.get_model_weights_file('BTCUSDT', '1h', version='v5')
torch.save(model.state_dict(), model_path)

# Save logs
log_file = paths.get_log_file('training')
with open(log_file, 'a') as f:
    f.write(log_message)
```

### Visualization Script

```python
from backend.path_config import PathConfig
import matplotlib.pyplot as plt

paths = PathConfig()

# Load model
model_path = paths.get_model_weights_file('BTCUSDT', '1h', version='v5')
model.load_state_dict(torch.load(model_path))

# Save visualization
results_dir = paths.results_visualizations_dir
plt.savefig(f'{results_dir}/plot.png')

# Save results CSV
df.to_csv(f'{results_dir}/summary.csv')
```

### Inference/Evaluation Script

```python
from backend.path_config import PathConfig

paths = PathConfig()

# Load data
data_file = paths.get_data_file('ETHUSDT', '1h')
df = pd.read_csv(data_file)

# Load model
model_file = paths.get_model_weights_file('ETHUSDT', '1h', version='v5')

# Save results
results_file = f"{paths.results_dir}/evaluation_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f)
```

## Running Scripts

All these work correctly now (no path errors):

```bash
# From project root
cd C:\Users\zong\Desktop\MOP
python quick_visualize_v5.py --all --device cuda
python train_v5.py 5 --device cuda

# From backend directory
cd C:\Users\zong\Desktop\MOP\backend
python train_v5_enhanced.py

# From anywhere (if added to Python path)
python C:\Users\zong\Desktop\MOP\quick_visualize_v5.py --all
```

## Debugging Paths

### Check Path Configuration

```bash
# Show all configured paths
python quick_visualize_v5.py --show-paths
```

### Verify in Script

```python
from backend.path_config import PathConfig

paths = PathConfig()
paths.print_summary()  # Shows all paths and their status
```

### Test Path Resolution

```python
from backend.path_config import PathConfig

paths = PathConfig()

# Check if files exist
data_file = paths.get_data_file('BTCUSDT')
print(f"Data file exists: {os.path.exists(data_file)}")

model_file = paths.get_model_weights_file('BTCUSDT', version='v5')
print(f"Model file exists: {os.path.exists(model_file)}")
```

## Adding New Path Types

If you need to add a new path type, edit `backend/path_config.py`:

```python
class PathConfig:
    # Add new property
    @property
    def new_dir(self):
        """New directory path"""
        return os.path.join(self.backend_dir, 'new_dir')
    
    # Or add getter method
    def get_new_file(self, name: str) -> str:
        """Get new file path"""
        return os.path.join(self.new_dir, f'{name}.ext')
```

## Common Path Errors - FIXED

### Before (Error):
```python
# Hard-coded path - breaks on different machine
model_path = 'C:\\Users\\zong\\Desktop\\MOP\\models\\BTCUSDT.pth'

# Relative path - breaks if run from different directory
model_path = './models/weights/BTCUSDT.pth'
```

### After (Works):
```python
from backend.path_config import PathConfig
paths = PathConfig()
model_path = paths.get_model_weights_file('BTCUSDT')
# Works from anywhere!
```

## Migration Guide

If you have old scripts, update them:

```python
# Old
import os
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'weights')
model_path = os.path.join(MODELS_DIR, 'BTCUSDT.pth')

# New
from backend.path_config import PathConfig
paths = PathConfig()
model_path = paths.get_model_weights_file('BTCUSDT')
```

## Summary

- Use `PathConfig()` for all directory access
- No more manual `os.path.join()` for standard paths
- Automatic directory detection
- Works from any location
- Consistent across all scripts
- One source of truth

All future scripts should use this pattern!
