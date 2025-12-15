# MOP 安裝指南

## 快速開始

### Step 1: 安裝依賴套件

```bash
pip install -r backend/requirements.txt
```

### Step 2: 配置環境變數

複製並編輯 `.env` 檔案：

```bash
cp .env.example .env
```

編輯 `.env` 並設定：
```
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
HF_TOKEN=your_huggingface_token_here
DISCORD_BOT_TOKEN=your_discord_token_here
```

### Step 3: 驗證安裝

```bash
python backend/train.py --help
```

---

## 完整依賴清單

### 核心機器學習框架

| 套件 | 版本 | 用途 |
|------|------|------|
| torch | 2.1.0 | PyTorch深度學習框架（需CUDA支持） |
| torchvision | 0.16.0 | 圖像處理工具 |
| torchaudio | 0.16.0 | 音頻處理工具 |

### 數據處理

| 套件 | 版本 | 用途 |
|------|------|------|
| numpy | 1.24.3 | 數值計算 |
| pandas | 2.0.3 | 數據操作和分析 |
| scikit-learn | 1.3.0 | 機器學習工具 |
| ta | 0.10.2 | 技術指標計算 |

### 加密貨幣數據

| 套件 | 版本 | 用途 |
|------|------|------|
| ccxt | 2.88.75 | 加密貨幣交易所API |
| requests | 2.31.0 | HTTP請求庫 |

### 模型部署和配置

| 套件 | 版本 | 用途 |
|------|------|------|
| huggingface-hub | 0.17.0 | Hugging Face模型庫 |
| python-dotenv | 1.0.0 | 環境變數管理 |
| redis | 5.0.0 | 緩存和隊列（可選） |

### 可選套件（用於可視化）

```bash
# 可視化依賴（強烈推薦）
pip install matplotlib>=3.5.0

# 高級繪圖
pip install seaborn>=0.12.0
pip install plotly>=5.0.0

# Jupyter支持
pip install jupyter>=1.0.0
```

---

## GPU 支持

### CUDA 支持版本

**強烈推薦使用 GPU 加速訓練**

#### Windows + NVIDIA GPU

```bash
# 安裝 CUDA 工具包（需要NVIDIA GPU）
# 訪問: https://developer.nvidia.com/cuda-downloads
# 建議版本: CUDA 11.8+

# 安裝 PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### macOS / Linux (無CUDA)

```bash
# CPU 版本
pip install torch torchvision torchaudio
```

### 驗證 GPU 支持

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 數量: {torch.cuda.device_count()}")
print(f"當前GPU: {torch.cuda.get_device_name(0)}")
```

---

## 完整安裝命令

### 方法1: 基礎安裝（推薦）

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安裝所有依賴
pip install -r backend/requirements.txt

# 安裝可視化支持
pip install matplotlib seaborn
```

### 方法2: GPU 加速（推薦用於訓練）

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate  # Windows

# 安裝 PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r backend/requirements.txt
pip install matplotlib seaborn
```

### 方法3: 完整開發環境

```bash
# 安裝所有依賴 + 開發工具
pip install -r backend/requirements.txt
pip install matplotlib seaborn plotly jupyter
pip install black flake8 pytest  # 代碼質量
```

---

## 驗證安裝

```bash
# 檢查 Python 版本
python --version  # 應該是 3.8+

# 檢查 PyTorch
python -c "import torch; print(torch.__version__)"

# 檢查 TensorFlow（可選）
python -c "import pandas; print(pandas.__version__)"

# 檢查 CUDA（如有GPU）
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 故障排除

### 問題1: ModuleNotFoundError

```bash
# 確認虛擬環境啟動
which python  # macOS/Linux
where python  # Windows

# 重新安裝依賴
pip install --upgrade -r backend/requirements.txt
```

### 問題2: GPU 不被識別

```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 重新安裝 PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 問題3: 導入錯誤

```bash
# 清除 Python 快取
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# 重新安裝
pip install --force-reinstall -r backend/requirements.txt
```

---

## 系統需求

### 最低配置
- Python 3.8+
- 4GB RAM
- 10GB 硬盤空間
- CPU: Intel i5/AMD Ryzen 5+

### 推薦配置
- Python 3.10+
- 16GB RAM
- 50GB SSD 硬盤空間
- NVIDIA GPU with CUDA 支持 (16GB+ VRAM)
- CPU: Intel i7/AMD Ryzen 7+

---

## 環境變數配置

### .env 文件模板

```env
# Binance API
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Hugging Face
HF_TOKEN=your_huggingface_token

# Discord（可選）
DISCORD_BOT_TOKEN=your_discord_bot_token

# Redis（可選）
REDIS_HOST=localhost
REDIS_PORT=6379

# 訓練配置
USE_GPU=true
DEVICE=cuda  # 或 cpu
```

---

## 下一步

1. **數據收集**: `python backend/data_fetcher_batch.py`
2. **模型訓練**: `python backend/train.py`
3. **模型評估**: `python backend/quick_eval.py`
4. **實時預測**: `python backend/inference.py`

---

## 支持和幫助

- GitHub Issues: https://github.com/caizongxun/MOP/issues
- 文檔: https://github.com/caizongxun/MOP/blob/main/README.md
