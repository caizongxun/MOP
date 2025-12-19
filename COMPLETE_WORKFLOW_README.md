# MOP 完整流程指南 - 从数据获取到 Hugging Face 上传

本指南包含所有需要在终端输入的命令，从零开始到完成上传。

---

## 目录

1. [前置准备](#前置准备)
2. [流程 1: 获取数据](#流程-1-获取数据)
3. [流程 2: 训练 V5 模型](#流程-2-训练-v5-模型)
4. [流程 3: 上传到 Hugging Face](#流程-3-上传到-hugging-face)
5. [完整工作流示例](#完整工作流示例)
6. [故障排除](#故障排除)

---

## 前置准备

### 1. 克隆项目

```bash
git clone https://github.com/caizongxun/MOP.git
cd MOP
```

### 2. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Hugging Face CLI (用于上传)
pip install huggingface-hub>=0.20.0
```

### 3. 创建 Hugging Face 账户（如果还没有）

访问 https://huggingface.co/join

### 4. 获取 Hugging Face Token

访问 https://huggingface.co/settings/tokens
- 点击 "New token"
- 选择 "Write" 权限
- 复制 token 值（格式: `hf_xxxxxxxxxxxxxxxxxxxxxxx`）

### 5. 创建 Hugging Face 仓库

访问 https://huggingface.co/new
- Repository name: `mop-v5-models` (或你喜欢的名字)
- License: `MIT`
- 完成后获得仓库 ID (格式: `username/repo-name`)

---

## 流程 1: 获取数据

### 选项 A: 获取所有 15 个 symbols 的数据

**耗时**: 约 10-15 分钟

```bash
python backend/data_fetcher_batch.py --all
```

### 选项 B: 获取特定 symbols 的数据

```bash
python backend/data_fetcher_batch.py --symbols BTCUSDT ETHUSDT BNBUSDT
```

### 选项 C: 快速测试（仅 5 个 symbols）

```bash
python backend/data_fetcher_batch.py --limit 5
```

### 验证数据已下载

```bash
ls -la data/raw/
```

应该看到类似的文件（每个 symbol 有两个时间框架的数据）：
```
BTCUSDT_1h.csv
BTCUSDT_15m.csv
ETHUSDT_1h.csv
ETHUSDT_15m.csv
...
```

---

## 流程 2: 训练 V5 模型

### 选项 A: 训练所有 15 个 symbols（推荐）

**耗时**: 约 45-60 分钟（取决于 GPU）

```bash
python train_v5.py --all --device cuda
```

### 选项 B: 训练特定 symbols

```bash
python train_v5.py --symbols BTCUSDT ETHUSDT BNBUSDT --device cuda
```

### 选项 C: 快速测试（训练 5 个 symbols）

**耗时**: 约 15-20 分钟

```bash
python train_v5.py 5 --device cuda
```

### 选项 D: 使用 CPU（如果没有 GPU）

```bash
python train_v5.py --all --device cpu
```

### 验证模型已训练

```bash
ls models/weights/*v5*.pth
```

应该看到类似的文件（每个 symbol 有一个 .pth 模型文件）：
```
BTCUSDT_1h_v5_lstm.pth
ETHUSDT_1h_v5_lstm.pth
BNBUSDT_1h_v5_lstm.pth
...
```

### 检查训练配置

```bash
ls models/config/*v5*.json
```

应该看到配置文件（包含模型性能指标）：
```
BTCUSDT_v5_config.json
ETHUSDT_v5_config.json
BNBUSDT_v5_config.json
...
```

---

## 流程 3: 上传到 Hugging Face

### 重要：设置环境变量（推荐）

设置一次后，后续命令可以直接使用 `$env:HF_TOKEN` 等

#### Linux/Mac:

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export HF_REPO="your_username/mop-v5-models"
```

#### Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
$env:HF_REPO="your_username/mop-v5-models"
```

#### Windows CMD:

```cmd
set HF_TOKEN=hf_YOUR_TOKEN_HERE
set HF_REPO=your_username/mop-v5-models
```

### 方式 1: 一行命令（快速）

#### 推送所有模型

```bash
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN_HERE" --repo "your_username/mop-v5-models" --all
```

#### 推送特定 symbols

```bash
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN_HERE" --repo "your_username/mop-v5-models" --symbols BTCUSDT ETHUSDT
```

### 方式 2: 使用环境变量（推荐）

先设置环境变量（见上面），然后：

```bash
python push_to_huggingface_batch.py --token $env:HF_TOKEN --repo $env:HF_REPO --all
```

### 方式 3: 使用反引号换行（Windows PowerShell 可读版本）

```powershell
python push_to_huggingface_batch.py `
  --token "hf_YOUR_TOKEN_HERE" `
  --repo "your_username/mop-v5-models" `
  --all
```

### 验证上传成功

访问你的 Hugging Face 仓库：

```
https://huggingface.co/your_username/mop-v5-models
```

应该看到 15 个文件夹（每个 symbol 一个），每个文件夹包含：
- `pytorch_model.bin` - 训练好的模型
- `config.json` - 配置和性能指标
- `README.md` - 自动生成的说明

---

## 完整工作流示例

### 场景：在新设备上从零开始

#### 第 1 步：准备环境

```bash
# 克隆项目
git clone https://github.com/caizongxun/MOP.git
cd MOP

# 安装依赖
pip install -r requirements.txt
pip install huggingface-hub>=0.20.0

# 设置环境变量 (Linux/Mac)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export HF_REPO="your_username/mop-v5-models"
```

#### 第 2 步：获取数据

```bash
# 获取所有 15 个 symbols 的数据（约 10-15 分钟）
python backend/data_fetcher_batch.py --all

# 验证数据
ls data/raw/ | wc -l  # 应该显示 30+ 个文件
```

#### 第 3 步：训练模型

```bash
# 训练所有 15 个 V5 模型（约 45-60 分钟）
python train_v5.py --all --device cuda

# 验证模型
ls models/weights/*v5*.pth | wc -l  # 应该显示 15 个文件
ls models/config/*v5*.json | wc -l   # 应该显示 15 个文件
```

#### 第 4 步：上传到 Hugging Face

```bash
# 一次性批量上传（约 2-5 分钟）
python push_to_huggingface_batch.py --token $env:HF_TOKEN --repo $env:HF_REPO --all

# 或者直接用 token
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN_HERE" --repo "your_username/mop-v5-models" --all
```

#### 完成！

访问 https://huggingface.co/your_username/mop-v5-models 查看结果

### 总耗时

- 数据获取: 10-15 分钟
- 模型训练: 45-60 分钟
- 上传 HF: 2-5 分钟
- **总计: 约 60-80 分钟**

---

## 快速命令速查

### Linux/Mac

```bash
# 获取数据
python backend/data_fetcher_batch.py --all

# 训练模型
python train_v5.py --all --device cuda

# 设置环境变量
export HF_TOKEN="hf_YOUR_TOKEN"
export HF_REPO="username/repo"

# 上传到 HF
python push_to_huggingface_batch.py --token $HF_TOKEN --repo $HF_REPO --all
```

### Windows PowerShell

```powershell
# 获取数据
python backend/data_fetcher_batch.py --all

# 训练模型
python train_v5.py --all --device cuda

# 设置环境变量
$env:HF_TOKEN="hf_YOUR_TOKEN"
$env:HF_REPO="username/repo"

# 上传到 HF (一行)
python push_to_huggingface_batch.py --token $env:HF_TOKEN --repo $env:HF_REPO --all
```

---

## 故障排除

### 问题 1: "ModuleNotFoundError: No module named 'xxx'"

**解决**:
```bash
pip install -r requirements.txt
```

### 问题 2: "CUDA out of memory"

**解决** (使用 CPU 或减少数量):
```bash
python train_v5.py 5 --device cpu  # 只训练 5 个，用 CPU
```

### 问题 3: "Cannot connect to internet" (获取数据失败)

**解决**:
- 检查网络连接
- 确保能访问 Binance API
- 重试命令

### 问题 4: "Token not recognized" (上传失败)

**解决**:
- 验证 token 是否正确 (https://huggingface.co/settings/tokens)
- 检查 token 是否有 "Write" 权限
- 检查环境变量是否正确设置

### 问题 5: "Repository not found" (上传失败)

**解决**:
- 确保仓库已创建 (https://huggingface.co/new)
- 检查仓库 ID 是否正确 (格式: `username/repo-name`)
- 检查 token 权限

### 问题 6: Windows PowerShell 中反斜杠导致错误

**错误信息**: `unrecognized arguments: \ \ \`

**解决** (三种方法):

方法 1 - 一行写完:
```powershell
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN" --repo "username/repo" --all
```

方法 2 - 使用反引号:
```powershell
python push_to_huggingface_batch.py `
  --token "hf_YOUR_TOKEN" `
  --repo "username/repo" `
  --all
```

方法 3 - 使用环境变量:
```powershell
$env:HF_TOKEN="hf_YOUR_TOKEN"
python push_to_huggingface_batch.py --token $env:HF_TOKEN --repo "username/repo" --all
```

---

## 参数说明

### 数据获取参数

```bash
python backend/data_fetcher_batch.py [OPTIONS]
```

- `--all` - 获取所有 15 个 symbols
- `--symbols BTCUSDT ETHUSDT ...` - 获取特定 symbols
- `--limit N` - 只获取前 N 个 symbols

### 训练参数

```bash
python train_v5.py [OPTIONS] [NUM_SYMBOLS]
```

- `NUM_SYMBOLS` - 训练前 N 个 symbols (可选，默认 15)
- `--all` - 训练所有 symbols
- `--symbols BTCUSDT ETHUSDT ...` - 训练特定 symbols
- `--device cuda/cpu` - 使用 GPU 或 CPU (默认 cuda)

### 上传参数

```bash
python push_to_huggingface_batch.py [OPTIONS]
```

- `--token TOKEN` - Hugging Face API token (必需)
- `--repo REPO` - 仓库 ID (必需，格式: `username/repo-name`)
- `--all` - 上传所有模型
- `--symbols BTCUSDT ETHUSDT ...` - 上传特定 symbols

---

## 支持的 Symbols

以下 15 个 symbols 都支持：

```
BTCUSDT    ETHUSDT    BNBUSDT    ADAUSDT    XRPUSDT
SOLUSDT    DOGEUSDT   AVAXUSDT   MATICUSDT  LINKUSDT
LTCUSDT    NEARUSDT   ATOMUSDT   UNIUSDT    APTUSDT
```

---

## 性能指标参考

### 数据获取
- 单个 symbol: 1-2 分钟
- 15 个 symbols: 10-15 分钟
- 网络需求: 稳定连接

### 模型训练
- 单个 symbol (GPU): 2-4 分钟
- 15 个 symbols (GPU): 45-60 分钟
- 15 个 symbols (CPU): 3-5 小时
- GPU 需求: NVIDIA GPU 8GB+ 显存

### 上传到 Hugging Face
- 单个 model: 10-30 秒
- 15 个 models: 2-5 分钟
- 网络需求: 稳定 1-2 Mbps
- 总大小: ~750 MB

---

## 下一步

1. **在其他项目中使用**
   ```python
   from huggingface_hub import hf_hub_download
   import torch
   
   model_path = hf_hub_download(
       repo_id="your_username/mop-v5-models",
       filename="BTCUSDT/pytorch_model.bin"
   )
   
   model = torch.load(model_path)
   model.eval()
   ```

2. **重新训练和更新**
   - 修改代码后重新训练
   - 运行相同的上传命令
   - 新版本自动覆盖旧版本

3. **分享模型**
   - 共享链接: https://huggingface.co/your_username/mop-v5-models
   - 邀请协作者
   - 查看社区讨论

---

## 相关文档

- [Hugging Face 批量推送指南](./HUGGINGFACE_BATCH_PUSH_GUIDE.md)
- [快速开始 V5](./V5_QUICK_START.md)
- [项目架构](./models_architecture.json)

---

## 支持

如有问题：

1. 查看本文档的"故障排除"部分
2. 检查 GitHub Issues
3. 查看日志文件 (logs/ 目录)

---

**祝您使用愉快！**
