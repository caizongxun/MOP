# MOP 完整工作流 - 三步快速开始

从零开始：**数据获取** -> **训练** -> **上传 Hugging Face**

需要的时间：**约 60-80 分钟**

---

## 前置准备（仅一次）

### 1. 克隆项目

```bash
git clone https://github.com/caizongxun/MOP.git
cd MOP
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install huggingface-hub>=0.20.0
```

### 3. 准备 Hugging Face

- 账户: https://huggingface.co/join
- Token: https://huggingface.co/settings/tokens (选择 Write 权限)
- 仓库: https://huggingface.co/new (名称: `mop-v5-models`)

---

## 流程 1: 数据获取

**耗时: 10-15 分钟**

```bash
python backend/data_fetcher_batch.py --all
```

### 验证成功

```bash
ls data/raw/ | wc -l  # 应该显示 30+ 个文件
```

---

## 流程 2: 训练 V5 模型

**耗时: 45-60 分钟 (GPU) 或 3-5 小时 (CPU)**

```bash
python train_v5.py --all --device cuda
```

如果没有 GPU：

```bash
python train_v5.py --all --device cpu
```

### 验证成功

```bash
ls models/weights/*v5*.pth | wc -l    # 应该显示 15
ls models/config/*v5*.json | wc -l    # 应该显示 15
```

---

## 流程 3: 上传到 Hugging Face

**耗时: 2-5 分钟**

### 方法 A: 一行命令 (最简单)

```bash
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN_HERE" --repo "your_username/mop-v5-models" --all
```

替换：
- `hf_YOUR_TOKEN_HERE` -> 你的 HF token
- `your_username` -> 你的 HF 用户名

### 方法 B: 使用环境变量 (推荐)

#### Linux/Mac:

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export HF_REPO="your_username/mop-v5-models"
python push_to_huggingface_batch.py --token $HF_TOKEN --repo $HF_REPO --all
```

#### Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
$env:HF_REPO="your_username/mop-v5-models"
python push_to_huggingface_batch.py --token $env:HF_TOKEN --repo $env:HF_REPO --all
```

#### Windows CMD:

```cmd
set HF_TOKEN=hf_YOUR_TOKEN_HERE
set HF_REPO=your_username/mop-v5-models
python push_to_huggingface_batch.py --token %HF_TOKEN% --repo %HF_REPO% --all
```

### 验证成功

访问: https://huggingface.co/your_username/mop-v5-models

应该看到 15 个文件夹，每个文件夹包含：
- `pytorch_model.bin` - 模型
- `config.json` - 配置
- `README.md` - 说明

---

## 快速图

| 方步 | 命令 | 耗时 |
|------|--------|--------|
| 数据获取 | `python backend/data_fetcher_batch.py --all` | 10-15m |
| 模型训练 | `python train_v5.py --all --device cuda` | 45-60m |
| 上传 HF | `python push_to_huggingface_batch.py ...` | 2-5m |
| **总计** | | **60-80m** |

---

## 一江水命令 (不推荐，仅检查使用)

Linux/Mac:

```bash
git clone https://github.com/caizongxun/MOP.git && cd MOP && pip install -r requirements.txt && pip install huggingface-hub>=0.20.0 && python backend/data_fetcher_batch.py --all && python train_v5.py --all --device cuda && export HF_TOKEN="hf_YOUR_TOKEN" && python push_to_huggingface_batch.py --token $HF_TOKEN --repo "your_username/mop-v5-models" --all
```

---

## 常见问题

**Q: 没有 GPU?**

A: 使用 CPU，但是会很慢 (3-5 小时)
```bash
python train_v5.py --all --device cpu
```

**Q: 上传失败?**

A: 脚本会自动重试 3 次，稍后重新运行

**Q: Windows PowerShell 反斜杠错误?**

A: 一行写完或用环境变量

**Q: 仅训练部分 symbols?**

A: 
```bash
python train_v5.py 5 --device cuda  # 前 5 个
python train_v5.py --symbols BTCUSDT ETHUSDT --device cuda  # 指定的
```

---

## 支持的 Symbols (15 个)

```
BTCUSDT   ETHUSDT   BNBUSDT   ADAUSDT   XRPUSDT
SOLUSDT   DOGEUSDT  AVAXUSDT  MATICUSDT LINKUSDT
LTCUSDT   NEARUSDT  ATOMUSDT  UNIUSDT   APTUSDT
```

---

## 开始使用

模型上传后，可以在其他项目中使用：

```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(
    repo_id="your_username/mop-v5-models",
    filename="BTCUSDT/pytorch_model.bin"
)

model = torch.load(model_path)
model.eval()

# 预测
with torch.no_grad():
    output = model(features)
```

---

## 更详细的文档

- [**完整工作流指南**](COMPLETE_WORKFLOW_README.md) - 带最全的终端命令
- [**Hugging Face 批量推送**](HUGGINGFACE_BATCH_PUSH_GUIDE.md) - 上传指南
- [**项目 README**](README.md) - 项目简介

---

**现在开始！**
