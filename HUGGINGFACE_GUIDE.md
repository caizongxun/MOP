# Hugging Face Hub Integration Guide

## 概述

现在可以直接推送训练好的V5模型到 Hugging Face Hub，与全球开发者共享！

## 前置条件

### 1. 创建 Hugging Face 账户

```bash
# 访问 https://huggingface.co/join
# 注册账户
```

### 2. 获取 API Token

```bash
# 访问 https://huggingface.co/settings/tokens
# 创建新 token（需要 write 权限）
# 复制 token 值
```

### 3. 安装依赖

```bash
pip install huggingface-hub>=0.20.0
```

## 步骤 1: 训练模型

如果还没有训练，先训练 V5 模型：

```bash
# 训练所有 15 个 symbols
python train_v5.py --all --device cuda

# 或训练特定数量
python train_v5.py 5 --device cuda
```

验证模型已创建：
```bash
ls models/weights/*v5*.pth
```

## 步骤 2: 创建 Hugging Face 仓库

### 方式 A: 网页创建（推荐）

1. 访问 https://huggingface.co/new
2. 填写信息：
   - **Repository name**: `mop-crypto-models` (或你喜欢的名字)
   - **License**: MIT
   - **Model card metadata**: 选择 "Cryptocurrency"
3. 点击 "Create repository"
4. 复制仓库 ID (例如: `username/mop-crypto-models`)

### 方式 B: CLI 创建

```bash
huggingface-cli repo create mop-crypto-models --type model
```

## 步骤 3: 推送模型到 Hugging Face

### 推送单个 Symbol

```bash
python push_to_huggingface.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --symbols BTCUSDT ETHUSDT
```

### 推送所有 Symbols

```bash
python push_to_huggingface.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --all
```

### 指定自定义仓库

```bash
python push_to_huggingface.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --repo "your_username/your_repo_name" \
  --all
```

## 预期输出

```
======================================================================
PUSHING TO HUGGING FACE
======================================================================
Repository: username/mop-crypto-models
Symbols: 2
======================================================================

[1/2] Pushing BTCUSDT...
  Model: /path/to/BTCUSDT_1h_v5_lstm.pth -> ./temp_hf_BTCUSDT/pytorch_model.bin
  Config: /path/to/BTCUSDT_v5_config.json -> ./temp_hf_BTCUSDT/config.json
  README: Created
  Uploading to Hugging Face...
  SUCCESS: Model uploaded to username/mop-crypto-models/BTCUSDT
  Cleaned up temporary files

[2/2] Pushing ETHUSDT...
  Model: /path/to/ETHUSDT_1h_v5_lstm.pth -> ./temp_hf_ETHUSDT/pytorch_model.bin
  Config: /path/to/ETHUSDT_v5_config.json -> ./temp_hf_ETHUSDT/config.json
  README: Created
  Uploading to Hugging Face...
  SUCCESS: Model uploaded to username/mop-crypto-models/ETHUSDT
  Cleaned up temporary files

======================================================================
PUSH COMPLETE
======================================================================

Successful: 2/2
Failed: 0/2

View models at: https://huggingface.co/username/mop-crypto-models
```

## 推送后的结构

Hugging Face 上的仓库结构：

```
username/mop-crypto-models/
├── BTCUSDT/
│   ├── pytorch_model.bin      (V5 模型)
│   ├── config.json            (训练配置)
│   └── README.md              (模型卡片)
├── ETHUSDT/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── README.md
├── BNBUSDT/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── README.md
└── ... (其他 symbols)
```

## 在其他项目中使用模型

### 下载模型

```python
from huggingface_hub import hf_hub_download
import torch

# 下载模型
model_path = hf_hub_download(
    repo_id="username/mop-crypto-models",
    filename="BTCUSDT/pytorch_model.bin"
)

# 加载模型
model = torch.load(model_path)
model.eval()
```

### 使用模型进行预测

```python
import torch
import numpy as np

# 准备输入数据 (60步 x 20+特征)
X = torch.randn(1, 60, 25)  # batch_size=1, seq_len=60, features=25

# 预测
with torch.no_grad():
    price_delta, uncertainty = model(X)
    print(f"预测价格差值: {price_delta.item():.6f}")
    print(f"不确定性: {uncertainty.item():.6f}")
```

## 高级用法

### 使用环境变量

```bash
# 设置 token 为环境变量
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# 然后只需要
python push_to_huggingface.py --all
```

### 登录 Hugging Face CLI

```bash
# 一次性登录
huggingface-cli login

# 然后脚本会自动使用存储的 token
python push_to_huggingface.py --all
```

### 批量更新模型

```bash
# 重新训练后，快速更新所有模型
python train_v5.py --all --device cuda
python push_to_huggingface.py --token $HF_TOKEN --all
```

## 模型卡片说明

每个模型都会生成一个 README.md，包含：

1. **模型详情**
   - 架构信息
   - 输入/输出说明
   - 参数配置

2. **训练数据**
   - 数据来源
   - 时间范围
   - 数据分割

3. **性能指标**
   - MAPE
   - MAE
   - RMSE

4. **特征说明**
   - 20+ 个工程特征
   - 特征计算方法

5. **使用示例**
   - Python 代码示例
   - 加载和预测

6. **限制说明**
   - 历史数据限制
   - 市场风险免责
   - 推荐使用方式

## 故障排除

### 问题 1: "Token 错误"

```bash
# 解决方案：检查 token 是否正确
echo $HF_TOKEN  # 检查环境变量

# 重新生成 token: https://huggingface.co/settings/tokens
```

### 问题 2: "权限不足"

```bash
# 解决方案：Token 需要 write 权限
# 重新创建 token 时选择 "Full write"
```

### 问题 3: "模型文件未找到"

```bash
# 解决方案：确保先训练了 V5 模型
python train_v5.py 5 --device cuda

# 验证文件存在
ls models/weights/*v5*.pth
```

### 问题 4: "huggingface-hub 未安装"

```bash
pip install --upgrade huggingface-hub>=0.20.0
```

## 性能考虑

### 上传速度

- 单个模型 (~50MB): 1-5 秒
- 15 个模型: 1-2 分钟
- 取决于网络速度

### 下载速度

- 单个模型: 5-10 秒（取决于网络）
- CDN 加速
- 缓存机制

## 安全最佳实践

1. **Token 管理**
   - 不要在代码中硬编码 token
   - 使用环境变量或文件存储
   - 定期更新 token
   - 不在 GitHub 上提交 token

2. **仓库权限**
   - 设置为公开（共享）或私有（仅限自己）
   - 仓库链接可在 HF 主页找到
   - 可以设置 collaborators

3. **版本控制**
   - 每次训练都推送新版本
   - 在 README 中注明训练日期
   - 跟踪性能指标变化

## 分享您的模型

### 推送到 HF Hub 后

1. **分享链接**
   ```
   https://huggingface.co/username/mop-crypto-models
   ```

2. **在项目中引用**
   - README 中添加链接
   - 论文/文章中提及
   - 社交媒体分享

3. **获得反馈**
   - HF 的讨论功能
   - Issues
   - Pull requests

## 下一步

1. **训练模型** (如果还没有)
   ```bash
   python train_v5.py --all --device cuda
   ```

2. **推送到 HF**
   ```bash
   python push_to_huggingface.py --token YOUR_TOKEN --all
   ```

3. **分享和协作**
   - 在 HF 上邀请合作者
   - 接收社区反馈
   - 改进模型

## 相关资源

- Hugging Face Hub: https://huggingface.co/
- huggingface-hub 文档: https://huggingface.co/docs/hub/
- 模型卡片指南: https://huggingface.co/docs/hub/model-cards
- Pytorch 模型上传: https://huggingface.co/docs/hub/adding-a-model

## 总结

现在你可以：
1. 训练 V5 模型
2. 将模型推送到 Hugging Face Hub
3. 与全球开发者共享
4. 在其他项目中使用

所有流程都已自动化！
