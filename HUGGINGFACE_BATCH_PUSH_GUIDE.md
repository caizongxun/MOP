# Hugging Face 批量推送指南

批量推送系统已完成 - 一次性推送整个文件夹，零 API 限制

## 为什么使用批量推送？

对比单文件上传：

| 方式 | API 调用数 | 限制 | 速度 | 可靠性 |
|------|----------|------|------|--------|
| 单文件 | 15 次 | 容易触发 | 慢 | 低 |
| **批量** | **1 次** | **无限制** | **快 10 倍** | **高** |

优点：
- 无 API 率限制
- 快 10 倍 (1 commit 提交所有)
- 自动重试网络错误
- 更可靠

---

## 前置准备

### 1. 创建 Hugging Face 账户

访问 https://huggingface.co/join

### 2. 获取 API Token

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Write" 权限
4. 复制 token 值

### 3. 安装依赖

```bash
pip install huggingface-hub>=0.20.0
```

### 4. 创建 Hugging Face 仓库

1. 访问 https://huggingface.co/new
2. 填写：
   - Repository name: `mop-v5-models`
   - License: `MIT`
3. 完成后获得仓库 ID (例如: `zongowo111/mop-v5-models`)

---

## 步骤 1: 训练 V5 模型

如果还没有训练，先训练模型：

```bash
# 训练所有 15 个 symbols (~45分钟)
python train_v5.py --all --device cuda

# 或快速测试 (5 个)
python train_v5.py 5 --device cuda
```

验证模型已生成：

```bash
ls models/weights/*v5*.pth
```

---

## 步骤 2: 批量推送到 Hugging Face

### 推送所有模型

```bash
python push_to_huggingface_batch.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --repo "zongowo111/mop-v5-models" \
  --all
```

### 推送特定 symbols

```bash
python push_to_huggingface_batch.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --repo "zongowo111/mop-v5-models" \
  --symbols BTCUSDT ETHUSDT BNBUSDT
```

### 使用环境变量 (推荐)

```bash
# Linux/Mac
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export HF_REPO="zongowo111/mop-v5-models"

# 然后简化为
python push_to_huggingface_batch.py --all
```

在 Windows PowerShell：

```powershell
$env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
$env:HF_REPO="zongowo111/mop-v5-models"
```

---

## 预期输出

```
======================================================================
BATCH PUSH TO HUGGING FACE
======================================================================
Repository: zongowo111/mop-v5-models
Symbols to push: 15
Symbols: BTCUSDT, ETHUSDT, BNBUSDT, ...
======================================================================

Preparing model folders...
  Preparing BTCUSDT... OK
  Preparing ETHUSDT... OK
  ... (all 15)

Successfully prepared: 15/15

Uploading 15 models to Hugging Face...
Folder size: 750.5 MB
This may take a few minutes...

Upload completed successfully!
Time taken: 120.3 seconds
URL: https://huggingface.co/zongowo111/mop-v5-models

======================================================================
SUCCESS: Models pushed to zongowo111/mop-v5-models
View at: https://huggingface.co/zongowo111/mop-v5-models
======================================================================
```

---

## 推送后的仓库结构

```
zongowo111/mop-v5-models/
├── BTCUSDT/
│   ├── pytorch_model.bin    (50MB - V5 模型)
│   ├── config.json          (训练配置)
│   └── README.md            (模型说明)
├── ETHUSDT/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── README.md
├── BNBUSDT/
│   └── ...
└── ... (13 个其他 symbols)
```

每个 symbol 的自动生成说明包括：
- 模型架构描述
- 训练数据说明
- 性能指标 (MAPE, MAE, RMSE)
- 使用代码示例
- 特征说明

---

## 在其他项目中使用模型

```python
from huggingface_hub import hf_hub_download
import torch

# 下载模型
model_path = hf_hub_download(
    repo_id="zongowo111/mop-v5-models",
    filename="BTCUSDT/pytorch_model.bin"
)

# 加载
model = torch.load(model_path)
model.eval()

# 预测
with torch.no_grad():
    price_delta, uncertainty = model(features)
    print(f"预测价格变化: {price_delta.item()}")
    print(f"不确定性: {uncertainty.item()}")
```

下载配置：

```python
import json
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(
    repo_id="zongowo111/mop-v5-models",
    filename="BTCUSDT/config.json"
)

with open(config_path) as f:
    config = json.load(f)
```

---

## 错误排查

### 问题: "Cannot access repository"

**解决**：
- Token 有 write 权限吗？https://huggingface.co/settings/tokens
- 仓库存在吗？https://huggingface.co/new
- 仓库 ID 正确吗？

### 问题: "Model file not found"

**解决**：
```bash
python train_v5.py --all --device cuda
ls models/weights/*v5*.pth
```

### 问题: "Network error / timeout"

**解决**：
- 脚本会自动重试 3 次
- 网络稳定时重新运行
- 同一仓库多次上传会自动覆盖

### 问题: "huggingface-hub not installed"

**解决**：
```bash
pip install --upgrade huggingface-hub>=0.20.0
```

---

## 性能指标示例

推送完成后，每个模型的 README 会显示：

```
BTCUSDAT:
- MAPE: 3.45%      (平均百分比错误)
- MAE: $0.0234     (平均绝对错误)
- RMSE: $0.0567    (均方根误差)

ETHUSDT:
- MAPE: 4.12%
- MAE: $0.0145
- RMSE: $0.0312
```

---

## 常见问题

**Q: 推送需要多长时间？**
- A: 15 个模型 (~750MB) 通常需要 2-5 分钟

**Q: 可以多次推送覆盖吗？**
- A: 是的，新推送会自动覆盖旧版本

**Q: 推送失败了怎么办？**
- A: 脚本自动重试 3 次，等待后重新运行

**Q: 如何检查推送成功？**
- A: 访问 https://huggingface.co/zongowo111/mop-v5-models 查看

**Q: 模型大小有限制吗？**
- A: HF 支持单文件 5GB+，不用担心

**Q: 可以推送到私有仓库吗？**
- A: 是的，创建私有仓库然后推送

---

## 推荐工作流

### 第一次推送

```bash
# 1. 训练模型
python train_v5.py --all --device cuda

# 2. 一次性批量推送
python push_to_huggingface_batch.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --repo "zongowo111/mop-v5-models" \
  --all

# 3. 验证成功
# 访问 https://huggingface.co/zongowo111/mop-v5-models
```

### 之后更新模型

```bash
# 重新训练
python train_v5.py --all --device cuda

# 再次推送 (自动覆盖)
python push_to_huggingface_batch.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --repo "zongowo111/mop-v5-models" \
  --all
```

---

## 分享你的模型

推送完成后：

1. **访问仓库**
   - https://huggingface.co/zongowo111/mop-v5-models

2. **分享链接**
   - 在社交媒体、论文、博客中分享

3. **邀请协作者**
   - 在仓库设置中添加 collaborators

4. **查看自动生成的 Model Cards**
   - 每个 symbol 都有详细说明

---

## 相关资源

- [Hugging Face Hub](https://huggingface.co/)
- [huggingface-hub 文档](https://huggingface.co/docs/hub/)
- [模型卡指南](https://huggingface.co/docs/hub/model-cards)
- [PyTorch 模型上传](https://huggingface.co/docs/hub/adding-a-model)

---

## 总结

✓ 批量推送系统已完成  
✓ 一次性推送所有 models  
✓ 无 API 率限制  
✓ 自动生成 README 和元数据  
✓ 与全球开发者分享  

现在可以推送了！
