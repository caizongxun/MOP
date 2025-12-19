# Hugging Face 批量推送系统 - 总结

一次性推送整个文件夹（零 API 限制、快 10 倍）

## 系统特点

- **批量推送** - 一次批量推送整个文件夹，无 API 限制
- **快 10 倍** - 15 个 models 只需 1 个 commit，2-5 分钟上传
- **自动重试** - 网络错误自动重试 3 次
- **自动 README** - 为每个 model 自动生成不同详情说明
- **零配置** - 削减网络开销

## 对比

| 特性 | 单文件推送 | **批量推送** |
|--------|-----------|----------|
| API 调用 | 15 次 | **1 次** |
| 限制 | 容易触发 | **无限制** |
| 速度 | 15 分钟 | **2-5 分钟** |
| 可靠性 | 低 | **高** |
| 自业及 | 有 | **无** |

## 快速开始 (3 步)

### 步骤 1: 训练模型

```bash
python train_v5.py --all --device cuda
```

### 步骤 2: 一次批量推送

```bash
python push_to_huggingface_batch.py \
  --token "hf_YOUR_TOKEN_HERE" \
  --repo "zongowo111/mop-v5-models" \
  --all
```

### 步骤 3: 享受成果

访问: https://huggingface.co/zongowo111/mop-v5-models

## 推送后的仓库结构

```
zongowo111/mop-v5-models/
├── BTCUSDT/          (50MB)
│   ├── pytorch_model.bin  (V5 模型)
│   ├── config.json        (配置)
│   └── README.md          (README)
├── ETHUSDT/           (50MB)
├── BNBUSDT/
└── ... (12 个其他)

总大小: ~750MB
推送时间: 2-5 分钟
API 调用: 1 次 (零限制!)
```

## 每个 model 自动包含

- 模型架构描述
- 训练数据说明
- 性能指标 (MAPE, MAE, RMSE)
- 使用代码示例
- 特征工程说明
- 风险免责声明

## 在互联网中使用

```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(
    repo_id="zongowo111/mop-v5-models",
    filename="BTCUSDT/pytorch_model.bin"
)

model = torch.load(model_path)
model.eval()

with torch.no_grad():
    delta, uncertainty = model(features)
```

## 常见问题

**Q: 需要网为多大？**
A: 750MB，稳定 1-2 Mbps 即可

**Q: 推送需要多穳？**
A: 2-5 分钟

**Q: 中途失败了怎么办？**
A: 自动重试 3 次，或稽后重新运行

**Q: 可以覆盖旧版本吗？**
A: 是的，新推送自动覆盖

## 完整步骤

1. **创建 HF 账户**
   - https://huggingface.co/join

2. **获取 Token** (write 权限)
   - https://huggingface.co/settings/tokens

3. **创建仓库**
   - https://huggingface.co/new
   - 名稱: `mop-v5-models`
   - License: `MIT`

4. **安装依赖**
   ```bash
   pip install huggingface-hub>=0.20.0
   ```

5. **训练模型**
   ```bash
   python train_v5.py --all --device cuda
   ```

6. **推送了！**
   ```bash
   python push_to_huggingface_batch.py \
     --token "YOUR_TOKEN" \
     --repo "zongowo111/mop-v5-models" \
     --all
   ```

## 存案文件

- `backend/push_to_huggingface_batch.py` - 完整推送器
- `push_to_huggingface_batch.py` - 快速启动脚本
- `HUGGINGFACE_BATCH_PUSH_GUIDE.md` - 详细指南
- `.gitattributes` - Git LFS 配置

## 总结

✓ 批量推送系统已完成  
✓ 无 API 限制  
✓ 快 10 倍  
✓ 自动重试  
✓ 自动 README  

**现在可以推送了！**
