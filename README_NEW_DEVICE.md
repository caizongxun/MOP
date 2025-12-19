# 新设备完整设置指南

在新设备上从零开始运行 MOP：**数据获取 -> 训练 -> 上传 Hugging Face**

---

## 快速导航

### 想快速开始？(推荐)

→ [**QUICK_START_THREE_STEPS.md**](QUICK_START_THREE_STEPS.md)

- 3 个流程
- 所有终端命令
- 5 分钟快速参考

### 需要详细步骤？

→ [**COMPLETE_WORKFLOW_README.md**](COMPLETE_WORKFLOW_README.md)

- 逐步详解
- 每个命令说明
- 故障排除
- 10000 字完整指南

### 处理特定问题？

- **上传到 Hugging Face** → [HUGGINGFACE_BATCH_PUSH_GUIDE.md](HUGGINGFACE_BATCH_PUSH_GUIDE.md)
- **Windows PowerShell** → [WINDOWS_POWERSHELL_COMMANDS.txt](WINDOWS_POWERSHELL_COMMANDS.txt)
- **批量推送系统** → [BATCH_PUSH_SUMMARY.md](BATCH_PUSH_SUMMARY.md)

---

## 三个流程概览

### 流程 1: 数据获取

```bash
python backend/data_fetcher_batch.py --all
```

**耗时**: 10-15 分钟

获取所有 15 个 symbols 的 1h 和 15m 时间框架数据

### 流程 2: 模型训练

```bash
python train_v5.py --all --device cuda
```

**耗时**: 45-60 分钟 (GPU) / 3-5 小时 (CPU)

训练所有 15 个 symbols 的 V5 LSTM 模型

### 流程 3: 上传到 Hugging Face

```bash
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN" --repo "your_username/mop-v5-models" --all
```

**耗时**: 2-5 分钟

一次性批量上传所有 15 个模型到 Hugging Face (零 API 限制)

---

## 前置准备

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

### 3. 创建 Hugging Face 账户和仓库

- 账户: https://huggingface.co/join
- Token: https://huggingface.co/settings/tokens
- 仓库: https://huggingface.co/new

---

## 耗时估计

| 步骤 | 耗时 | 条件 |
|------|------|------|
| **环境准备** | 5-10 分钟 | 一次性 |
| **数据获取** | 10-15 分钟 | 网络 |
| **模型训练** | 45-60 分钟 | GPU |
| **上传 HF** | 2-5 分钟 | 网络 |
| **总计** | 60-80 分钟 | 第一次 |

---

## 选择适合您的文档

### 情况 1: "我想快速上手"

**推荐**: 打开 [QUICK_START_THREE_STEPS.md](QUICK_START_THREE_STEPS.md)

- 所有命令已列出
- Windows/Linux/Mac 用法
- 可直接复制粘贴

### 情况 2: "我想了解详细步骤"

**推荐**: 打开 [COMPLETE_WORKFLOW_README.md](COMPLETE_WORKFLOW_README.md)

- 完整的分步指南
- 每个命令的解释
- 验证方法
- 故障排除

### 情况 3: "我在 Windows PowerShell 上有问题"

**推荐**: 查看 [WINDOWS_POWERSHELL_COMMANDS.txt](WINDOWS_POWERSHELL_COMMANDS.txt)

- PowerShell 特殊语法
- 反引号用法
- 环境变量设置

### 情况 4: "我想了解批量上传机制"

**推荐**: 读 [BATCH_PUSH_SUMMARY.md](BATCH_PUSH_SUMMARY.md)

- 为什么使用批量上传
- 性能对比
- 系统架构

---

## 立即开始

### 最快的方式 (复制粘贴)

#### Linux/Mac:

```bash
git clone https://github.com/caizongxun/MOP.git && cd MOP && \
pip install -r requirements.txt && \
pip install huggingface-hub>=0.20.0 && \
python backend/data_fetcher_batch.py --all && \
python train_v5.py --all --device cuda && \
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN" --repo "your_username/mop-v5-models" --all
```

#### Windows PowerShell (分步版本更好):

第 1-4 行: 安装和准备
```powershell
git clone https://github.com/caizongxun/MOP.git
cd MOP
pip install -r requirements.txt
pip install huggingface-hub>=0.20.0
```

第 5 行: 获取数据
```powershell
python backend/data_fetcher_batch.py --all
```

第 6 行: 训练模型
```powershell
python train_v5.py --all --device cuda
```

第 7-8 行: 设置环境变量
```powershell
$env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
$env:HF_REPO="your_username/mop-v5-models"
```

第 9 行: 上传到 HF
```powershell
python push_to_huggingface_batch.py --token $env:HF_TOKEN --repo $env:HF_REPO --all
```

---

## 支持的 Symbols

所有 15 个主要加密货币对:

```
BTCUSDT    ETHUSDT    BNBUSDT    ADAUSDT    XRPUSDT
SOLUSDT    DOGEUSDT   AVAXUSDT   MATICUSDT  LINKUSDT
LTCUSDT    NEARUSDT   ATOMUSDT   UNIUSDT    APTUSDT
```

可以只训练部分:
```bash
python train_v5.py 5 --device cuda                    # 前 5 个
python train_v5.py --symbols BTCUSDT ETHUSDT --device cuda  # 指定的
```

---

## 常见问题快速回答

**Q: 我没有 GPU 怎么办？**

A: 可以用 CPU，但会很慢 (3-5 小时)
```bash
python train_v5.py --all --device cpu
```

**Q: 网络断了怎么办？**

A: 脚本会自动重试 3 次，或稍后重新运行

**Q: 出错了我看不懂怎么办？**

A: 查看 [COMPLETE_WORKFLOW_README.md](COMPLETE_WORKFLOW_README.md) 的"故障排除"部分

**Q: 我想只上传部分模型？**

A:
```bash
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN" --repo "your_username/mop-v5-models" --symbols BTCUSDT ETHUSDT
```

**Q: Windows PowerShell 反斜杠报错？**

A: 一行写完或查看 [WINDOWS_POWERSHELL_COMMANDS.txt](WINDOWS_POWERSHELL_COMMANDS.txt)

---

## 文档结构

```
新设备入门:
  ├─ README_NEW_DEVICE.md (本文件) - 导航
  ├─ QUICK_START_THREE_STEPS.md - 快速参考
  └─ COMPLETE_WORKFLOW_README.md - 完整指南

特定主题:
  ├─ HUGGINGFACE_BATCH_PUSH_GUIDE.md - HF 上传
  ├─ BATCH_PUSH_SUMMARY.md - 批量系统
  ├─ WINDOWS_POWERSHELL_COMMANDS.txt - PowerShell
  └─ HUGGINGFACE_GUIDE.md - HF 集成

代码:
  ├─ backend/data_fetcher_batch.py - 数据获取
  ├─ train_v5.py - 模型训练
  └─ push_to_huggingface_batch.py - HF 上传
```

---

## 下一步

1. **选择您的起点**: 上面的导航部分
2. **按照步骤执行**: 所有命令都已准备好
3. **验证结果**: 
   - 数据: `ls data/raw/ | wc -l` (应该 30+)
   - 模型: `ls models/weights/*v5*.pth | wc -l` (应该 15)
   - HF: https://huggingface.co/your_username/mop-v5-models

---

## 获取帮助

1. 查看相关的 .md 文件 (见导航)
2. 检查日志: `ls logs/`
3. 验证依赖: `pip list | grep -i torch`
4. 查看完整的 README.md: [README.md](README.md)

---

**准备好了？打开 [QUICK_START_THREE_STEPS.md](QUICK_START_THREE_STEPS.md) 开始吧！**
