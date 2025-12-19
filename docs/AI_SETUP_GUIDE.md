# AI 系统设置指南 - 自动模式配置

## 快速开始

### 步骤 1: 复制并配置 .env

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
nano .env  # 或使用你喜欢的编辑器
```

**关键配置项：**
```bash
# Binance API (必须)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Hugging Face (推荐)
HF_TOKEN=your_hf_token_here

# AI 自动模式 (重要！)
REQUIRES_USER_APPROVAL=false
AI_AUTO_MODE=true
AUTO_PUSH_ENABLED=True
```

### 步骤 2: 使用 AI 系统提示词

当你与 AI 交互时，使用以下提示词模板：

```markdown
请使用 docs/AI_SYSTEM_PROMPT.md 中定义的专业标准来处理这个任务：

[你的具体需求]

例如：
"请按照 AI_SYSTEM_PROMPT.md 的标准，训练 ETHUSDT 15m 的 LSTM 模型。
使用 3000 根 K 棒的训练数据，执行网格搜索超参数优化，
生成完整的性能报告，并自动推送到 GitHub。"
```

---

## 自动模式详解

### 什么是自动模式？

自动模式允许 AI 在以下情况下不需要用户手动批准：
- 执行数据采集任务
- 训练和优化模型
- 生成报告和分析
- **自动推送结果到 GitHub**

### 启用自动模式

**在 .env 中设置：**
```bash
REQUIRES_USER_APPROVAL=false
AI_AUTO_MODE=true
AUTO_PUSH_ENABLED=True
```

**在代码中启用：**
```python
import os
os.environ['REQUIRES_USER_APPROVAL'] = 'false'
os.environ['AI_AUTO_MODE'] = 'true'
```

### 自动模式的工作流程

```
用户命令
   ↓
AI 分析任务
   ↓
执行任务（无需确认）
   ↓
生成结果
   ↓
自动推送到 GitHub
   ↓
完成！
```

---

## AI 系统提示词使用示例

### 示例 1: 获取训练数据

```markdown
根据 AI_SYSTEM_PROMPT.md，执行数据采集任务：

1. 从 Binance 采集以下币种：BTCUSDT, ETHUSDT, BNBUSDT
2. 时间框架：15m, 1h
3. 每个币种 3000 根 K 棒
4. 验证数据完整性和时间戳格式正确
5. 生成 metadata.json 报告
6. 自动推送到 GitHub
```

### 示例 2: 训练 LSTM 模型

```markdown
按照 AI_SYSTEM_PROMPT.md 的标准流程，训练加密货币预测模型：

数据：ETHUSDT_15m.csv (3000 根 K 棒)

任务：
1. 计算 35 项技术指标
2. 构建 LSTM 模型 (128->64 单元)
3. 执行网格搜索优化 lookback_period 和 dropout_rate
4. 训练 200 epochs（含早停）
5. 在测试集上评估，目标方向准确率 > 65%
6. 生成性能报告（混淆矩阵、ROC 曲线、损失曲线）
7. 保存模型和 scaler
8. 上传到 Hugging Face
9. 推送脚本和报告到 GitHub
```

### 示例 3: 全自动化流程

```markdown
执行完整的 ML pipeline，按照 AI_SYSTEM_PROMPT.md 的标准：

Phase 1: 采集 40+ 币种的训练数据
Phase 2: 特征工程（35+ 指标）
Phase 3: 训练多个 LSTM 模型
Phase 4: 超参数优化和对比
Phase 5: 生成综合报告
Phase 6: 部署和版本管理

配置：
- lookback_period: 60
- batch_size: 32
- epochs: 200
- early_stopping_patience: 20

输出：
- 40+ 训练好的模型
- 性能对比报告
- 推送到 GitHub 和 Hugging Face
```

---

## 配置 GitHub 自动推送

### 生成 GitHub Personal Access Token

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token"
3. 选择 "classic" token
4. 授予权限：
   - repo (完整仓库访问)
   - read:user
5. 生成并复制 token

### 在 .env 中配置

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
GITHUB_REPO_OWNER=caizongxun
GITHUB_REPO_NAME=MOP
AUTO_PUSH_ENABLED=True
```

### 验证配置

```bash
# 测试 GitHub 连接
python -c "
from backend.data.data_manager import DataManager
import os
token = os.getenv('GITHUB_TOKEN')
if token:
    print('GitHub token 已配置')
else:
    print('GitHub token 未配置')
"
```

---

## AI 提示词结构

### 核心组件

**AI_SYSTEM_PROMPT.md 包含：**

1. **系统角色定义**
   - 数据采集专家
   - 模型工程师
   - 质量保证官
   - 项目管理者

2. **核心能力**
   - 数据采集与预处理
   - 特征工程（35+ 指标）
   - 模型架构设计
   - 超参数优化
   - 模型训练和评估
   - 版本管理和部署

3. **工作流程**
   - Phase 1-5 的详细步骤
   - 每个阶段的关键检查点
   - 输出和交付物列表

4. **约束与最佳实践**
   - 时间戳处理标准
   - 数据分割方法
   - 过拟合防止策略
   - 错误处理流程

---

## 常见命令

### 启动数据采集

```bash
# 使用 AI 获取数据
python -c "
from backend.data.data_manager import DataManager
dm = DataManager()
results = dm.fetch_and_store_all_batch(
    timeframes=['15m', '1h'],
    total_limit=3000
)
dm.print_statistics()
"
```

### 启动模型训练

```bash
# 训练单个模型
python training_scripts/train_lstm.py \
    --symbol ETHUSDT \
    --timeframe 15m \
    --epochs 200 \
    --batch_size 32

# 批量训练
python training_scripts/train_all_models.py \
    --symbols all \
    --hyperparameter_search
```

### 自动推送结果

```bash
# 推送模型和报告
python scripts/push_to_github.py \
    --model models/lstm_model_v*.h5 \
    --report reports/training_report_*.pdf \
    --message "Model training completed"
```

---

## 监控和调试

### 查看日志

```bash
# 实时查看日志
tail -f logs/crypto_predictor.log

# 查看特定级别的日志
grep "ERROR" logs/crypto_predictor.log
grep "WARNING" logs/crypto_predictor.log
```

### 检查 AI 执行状态

```bash
# 检查是否启用自动模式
python -c "import os; print('Auto mode:', os.getenv('AI_AUTO_MODE'))"

# 检查 GitHub 连接
python -c "from github import Github; print('GitHub OK')"

# 检查 Hugging Face 连接
python -c "from huggingface_hub import HfApi; api = HfApi(); print('HF OK')"
```

---

## 故障排除

### 问题 1: AI 无法推送到 GitHub

**解决：**
```bash
# 检查 token
echo $GITHUB_TOKEN

# 重新生成 token
# https://github.com/settings/tokens

# 更新 .env
echo "GITHUB_TOKEN=your_new_token" >> .env
```

### 问题 2: 自动模式未生效

**解决：**
```bash
# 检查 .env 配置
grep REQUIRES_USER_APPROVAL .env
grep AI_AUTO_MODE .env

# 重启应用
pkill -f python
sleep 2
python your_script.py
```

### 问题 3: 模型训练中断

**解决：**
```bash
# 检查日志
tail -100 logs/crypto_predictor.log

# 使用检查点恢复
python training_scripts/train_lstm.py \
    --resume_from_checkpoint models/checkpoint_latest.h5
```

---

## 下一步

1. **配置你的 .env 文件**
   - 填入 Binance API 密钥
   - 配置 GitHub token
   - 启用自动模式

2. **运行示例**
   ```bash
   python fetch_all.py  # 获取训练数据
   python training_scripts/train_lstm.py  # 训练模型
   ```

3. **与 AI 交互**
   - 使用 AI_SYSTEM_PROMPT.md 中的提示词
   - 让 AI 自动执行和推送
   - 查看 GitHub 上的结果

4. **监控和优化**
   - 查看性能报告
   - 调整超参数
   - 改进模型

---

## 支持

如有问题，请：
1. 查看完整的 AI_SYSTEM_PROMPT.md
2. 检查日志文件 logs/crypto_predictor.log
3. 创建 GitHub Issue

---

**版本**: v1.0  
**最后更新**: 2025-12-19  
**维护者**: zongowo111
