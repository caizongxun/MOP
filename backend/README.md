# MOP 加密貨幣價格預測系統

## 版本對比

| 版本 | 方法 | 預期 MAPE | 特徵 | 推薦 |
|------|------|---------|------|------|
| **V2 BB-Focused** | LSTM | 10-12% | 18 個 Bollinger Bands 特徵 | 基礎版 |
| **V3 LSTM+XGBoost** | 混合 | **1-2%** | LSTM 特徵 + XGBoost | ⭐ 推薦 |

---

## V3 LSTM+XGBoost 快速開始（推薦）

### 安裝依賴

```bash
pip install xgboost scikit-learn
```

### 訓練（2 階段）

```bash
# 訓練 3 個幣種測試
python -m backend.train_v3_lstm_xgboost --symbols BTCUSDT ETHUSDT BNBUSDT

# 訓練所有 15 個幣種
python -m backend.train_v3_lstm_xgboost
```

**訓練流程：**
1. 第1階段：LSTM 提取時間特徵 (50 epochs)
2. 第2階段：XGBoost 在 LSTM 特徵上預測價格

### 預測

```bash
# 單個預測
python -m backend.inference_v3_lstm_xgboost --symbol BTCUSDT

# 批量預測（所有 15 個幣種）
python -m backend.inference_v3_lstm_xgboost --batch
```

**輸出：**
- 預測價格
- 預測漲跌
- 技術指標參考
- CSV 報告

---

## V2 Regression 版本（基礎）

### 訓練

```bash
# 訓練 3 個幣種
python -m backend.train_v2_bb_focused --symbols BTCUSDT ETHUSDT BNBUSDT

# 訓練所有 15 個幣種
python -m backend.train_v2_bb_focused --epochs 100
```

### 預測

```bash
# 單個預測
python -m backend.inference_v2_bb --symbol BTCUSDT

# 批量預測
python -m backend.inference_v2_bb --batch
```

---

## 模型文件位置

### V3 (LSTM+XGBoost)
```
backend/models/weights/
  ├─ {symbol}_1h_v3_lstm.pth       # LSTM 模型
  └─ {symbol}_1h_v3_xgb.pkl       # XGBoost 模型
```

### V2 (BB-Focused)
```
backend/models/weights/
  └─ {symbol}_1h_v2_best.pth      # LSTM 模型
```

### 預測結果
```
backend/predictions/
  ├─ batch_v3_lstm_xgboost_YYYYMMDD_HHMMSS.csv
  └─ batch_directional_YYYYMMDD_HHMMSS.csv
```

---

## 架構細節

### V3 LSTM+XGBoost 架構

```
輸入特徵 (18 個技術指標)
       ↓
  ┌─────────┐
  │  LSTM   │ (128 隱藏單元，2 層)
  │         │ 提取時間依賴特徵
  └────┬────┘
       ↓
  隱藏狀態 (128 維)
       ↓
  ┌─────────────┐
  │  XGBoost    │ (500 樹，max_depth=6)
  │  Regressor  │ 預測最終價格
  └────┬────────┘
       ↓
  預測價格 ($)
```

### 為什麼 LSTM+XGBoost 更好

1. **LSTM 優勢**：捕捉時間序列的長期依賴
2. **XGBoost 優勢**：在非線性關係上無敵
3. **結合優勢**：>  單個模型

---

## 超參數設置

### V3 超參數

**LSTM 配置：**
```python
hidden_size: 128
num_layers: 2
dropout: 0.2
lr: 0.001
epochs: 50
batch_size: 16
```

**XGBoost 配置：**
```python
max_depth: 6
learning_rate: 0.05
n_estimators: 500
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 1
reg_lambda: 1
```

### V2 超參數

```python
hidden_size: 128
num_layers: 2
dropout: 0.2
lr: 0.001
batch_size: 16
weight_decay: 1e-5
```

---

## 特徵工程

### 核心特徵 (5 個)

```
Bollinger Bands:
- bb_upper   # 上軌
- bb_middle  # 中軌 (SMA 20)
- bb_lower   # 下軌
- bb_percent_b  # 價格在通道的百分位
- bb_width      # 通道寬度 (波動性)
```

### 支持特徵 (13 個，由相關性選擇)

```
趨勢：    EMA_10, EMA_20, EMA_50
動能：    MACD, MACD_signal, MACD_histogram
超買超賣：RSI_14, RSI_21
成交量：  Volume, Volume_change
波動性：  ATR_14, DI+, DI-
其他：    MFI, OBV
```

---

## 預期性能

### V3 LSTM+XGBoost
```
整體 MAPE：        1-2%
每類資產 MAPE：
  - BTC/ETH：      0.8-1.5%
  - BNB/SOL：      1-1.8%
  - DOGE/NEAR：    0.5-1.2%
```

### V2 BB-Focused
```
整體 MAPE：        10-12%
每類資產 MAPE：
  - BTC/ETH：      8-12%
  - BNB/SOL：      9-11%
  - DOGE/NEAR：    3-8%
```

---

## 訓練時間

| 版本 | 每個幣種 | 15 個幣種 |
|------|---------|----------|
| V3   | 2-3 分鐘 | 30-45 分鐘 |
| V2   | 30-40 秒 | 8-10 分鐘 |

---

## 推薦工作流程

### 第 1 天：測試 V3
```bash
# 快速測試 3 個幣種
python -m backend.train_v3_lstm_xgboost --symbols BTCUSDT ETHUSDT BNBUSDT

# 檢查 MAPE 是否在 1-2%
python -m backend.inference_v3_lstm_xgboost --batch
```

### 第 2 天：完整訓練（如果結果好）
```bash
# 訓練所有 15 個幣種
python -m backend.train_v3_lstm_xgboost

# 生成完整預測報告
python -m backend.inference_v3_lstm_xgboost --batch
```

---

## 故障排除

### CUDA 記憶體不足
```bash
# 用 CPU 訓練
python -m backend.train_v3_lstm_xgboost --device cpu
```

### XGBoost 不在環境中
```bash
pip install xgboost
```

### 模型不存在
```bash
# 重新訓練缺失的幣種
python -m backend.train_v3_lstm_xgboost --symbols BTCUSDT
```

---

## 檔案清單

### 訓練腳本
- `train_v3_lstm_xgboost.py` - **V3 LSTM+XGBoost（推薦）**
- `train_v2_bb_focused.py` - V2 Regression（基礎）

### 推論腳本
- `inference_v3_lstm_xgboost.py` - **V3 推論（推薦）**
- `inference_v2_bb.py` - V2 推論

### 配置文件
- `models/lstm_model.py` - LSTM 架構
- `data/data_manager.py` - 數據管理
- `data/data_loader.py` - 技術指標計算

---

## 下一步優化

1. **更多數據**：增加訓練數據時間範圍
2. **特徵工程**：嘗試新的技術指標組合
3. **超參優化**：自動搜索最佳超參
4. **多時間框架**：融合 5m、1h、4h 數據
5. **集成模型**：結合多個模型的預測

---

## 聯繫方式

如有問題或建議，請提交 Issue。
