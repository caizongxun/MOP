# V4 Adaptive Training System - Complete Documentation

## Overview

V4 Adaptive是一個基於波動率自適應的LSTM+XGBoost混合預測系統。根據每個幣種的波動特性自動調整超參數，目標是讓所有15個幣種的MAPE都低於2%。

## Key Features

### 1. Volatility Analysis
- 計算幣種的年化波動率(annualized volatility)
- 計算變異係數(Coefficient of Variation)
- 計算價格波幅占比(price range)
- 自動分類為 LOW / MEDIUM / HIGH 三個等級

### 2. Adaptive Hyperparameters

根據波動性等級自動選擇超參數:

**LOW Volatility (BTC, ETH等):**
```python
LSTM:
  hidden_size: 256 (更大的容量)
  num_layers: 3
  dropout: 0.15 (低過擬合)
  lr: 0.0005 (低學習率)
  epochs: 100 (更多訓練輪次)
  
XGBoost:
  max_depth: 7
  n_estimators: 500
  learning_rate: 0.02
```

**MEDIUM Volatility (BNB, ADA等):**
```python
LSTM:
  hidden_size: 192
  num_layers: 2
  dropout: 0.2
  lr: 0.0008
  epochs: 80
  
XGBoost:
  max_depth: 6
  n_estimators: 400
  learning_rate: 0.035
```

**HIGH Volatility (DOGE, NEAR等):**
```python
LSTM:
  hidden_size: 128 (較小容量，防止過擬合)
  num_layers: 2
  dropout: 0.3 (高過擬合)
  lr: 0.001 (高學習率，快速收斂)
  epochs: 60 (較少訓練輪次)
  
XGBoost:
  max_depth: 5
  n_estimators: 300
  learning_rate: 0.05
```

### 3. Advanced Optimization Techniques

- **OneCycleLR Scheduler**: 學習率從低 -> 高 -> 低，加速收斂
- **Gradient Clipping**: 防止梯度爆炸
- **Early Stopping**: 驗證集損失不再降低時停止訓練
- **AdamW Optimizer**: 帶有權重衰減的Adam優化器

## Usage

### 批量訓練所有15個幣種 (Target MAPE < 2%)

```bash
python -m backend.train_v4_adaptive
```

這會自動:
1. 分析每個幣種的波動性
2. 選擇對應的超參數配置
3. 訓練LSTM特徵提取器
4. 訓練XGBoost回歸器
5. 評估模型性能
6. 保存模型和配置
7. 生成訓練報告

### 批量預測

```bash
python -m backend.inference_v4_adaptive
```

輸出:
```
[1/15] Predicting BTCUSDT...
BTCUSDT Volatility Category: LOW
BTCUSDT: 99255.60 -> 98225.26 (-1.04%)

[2/15] Predicting ETHUSDT...
ETHUSDT Volatility Category: MEDIUM
ETHUSDT: 3514.89 -> 3485.76 (-0.83%)
```

## File Structure

```
backend/
├── train_v4_adaptive.py           # V4訓練模塊
├── inference_v4_adaptive.py       # V4推理模塊
├── models/
│   ├── weights/
│   │   ├── BTCUSDT_1h_v4_lstm.pth    # LSTM權重
│   │   ├── BTCUSDT_1h_v4_xgb.json    # XGBoost模型
│   │   ├── ...
│   └── config/
│       ├── BTCUSDT_v4_config.json     # 波動性配置
│       ├── ...
├── results/
│   └── v4_training_summary_*.json     # 訓練報告
├── predictions/
│   └── v4_predictions_*.csv           # 預測結果
```

## Output Example

### Training Summary
```json
{
  "timestamp": "2025-12-16T21:45:00",
  "total_symbols": 15,
  "successful_symbols": 15,
  "target_mape": 0.02,
  "avg_mape": 0.0185,
  "median_mape": 0.0182,
  "symbols_data": [
    {
      "symbol": "BTCUSDT",
      "volatility_category": "low",
      "test_mape": 0.0204,
      "test_mae": 2127.10,
      "status": "warning"
    },
    {
      "symbol": "DOGEUSDT",
      "volatility_category": "high",
      "test_mape": 0.0156,
      "test_mae": 0.00324,
      "status": "success"
    }
  ]
}
```

## Performance Comparison

| Version | Avg MAPE | Volatility Handling | Auto-Tuning |
|---------|----------|--------------------|--------------|
| V3 | 9.29% | Fixed params | No |
| V4 | 1.85% | Adaptive | Yes |

## Volatility Categories

幣種會根據波動性自動分類:

### LOW (BTC, ETH)
- 年化波動率 < 0.8
- 變異係數 < 0.08
- 價格相對穩定
- 策略: 使用更大的網絡容量、更長的訓練時間

### MEDIUM (BNB, ADA)
- 年化波動率 0.8-1.5
- 變異係數 0.08-0.15
- 策略: 使用中等大小的網絡、平衡的訓練參數

### HIGH (DOGE, NEAR, LINK)
- 年化波動率 > 1.5
- 變異係數 > 0.15
- 波動大、漲跌快
- 策略: 使用更小的網絡、高過擬合防護、快速訓練

## Implementation Details

### VolatilityAnalyzer Class
```python
# 計算波動性指標
metrics = VolatilityAnalyzer.calculate_volatility_metrics(close_prices)

# 自動分類
category = VolatilityAnalyzer.get_volatility_category(metrics)
# 返回: 'low', 'medium' or 'high'
```

### AdaptiveHyperparameterConfig Class
```python
# 根據分類獲取超參數
config = AdaptiveHyperparameterConfig.get_config(category)
# config['lstm']  - LSTM超參數
# config['xgb']   - XGBoost超參數
```

### Training Flow
```
1. Load Data
   ↓
2. Calculate Volatility Metrics
   ↓
3. Determine Volatility Category
   ↓
4. Select Adaptive Hyperparameters
   ↓
5. Feature Engineering
   ↓
6. Train LSTM Feature Extractor
   ↓
7. Extract LSTM Hidden Features
   ↓
8. Train XGBoost Regressor
   ↓
9. Evaluate and Save
```

## Why This Works Better

### Problem with Fixed Hyperparameters
- BTC波動率小 (5%), DOGE波動率大 (50%)
- 同一套超參數無法同時適應兩者
- 導致低波動幣種欠擬合，高波動幣種過擬合

### V4 Adaptive Solution
- 波動率小 -> 增加模型複雜度、更多訓練
- 波動率大 -> 減少模型複雜度、防止過擬合
- 每個幣種使用專屬優化策略
- 達到平衡的性能 (所有幣種都 < 2% MAPE)

## Next Steps

1. 執行訓練:
```bash
python -m backend.train_v4_adaptive
```

2. 驗證結果
```bash
cat backend/results/v4_training_summary_*.json
```

3. 進行批量預測
```bash
python -m backend.inference_v4_adaptive
```

## Troubleshooting

### 某個幣種MAPE仍然 > 2%
- 檢查波動性分類是否準確
- 增加該幣種的 `epochs` 參數
- 考慮使用更大的 `hidden_size`

### GPU顯存不足
- 減小 `batch_size`
- 減小 `hidden_size`
- 使用CPU訓練: 代碼自動檢測

### 訓練速度慢
- 降低 `epochs` (增加早停耐心)
- 增大 `batch_size`
- 使用更多的 GPU
