# 紧急修复 - Windows Python 路径错误

## 您遇到的错误

```
ModuleNotFoundError: No module named 'backend'
```

## 原因

您不在项目根目录运行命令

---

## 快速修复

### 步骤 1: 进入项目根目录

```powershell
cd C:\Users\omt23\PycharmProjects\MOP
```

### 步骤 2: 验证位置

```powershell
pwd
```

应该显示:
```
C:\Users\omt23\PycharmProjects\MOP
```

### 步骤 3: 运行正确的命令

```powershell
python backend/data_fetcher_batch.py --all
```

---

## ❌ 常见错误

### 错误 1: 使用了 `-m` 标记

```powershell
python -m data_fetcher_batch.py --all  # ❌ 错误
```

### 错误 2: 在子目录中运行

```powershell
cd backend
python data_fetcher_batch.py --all  # ❌ 错误
```

### 错误 3: 不在项目根目录

```powershell
# ❌ 如果 pwd 输出不是项目根目录
```

---

## ✓ 正确的做法

```powershell
# 确保在项目根目录
cd C:\Users\omt23\PycharmProjects\MOP

# 验证
pwd  # 应该显示项目根目录

# 运行正确的命令
python backend/data_fetcher_batch.py --all
```

---

## 验证您在正确位置

```powershell
ls backend
```

应该看到:
```
__init__.py
data/
models/
data_fetcher_batch.py
data_manager.py
etc.
```

---

## 三个核心命令

### 所有命令都在项目根目录运行

#### 流程 1: 获取数据

```powershell
python backend/data_fetcher_batch.py --all
```

#### 流程 2: 训练模型

```powershell
python train_v5.py --all --device cuda
```

#### 流程 3: 上传到 HF

```powershell
python push_to_huggingface_batch.py --token "hf_YOUR_TOKEN" --repo "your_username/mop-v5-models" --all
```

---

## 使用 PyCharm Terminal (推荐)

如果您在 PyCharm 中工作:

1. 点击 **View** -> **Tool Windows** -> **Terminal**
2. PyCharm 会自动设置正确的工作目录
3. 直接运行命令，不需要 `cd`

这样最不容易出错！

---

## 一行快速测试

```powershell
cd C:\Users\omt23\PycharmProjects\MOP && pwd && ls backend && echo "✓ 正确位置！"
```

如果看到文件列表和 "✓ 正确位置！"，说明您准备好了。

---

## 检查清单

运行前确认:

- [ ] 当前目录: `C:\Users\omt23\PycharmProjects\MOP`
- [ ] 验证方式: `pwd` 或 `ls backend`
- [ ] 命令格式: `python backend/data_fetcher_batch.py --all`
- [ ] 不使用: `-m` 标记
- [ ] 位置正确: backend 文件夹可见

---

## 现在就开始

```powershell
cd C:\Users\omt23\PycharmProjects\MOP
python backend/data_fetcher_batch.py --all
```

应该能看到数据开始下载！

---

## 需要详细帮助？

查看完整指南: [WINDOWS_SETUP_FIX.md](WINDOWS_SETUP_FIX.md)
