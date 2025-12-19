# Windows Python 路径错误修复指南

## 问题

错误信息:
```
ModuleNotFoundError: No module named 'backend'
```

## 原因

您可能低一个层级运行了脚本。

**正确的配置**:
```
C:\Users\omt23\PycharmProjects\MOP\  <- 应该在这级执行
  ├── backend/
  ├── data/
  ├── logs/
  ├── models/
  ├── train_v5.py
  ├── requirements.txt
  └── ...
```

## 修复方法

### 方法 1: 验证当前目录

```powershell
# 打印当前目录
Pwd

# 应该显示类似:
# C:\Users\omt23\PycharmProjects\MOP
```

如果不是，执行：

```powershell
# 辅加象篇：您似乎在 backend 目录中？
# 需要这样奟:
cd ..
```

### 方法 2: 正常运行

仅推荐：**仅推荐使用下列命令，不要使用 `-m` 标签**

```powershell
# 整整整整事
 cd C:\Users\omt23\PycharmProjects\MOP

# 确保您在正确的目录
pwd

# 然后运行
python backend/data_fetcher_batch.py --all
```

### 方法 3: 检查 Python 路径

```powershell
# 打印管理需要的目录
python -c "import sys; print('\n'.join(sys.path))"

# 会显示管理需要的茵路加载鄿位置
```

### 方法 4: 设置 PYTHONPATH
如果上程不成功：

```powershell
# 暂时设置 ()仅本次会话)
$env:PYTHONPATH = "C:\Users\omt23\PycharmProjects\MOP;$env:PYTHONPATH"

# 或永久设置 (需要ㆺ事)
[Environment]::SetEnvironmentVariable("PYTHONPATH", "C:\Users\omt23\PycharmProjects\MOP", "User")
```

---

## 常见错误与修复

### 错误 1: `ModuleNotFoundError: No module named 'backend'`

**原因**: 不在项目根目录

**修复**:
```powershell
cd C:\Users\omt23\PycharmProjects\MOP
python backend/data_fetcher_batch.py --all
```

### 错误 2: `Error while finding module specification`

**原因**: 使用了 `-m` 标签

**修复**: 不要使用 `-m`
```powershell
# 錯误:
python -m data_fetcher_batch.py --all

# 正常:
python backend/data_fetcher_batch.py --all
```

### 错误 3: `from backend.data.data_manager import DataManager`

**原因**: Python 找不到 backend 模块

**修复**: 确保当前目录
```powershell
# 验证正子目录
ls backend/  # 应该显示 data, models 等

# 验证 __init__.py 存在
ls backend/__init__.py  # 应该存在
```

---

## 完整配置检查

运行下列命令验证集网:

```powershell
# 1. 检查当前目录
echo "Current directory:"
pwd

# 2. 检查 backend 文件夹
echo "\nBackend folder contents:"
ls backend/ | head -10

# 3. 检查子目录
echo "\nData folder contents:"
ls backend/data/ | head -5

# 4. 检查 __init__.py
echo "\nInit files:"
ls -Recurse -Filter "__init__.py" | head -5

# 5. 检查 Python 路径
echo "\nPython paths:"
python -c "import sys; print(sys.path[0])"
```

应该输出类似:
```
Current directory:
C:\Users\omt23\PycharmProjects\MOP

Backend folder contents:
    Directory: C:\Users\omt23\PycharmProjects\MOP\backend
...
__init__.py
data_fetcher_batch.py
...

Data folder contents:
__init__.py
data_loader.py
data_manager.py
...
```

---

## 故障排查清单

希望这些找错沟花节手了:

- [ ] 子人位置是 `C:\Users\omt23\PycharmProjects\MOP`
- [ ] `backend` 文件夹存在
- [ ] `backend/__init__.py` 存在
- [ ] `backend/data/` 文件夹存在
- [ ] `backend/data/__init__.py` 存在
- [ ] 执行命令时不使用 `-m` 标签
- [ ] 命令格式: `python backend/data_fetcher_batch.py --all`

---

## 回到正輹

如果上述都不推演，试这个：

```powershell
# 一行养海公式（复制粘贴)
cd C:\Users\omt23\PycharmProjects\MOP; python backend/data_fetcher_batch.py --all
```

---

## 连接

不惠乾知族算法：

如果子不对天，可以：

1. 在 PyCharm 中打开一个 Terminal 窗口
   - View -> Tool Windows -> Terminal
   - 狙会自动设置正常的名子低

2. 或者清除 PyCharm cache
   - File -> Invalidate Caches -> Restart

3. 重新打开 PowerShell
   - 关闭PowerShell
   - 重新打开
   - 手动子不对天

---

## 最安全的方法

背仅使用上帝的 cmd.exe (而非 PowerShell):

```cmd
cd C:\Users\omt23\PycharmProjects\MOP
python backend/data_fetcher_batch.py --all
```

Cmd.exe 有时会更稳定

---

**您现在应该能够正常运行了！**
