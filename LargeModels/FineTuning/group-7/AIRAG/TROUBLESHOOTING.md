# 🔧 故障排除指南

本文档帮助解决常见问题。

---

## 🚨 问题1: API密钥错误

### 错误信息
```
LLM初始化失败: 使用API模式需要提供API密钥
Error code: 401 - Incorrect API key provided
```

### ✅ 解决方案

**方法1: 使用修复后的配置（推荐）**

`config.py` 已更新，包含默认的小航API密钥，直接使用即可：

```bash
python web_demo_enhanced.py
```

**方法2: 创建 .env 文件**

```bash
# 复制配置文件
copy env.example .env

# 编辑 .env 文件，确保包含：
MODEL_TYPE=api
LLM_MODEL_NAME=xhang
OPENAI_API_KEY=f93082e1-2cbf-4f81-af8f-9c98d528b6b1
OPENAI_BASE_URL=https://xhang.buaa.edu.cn/xhang/v1
```

**方法3: 检查配置**

打开 `config.py` 文件，确认第26行：
```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "f93082e1-2cbf-4f81-af8f-9c98d528b6b1")
```

---

## 🚨 问题2: HuggingFace 模型下载失败

### 错误信息
```
Connection aborted
ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。')
thrown while requesting HEAD https://huggingface.co/...
```

### ✅ 解决方案

**方法1: 使用修复版启动脚本（推荐）**

```bash
run_web_fixed.bat
```

这个脚本会自动配置 HuggingFace 镜像。

**方法2: 手动设置环境变量**

在运行前执行：
```bash
set HF_ENDPOINT=https://hf-mirror.com
python web_demo_enhanced.py
```

**方法3: 使用已下载的模型（最快）**

如果模型已经下载过一次，会自动使用本地缓存，无需重新下载。

模型缓存位置：
```
C:\Users\你的用户名\.cache\huggingface\hub\
```

**方法4: 换用更小的嵌入模型**

编辑 `config.py` 或 `.env`：
```bash
EMBEDDING_MODEL=shibing624/text2vec-base-chinese-paraphrase
```

或者使用已经在系统中的模型。

---

## 🚨 问题3: 日志乱码

### 问题描述
```
��ʼ��ʼ��RAGϵͳ...
�����ļ�: sample_ai.txt
```

### ✅ 解决方案

**方法1: 已在启动脚本中修复**

`run_web_fixed.bat` 第2行：
```batch
chcp 65001 > nul
```
会自动设置UTF-8编码。

**方法2: 手动设置编码**

在运行前执行：
```bash
chcp 65001
python web_demo_enhanced.py
```

**方法3: 查看日志文件（推荐）**

日志文件是UTF-8编码，不会乱码：
```bash
type data\logs\web_demo.log
```

---

## 🚨 问题4: 端口占用错误

### 错误信息
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 7860)
通常每个套接字地址(协议/网络地址/端口)只允许使用一次。
```

### ✅ 解决方案（已自动修复）

**方法1: 使用新版启动脚本（推荐）**

程序会自动尝试多个端口（7860-7864），找到可用的端口：

```bash
START.bat
```

或
```bash
run_web_fixed.bat
```

**方法2: 手动关闭占用的进程**

运行清理工具：
```bash
kill_port.bat
```

这会自动关闭所有占用7860-7864端口的进程。

**方法3: 手动查找并关闭**

```bash
# 1. 查找占用端口的进程
netstat -ano | findstr :7860

# 2. 记下进程ID（PID），例如 5748

# 3. 关闭进程（需要管理员权限）
taskkill /F /PID 5748
```

**方法4: 重启电脑**

最简单但最慢的方法。

---

## 🚨 问题5: 初始化时间过长

### 问题描述
初始化需要几分钟，不断重试连接 HuggingFace。

### ✅ 解决方案

**核心原因**: 首次运行需要下载嵌入模型（约 400MB）

**方法1: 使用 HuggingFace 镜像（推荐）**

运行修复版脚本：
```bash
run_web_fixed.bat
```

**方法2: 提前下载模型**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
```

**方法3: 使用更小的模型**

编辑 `config.py`：
```python
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese-paraphrase"  # 更小更快
```

**方法4: 耐心等待**

首次下载需要5-10分钟（取决于网速）。模型下载完成后，后续启动只需几秒。

**进度提示**:
- 第1次运行: 5-10分钟（下载模型）
- 第2次运行: 30秒（加载本地模型）
- 第3次运行: 10秒（使用缓存）

---

## 🚨 问题6: 查询时小航API调用失败

### 错误信息
```
Error code: 401 - Incorrect API key provided
```

### ✅ 解决方案

**检查配置加载**

运行测试脚本：
```bash
python test_xiaohang.py
```

如果测试通过，说明配置正确。

**强制刷新配置**

1. 关闭所有 Python 进程
2. 删除 `__pycache__` 目录
3. 重新启动

**验证API Key**

打开 `config.py`，确认第26行有正确的API Key。

---

## 📋 快速检查清单

运行前检查：

- [ ] 已安装依赖: `install_deps.bat`
- [ ] 已配置API Key: 检查 `config.py` 第26行
- [ ] 已设置HF镜像: 使用 `run_web_fixed.bat`
- [ ] 文档目录有文件: 检查 `data/documents/`
- [ ] 端口未占用: 7860端口可用

---

## 🚀 推荐启动流程

### 首次运行

```bash
# 1. 安装依赖（只需一次）
install_deps.bat

# 2. 等待完成（5-10分钟）

# 3. 使用修复版启动
run_web_fixed.bat

# 4. 打开浏览器
http://localhost:7860

# 5. 初始化系统（Web界面中操作）
```

### 后续运行

```bash
# 直接启动即可（快速）
run_web_fixed.bat
```

---

## 📞 仍然有问题？

### 查看详细日志

```bash
# 查看最新日志
type data\logs\web_demo.log

# 查看所有日志
dir data\logs\
```

### 测试各个组件

```bash
# 测试小航API
python test_xiaohang.py

# 测试文档加载
python -c "from src.document_loader import DocumentProcessor; print('OK')"

# 测试嵌入模型
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

### 完全重置

```bash
# 1. 删除虚拟环境
rmdir /s /q ai_rag
rmdir /s /q venv

# 2. 删除缓存
rmdir /s /q __pycache__
rmdir /s /q src\__pycache__

# 3. 重新安装
install_deps.bat
```

---

## 📚 相关文档

- **快速开始**: `QUICK_START_XIAOHANG.md`
- **小航API指南**: `docs/XIAOHANG_API_GUIDE.md`
- **项目README**: `README.md`
- **环境配置**: `env.example`

---

## 💡 常用命令

```bash
# 查看Python版本
python --version

# 查看已安装包
pip list

# 查看端口占用
netstat -ano | findstr :7860

# 设置UTF-8编码
chcp 65001

# 设置HF镜像
set HF_ENDPOINT=https://hf-mirror.com

# 激活虚拟环境
ai_rag\Scripts\activate

# 启动Web界面
python web_demo_enhanced.py
```

---

**最后更新**: 2025-10-19

如问题依然存在，请提供：
1. 错误信息截图
2. `data/logs/web_demo.log` 文件内容
3. `python --version` 输出
4. `pip list` 输出

