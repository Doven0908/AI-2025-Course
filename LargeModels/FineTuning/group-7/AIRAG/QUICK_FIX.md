# ⚡ 一键修复指南

遇到问题？按顺序尝试以下解决方案。

---

## 🎯 方案1: 使用新的启动脚本（推荐）

**所有问题已自动修复！**

### 只需运行：

```bash
START.bat
```

这个脚本会：
- ✅ 自动配置 HuggingFace 镜像
- ✅ 自动激活虚拟环境  
- ✅ 自动寻找可用端口（7860-7864）
- ✅ 自动启动系统

---

## 🎯 方案2: 端口被占用？

### 运行清理工具：

```bash
kill_port.bat
```

然后重新启动：

```bash
START.bat
```

---

## 🎯 方案3: 完全重启

### 步骤：

1. **关闭所有 Python 进程**
   - 按 `Ctrl+C` 停止运行中的程序
   - 或任务管理器中结束 Python 进程

2. **重新启动**
   ```bash
   START.bat
   ```

---

## 🎯 方案4: 首次运行？

### 完整流程：

```bash
# 1. 安装依赖（仅首次需要）
install_deps.bat

# 2. 启动系统
START.bat

# 3. 打开浏览器
# 会自动显示访问地址，例如：
# http://localhost:7860
# 或
# http://localhost:7861
```

---

## 🎯 方案5: 仍然有问题？

### 检查清单：

- [ ] 已运行 `install_deps.bat`？
- [ ] Python 版本 3.8+ ？
- [ ] 网络正常？
- [ ] 有足够磁盘空间？（至少2GB）

### 查看详细错误：

```bash
type data\logs\web_demo.log
```

### 完全重置：

```bash
# 1. 删除缓存
rmdir /s /q __pycache__
rmdir /s /q src\__pycache__

# 2. 重新安装
install_deps.bat

# 3. 启动
START.bat
```

---

## 📊 常见错误速查表

| 错误信息 | 解决方案 |
|---------|---------|
| `端口 7860 已被占用` | 运行 `kill_port.bat` 或 `START.bat`（会自动换端口） |
| `API密钥错误` | 已修复，直接运行 `START.bat` |
| `模型下载失败` | 已配置镜像，耐心等待首次下载 |
| `日志乱码` | 不影响使用，查看 `data/logs/web_demo.log` |
| `虚拟环境未找到` | 运行 `install_deps.bat` |

---

## 🚀 最简单的方法

**只需记住一个命令：**

```bash
START.bat
```

一切都会自动处理！

---

## 💡 提示

### 首次运行需要等待

- 第1次：5-10分钟（下载模型）
- 第2次：30秒（加载模型）
- 第3次：10秒（使用缓存）

### 打开浏览器

程序启动后会显示：
```
Running on local URL:  http://0.0.0.0:7861
```

在浏览器中访问：
- `http://localhost:7861`
- 或程序显示的任何端口

---

## 📞 需要帮助？

1. **查看详细文档**: `TROUBLESHOOTING.md`
2. **查看日志**: `data/logs/web_demo.log`
3. **测试系统**: `python test_xiaohang.py`

---

**记住：遇到任何问题，先试试 `START.bat`！** 🚀





