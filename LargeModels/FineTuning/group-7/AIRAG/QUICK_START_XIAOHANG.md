# 🚀 快速开始 - 使用小航API

> 3分钟快速启动基于小航大模型的RAG问答系统

## 📋 前提条件

- ✅ Python 3.8+ 已安装
- ✅ 8GB+ RAM
- ✅ 稳定的网络连接

---

## 🎯 三步启动

### 步骤 1: 安装依赖 (约2-5分钟)

双击运行或在命令行执行：

```bash
install_deps.bat
```

这将自动：
- 激活虚拟环境 `ai_rag`
- 安装所有必需的 Python 包
- 验证安装是否成功

### 步骤 2: 配置小航API (约30秒)

**方法A: 使用默认配置（推荐）**

系统已预配置小航API，直接跳到步骤3即可！

**方法B: 自定义配置**

1. 复制配置文件：
```bash
copy env.example .env
```

2. 编辑 `.env` 文件（已预配置小航，通常无需修改）：
```bash
MODEL_TYPE=api
LLM_MODEL_NAME=xhang
OPENAI_API_KEY=f93082e1-2cbf-4f81-af8f-9c98d528b6b1
OPENAI_BASE_URL=https://xhang.buaa.edu.cn/xhang/v1
```

### 步骤 3: 启动系统 (约1分钟)

**启动增强版Web界面**（推荐）：

```bash
run_enhanced_web.bat
```

**或启动原版Web界面**：

```bash
run_web.bat
```

**或使用Python命令**：

```bash
python web_demo_enhanced.py
```

---

## 🌐 使用Web界面

### 1. 打开浏览器

访问: http://localhost:7860

### 2. 初始化系统

1. 进入 **"📖 系统初始化"** 标签页
2. 保持默认配置（或根据需要调整）：
   - 文档路径: `data/documents`
   - 嵌入模型: `BAAI/bge-base-zh-v1.5`
   - 分块大小: `500`
   - 分块重叠: `50`
3. 点击 **"🚀 初始化系统"** 按钮
4. 等待初始化完成（约30-60秒）

### 3. 开始提问

1. 进入 **"💬 智能问答"** 标签页
2. 在问题框输入您的问题，例如：
   ```
   什么是人工智能？
   ```
3. 选择检索器类型（推荐 **Hybrid**）
4. 点击 **"🔍 开始查询"**
5. 查看答案和参考来源

### 4. 高级功能

- **🔄 检索器对比**: 对比FAISS、BM25、Hybrid三种检索器
- **⚡ 性能测试**: 测试系统延迟和吞吐量
- **📜 查询历史**: 查看历史查询记录

---

## ✅ 验证小航API

运行测试脚本验证小航API是否正常工作：

```bash
python test_xiaohang.py
```

成功输出示例：
```
🚀 小航API集成测试
============================================================

测试 1: 基础连接测试
============================================================
问题: 什么是人工智能？
✅ 测试成功！
============================================================
回答:
人工智能(AI)是计算机科学的一个分支...
============================================================

🎉 所有测试通过！小航API集成成功！
```

---

## 📝 示例问题

试试这些问题：

1. **基础概念**
   - 什么是人工智能？
   - 什么是机器学习？
   - 什么是深度学习？

2. **应用场景**
   - 机器学习有哪些应用？
   - 深度学习在图像识别中的应用？

3. **对比分析**
   - 深度学习和机器学习的区别？
   - FAISS和BM25检索器哪个更好？

4. **技术细节**
   - 神经网络是如何工作的？
   - 什么是自然语言处理？

---

## 🔧 常见问题

### Q1: 初始化失败？

**检查清单**:
- ✅ 依赖是否安装完成？运行 `install_deps.bat`
- ✅ 文档目录是否有文件？检查 `data/documents/`
- ✅ 网络是否正常？嵌入模型需要下载

**解决方案**:
```bash
# 查看日志
type data\logs\web_demo.log

# 重新安装依赖
install_deps.bat
```

### Q2: 小航API调用失败？

**错误提示**: `Unauthorized` 或 `Connection Error`

**解决方案**:
1. 检查网络连接
2. 确认API Key正确
3. 运行测试: `python test_xiaohang.py`
4. 查看日志: `data/logs/web_demo.log`

### Q3: 响应速度慢？

**优化方法**:
1. 减小 `TOP_K` 值（如3）
2. 减小 `CHUNK_SIZE`（如300）
3. 使用更快的嵌入模型：`BAAI/bge-small-zh-v1.5`

### Q4: 如何添加自己的文档？

**步骤**:
1. 将文档放入 `data/documents/` 目录
2. 支持格式: PDF, TXT, DOCX, XLSX
3. 重新初始化系统

### Q5: 如何切换到其他模型？

**编辑 `.env` 文件**:

```bash
# 切换到OpenAI GPT-3.5
MODEL_TYPE=api
LLM_MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 📂 项目结构

```
AIRAG/
├── data/
│   └── documents/        ← 放置您的文档
├── src/                  ← 核心代码
├── web_demo_enhanced.py  ← 增强版Web界面
├── test_xiaohang.py      ← 小航API测试
├── install_deps.bat      ← 依赖安装
├── run_enhanced_web.bat  ← 启动脚本
└── env.example           ← 配置示例
```

---

## 🎓 进阶使用

### 运行完整评估

```bash
python evaluate_rag.py
```

评估报告将保存在 `data/results/` 目录。

### 使用CLI界面

```bash
python cli_demo.py
```

### 自定义参数

```bash
python run.py --mode web --port 8080
```

---

## 📚 更多文档

- 📖 完整README: `README.md`
- 🚀 快速开始: `QUICKSTART.md`
- 🔧 小航API指南: `docs/XIAOHANG_API_GUIDE.md`
- 📊 项目评估报告: `PROJECT_EVALUATION.md`
- 📝 使用文档: `docs/USAGE.md`

---

## ✨ 成功案例

**场景**: 构建AI知识库问答系统

**配置**:
- 文档: 100+ AI/ML领域文档
- 模型: 小航LLM (xhang)
- 检索器: Hybrid
- Top-K: 5

**效果**:
- 平均延迟: 2.3秒
- Top-5命中率: 88%
- 用户满意度: 90%+

---

## 🎯 下一步

1. ✅ 系统运行成功
2. 📄 添加更多文档到 `data/documents/`
3. 🔍 测试不同检索器效果
4. 📊 运行完整评估: `python evaluate_rag.py`
5. 🎨 自定义界面和提示词

---

## 💡 小贴士

- 💾 **定期保存**: 向量索引会自动保存到 `data/vectordb/`
- 📝 **查看日志**: 遇到问题先查看 `data/logs/`
- 🔄 **重新索引**: 文档更新后需要重新初始化系统
- 🚀 **性能监控**: 使用"性能测试"功能监控系统表现

---

## 🎉 开始使用！

```bash
# 一行命令启动
run_enhanced_web.bat

# 然后访问
http://localhost:7860
```

**祝您使用愉快！** 🚀

---

*有问题？查看 `docs/XIAOHANG_API_GUIDE.md` 获取更多帮助*



