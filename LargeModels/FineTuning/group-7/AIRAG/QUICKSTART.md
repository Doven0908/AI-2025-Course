# 🚀 快速开始指南

## 5分钟快速上手

### Step 1: 安装依赖

#### Windows
```bash
# 双击运行
start.bat
```

#### Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

### Step 2: 准备文档

将您的文档放入 `data/documents/` 目录：

```
data/documents/
├── 文档1.pdf
├── 文档2.txt
└── 文档3.docx
```

**已包含示例文档**：
- `sample_ai.txt` - 人工智能简介
- `sample_ml.txt` - 机器学习详解

### Step 3: 开始使用

启动脚本会自动显示菜单，选择运行模式：

```
请选择运行模式:
1. Web界面 (推荐)    ← 输入 1
2. CLI命令行
3. 索引文档
4. 运行评估
5. 退出
```

选择 **1** 后，访问 http://localhost:7860

## 常见问题

### Q1: 依赖安装失败？

**解决方案**：
```bash
# 手动安装
pip install langchain langchain-community langchain-openai
pip install faiss-cpu sentence-transformers
pip install gradio rich loguru
pip install pypdf python-docx pandas numpy
```

### Q2: 模型下载慢？

**解决方案**：使用国内镜像
```bash
# 设置HuggingFace镜像
set HF_ENDPOINT=https://hf-mirror.com

# 或下载后手动指定路径
EMBEDDING_MODEL=/path/to/local/model
```

### Q3: 中文显示乱码？

**解决方案**：
- Windows: 确保命令行使用UTF-8编码
- 已自动设置：`chcp 65001`

### Q4: 内存不足？

**解决方案**：在 `.env` 中调整参数
```bash
CHUNK_SIZE=300        # 减小块大小
TOP_K=3               # 减少检索数量
```

## 快速测试

运行系统测试：
```bash
python test_system.py
```

## Web界面使用

1. **初始化系统**
   - 进入"系统初始化"标签
   - 点击"初始化系统"
   - 等待完成

2. **提问**
   - 进入"问答"标签
   - 输入问题
   - 选择检索器（推荐：Hybrid）
   - 点击"查询"

3. **查看结果**
   - 阅读生成的答案
   - 查看参考来源
   - 检查相关度评分

## CLI使用

```bash
# 启动交互模式
python cli_demo.py --interactive

# 使用命令
>>> 什么是人工智能？           # 提问
>>> /compare 机器学习是什么？   # 对比检索器
>>> /retriever hybrid          # 切换检索器
>>> /topk 5                    # 设置Top-K
>>> /help                      # 帮助
>>> /quit                      # 退出
```

## 配置API（可选）

如需使用LLM生成功能，编辑 `.env`：

```bash
MODEL_TYPE=api
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 下一步

- 📖 查看 [完整文档](README.md)
- 📚 阅读 [使用指南](docs/USAGE.md)
- 🔧 参考 [API文档](docs/API.md)

## 获取帮助

- 💬 提交Issue
- 📧 联系: your.email@example.com

---

**提示**: 首次运行需要下载模型，可能需要几分钟时间。

