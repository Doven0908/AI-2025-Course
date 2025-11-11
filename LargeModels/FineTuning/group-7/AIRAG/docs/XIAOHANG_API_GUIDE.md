# 🚀 小航API集成指南

本文档介绍如何在 RAG 问答系统中使用**小航大模型**。

## 📋 目录

- [小航API介绍](#小航api介绍)
- [快速配置](#快速配置)
- [详细说明](#详细说明)
- [测试验证](#测试验证)
- [常见问题](#常见问题)

---

## 🤖 小航API介绍

小航是由**北京航空航天大学**开发的大语言模型，提供两种API接口：

### 1. 小航 LLM API（推荐用于RAG）

- **模型名称**: `xhang`
- **API Key**: `f93082e1-2cbf-4f81-af8f-9c98d528b6b1`
- **Base URL**: `https://xhang.buaa.edu.cn/xhang/v1`
- **接口格式**: 兼容 OpenAI 格式
- **用途**: 适合生成式任务，如RAG问答系统

### 2. 小航 Agent API（可选）

- **URL**: `https://api.xhang.buaa.edu.cn:28119/apps/llm/chat/agent`
- **API Key**: `c7bcd1f5-8cb4-4541-a4e2-510fee59ae70`
- **接口格式**: SSE 流式响应
- **用途**: 适合对话助手，支持联网搜索、画图等功能

---

## ⚡ 快速配置

### 方法一：使用环境变量文件（推荐）

1. **复制配置文件**

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

2. **编辑 `.env` 文件，使用小航配置**

```bash
# LLM 模型配置
MODEL_TYPE=api
LLM_MODEL_NAME=xhang
OPENAI_API_KEY=f93082e1-2cbf-4f81-af8f-9c98d528b6b1
OPENAI_BASE_URL=https://xhang.buaa.edu.cn/xhang/v1

# 嵌入模型配置
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5

# 向量数据库配置
VECTOR_DB_TYPE=faiss

# 检索配置
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

3. **启动系统**

```bash
# Windows
run_enhanced_web.bat

# Linux/Mac
python web_demo_enhanced.py
```

### 方法二：直接修改 config.py

编辑 `config.py` 文件：

```python
# 模型配置
MODEL_TYPE = "api"
LLM_MODEL_NAME = "xhang"
OPENAI_API_KEY = "f93082e1-2cbf-4f81-af8f-9c98d528b6b1"
OPENAI_BASE_URL = "https://xhang.buaa.edu.cn/xhang/v1"
```

---

## 📖 详细说明

### 配置参数解释

| 参数 | 值 | 说明 |
|------|-----|------|
| `MODEL_TYPE` | `api` | 使用在线API模式 |
| `LLM_MODEL_NAME` | `xhang` | 指定小航模型 |
| `OPENAI_API_KEY` | `f93082e1-...` | 小航LLM的API密钥 |
| `OPENAI_BASE_URL` | `https://xhang.buaa.edu.cn/xhang/v1` | 小航LLM的API地址 |

### 兼容性说明

小航LLM API **完全兼容 OpenAI 接口格式**，因此无需修改代码，只需更改配置即可。

本系统使用 LangChain 的 `ChatOpenAI` 类，通过修改 `openai_api_base` 和 `model` 参数即可无缝切换到小航API。

### 工作流程

```
用户问题
    ↓
文档检索 (FAISS/BM25/Hybrid)
    ↓
上下文准备
    ↓
调用小航LLM API ← 使用 xhang 模型
    ↓
生成带引用的答案
    ↓
返回给用户
```

---

## 🧪 测试验证

### 1. 命令行测试

创建测试脚本 `test_xiaohang.py`：

```python
from src.generator import AnswerGenerator
from langchain.schema import Document

# 初始化小航生成器
generator = AnswerGenerator(
    model_type="api",
    api_key="f93082e1-2cbf-4f81-af8f-9c98d528b6b1",
    base_url="https://xhang.buaa.edu.cn/xhang/v1",
    model_name="xhang"
)

# 测试文档
test_docs = [
    Document(
        page_content="人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
        metadata={"source": "test.txt"}
    )
]

# 测试问题
question = "什么是人工智能？"

# 生成答案
result = generator.generate_answer(question, test_docs)

print(f"问题: {result['question']}")
print(f"\n答案:\n{result['answer']}")
print(f"\n来源数量: {len(result['sources'])}")
```

运行测试：

```bash
python test_xiaohang.py
```

### 2. Web界面测试

1. 启动增强版Web界面：

```bash
# Windows
run_enhanced_web.bat

# Linux/Mac
python web_demo_enhanced.py
```

2. 打开浏览器访问：`http://localhost:7860`

3. 在"系统初始化"标签页初始化系统

4. 在"智能问答"标签页测试提问

### 3. 验证小航API是否正常工作

查看日志文件 `data/logs/web_demo.log`，应该看到类似信息：

```
INFO - 使用在线API初始化LLM: xhang
INFO - API Base URL: https://xhang.buaa.edu.cn/xhang/v1
INFO - 开始生成答案，问题: 什么是人工智能？
INFO - 答案生成完成
```

---

## ❓ 常见问题

### Q1: 如何切换回 OpenAI API？

**解决方案**: 修改 `.env` 文件：

```bash
MODEL_TYPE=api
LLM_MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Q2: 小航API响应速度慢？

**原因**: 
- 网络延迟
- 模型计算时间
- 并发请求过多

**解决方案**:
1. 减少 `CHUNK_SIZE` 和 `TOP_K` 以减少上下文长度
2. 使用缓存机制（可扩展实现）
3. 检查网络连接

### Q3: 出现 API Key 错误？

**错误信息**: `Unauthorized` 或 `Invalid API Key`

**解决方案**:
1. 确认API Key正确无误
2. 检查 `.env` 文件是否被正确加载
3. 联系小航API管理员确认Key状态

### Q4: 如何查看API调用详情？

**解决方案**: 设置日志级别为 DEBUG：

```bash
# .env 文件中
LOG_LEVEL=DEBUG
```

然后查看 `data/logs/` 目录下的日志文件。

### Q5: 小航LLM vs 小航Agent，选哪个？

| 特性 | 小航LLM | 小航Agent |
|------|---------|-----------|
| **用途** | 文本生成 | 对话助手 |
| **接口** | OpenAI格式 | SSE流式 |
| **RAG适用性** | ✅ 推荐 | ⚠️ 不推荐 |
| **功能** | 纯生成 | 搜索、画图等 |
| **集成难度** | 简单 | 复杂 |

**结论**: RAG系统使用**小航LLM**即可。

### Q6: 支持流式输出吗？

**当前状态**: 本系统暂不支持流式输出

**原因**: LangChain 的 `LLMChain` 默认是批量响应

**未来计划**: 可以扩展支持 `stream=True` 实现流式输出

---

## 🎯 性能优化建议

### 1. 嵌入模型优化

使用更快的嵌入模型：

```bash
# 轻量级，速度更快
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 或使用更小的模型
EMBEDDING_MODEL=shibing624/text2vec-base-chinese-paraphrase
```

### 2. 检索优化

调整检索参数：

```bash
# 减少返回文档数量
TOP_K=3

# 减小分块大小
CHUNK_SIZE=300
CHUNK_OVERLAP=30
```

### 3. API 调用优化

- 减少不必要的API调用
- 实现答案缓存机制
- 批量处理多个问题

---

## 📚 参考资源

- [小航API官方文档](https://xhang.buaa.edu.cn/)
- [LangChain文档](https://python.langchain.com/)
- [OpenAI API参考](https://platform.openai.com/docs/api-reference)

---

## 💡 技术支持

如有问题，请：

1. 查看日志文件：`data/logs/web_demo.log`
2. 检查配置文件：`.env` 或 `config.py`
3. 参考本文档的"常见问题"部分
4. 联系小航API技术支持

---

## 🎉 总结

使用小航API只需三步：

1. ✅ 配置 `.env` 文件（或修改 `config.py`）
2. ✅ 启动Web界面或CLI
3. ✅ 开始使用RAG问答系统

**小航API的优势**：
- 🇨🇳 中文优化
- 🔌 兼容OpenAI格式
- 🎓 学术研究友好
- 💰 成本可控

祝使用愉快！ 🚀


