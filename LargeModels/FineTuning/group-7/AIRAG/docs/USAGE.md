# RAG问答系统使用指南

## 目录
1. [快速入门](#快速入门)
2. [详细配置](#详细配置)
3. [使用示例](#使用示例)
4. [高级功能](#高级功能)
5. [故障排除](#故障排除)

## 快速入门

### 第一步：环境准备

#### Windows用户
```bash
# 双击运行或在命令行执行
start.bat
```

#### Linux/Mac用户
```bash
# 在终端执行
chmod +x start.sh
./start.sh
```

### 第二步：准备文档

将您的文档放入 `data/documents/` 目录：

```
data/documents/
├── AI基础知识.pdf
├── 机器学习教程.docx
└── 深度学习笔记.txt
```

支持的文档格式：
- PDF (.pdf)
- Word文档 (.docx, .doc)
- 文本文件 (.txt)

### 第三步：开始使用

选择您喜欢的方式：

#### 方式1：Web界面（推荐）
```bash
python run.py --mode web
```
然后访问 http://localhost:7860

#### 方式2：命令行
```bash
python run.py --mode cli
```

## 详细配置

### 环境变量配置

编辑 `.env` 文件：

```bash
# ============ 模型配置 ============

# 使用模式：local（本地）或 api（在线API）
MODEL_TYPE=local

# 如果使用在线API，配置以下参数
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# ============ 嵌入模型 ============

# 中文场景推荐
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
# 或使用：BAAI/bge-large-zh-v1.5 (更好但更慢)

# 多语言场景
# EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# ============ 向量数据库 ============

# 类型：faiss 或 chroma
VECTOR_DB_TYPE=faiss

# 存储路径
VECTOR_DB_PATH=./data/vectordb

# ============ 检索配置 ============

# Top-K：返回最相关的K个文档片段
TOP_K=5

# 文档分块大小（字符数）
CHUNK_SIZE=500

# 文档分块重叠（字符数）
CHUNK_OVERLAP=50

# ============ 日志配置 ============

# 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

### 嵌入模型选择指南

| 模型 | 语言 | 维度 | 性能 | 推荐场景 |
|------|------|------|------|---------|
| BAAI/bge-base-zh-v1.5 | 中文 | 768 | 快速 | 中文文档，平衡性能 |
| BAAI/bge-large-zh-v1.5 | 中文 | 1024 | 较慢 | 中文文档，高质量 |
| paraphrase-multilingual-MiniLM-L12-v2 | 多语言 | 384 | 最快 | 多语言，快速检索 |
| all-MiniLM-L6-v2 | 英文 | 384 | 最快 | 英文文档 |

### 文档分块参数调优

**CHUNK_SIZE（分块大小）**
- 较小（200-300）：检索更精确，但可能丢失上下文
- 中等（500-800）：平衡精确度和上下文（推荐）
- 较大（1000+）：保留更多上下文，但检索可能不够精确

**CHUNK_OVERLAP（重叠大小）**
- 建议设置为 chunk_size 的 10-20%
- 避免在分块边界丢失重要信息

## 使用示例

### 1. Web界面使用

#### 步骤1：初始化系统
1. 打开Web界面
2. 进入"系统初始化"标签
3. 配置参数：
   - 文档路径：`data/documents`
   - 嵌入模型：`BAAI/bge-base-zh-v1.5`
   - 文档块大小：`500`
   - 文档块重叠：`50`
4. 点击"初始化系统"按钮
5. 等待初始化完成

#### 步骤2：问答
1. 进入"问答"标签
2. 输入问题，例如："什么是机器学习？"
3. 选择检索器类型：FAISS / BM25 / Hybrid
4. 设置Top-K值（建议3-5）
5. 勾选"显示来源文档"
6. 点击"查询"按钮
7. 查看答案和参考来源

#### 步骤3：对比检索器
1. 进入"检索器对比"标签
2. 输入相同的问题
3. 点击"开始对比"
4. 查看不同检索器的结果差异

### 2. CLI命令行使用

#### 交互模式

```bash
# 启动交互模式
python cli_demo.py --interactive

# 在交互模式中：
>>> 什么是人工智能？        # 直接输入问题
>>> /retriever faiss        # 切换检索器
>>> /topk 3                 # 设置Top-K
>>> /compare 机器学习是什么？  # 对比检索器
>>> /help                   # 显示帮助
>>> /quit                   # 退出
```

#### 单次查询模式

```bash
# 基本查询
python cli_demo.py --question "什么是深度学习？"

# 指定检索器
python cli_demo.py \
    --question "机器学习的应用有哪些？" \
    --retriever hybrid \
    --topk 5

# 对比检索器
python cli_demo.py \
    --question "人工智能的未来趋势" \
    --compare
```

### 3. 编程方式使用

```python
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import RetrieverManager
from src.generator import AnswerGenerator, RAGPipeline
import config

# 1. 加载和索引文档
processor = DocumentProcessor()
chunks = processor.process_documents("data/documents")

# 2. 创建向量索引
vs_manager = VectorStoreManager()
vs_manager.create_vectorstore(chunks)
vs_manager.save("my_index")

# 3. 初始化检索器
retriever = RetrieverManager(vs_manager)
retriever.setup_bm25(chunks)

# 4. 创建RAG流水线
generator = AnswerGenerator()
rag = RAGPipeline(retriever, generator)

# 5. 执行查询
result = rag.query(
    question="什么是机器学习？",
    retriever_type="hybrid",
    k=5
)

# 6. 获取结果
print(f"答案: {result['answer']}")
print(f"来源数量: {len(result['sources'])}")
```

## 高级功能

### 1. 自定义检索策略

#### 混合检索权重调整

```python
from src.retriever import RetrieverManager

retriever = RetrieverManager(vs_manager)

# 调整FAISS和BM25的权重
results = retriever.hybrid_retrieve(
    query="你的问题",
    k=5,
    faiss_weight=0.7  # FAISS权重70%，BM25权重30%
)
```

### 2. 批量评估

```python
# 创建测试用例文件 test_cases.json
[
    {
        "question": "什么是人工智能？",
        "ground_truth": "人工智能是...",
        "expected_keywords": ["AI", "计算机", "智能"]
    },
    {
        "question": "机器学习的应用？",
        "ground_truth": "机器学习应用包括...",
        "expected_keywords": ["应用", "图像", "语音"]
    }
]

# 运行评估
python evaluate_rag.py --test-file test_cases.json
```

### 3. 自定义提示模板

```python
from langchain.prompts import PromptTemplate

# 自定义提示模板
custom_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""基于以下上下文回答问题，要求：
1. 答案简洁明了
2. 用中文回答
3. 标注信息来源

上下文：
{context}

问题：{question}

回答："""
)

# 使用自定义模板
from src.generator import AnswerGenerator
generator = AnswerGenerator()
generator.prompt_template = custom_template
```

### 4. 增量索引

```python
# 加载已有索引
vs_manager = VectorStoreManager()
vs_manager.load("existing_index")

# 处理新文档
processor = DocumentProcessor()
new_chunks = processor.process_documents("new_documents/")

# 添加到现有索引（FAISS）
from langchain_community.vectorstores import FAISS
new_db = FAISS.from_documents(new_chunks, vs_manager.embeddings)
vs_manager.vectorstore.merge_from(new_db)

# 保存更新后的索引
vs_manager.save("existing_index")
```

## 故障排除

### 常见问题

#### 1. 模块导入错误

**问题**: `ModuleNotFoundError: No module named 'langchain'`

**解决方案**:
```bash
# 确保在虚拟环境中
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 重新安装依赖
pip install -r requirements.txt
```

#### 2. 嵌入模型下载失败

**问题**: 无法下载HuggingFace模型

**解决方案**:
```bash
# 方法1：使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法2：手动下载模型
# 1. 访问 https://hf-mirror.com/BAAI/bge-base-zh-v1.5
# 2. 下载所有文件到本地目录
# 3. 修改.env中的EMBEDDING_MODEL为本地路径
EMBEDDING_MODEL=/path/to/local/model
```

#### 3. 内存不足

**问题**: `MemoryError` 或系统卡顿

**解决方案**:
```bash
# 1. 减小chunk_size
CHUNK_SIZE=300

# 2. 减小top_k
TOP_K=3

# 3. 使用更小的嵌入模型
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 4. 批量处理文档
# 分批索引大量文档
```

#### 4. FAISS索引错误

**问题**: `RuntimeError: Error in void faiss::...`

**解决方案**:
```bash
# 删除现有索引重新创建
rm -rf data/vectordb/*

# 或强制重新索引
python cli_demo.py --force-reindex
```

#### 5. LLM API错误

**问题**: `openai.error.AuthenticationError`

**解决方案**:
```bash
# 检查.env配置
# 1. 确保API密钥正确
OPENAI_API_KEY=sk-your-correct-key

# 2. 检查API地址
OPENAI_BASE_URL=https://api.openai.com/v1

# 3. 如果使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

#### 6. 向量检索结果不准确

**解决方案**:

1. **优化文档分块**
```bash
# 调整分块参数
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

2. **更换嵌入模型**
```bash
# 使用更强的模型
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

3. **使用混合检索**
```python
# 结合向量检索和关键词检索
result = rag.query(
    question="你的问题",
    retriever_type="hybrid"
)
```

4. **增加Top-K**
```bash
TOP_K=10  # 检索更多文档
```

### 性能优化建议

#### 1. 加速嵌入编码

```python
# 使用GPU
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cuda'},  # 使用GPU
    encode_kwargs={'normalize_embeddings': True}
)
```

#### 2. 批量处理

```python
# 批量编码文档
batch_size = 32
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    # 处理批次
```

#### 3. 使用更快的向量数据库

```bash
# Chroma可能比FAISS更快（取决于场景）
VECTOR_DB_TYPE=chroma
```

### 日志和调试

#### 启用调试日志

```bash
# 修改.env
LOG_LEVEL=DEBUG
```

#### 查看日志文件

```bash
# 日志文件位置
data/logs/
├── web_demo.log
├── cli_demo.log
├── evaluation.log
└── run.log

# 查看最新日志
tail -f data/logs/web_demo.log
```

## 更多帮助

- 📖 查看 [README.md](../README.md) 了解项目概述
- 🐛 遇到问题？提交 [Issue](https://github.com/your-repo/issues)
- 💬 加入讨论：[Discussions](https://github.com/your-repo/discussions)
- 📧 联系作者：your.email@example.com

---

**提示**: 本指南持续更新中，欢迎提出改进建议！

