# RAG问答系统 API文档

## 核心模块API

### 1. DocumentProcessor (文档处理器)

处理文档加载和分块。

```python
from src.document_loader import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=500,      # 文档块大小
    chunk_overlap=50     # 文档块重叠
)
```

#### 方法

**load_documents(path: str) -> List[Document]**

加载文档。

- **参数**:
  - `path`: 文档路径（文件或目录）
- **返回**: Document对象列表
- **支持格式**: PDF, TXT, DOCX

```python
documents = processor.load_documents("data/documents")
```

**split_documents(documents: List[Document]) -> List[Document]**

分块文档。

- **参数**:
  - `documents`: Document对象列表
- **返回**: 分块后的Document对象列表

```python
chunks = processor.split_documents(documents)
```

**process_documents(path: str) -> List[Document]**

完整处理流程：加载 + 分块。

```python
chunks = processor.process_documents("data/documents")
```

---

### 2. VectorStoreManager (向量数据库管理器)

管理向量数据库的创建、保存和加载。

```python
from src.vector_store import VectorStoreManager

vs_manager = VectorStoreManager(
    embedding_model_name="BAAI/bge-base-zh-v1.5",
    db_type="faiss",
    db_path="./data/vectordb"
)
```

#### 方法

**create_vectorstore(documents: List[Document]) -> None**

创建向量数据库。

```python
vs_manager.create_vectorstore(chunks)
```

**save(index_name: str = "default") -> None**

保存向量数据库。

```python
vs_manager.save("my_index")
```

**load(index_name: str = "default") -> None**

加载向量数据库。

```python
vs_manager.load("my_index")
```

**similarity_search(query: str, k: int = 5) -> List[Document]**

相似度搜索。

```python
results = vs_manager.similarity_search("查询问题", k=5)
```

**similarity_search_with_score(query: str, k: int = 5) -> List[Tuple[Document, float]]**

带评分的相似度搜索。

```python
results = vs_manager.similarity_search_with_score("查询问题", k=5)
# 返回: [(Document, score), ...]
```

---

### 3. RetrieverManager (检索器管理器)

管理多种检索策略。

```python
from src.retriever import RetrieverManager

retriever = RetrieverManager(vector_store_manager=vs_manager)
```

#### 方法

**setup_bm25(documents: List[Document]) -> None**

初始化BM25检索器。

```python
retriever.setup_bm25(chunks)
```

**retrieve_with_faiss(query: str, k: int = 5) -> List[Tuple[Document, float]]**

FAISS向量检索。

```python
results = retriever.retrieve_with_faiss("查询", k=5)
```

**retrieve_with_bm25(query: str, k: int = 5) -> List[Document]**

BM25关键词检索。

```python
results = retriever.retrieve_with_bm25("查询", k=5)
```

**hybrid_retrieve(query: str, k: int = 5, faiss_weight: float = 0.5) -> List[Tuple[Document, float]]**

混合检索。

- **参数**:
  - `query`: 查询文本
  - `k`: 返回结果数量
  - `faiss_weight`: FAISS权重（0-1）
- **返回**: [(Document, score), ...]

```python
results = retriever.hybrid_retrieve(
    query="查询",
    k=5,
    faiss_weight=0.7  # FAISS 70%, BM25 30%
)
```

**compare_retrievers(query: str, k: int = 5) -> dict**

对比不同检索器。

```python
comparison = retriever.compare_retrievers("查询", k=5)
# 返回: {
#     'query': "查询",
#     'faiss': [...],
#     'bm25': [...],
#     'hybrid': [...]
# }
```

---

### 4. AnswerGenerator (答案生成器)

基于检索结果生成带引用的答案。

```python
from src.generator import AnswerGenerator

generator = AnswerGenerator(
    model_type="api",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)
```

#### 方法

**generate_answer(question: str, retrieved_docs: List[Document], scores: Optional[List[float]] = None) -> Dict**

生成答案。

- **参数**:
  - `question`: 用户问题
  - `retrieved_docs`: 检索到的文档
  - `scores`: 文档相关度评分（可选）
- **返回**: 包含答案、来源等信息的字典

```python
result = generator.generate_answer(
    question="什么是AI？",
    retrieved_docs=retrieved_docs,
    scores=scores
)

# result结构:
# {
#     'question': "什么是AI？",
#     'answer': "答案内容...",
#     'sources': [
#         {
#             'index': 1,
#             'content': "文档内容",
#             'metadata': {...},
#             'score': 0.85
#         },
#         ...
#     ],
#     'context': "格式化的上下文"
# }
```

---

### 5. RAGPipeline (RAG流水线)

完整的端到端RAG流程。

```python
from src.generator import RAGPipeline

rag = RAGPipeline(
    retriever_manager=retriever,
    generator=generator
)
```

#### 方法

**query(question: str, retriever_type: str = "faiss", k: int = 5) -> Dict**

执行完整RAG查询。

- **参数**:
  - `question`: 用户问题
  - `retriever_type`: 检索器类型 ("faiss", "bm25", "hybrid")
  - `k`: Top-K值
- **返回**: 包含答案、来源、检索分析等的字典

```python
result = rag.query(
    question="什么是机器学习？",
    retriever_type="hybrid",
    k=5
)

# result结构:
# {
#     'question': "什么是机器学习？",
#     'answer': "机器学习是...",
#     'sources': [...],
#     'retriever_type': "hybrid",
#     'retrieval_analysis': {
#         'num_retrieved': 5,
#         'avg_score': 0.78,
#         'score_distribution': [0.85, 0.82, ...]
#     }
# }
```

---

### 6. RAGEvaluator (RAG评估器)

评估RAG系统性能。

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
```

#### 方法

**prepare_evaluation_dataset(questions, answers, contexts, ground_truths=None) -> Dataset**

准备评估数据集。

```python
dataset = evaluator.prepare_evaluation_dataset(
    questions=["Q1", "Q2"],
    answers=["A1", "A2"],
    contexts=[["C1"], ["C2"]],
    ground_truths=["GT1", "GT2"]  # 可选
)
```

**evaluate_rag_system(dataset: Dataset, metrics=None) -> Dict**

评估RAG系统（使用RAGAS）。

```python
result = evaluator.evaluate_rag_system(dataset)
# 返回RAGAS评分
```

**calculate_retrieval_metrics(retrieved_docs, relevant_docs) -> Dict**

计算检索指标。

```python
metrics = evaluator.calculate_retrieval_metrics(
    retrieved_docs=[["doc1", "doc2"], ...],
    relevant_docs=[["doc1"], ...]
)

# 返回:
# {
#     'avg_precision': 0.75,
#     'avg_recall': 0.80,
#     'avg_f1': 0.77,
#     'avg_hit_rate': 0.85,
#     'avg_mrr': 0.70
# }
```

**measure_performance(rag_pipeline, test_questions, retriever_types) -> Dict**

测量性能指标。

```python
perf = evaluator.measure_performance(
    rag_pipeline,
    test_questions=["Q1", "Q2", "Q3"],
    retriever_types=['faiss', 'bm25']
)

# 返回:
# {
#     'faiss': {
#         'avg_latency': 0.5,
#         'min_latency': 0.3,
#         'max_latency': 0.8,
#         'throughput': 2.0
#     },
#     ...
# }
```

---

### 7. FailureAnalyzer (失败案例分析器)

分析失败案例并归因错误。

```python
from src.evaluation import FailureAnalyzer

analyzer = FailureAnalyzer()
```

#### 方法

**analyze_failure(question, generated_answer, expected_answer, retrieved_docs, relevant_docs, scores=None) -> Dict**

分析单个失败案例。

```python
analysis = analyzer.analyze_failure(
    question="什么是AI？",
    generated_answer="生成的答案",
    expected_answer="期望的答案",
    retrieved_docs=["doc1", "doc2"],
    relevant_docs=["doc1"],
    scores=[0.8, 0.6]
)

# 返回:
# {
#     'question': "什么是AI？",
#     'generated_answer': "...",
#     'expected_answer': "...",
#     'errors': [
#         {
#             'type': 'retrieval_error',
#             'description': "...",
#             'severity': 'high'
#         },
#         ...
#     ]
# }
```

**batch_analyze_failures(test_cases: List[Dict]) -> Dict**

批量分析失败案例。

```python
test_cases = [
    {
        'question': "Q1",
        'generated_answer': "A1",
        'expected_answer': "E1",
        'retrieved_docs': ["d1"],
        'relevant_docs': ["d1"],
        'scores': [0.8]
    },
    ...
]

result = analyzer.batch_analyze_failures(test_cases)

# 返回:
# {
#     'analyses': [...],  # 每个案例的分析
#     'error_statistics': {
#         'retrieval_error': 3,
#         'ranking_error': 2,
#         'generation_error': 1,
#         'context_error': 0
#     },
#     'total_cases': 6
# }
```

---

## 完整示例

### 示例1：基础RAG流程

```python
import config
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import RetrieverManager
from src.generator import AnswerGenerator, RAGPipeline

# 1. 处理文档
processor = DocumentProcessor()
chunks = processor.process_documents(config.DOCUMENTS_PATH)

# 2. 创建向量索引
vs_manager = VectorStoreManager()
vs_manager.create_vectorstore(chunks)
vs_manager.save()

# 3. 初始化检索器
retriever = RetrieverManager(vs_manager)
retriever.setup_bm25(chunks)

# 4. 创建RAG流水线
generator = AnswerGenerator()
rag = RAGPipeline(retriever, generator)

# 5. 查询
result = rag.query("什么是人工智能？", retriever_type="hybrid")
print(result['answer'])
```

### 示例2：评估RAG系统

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator()

# 准备测试数据
test_questions = ["Q1", "Q2", "Q3"]

# 性能评估
performance = evaluator.measure_performance(
    rag,
    test_questions,
    retriever_types=['faiss', 'bm25', 'hybrid']
)

# 打印结果
for retriever, metrics in performance.items():
    print(f"{retriever}:")
    print(f"  延迟: {metrics['avg_latency']:.3f}s")
    print(f"  吞吐: {metrics['throughput']:.2f} q/s")
```

### 示例3：失败案例分析

```python
from src.evaluation import FailureAnalyzer

analyzer = FailureAnalyzer()

# 准备测试案例
test_cases = [...]  # 失败的查询案例

# 批量分析
analysis = analyzer.batch_analyze_failures(test_cases)

# 查看错误统计
print(analysis['error_statistics'])

# 生成报告
df = analyzer.generate_failure_report(
    analysis,
    output_path="failure_report.csv"
)
```

---

## 数据结构

### Document对象

```python
from langchain.schema import Document

doc = Document(
    page_content="文档内容",
    metadata={
        'source': '文件路径',
        'page': 页码,
        ...
    }
)
```

### 查询结果结构

```python
{
    'question': str,           # 用户问题
    'answer': str,             # 生成的答案
    'sources': [               # 来源文档
        {
            'index': int,      # 序号
            'content': str,    # 文档内容
            'metadata': dict,  # 元数据
            'score': float     # 相关度评分（可选）
        },
        ...
    ],
    'context': str,            # 格式化的上下文
    'retriever_type': str,     # 检索器类型
    'retrieval_analysis': {    # 检索分析
        'num_retrieved': int,
        'avg_score': float,
        'score_distribution': List[float]
    }
}
```

---

## 错误处理

所有主要方法都包含异常处理，失败时会记录日志并返回合适的默认值。

```python
try:
    result = rag.query("问题")
except Exception as e:
    logger.error(f"查询失败: {e}")
    # 系统会返回错误信息而不是崩溃
```

---

## 日志记录

使用loguru进行日志记录：

```python
from loguru import logger

# 日志会自动记录到文件
# 位置: data/logs/*.log
```

---

更多信息请参考[使用指南](USAGE.md)和[README](../README.md)。

