## 检索与精排功能说明

本文聚焦 RAGQA 系统中“FAISS 检索 + ColBERT 精排”两大环节，详细说明逻辑、关键代码与核心参数。

---

### 1. 检索逻辑

#### 1.1 组件与参数

- **入口类**：`app/retriever.py` 中的 `Retriever`。  
- **依赖**：`VectorStoreManager`（负责加载 FAISS 索引）。  
- **主要参数**：`top_k` 为一次性从向量库召回的候选数，通常由配置项 `RERANK_TOP_K` 提供。

```python
class Retriever:
    def __init__(self, vector_store: VectorStoreManager, *, top_k: int):
        self.vector_store = vector_store
        self.top_k = top_k
```

#### 1.2 执行流程

1. **相似度检索**  
   调用 `vector_store.similarity_search_with_score(query, top_k)`，底层使用 FAISS 余弦/内积相似度（取决于生成索引时的配置）。  
2. **构造结果对象**  
   - 将 `LangChain Document` 转换为 `DocumentChunk`，保留 `doc_id/chunk_id` 等元数据。  
   - 将原始得分写入 `metadata["score"]`，用于后续展示。

核心代码（`app/retriever.py:25-51`）：

```python
documents_with_scores = self.vector_store.similarity_search_with_score(query, self.top_k)
for doc, score in documents_with_scores:
    metadata = doc.metadata or {}
    metadata["score"] = float(score)
    chunk = DocumentChunk(
        doc_id=metadata.get("doc_id", ""),
        chunk_id=metadata.get("chunk_id", ""),
        text=doc.page_content,
        start_index=int(metadata.get("start_index", 0)),
        end_index=int(metadata.get("end_index", 0)),
        metadata=metadata,
    )
    results.append(RetrievalResult(chunk=chunk, score=score))
```

3. **性能记录**  
   检索耗时通过 `logger.info("Retrieved %s documents in %.2f ms", ...)` 输出，可被指标系统采集。

---

### 2. 精排逻辑

#### 2.1 组件与参数

- **入口类**：`app/reranker.py` 中的 `ColBERTReranker`。  
- **参数配置**：  
  - `RERANKER`（`colbert` 或 `none`）决定是否启用精排。  
  - `RERANKER_MODEL` 指定 ColBERT checkpoint。  

#### 2.2 模型加载

```python
ensure_ragatouille_dependencies()
from ragatouille import RAGPretrainedModel
self._model = RAGPretrainedModel.from_pretrained(model_name)
logger.info("Loading ColBERTv2 reranker model %s", model_name)
```

`RAGPretrainedModel` 会自动处理 HuggingFace 权重缓存，并提供 `rerank` 接口执行 late interaction 排序。

#### 2.3 rerank 执行

1. **准备候选**  
   - 从 `RetrievalResult` 中取出片段文本，构建 `passages` 列表。  
   - 记录 `index -> RetrievalResult` 的映射，便于 ColBERT 结果回写。

2. **模型推理**  
   ```python
   results = self._model.rerank(query=query, documents=passages, k=top_k)
   ```
   返回的 `results` 包含 `content/score/result_index` 等字段。

3. **整理输出**  
   - 使用 `result_index` 找到对应候选，写入精排得分。  
   - 用 `seen_texts` 去重，防止相同片段因 ColBERT 的细粒度切分而重复出现。  
   - 若 `result_index` 缺失，则 fallback 到基于文本匹配的搜索。

关键代码（`app/reranker.py:56-85`）：

```python
ranked: List[RetrievalResult] = []
seen_texts = set()
for result in results:
    score = float(result.get("score", 0.0))
    index = result.get("result_index")
    text = result.get("content") or result.get("text")
    if index is None and text:
        try:
            index = passages.index(text)
        except ValueError:
            continue
    item = mapping.get(index)
    if item is None:
        continue
    if text and text in seen_texts:
        continue
    if text:
        seen_texts.add(text)
    item.chunk.metadata["score"] = score
    ranked.append(item)
```

4. **补齐数量**  
   如果 ColBERT 结果不足 `top_k`，则按原检索顺序添加剩余候选，保证输出长度稳定。

#### 2.4 关键参数影响

- **`top_k`（精排输出数）**：由 `RAGPipeline.answer` 传入，通常等于 `Settings.top_k`。  
- **`retrieval_depth`（检索深度）**：在调用 `Retriever` 时取 `max(rerank_top_k, top_k)`，确保精排候选数量充足。  
- **`metadata` 字段**：`retrieval_rank`、`rerank_rank`、`score` 等字段为后续生成与 UI 提供可视化依据。

