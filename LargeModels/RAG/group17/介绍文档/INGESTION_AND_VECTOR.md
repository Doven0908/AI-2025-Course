## 文档摄取、向量化与存储说明

本文档总结 RAGQA 项目中“输入文档 → 语义向量”整条链路，涵盖数据来源、分块策略、向量化与 FAISS 存储实现。

---

### 1. 文档来源与摄取流程

1. **来源目录**  
   `Settings.document_dir` 默认指向 `Document/`。将 `.txt/.md/.docx/.pdf` 文件放入该目录即可被识别，未列出的类型会在日志中提示跳过（`app/data_ingestion.py:41`）。

2. **解析逻辑**  
   - 纯文本/Markdown：使用 `Path.read_text` 直接读取。  
   - DOCX：优先 `docx2txt.process`，若依赖缺失回退到 `python-docx`（`app/data_ingestion.py:24-38`）。  
   - PDF：优先 `pypdf.PdfReader`，回退至 `PyPDF2.PdfReader`（`app/data_ingestion.py:9-21`）。

3. **封装为 `RawDocument`**  
   核心代码（节选自 `app/data_ingestion.py:65-109`）：

```python
for path in sorted(document_dir.glob("**/*")):
    reader = READERS.get(path.suffix.lower())
    ...
    doc_id = uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())).hex
    metadata = {
        "source_name": path.name,
        "source_path": str(path),
        "file_extension": path.suffix.lower(),
    }
    documents.append(
        RawDocument(doc_id=doc_id, text=text, source_path=str(path), metadata=metadata)
    )
```

   - doc_id 基于文件路径生成，可复现。  
   - metadata 保存原始文件名、路径、扩展名，后续在 UI 引用与 RAGAS 评估时展示来源。

4. **导出索引**  
   `export_documents` 会把 doc_id 与 metadata 序列化到 `storage/document_index.jsonl`，方便离线核对（`app/data_ingestion.py:116-128`）。

---

### 2. Chunking 策略

`app/chunking.py` 实现了中文友好的滑动窗口分块，流程如下：

1. **分句**  
   **`_split_sentences` 使用正则 `(?<=[。！？?!])` 进行粗粒度断句**，保证 chunk 在句边界处切分（`app/chunking.py:8-28`）。

2. **滑动窗口**  
   **`chunk_document` 以 `chunk_size` 为阈值累计句子；超过阈值时输出一个 chunk**，并保留 `overlap` 字符作为上下文（`app/chunking.py:30-90`）。

3. **Chunk 元数据**  
   - `chunk_id = f"{doc_id}_{chunk_index:05d}"`  
   - `start_index/end_index` 记录原文中的偏移，便于 UI 定位。  
   - metadata 继承原文信息，并新增 `chunk_index`（示例见 `app/chunking.py:55-76`）。

4. **批量处理**  
   `chunk_corpus` 遍历 `RawDocument` 列表，返回全部 `DocumentChunk`（`app/chunking.py:94-106`）。

> 配置入口：`Settings.chunk_size` 与 `Settings.chunk_overlap`，可通过环境变量 `CHUNK_SIZE / CHUNK_OVERLAP` 调整。

---

### 3. 向量化与缓存

`EmbeddingService`（`app/embedding_service.py`）负责将 chunk 文本转换为向量。

1. **模型加载**  
   
   ```python
   from FlagEmbedding import FlagAutoModel
   self._model = FlagAutoModel.from_finetuned(
       settings.embedding_model,
       query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
       use_fp16=True,
   )
   ```
   - **默认模型：`BAAI/bge-large-zh-v1.5`**，可通过 `EMBEDDING_MODEL` 环境变量覆盖。  
   - 自动检测 GPU/CPU，保存到 `self._device`。
   
3. **批量编码**  
   `_embed_batch` 将文本列表送入 FlagEmbedding，并限制 `max_length=512`。

4. **LangChain 适配器**  
   `as_langchain_embedding` 返回实现了 `Embeddings` 接口的内部类，使 FAISS 能直接调用（`app/embedding_service.py:164-183`）。

---

### 4. FAISS 存储与检索

`VectorStoreManager`（`app/vector_store.py`）封装了“建库 -> 保存 -> 加载 -> 检索”整个过程。

1. **构建索引**

```python
docs.append(Document(page_content=chunk.text, metadata=metadata))
index = FAISS.from_documents(documents=docs, embedding=adapter)
index.save_local(str(self.index_path))
with self.metadata_path.open("w", encoding="utf-8") as f:
    for doc in docs:
        f.write(json.dumps(doc.metadata, ensure_ascii=False) + "\n")
```

- `metadata` 包含 `doc_id/chunk_id/source/start_index/end_index`。  
- 向量索引存储在 `storage/vector_index.faiss`；元数据单独保存在 `storage/chunks.jsonl` 以便调试。

2. **加载索引**  
   `_ensure_vector_store` 在首次检索时调用 `FAISS.load_local` 恢复索引，并复用上文的 LangChain Embedding 适配器（`app/vector_store.py:38-63`）。

3. **检索 API**  
   - `similarity_search(query, k)`：返回 `Document` 列表。  
   - `similarity_search_with_score(query, k)`：附带相似度分数，供 `Retriever` 构造 `RetrievalResult`（`app/vector_store.py:90-102`）。



