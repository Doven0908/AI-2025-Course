## 生成模块说明

本文介绍 RAGQA 中负责“根据精排上下文调用大模型生成带引用回答”的实现细节，重点覆盖 `Generator` 的逻辑、参数配置与关键代码。

---

### 1. 组件与依赖

- **核心类**：`app/generator.py` 中的 `Generator`。  
- **外部依赖**：`openai.OpenAI` SDK。  
- **配置来源**：`Settings`。

初始化流程（`app/generator.py:17-33`）：

```python
class Generator:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError("API_KEY is required ...")
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model
```

若 `API_KEY` 为空将立即抛出异常。

---

### 2. 上下文构建

`Generator._build_context_prompt` 在调用 LLM 前会将精排结果转化为结构化提示词。

1. **编号绑定**：遍历精排后的 `RetrievalResult`，按照顺序生成 `[idx]` 标签，未来引用直接使用该编号。  
2. **来源展示**：优先输出 `metadata["source_name"]`（文件名或别名），若缺失则回退到 `metadata["source"]`。  
3. **文本清洗**：`result.chunk.text.strip()` 去除首尾空白；片段长度由 chunking 控制，无需额外截断。  
4. **顺序保持**：上下文顺序与 ColBERT 精排输出一致，让高得分片段排在前面。  
5. **空结果兜底**：若候选为空，稍后 prompt 会显示 “No context available.”，引导模型说明知识库未命中。

关键方法（`app/generator.py:35-52`）：

```python
def _build_context_prompt(self, contexts: Sequence[RetrievalResult]) -> str:
    parts: List[str] = []
    for idx, result in enumerate(contexts, start=1):
        metadata = result.chunk.metadata
        source_name = metadata.get("source_name") or metadata.get("source") or ""
        parts.append(
            f"[{idx}] Source: {source_name}\n"
            f"Content: {result.chunk.text.strip()}\n"
        )
    return "\n".join(parts)
```

---

### 3. Prompt 与请求格式

`generate_answer` 将上下文、问题和指令拼装成 Chat Completion 请求，并记录性能指标。

#### 3.1 消息模板

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are an expert QA assistant. Only answer using the provided "
            "context. Use numbered citations like [1], [2] matching the context "
            "section. If information is missing, explicitly say you cannot find it "
            "in the knowledge base."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Question: {query}\n\n"
            f"Context sections:\n{context_prompt or 'No context available.'}\n\n"
            "Answer in Chinese. Provide concise bullet points when suitable. "
            "Always attach the relevant citation markers."
        ),
    },
]
```

- **System 指令**：限制模型只能引用给定上下文、必须输出 `[n]` 引用，并在缺少信息时明确说明。  
- **User 指令**：包含原始问题与上下文列表，同时强调中文回答和列点格式。  
- 需要调整语气或长度时，仅需修改相应 `content` 字符串即可。

#### 3.2 请求与监控

```python
start = time.perf_counter()
response = self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    extra_body={"enable_thinking": False},
)
latency = (time.perf_counter() - start) * 1000
```

- `model` 来自 `Settings.llm_model`。  
- 借助 `perf_counter` 记录总耗时，与 `response.usage.total_tokens` 一起写入日志。  

---

### 4. 引用与响应结构

为保证回答可追溯，模块会生成引用数组并返回统一格式的结果字典。

#### 4.1 引用构建

```python
citations: List[Dict] = []
for idx, result in enumerate(contexts, start=1):
    metadata = result.chunk.metadata
    citations.append(
        {
            "label": f"[{idx}]",
            "source": metadata.get("source_name") or metadata.get("source"),
            "doc_id": result.chunk.doc_id,
            "chunk_id": result.chunk.chunk_id,
            "excerpt": result.chunk.text.strip()[:200],
        }
    )
```

- `label` 与 prompt 中的 `[idx]` 对应，方便前端高亮引用位置。  
- `source` 与 `excerpt` 用于 UI 卡片展示，帮助用户快速定位原文。  
- `doc_id/chunk_id` 便于日志和评估工具进一步追踪或复查。

#### 4.2 返回字典

```python
return {
    "answer": message.content,
    "latency_ms": latency,
    "citations": citations,
    "raw_response": response.model_dump(),
}
```

- `answer`：最终呈现给终端用户的文本。  
- `latency_ms`：供指标面板与性能日志使用。  
- `citations`：驱动前端引用显示及 RAGAS 等评估工具。  
- `raw_response`：保存 LLM 原始输出（包含 token 统计、finish_reason 等），便于调试和审计。

---

### 5. 关键参数影响

- **`Settings.api_key` / `llm_base_url` / `llm_model`**：决定调用哪家 LLM 服务及鉴权方式。  
- **上下文长度**：由精排后选中的 chunk 数量与 `chunk_size` 决定；上下文越长引用覆盖越广，但推理耗时也会提升。  
- **引用策略**：默认 `[1..n]` 顺序编号，若需自定义格式或多段引用，可同时修改 `_build_context_prompt` 和引用生成逻辑。
