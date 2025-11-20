## 评估与反馈功能说明

本文依据生成模块文档的写法，介绍 RAGQA 项目中“质量评估（RAGAS）与用户反馈”相关组件，涵盖方法、关键代码与数据结构。

---

### 1. 核心组件

- **评估管理器**：`app/ragas_evaluator.py` 中的 `RagasEvaluationManager`，负责准备评估数据集、调用 pipeline 生成回答，并使用 RAGAS 指标打分。  
- **评估流程封装**：`app/evaluation.py` 提供了对外函数 `run_ragas_evaluation`，供 CLI/服务层调用。  
- **数据与配置**：`DEFAULT_DATASET`（内置示例问答）、`settings.storage_dir/"evaluation_dataset.json"`（持久化用户导入的数据）。  
- **反馈存储**：`server.py` 中 `feedback_path = storage/feedback.jsonl`，`POST /feedback` 接口以 JSONL 形式保存用户回馈。

---

### 2. 评估数据与任务构建

#### 2.1 默认数据集 (`DEFAULT_DATASET`)

- 每条记录包含 `question/user_input/ground_truths/top_k` 等字段，可直接用于快速评估。  
- `retrieved_contexts/response/reference` 默认为空，运行评估后会写入。

示例（`app/ragas_evaluator.py:14-57`）：

```python
DEFAULT_DATASET: List[Dict[str, object]] = [
     {
        "question": "根据2022版《中国居民膳食指南》，“食物多样，合理搭配”具体要求：每天膳食至少应包括哪些主要食物类别？从一周角度看，平均每天和每周分别建议吃多少种不同的食物？",
        "user_input": "根据2022版膳食指南说明：每天应该吃哪些大类食物？一周内每天和每周各要吃多少种食物才算食物多样？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": ["膳食指南提出每天饮食应至少包括谷薯类、蔬菜、水果、畜禽鱼蛋奶和大豆等几大类食物；在种类数量上，平均每天应摄入12种以上不同食物，每周应达到25种以上不同食物，以保证膳食多样化。"],
        "top_k": 5,
    },
    ...
]
```

#### 2.2 管理器初始化

```python
manager = RagasEvaluationManager(
    dataset_path=settings.storage_dir / "evaluation_dataset.json",
    settings=settings,
)
```

- `dataset_path` 存放评估集；若不存在则回退到 `DEFAULT_DATASET`。  
- `settings` 用于共享 API Key/模型/向量索引路径等配置。

#### 2.3 任务配置

`RagasEvaluationManager.create_job`（或类似方法）会：

1. 读取数据集并按需过滤 `top_k`。  
2. 注入运行时上下文（timestamp、job_id 等）。  
3. 将任务状态持久化，便于前端轮询进度。

---

### 3. 评估执行与指标

评估核心逻辑位于 `app/evaluation.py`。

#### 3.1 生成回答

```python
pipeline = RAGPipeline(settings)
answer = pipeline.answer(query=item["question"], top_k=item.get("top_k"))
```

- 对每条样本调用完整的 RAG 流程，收集 `answer/contexts/citations` 等信息。  
- `retrieved_contexts` & `response` 字段会在运行后被填充，以供复查。

#### 3.2 调用 RAGAS

```python
dataset = Dataset.from_dict({...})
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevance, context_precision, answer_similarity],
    llm=ChatOpenAI(model=settings.eval_llm_model, base_url=settings.eval_llm_base_url),
)
```

- 使用 `ragas`, `datasets`, `langchain_openai` 等组合完成评估。  
- `eval_llm_*` 与 `eval_embedding_model` 支持与主推理模型不同的配置，避免串扰。  
- 评估结果支持导出 DataFrame（`ragas_result.to_pandas()`）并保存至本地。

#### 3.3 结果落盘

- 指标详情写入 `storage/metrics.db`（通过 `MetricsCollector`）。  
- RAGAS 评估输出可存为 JSON/CSV，或直接通过 API 返回给前端。

#### 3.4 常用 RAGAS 指标释义

- **Faithfulness（忠实度）**：衡量回答是否完全依据检索上下文，分值越高代表幻觉越少。若该指标持续低于 0.6，通常需要审查检索内容是否充分或回答提示是否限制了引用。  
- **Answer Relevance（答案相关性）**：关注模型回答与问题的匹配度，可发现“答非所问”或生成阶段跑题的问题。低分时优先检查提示词和 rerank 策略。  
- **Context Precision（上下文精准度）**：统计真正被回答用到的上下文比例，用来评估召回是否过宽、噪声是否过多。Precision 低意味着 chunk/`top_k` 需要收紧。  
- **Context Recall（上下文召回率）**：衡量回答所需的重要信息是否都包含在提供的上下文中，低分多半是语料缺失或检索没有覆盖关键片段。  
- **（可选）Answer Similarity / Semantic Similarity**：针对有标准答案的数据集，检查回答与 `ground_truths` 的语义距离，常用于考试题或 FAQ 复写评估。

评估任务会把上述指标及逐条样本得分写入 `storage/ragas_results/`，前端 “RAGAS” 页面读取该目录，展示折线/表格，便于定位检索→生成链路中的瓶颈。

---

### 4. 快速触发评估

在 FastAPI 中，`POST /ragas/evaluate` 会：

1. 解析用户上传的评估集或使用默认数据。  
2. 通过 `asyncio.to_thread` 调用评估函数，避免阻塞事件循环。  
3. 将结果（指标表、样本详情）返回给前端，并在页面显示进度条。

```python
@app.post("/ragas/evaluate", ...)
async def evaluate(request: EvaluateRequest):
    manager = RagasEvaluationManager(...)
    result = await asyncio.to_thread(manager.run, request.payload)
    return result
```

---

### 5. 反馈机制

#### 5.1 API 结构

`server.py` 定义了 `POST /feedback` 接口，接收 `FeedbackRequest`（问题、回答、满意度、备注等字段）。

```python
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(request.model_dump(), ensure_ascii=False) + "\n")
    return {"status": "ok"}
```

- 采用 JSONL 追加写入，便于后续用脚本分析。  
- 若需要实时通知或入库，可在此位置扩展 MQ/DB 接入。

#### 5.2 前端交互

- QA 页面提供反馈按钮，用户可填写“是否有用/期望补充内容”等信息。  
- 后端存储的 `feedback.jsonl` 可与 RAGAS 指标对照，定位低分场景并优化文档或模型。

---

### 6. 参数与扩展建议

- **`EVAL_LLM_MODEL / EVAL_EMBED_MODEL`**：可独立于主流水线配置，用于 RAGAS 评估。  
- **`evaluation_dataset.json`**：支持用户上传自定义评估问题集。  
- **反馈处理**：可定期读取 `feedback.jsonl`，结合指标分析模型表现，也可以接入 BI 系统。  
- **异步执行**：若评估样本较多，可将 `manager.run` 部署为后台任务（Celery/Redis Queue）以提升响应速度。
