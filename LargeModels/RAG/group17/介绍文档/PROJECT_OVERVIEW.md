## RAGQA 项目总览

本项目实现了一个“检索增强生成”(Retrieval-Augmented Generation, RAG) 系统，涵盖文档摄取、向量化检索、ColBERTv2 精排、LLM 回答生成以及 RAGAS 质量评估和网页交互界面。以下内容帮助新成员快速理解核心方法、结构与功能。

---

### 1. 方法论与高阶流程

1. **文档摄取**  
   - `app/data_ingestion.py` 读取 `Document/` 目录中的 `.txt/.md/.docx/.pdf` 文件，统一封装成 `RawDocument` 并保留来源元数据。  
   - `app/chunking.py` 将长文档切分为大小与重叠可配置的语义片段，兼顾上下文完整性与索引粒度。

2. **向量化与存储**  
   - `app/embedding_service.py` 利用 FlagEmbedding（默认 `BAAI/bge-large-zh-v1.5`）对片段编码，并在本地维护哈希缓存减少重复计算。  
   - `app/vector_store.py` 使用 LangChain 的 FAISS 封装构建/加载向量索引，同时将片段元数据以 JSONL 形式持久化，支持重启后快速恢复。

3. **检索与精排**  
   - `app/retriever.py` 基于 FAISS 相似度检索初选片段，记录检索耗时并回写得分。  
   - `app/reranker.py` 默认支持 NoOp 与 **ColBERTv2 精排**：通过 `ragatouille` 的 `RAGPretrainedModel` 对候选片段做 token-level 交互式重排。在保持新版 LangChain 的同时，`app/ragatouille_compat.py` 注入兼容 shim，避免旧接口缺失。

4. **生成回答**  
   - `app/generator.py` 调用兼容 OpenAI 的 API（默认指向 DashScope/Qwen），构造包含检索片段与引用标签的 prompt，输出中文回答并附带 `[n]` 引用。

5. **评估与反馈**  
   - `app/ragas_evaluator.py` & `app/evaluation.py` 通过 RAGAS 指标收集 Faithfulness、Answer Relevance 等质量信号，可异步触发评估任务，结果写入 SQLite。  
   - `app/metrics.py` 记录每次问答的延迟、检索深度等运行指标，并在前端提供滚动窗口视图。

整个流程由 `app/pipeline.py` 的 `RAGPipeline` 串联。`scripts/run_pipeline.ps1` 提供 CLI 入口，可执行文档摄取、服务启动等命令。

---

### 2. 模块结构

| 模块 | 主要职责 |
| --- | --- |
| `config.py` | 统一读取 `.env/环境变量`，生成 `Settings`（文档目录、模型名称、Top-K、ColBERT 配置等）。 |
| `embedding_service.py` | FlagEmbedding 包装、设备管理、磁盘缓存与 LangChain Adapter。 |
| `vector_store.py` | FAISS 构建/加载/存取，暴露 `similarity_search` API。 |
| `retriever.py` | 拉取 Top-K 候选、构造 `RetrievalResult`。 |
| `reranker.py` | NoOp/ColBERT 精排策略；利用 `ragatouille_compat` 解决依赖冲突。 |
| `generator.py` | 调用 LLM 生成含引用的最终答复。 |
| `evaluation.py` & `ragas_evaluator.py` | RAGAS 数据集构建、评估调度及结果落盘。 |
| `server.py` | FastAPI 服务：提供 UI 页面、`/ask/{variant}` 问答接口、`/ragas/evaluate` 等。 |
| `schemas.py` & `types.py` | Pydantic/TypedDict 定义，约束请求响应与内部数据结构。 |
| `metrics.py` | SQLite 指标表、线程安全写入与聚合。 |

UI 资源位于 `templates/` + `static/`，包含 landing、QA、RAGAS 等页面，前端通过 `/metrics/{variant}` 轮询状态并允许切换检索器。

---

### 3. 项目功能亮点

1. **多检索器并存**  
   - `server.create_app` 初始化 `faiss` 与 `colbert` 两种变体，前端下拉可切换。  
   - 通过 `Settings` 中的 `RERANKER`/`RERANKER_MODEL` 控制默认精排器；ColBERT pipeline 会生成独立的 `metrics_colbert.db`，便于 A/B 对比。

2. **运行时诊断与监控**  
   - 日志使用统一的 `app/logger.py`，输出到控制台（可扩展至文件）。  
   - `/metrics/{variant}` 返回最近查询列表与聚合统计，前端使用图表展示响应耗时。

3. **离线/在线评估**  
   - `RagasEvaluationManager` 可以加载自定义评估数据集，调用 RAG Pipeline 生成答案后计算指标。  
   - 评估任务通过 FastAPI 接口触发，前端展示进度条。

4. **兼容多厂商 LLM**  
   - 通过 `Settings.llm_base_url/llm_model/api_key` 与 `eval_*` 字段，实现推理与评估阶段使用不同供应商（如 DashScope + 自建 OpenAI 兼容 API）。

5. **易扩展的脚本与 CLI**  
   - `app/cli.py`（基于 Typer）提供命令行操作：构建索引、执行查询等。  
   - `scripts/run_pipeline.ps1` 将常用操作包装为脚本，便于 Windows 环境快速启动。

---

### 4. 关键配置与依赖

1. **环境变量**（示例）

```
DOCUMENT_DIR=Document
STORAGE_DIR=storage
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-32b
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
TOP_K=5
RERANK_TOP_K=12
RERANKER=colbert          # 或 none
RERANKER_MODEL=colbert-ir/colbertv2.0
API_KEY=sk-xxx
EVAL_API_KEY=sk-eval-xxx
```

2. **依赖**  
   - Web/API：FastAPI + Uvicorn。  
   - LLM & LangChain：`openai`, `langchain`, `langchain-core`, `langchain-openai`。  
   - 检索：`FlagEmbedding`, `faiss-cpu`, `ragatouille`, `torch`, `huggingface-hub`。  
   - 评估：`ragas`, `datasets`, `pandas`。  
   - 文档解析：`docx2txt`, `python-docx`, `pypdf`, `PyPDF2`。  
   - 其他：`python-dotenv`, `typer[all]`, `rich`, `jinja2`。

---

### 5. 服务与接口

| Endpoint | 描述 |
| --- | --- |
| `GET /` | Landing Page，展示可用检索器及状态。 |
| `GET /qa/{variant}` | QA 页面，用户输入问题并选择检索器。 |
| `POST /ask/{variant}` | 主问答接口，与前端/CLI 交互；内部调用 `RAGPipeline.answer`。 |
| `GET /metrics/{variant}` | 返回该检索器的实时性能指标。 |
| `POST /ragas/evaluate` | 触发 RAGAS 评估任务，异步执行后返回结果 ID。 |
| `POST /feedback` | 收集用户反馈并写入 `storage/feedback.jsonl`（如启用）。 |

FastAPI 还自动暴露 `/docs`（Swagger）与 `/openapi.json` 供调试。

---

### 6. 运行方式

1. **准备数据与依赖**  
   - 将原始文档放入 `Document/`。  
   - 使用 `conda` 或 `venv`，执行 `pip install -r requirements.txt`。  
   - 设置 `.env` 或环境变量（API key、模型、ColBERT 开关等）。

2. **构建索引**  
   ```
   python -m app.cli ingest
   ```
   或 `python scripts/run_pipeline.py --Ingest`（以 Typer/PS 脚本为准）。

3. **启动服务**  
   ```
   python -m app.cli serve
   ```
   或 PowerShell 脚本：  
   ```
   powershell -File .\scripts\run_pipeline.ps1 -Serve -PythonPath "<env_python>"
   ```

4. **访问前端**  
   打开 `http://<host>:8000`，选择 “FAISS 检索” 或 “ColBERTv2 精排” 进行问答；可在 “RAGAS” 页签启动评估。

---

### 7. 二次开发建议

1. **扩展检索/精排**：在 `reranker.py` 中新增 `BaseReranker` 子类即可接入其他重排模型，可通过环境变量切换。  
2. **接入新 LLM**：调整 `Settings.llm_base_url` 与 `llm_model`；如需完全不同的 SDK，可扩展 `generator.py`。  
3. **增加多模态文档**：在 `data_ingestion.py` 的 `READERS` 中注册新的文件解析器，并确保 `chunking` 逻辑可处理对应格式。  
4. **部署模式**：目前默认单机/单卡，可在 `embedding_service` 与 `ragatouille` 中配置 GPU 数量或使用量化模型以降低资源成本。

---

通过以上结构，本项目将传统知识库、现代向量检索与大模型回答有机结合，并提供评估闭环和可视化界面，适合企业内部知识问答、会议纪要搜索等场景。欢迎在此基础上继续扩展，如接入多语言支持、细粒度权限控制或自定义指标仪表盘。
