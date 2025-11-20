# RAG 问答系统使用说明

本文档介绍如何在当前项目中配置环境、构建索引、启动服务与执行评估。

## 1. 环境准备

1. 确保系统已安装 Python 3.10 及以上版本，并在 Conda 环境中手动安装 `requirements.txt` 列出的依赖：
   ```powershell
   pip install -r requirements.txt
   ```
2. 在项目根目录创建 `.env`（或直接设置环境变量），填入必要配置：
   ```
   API_KEY=你的Qwen接口密钥
   LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```
   若需调整模型或检索参数，同样可在此处配置（详见第 6 节）。

## 2. 文档准备与索引构建

1. 将需要检索的资料放入 `Document/` 目录，当前支持 `.txt`、`.md`、`.docx`、`.pdf` 等格式。
2. 在已激活的 Conda 环境中，执行以下命令构建向量索引：
   ```powershell
   powershell -File ./scripts/run_pipeline.ps1 -Ingest -PythonPath "D:\ProgramData\anaconda3\envs\rag\python.exe"
   ```
   索引构建完成后，结果保存在 `storage/` 目录内（包括 FAISS 索引、分块元数据、缓存与指标数据库）。

## 3. 运行方式与演示

### 3.1 Web Demo

```powershell
powershell -File ./scripts/run_pipeline.ps1 -Serve -PythonPath "D:\ProgramData\anaconda3\envs\rag\python.exe"
```

默认监听 `http://0.0.0.0:8000`。在浏览器访问根路径，可体验 Web 演示：
- 输入问题触发 RAG 问答；
- 查看模型回答和引用片段；
- 观察最近的调用延迟、检索耗时等指标。

可通过参数 `-BindHost` / `-BindPort` 调整监听地址与端口，例如：
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -Serve -BindHost 127.0.0.1 -BindPort 8080 -PythonPath "\你的\conda\环境\路径\python.exe"
```

### 3.2 CLI 交互

```powershell
python -m app.cli ask "洛阳纸贵是什么意思？" --top-k 6
```


CLI 输出将包含模型回答、引用表格以及整体耗时。

## 4. API 接口

FastAPI 自动提供 Swagger 文档（`/docs`）。常用接口：

- `POST /ask`：问答入口，请求体示例 `{"query": "...", "top_k": 5}`，返回回答、引用、延迟等信息。
- `POST /feedback`：记录用户反馈（问题、是否有帮助、备注），写入 `storage/feedback.jsonl`。
- `GET /metrics`：返回聚合指标与最近调用记录。
- `GET /health`：健康检查。

## 5. 离线评估与误差分析

1. 准备评估数据集（JSON/JSONL），格式示例：
   ```json
   {
     "question": "示例问题？",
     "ground_truths": ["标准答案1", "参考答案2"]
   }
   ```
2. 在 Conda 环境中执行：
   ```powershell
   python -m app.cli evaluate data/qa_eval.jsonl --limit 100 --output-path reports/result.json
   ```
3. 生成的报告包括：
   - 检索召回率、回答命中率；
   - RAGAS 指标（Faithfulness、Answer Relevance、Context Precision 等）；
   - 失败案例按检索错误、重排错误、生成错误分类的详细列表。

## 6. 主要配置项

| 变量名 | 说明 | 默认值 |
| --- | --- | --- |
| `DOCUMENT_DIR` | 原始文档目录 | `Document` |
| `STORAGE_DIR` | 存储目录 | `storage` |
| `LLM_MODEL` | 生成模型名称 | `qwen3-32b` |
| `EMBEDDING_MODEL` | 向量模型 | `BAAI/bge-large-zh-v1.5` |
| `TOP_K` | 返回的上下文数量 | `5` |
| `RERANK_TOP_K` | 初次检索深度 | `12` |
| `RERANKER` | 重排器名称（`none` 或 `colbert`） | `none` |
| `RERANKER_MODEL` | ColBERT 模型名称 | `colbert-ir/colbertv2.0` |

可通过环境变量或 `.env` 文件覆盖上述配置；如需切换模型、调节分块大小，可继续使用 `CHUNK_SIZE`、`CHUNK_OVERLAP` 等变量。

## 7. 常见问题

- **无法调用模型**：确保网络能够访问模型服务端点，且 `API_KEY` 已配置。
- **检索质量不佳**：尝试提高 `RERANK_TOP_K`，启用 ColBERT 重排，或调整分块参数。
- **评估失败**：确认数据集格式正确，并已安装 `ragas`、`datasets` 等依赖。

