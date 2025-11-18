# SELF-RAG & SelfCheckGPT 实验复现说明

该目录包含用于复现 SELF-RAG（按需检索-反思-再生成）与 SelfCheckGPT（生成后自检）相关的演示与脚本。目标是定量对比两类策略在开放域问答 / 长文本生成任务上的事实性与一致性改进，以及测量时间开销。

主要文件
- `run_long_form_static.py`：主入口脚本，用于批量运行 SELF-RAG 风格的长文本生成（支持 `factscore`, `asqa`, `eli5` 等任务）。已添加轻量级计时输出，用于记录总耗时与平均每样本耗时。
- `self-rag.py`：交互式 demo，展示如何用本地 `selfrag/selfrag_llama2_7b` + 简单 Retriever 做检索增强生成（保留 IPython 交互）。
- `ragas_eval.py`：演示如何用 `ragas` 的评估接口（LLMContextRecall、Faithfulness、FactualCorrectness）对小批样本进行评估。已添加评估耗时打印。

快速复现实验（示例）
1. 环境要求
- Python 3.8+
- 安装依赖（vllm、transformers、ragas 及其依赖等）。请依据仓库顶层 `pyproject.toml` 或 `requirements.txt` 安装。
- 配置模型缓存目录：默认使用 `/openbayes/home/model_cache`（可通过脚本参数覆盖）。
- 如果使用 `ragas_eval.py`，请设置评估所需的环境变量（见下文）。

2. 运行示例（SELF-RAG）
下面命令展示了如何运行 `run_long_form_static.py`（按需检索 + 生成）：

```bash
python run_long_form_static.py \
  --model_name selfrag/selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --task asqa --input_file eval_data/asqa_eval_gtr_top100.json \
  --output_file self_rag_output_20251026 --max_depth 7 --mode always_retrieve
```

运行时脚本会打印每个主任务分支（`factscore` / `asqa` 等）的总耗时和平均每样本耗时，便于比较不同策略的时间开销。

3. 交互式运行示例：python self-rag.py
- 说明：该脚本为交互式演示，便于手动输入查询并观察检索证据与生成结果。可直接在终端或 Jupyter 中运行。
- 运行命令：
```bash
python self-rag.py
```
- 预期行为：脚本会提示输入问题，随后展示检索到的片段、模型生成结果以及相关中间信息（如 split sentences / ctxs）。适合逐条调试和人工检查生成质量。

4. 使用 RAGAS 自动评测：python ragas_eval.py
- 说明：`ragas_eval.py` 演示如何把模型生成结果组织为 `EvaluationDataset` 并调用 `evaluate(...)`，支持 `LLMContextRecall`, `Faithfulness`, `FactualCorrectness` 等度量。
- 运行命令（示例）：
```bash
python ragas_eval.py --preds_file path/to/predictions.json --refs_file path/to/references.json
```
- 示例评估输出（示意）：
```
Evaluation Results:
{'context_recall': 1.0000, 'faithfulness': 0.8571, 'factual_correctness(mode=f1)': 0.7940}
```
- 注意：实际结果会根据输入数据与配置的评估模型有所不同。

5. 环境变量（评估/embeddings/外部 LLM）
- 推荐通过环境变量配置评估/embeddings 所需的 API 地址与密钥。示例（请替换为你自己的密钥，不要在公共仓库中明文提交）：
```bash
export OPENAI_API_KEY="你的_OPENAI_API_KEY_请勿公开提交"
```
- 强烈建议将实际密钥放在本地安全存储（如 ~/.bashrc / .env / CI secrets）中，避免直接提交到版本控制系统。

结果与上报建议
- 事实性/一致性指标：使用 `ragas.metrics` 中的 `Faithfulness`, `FactualCorrectness` 等度量。
- 幻觉检测效果（句级 / 段级）：可基于生成的 `intermediate` 字段（包含 split sentences / ctxs）标注或用外部参考进行句级对齐比对；
- 时间开销：记录每个方法（Baseline / SelfCheckGPT / SELF-RAG）在相同数据集上的总耗时与平均单样本耗时；并报告检索与重生成造成的额外延迟（秒）。

附录：eval_data 文件说明
- 目录位置：`self-rag&selfcheckgpt/eval_data/`
- 说明：下列文件为常用的评估/复现实验数据集，文件格式以 json 或 jsonl 为主。每个数据集的字段结构可能略有差异，使用前请查看样本以确认字段名称（例如 question、answer、ctxs、gold、docs 等）。

数据集清单与简要说明：
- arc_challenge_processed.jsonl
  - 格式：jsonl（每行一个 JSON 对象）
  - 用途：ARC Challenge 问题的处理版本，适合评估多选/复杂推理问答场景。每条通常包含问题文本与候选答案/参考答案字段，可用于检索增强模型的准确性评估。
- asqa_eval_gtr_top100.json
  - 格式：json（按任务组织）
  - 用途：ASQA（Ambiguous/Answer-seeking QA）评估集合，通常包含问题、参考答案及基于 dense retriever（如 GTR）检索到的 top-k 文档。适合检验检索质量对长文本生成及事实性影响。
- factscore_unlabeled_alpaca_13b_retrieval.jsonl
  - 格式：jsonl
  - 用途：用于事实性评估的未标注生成文本集合，通常附带检索到的证据片段（用于后续用 RAGAS 或 verifier 对生成进行 factuality 评分）。适合 FactScore 风格的自动化评估流程。
- health_claims_processed.jsonl
  - 格式：jsonl
  - 用途：与健康相关的陈述/问题集合，经过预处理以适配事实性检测与检索增强评估。适合检验在敏感领域（医疗健康）中模型输出的准确性与保守性。
- popqa_longtail.json / popqa_longtail_w_gs.jsonl
  - 格式：`popqa_longtail.json`（整体集合） / `popqa_longtail_w_gs.jsonl`（带 gold 标注的逐行 JSON）
  - 用途：PopQA 长尾问题集合；带 _w_gs 文件包含金标准答案（gold standard），便于对比生成结果的正确率与检索证据召回。

