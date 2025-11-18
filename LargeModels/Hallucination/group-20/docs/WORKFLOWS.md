# 🔄 常用工作流

整理各模块的典型运行方式，方便快速复现实验或 Demo。

## 1. Comprehensive 统一评测

1. 准备配置：编辑 `modules/comprehensive/config.yaml` 调整数据集、Solver、Evaluator。
2. 激活虚拟环境并确保安装 `ragas`, `litellm`, `faiss-cpu` 等依赖。
3. 运行：
   ```bash
   bash scripts/run_comprehensive.sh
   ```
4. 输出：结果保存到 `modules/comprehensive/outputs/eval_results/...yaml`，日志写入 `log.txt`。

## 2. SELF-RAG 官方复现

1. 安装 `modules/selfrag-official/requirements.txt` 或使用其 `environment.yml`。
2. 按需运行训练/评估脚本，如：
   ```bash
   bash modules/selfrag-official/setup.sh
   python modules/selfrag-official/run_long_form_static.py ...
   ```
3. 通过 `modules/selfrag-official/visualize_results.py` 生成图表，或阅读 `presentation_*.md` 分享材料。

## 3. SELF-RAG Lite + RAGAS

1. 安装轻量依赖：`pip install vllm ragas datasets`.
2. 准备输入：将 ASQA/FactScore 等文件放在 `modules/selfrag-lite/eval_data/`。
3. 执行：
   ```bash
   bash scripts/run_selfrag_lite.sh
   ```
4. 若需评估幻觉，可运行：
   ```bash
   python modules/selfrag-lite/hallucination_detection.py --input self_rag_output/output.json
   ```

## 4. CoVe + LettuceDetect

1. 安装 `lettucedetect`，确保 GPU/CPU 可用。
2. 准备 `modules/cove/question.json`，包含 `question` 与 `context` 字段。
3. 执行：
   ```bash
   bash scripts/run_cove.sh
   ```
4. 输出对比保存在 `modules/cove/hallucination_results*.json`。

## 5. SelfCheckGPT 独立评测

1. 在仓库根目录运行 `pip install -e modules/selfcheckgpt`。
2. 参考 `modules/selfcheckgpt/demo.py`，将其集成到任意 pipeline。
3. 可结合 comprehensive solver 作为后处理校验器。

---

> 若需添加新的工作流，请复制以上格式：说明依赖、输入、运行命令与输出位置，便于他人快速跟进。
