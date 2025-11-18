# Quick Start: RAGAS Evaluation for Self-RAG

## 中文说明

本项目已集成 RAGAS 评估框架,使用 DeepSeek-R1 API 对 Self-RAG 的三个实验结果进行自动化评估。

### 快速开始

#### 1. 安装依赖

```bash
# 进入 RAGAS 目录并安装
cd /data/self-rag/ragas
pip install -e .

# 或者从 PyPI 安装
pip install ragas openai datasets
```

#### 2. 设置 DeepSeek API 密钥

```bash
export DEEPSEEK_API_KEY='你的-deepseek-api-密钥'
```

获取 API 密钥: https://platform.deepseek.com/

#### 3. 运行评估

**测试模式(推荐先运行):**
```bash
cd /data/self-rag
python evaluate_with_ragas_simple.py
```
此脚本只评估每个实验的前10个样本,用于快速测试。

**完整评估:**
```bash
python evaluate_with_ragas.py
```
此脚本评估所有样本(可能需要较长时间和较多 API 调用)。

#### 4. 查看结果

结果保存在 `ragas_results/` 目录:
- `exp1_popqa_ragas_eval.json` - PopQA 实验评估结果
- `exp2_arc_ragas_eval.json` - ARC Challenge 实验评估结果
- `exp3_health_ragas_eval.json` - Health Claims 实验评估结果
- `summary.json` - 总结

### 评估指标说明

1. **Answer Relevancy (回答相关性)** - 衡量生成答案与问题的相关程度
2. **Answer Correctness (回答正确性)** - 将生成答案与标准答案对比
3. **Faithfulness (忠实度)** - 检查答案是否忠实于检索的上下文,是否有幻觉
4. **Context Precision (上下文精确度)** - 衡量相关上下文是否排名靠前
5. **Context Recall (上下文召回率)** - 衡量是否检索到所有相关信息

所有指标范围 0-1,越高越好。

---

## English Instructions

### Prerequisites

- Python 3.8+
- DeepSeek API key
- Self-RAG experiment results (exp1, exp2, exp3_debug)

### Installation

```bash
# Install RAGAS
cd /data/self-rag/ragas
pip install -e .

# Or install from PyPI
pip install ragas openai datasets
```

### Setup

1. **Get DeepSeek API Key**
   - Sign up at https://platform.deepseek.com/
   - Create an API key

2. **Set Environment Variable**
   ```bash
   export DEEPSEEK_API_KEY='your-api-key-here'
   ```

### Usage

#### Quick Test (Recommended First)

Evaluate only 10 samples per experiment:

```bash
cd /data/self-rag
python evaluate_with_ragas_simple.py
```

**Expected output:**
```
============================================================
Evaluating: exp1_popqa
============================================================

Loading 10 samples from retrieval_lm/exp1...
Loaded 10 samples

Sample:
  Q: What is Henry Feilden's occupation?...
  A: Henry Feilden is a British Army officer....
  GT: politician...

Initializing DeepSeek-R1...
Evaluating Answer Relevancy...
  Sample 1: 0.856
  Sample 2: 0.792
  ...

============================================================
Results for exp1_popqa:
============================================================
  Average Answer Relevancy: 0.8234
  Samples evaluated: 10
============================================================

Results saved to: ragas_results/exp1_popqa_simple_eval.json
```

#### Full Evaluation

Evaluate all samples (may take longer):

```bash
python evaluate_with_ragas.py
```

### Understanding Results

Results are saved in JSON format in `ragas_results/` directory:

```json
{
  "experiment": "exp1_popqa",
  "num_samples": 10,
  "metrics": {
    "answer_relevancy": 0.8234
  },
  "individual_scores": [0.856, 0.792, ...]
}
```

### Metrics Explained

- **Answer Relevancy**: How well the answer addresses the question (0-1, higher is better)
- **Answer Correctness**: Semantic and factual similarity to ground truth (0-1)
- **Faithfulness**: Whether the answer is faithful to retrieved context (0-1)
- **Context Precision**: Quality of context ranking (0-1)
- **Context Recall**: Coverage of relevant information (0-1)

### Troubleshooting

**Problem**: `DEEPSEEK_API_KEY environment variable not set`
**Solution**: Run `export DEEPSEEK_API_KEY='your-key'`

**Problem**: Out of memory
**Solution**: Use the simple script or reduce sample size

**Problem**: Rate limiting
**Solution**:
- Wait and retry
- Upgrade API tier
- Reduce evaluation frequency

### Project Files

```
/data/self-rag/
├── evaluate_with_ragas.py          # Full evaluation script
├── evaluate_with_ragas_simple.py   # Quick test script (10 samples)
├── RAGAS_EVALUATION_README.md      # Detailed documentation
├── QUICKSTART_RAGAS.md            # This file
└── ragas_results/                 # Output directory (created automatically)
```

### Next Steps

1. Run the simple test to verify setup
2. Review the results in `ragas_results/`
3. If satisfied, run full evaluation
4. Compare metrics across experiments

### API Costs

- DeepSeek-R1 API pricing: Check https://platform.deepseek.com/api-docs/pricing/
- Estimated cost per 1000 samples: ~$X (varies by complexity)
- Simple test (30 samples total): Minimal cost

### Additional Resources

- Full documentation: `RAGAS_EVALUATION_README.md`
- RAGAS docs: https://docs.ragas.io/
- DeepSeek docs: https://platform.deepseek.com/api-docs/
- Self-RAG paper: https://arxiv.org/abs/2310.11511
