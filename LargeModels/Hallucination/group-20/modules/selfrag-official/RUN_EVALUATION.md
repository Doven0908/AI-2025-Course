# 运行 RAGAS 评估 / Run RAGAS Evaluation

## 快速开始 / Quick Start

### 1. 设置 API 密钥 / Set API Key

```bash
export DEEPSEEK_API_KEY='your-deepseek-api-key-here'
```

获取密钥 / Get key: https://platform.deepseek.com/

### 2. 检查设置 / Check Setup

```bash
cd /data/self-rag
./setup_ragas_eval.sh
```

### 3. 运行评估 / Run Evaluation

```bash
# 20 samples per experiment (recommended - 推荐)
python3 evaluate_with_ragas.py

# Or 10 samples per experiment (faster - 更快)
python3 evaluate_with_ragas_simple.py
```

## 当前配置 / Current Configuration

### evaluate_with_ragas.py

- **每个实验样本数 Samples per experiment**: 20
- **总样本数 Total samples**: 60 (20 × 3)
- **预计时间 Estimated time**: 5-10 分钟 minutes
- **API 成本 API cost**: 低 Low (~98% reduction vs full)

### evaluate_with_ragas_simple.py

- **每个实验样本数 Samples per experiment**: 10
- **总样本数 Total samples**: 30 (10 × 3)
- **预计时间 Estimated time**: 2-5 分钟 minutes
- **API 成本 API cost**: 极低 Very Low

## 评估的实验 / Experiments Evaluated

| # | 实验名称<br>Experiment | 任务类型<br>Task | 样本数<br>Samples |
|---|---|---|---|
| 1 | exp1_popqa | PopQA 问答<br>Question Answering | 20 |
| 2 | exp2_arc | ARC Challenge 选择题<br>Multiple Choice | 20 |
| 3 | exp3_health | 健康声明验证<br>Health Claims | 20 |

## 结果位置 / Results Location

评估完成后，结果将保存在：

**Results will be saved to:**

```
ragas_results/
├── exp1_popqa_ragas_eval.json     ← PopQA 评估结果
├── exp2_arc_ragas_eval.json       ← ARC Challenge 评估结果
├── exp3_health_ragas_eval.json    ← Health Claims 评估结果
└── summary.json                   ← 总结 Summary
```

## 查看结果 / View Results

### 方法1: 直接查看 JSON / View JSON Directly

```bash
# 查看单个实验结果
cat ragas_results/exp1_popqa_ragas_eval.json

# 查看总结
cat ragas_results/summary.json
```

### 方法2: 使用 Python / Using Python

```python
import json

# 读取结果
with open('ragas_results/summary.json', 'r') as f:
    results = json.load(f)

# 显示所有实验的指标
for exp in results['experiments']:
    print(f"\n{exp['experiment']}:")
    for metric, score in exp['metrics'].items():
        print(f"  {metric}: {score:.4f}")
```

### 方法3: 使用 jq 工具 / Using jq

```bash
# 安装 jq (如果未安装)
# sudo apt-get install jq  # Ubuntu/Debian
# brew install jq          # macOS

# 查看所有实验的 answer_relevancy 分数
jq '.experiments[] | {experiment: .experiment, relevancy: .metrics.answer_relevancy}' ragas_results/summary.json

# 查看某个实验的所有指标
jq '.experiments[] | select(.experiment == "exp1_popqa") | .metrics' ragas_results/summary.json
```

## 预期输出示例 / Expected Output Example

```
================================================================================
RAGAS Evaluation - Limited Mode
Evaluating 20 samples per experiment
================================================================================

================================================================================
Evaluating exp1_popqa
(Limited to 20 samples)
================================================================================

Loading results from: retrieval_lm/exp1
Loaded 20 samples
Sample preview:
  Question: What is Henry Feilden's occupation?...
  Answer: Henry Feilden is a British Army officer....
  Ground Truth: politician...

Initializing DeepSeek-R1 LLM...
Creating RAGAS dataset...
Metrics to evaluate: ['AnswerRelevancy', 'AnswerCorrectness']

Running RAGAS evaluation on 20 samples (this may take a while)...
Evaluating: 100%|████████████████████| 20/20 [02:15<00:00,  6.78s/it]

================================================================================
Results for exp1_popqa:
================================================================================
  answer_relevancy: 0.8234
  answer_correctness: 0.7156

Results saved to: ragas_results/exp1_popqa_ragas_eval.json
```

## 结果解读 / Understanding Results

### 指标含义 / Metrics Explained

| 指标<br>Metric | 含义<br>Meaning | 范围<br>Range | 越高越好<br>Higher is Better |
|---|---|---|---|
| **answer_relevancy** | 答案与问题的相关性<br>Answer relevance to question | 0-1 | ✓ |
| **answer_correctness** | 答案的正确性<br>Answer correctness | 0-1 | ✓ |
| **faithfulness** | 答案对上下文的忠实度<br>Faithfulness to context | 0-1 | ✓ |
| **context_precision** | 上下文排序质量<br>Context ranking quality | 0-1 | ✓ |
| **context_recall** | 上下文信息覆盖<br>Context information coverage | 0-1 | ✓ |

### 分数参考 / Score Reference

- **0.9-1.0**: 优秀 Excellent
- **0.8-0.9**: 良好 Good
- **0.7-0.8**: 中等 Fair
- **0.6-0.7**: 需改进 Needs Improvement
- **< 0.6**: 较差 Poor

## 常见问题 / FAQ

### Q1: 如何修改样本数量？

**A1:** 编辑 `evaluate_with_ragas.py` 第263行：

```python
MAX_SAMPLES = 20  # 改为你想要的数量 Change to desired number
```

或设置为 `None` 评估所有样本：

```python
MAX_SAMPLES = None  # Evaluate all samples
```

### Q2: 评估时间太长怎么办？

**A2:**
- 使用更少的样本 (如10个)
- 使用 `evaluate_with_ragas_simple.py`
- 检查网络连接

### Q3: API 调用失败怎么办？

**A3:**
- 检查 API 密钥是否正确
- 确认 API 配额充足
- 检查网络连接
- 查看错误日志

### Q4: 如何比较不同实验？

**A4:** 创建一个简单的比较脚本：

```python
import json

with open('ragas_results/summary.json', 'r') as f:
    data = json.load(f)

print("Experiment Comparison:")
print("-" * 60)
for exp in data['experiments']:
    name = exp['experiment']
    rel = exp['metrics'].get('answer_relevancy', 0)
    cor = exp['metrics'].get('answer_correctness', 0)
    print(f"{name:20} | Relevancy: {rel:.4f} | Correctness: {cor:.4f}")
```

## 下一步 / Next Steps

1. ✅ 运行评估
2. ✅ 查看结果
3. ✅ 分析各实验的性能
4. ✅ 根据需要调整样本数量
5. ✅ 比较不同实验的表现

## 故障排除 / Troubleshooting

### 问题: 导入错误

```
ImportError: No module named 'ragas'
```

**解决:**
```bash
pip install ragas openai datasets
```

### 问题: API 密钥未设置

```
Error: DEEPSEEK_API_KEY environment variable not set
```

**解决:**
```bash
export DEEPSEEK_API_KEY='your-key'
```

### 问题: 文件未找到

```
FileNotFoundError: retrieval_lm/exp1
```

**解决:** 确认你在正确的目录：
```bash
cd /data/self-rag
pwd  # 应显示 /data/self-rag
```

## 获取帮助 / Get Help

- 📖 详细文档: `RAGAS_EVALUATION_README.md`
- 🚀 快速入门: `QUICKSTART_RAGAS.md`
- 📝 项目总结: `RAGAS_INTEGRATION_SUMMARY.md`
- 📋 修改说明: `CHANGES.md`

---

**最后更新 Last Updated**: 2025-10-29

**状态 Status**: ✅ 就绪 Ready to Run
