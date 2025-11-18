# Self-RAG RAGAS 评估系统 / Self-RAG RAGAS Evaluation System

## ✅ 完成状态 / Completion Status

已完成 Self-RAG 实验结果与 RAGAS 评估框架的集成，使用 DeepSeek-R1 API 进行自动化评估。

**Integration of Self-RAG experiment results with RAGAS evaluation framework using DeepSeek-R1 API is complete.**

---

## 📁 项目文件 / Project Files

### 核心脚本 / Core Scripts

| 文件 File | 说明 Description | 样本数 Samples |
|---|---|---|
| `evaluate_with_ragas.py` | **主评估脚本** (推荐)<br>Main evaluation script (recommended) | 20 per exp |
| `evaluate_with_ragas_simple.py` | 快速测试脚本<br>Quick test script | 10 per exp |
| `setup_ragas_eval.sh` | 自动设置检查<br>Automated setup check | - |

### 文档文件 / Documentation

| 文件 File | 说明 Description |
|---|---|
| `RUN_EVALUATION.md` | **⭐ 运行指南 (从这里开始)**<br>**⭐ Run guide (start here)** |
| `QUICKSTART_RAGAS.md` | 快速入门指南<br>Quick start guide |
| `RAGAS_EVALUATION_README.md` | 详细技术文档<br>Detailed technical docs |
| `RAGAS_INTEGRATION_SUMMARY.md` | 项目架构总结<br>Project architecture summary |
| `CHANGES.md` | 修改日志<br>Change log |
| `README_RAGAS.md` | 本文件<br>This file |

---

## 🚀 快速开始 / Quick Start

### 1️⃣ 设置 API 密钥

```bash
export DEEPSEEK_API_KEY='your-api-key'
```

### 2️⃣ 运行评估 (推荐配置)

```bash
cd /data/self-rag
python3 evaluate_with_ragas.py
```

这将评估：
- exp1_popqa: 20 samples
- exp2_arc: 20 samples  
- exp3_health: 20 samples
- **总计 Total: 60 samples**

### 3️⃣ 查看结果

```bash
cat ragas_results/summary.json
```

---

## 📊 评估配置 / Evaluation Configuration

### 当前设置 / Current Settings

```
每个实验样本数 Samples per experiment: 20
评估时间 Evaluation time:             5-10 分钟 minutes
API 成本 API cost:                    低 Low
```

### 评估指标 / Metrics

✓ Answer Relevancy (回答相关性)
✓ Answer Correctness (回答正确性)
✓ Faithfulness (忠实度) *
✓ Context Precision (上下文精确度) *
✓ Context Recall (上下文召回率) *

*仅在上下文可用时计算

---

## 📈 使用流程 / Workflow

```
1. 设置 API 密钥
   Set API Key
   ↓
2. 运行设置检查 (可选)
   Run setup check (optional)
   ./setup_ragas_eval.sh
   ↓
3. 运行评估
   Run evaluation
   python3 evaluate_with_ragas.py
   ↓
4. 查看结果
   View results
   ragas_results/*.json
   ↓
5. 分析和比较
   Analyze and compare
```

---

## 🎯 三个实验 / Three Experiments

| 实验 | 数据集 | 任务类型 | 样本数 |
|---|---|---|---|
| **exp1_popqa** | PopQA longtail | 问答<br>QA | 20 |
| **exp2_arc** | ARC Challenge | 选择题<br>Multiple Choice | 20 |
| **exp3_health** | Health Claims | 验证<br>Verification | 20 |

---

## 💡 常用命令 / Common Commands

### 评估 / Evaluation

```bash
# 20 samples (推荐 recommended)
python3 evaluate_with_ragas.py

# 10 samples (更快 faster)
python3 evaluate_with_ragas_simple.py

# 检查设置 check setup
./setup_ragas_eval.sh
```

### 查看结果 / View Results

```bash
# 单个实验 single experiment
cat ragas_results/exp1_popqa_ragas_eval.json

# 总结 summary
cat ragas_results/summary.json

# 格式化输出 formatted output
python3 -m json.tool ragas_results/summary.json
```

### 修改样本数 / Change Sample Count

编辑 `evaluate_with_ragas.py` 第263行:

```python
MAX_SAMPLES = 20  # 改为任意数字或 None (全部)
                  # Change to any number or None (all)
```

---

## 📚 详细文档 / Detailed Documentation

想了解更多？查看这些文档：

**Want to know more? Check these docs:**

1. **RUN_EVALUATION.md** - 详细运行指南和故障排除
2. **QUICKSTART_RAGAS.md** - 安装和设置说明
3. **RAGAS_EVALUATION_README.md** - 技术细节和自定义
4. **CHANGES.md** - 修改日志

---

## 🔍 结果示例 / Result Example

```json
{
  "experiment": "exp1_popqa",
  "num_samples": 20,
  "metrics": {
    "answer_relevancy": 0.8234,
    "answer_correctness": 0.7156
  }
}
```

---

## ⚙️ 技术栈 / Tech Stack

- **评估框架 Evaluation Framework**: RAGAS
- **LLM**: DeepSeek-R1 (deepseek-reasoner)
- **语言 Language**: Python 3.8+
- **主要依赖 Dependencies**: ragas, openai, datasets

---

## 🆘 需要帮助? / Need Help?

### 问题诊断 / Issue Diagnosis

```bash
# 1. 检查设置
./setup_ragas_eval.sh

# 2. 查看详细错误
python3 evaluate_with_ragas.py 2>&1 | tee error.log

# 3. 验证文件
ls -la retrieval_lm/exp* eval_data/*.jsonl
```

### 常见问题 / Common Issues

**Q: API 密钥未设置**
```bash
export DEEPSEEK_API_KEY='your-key'
```

**Q: 包未安装**
```bash
pip install ragas openai datasets
```

**Q: 文件未找到**
```bash
cd /data/self-rag  # 确保在正确目录
```

---

## 📞 支持 / Support

- 📖 查看文档 Check documentation
- 🐛 报告问题 Report issues  
- 💬 提出建议 Suggest improvements

---

## ✨ 特性 / Features

✅ 自动化评估 Automated evaluation
✅ 多指标支持 Multiple metrics
✅ 灵活配置 Flexible configuration
✅ 详细文档 Comprehensive documentation
✅ 中英双语 Bilingual support
✅ 快速测试模式 Quick test mode
✅ 批量评估 Batch evaluation
✅ 结果保存 Result persistence

---

## 🎉 就绪运行! / Ready to Run!

所有设置已完成，你可以立即开始评估！

**All setup is complete, you can start evaluation immediately!**

```bash
cd /data/self-rag
export DEEPSEEK_API_KEY='your-key'
python3 evaluate_with_ragas.py
```

**祝你评估顺利! Good luck with your evaluation!** 🚀

---

**版本 Version**: 1.0
**更新日期 Last Updated**: 2025-10-29
**状态 Status**: ✅ 生产就绪 Production Ready
