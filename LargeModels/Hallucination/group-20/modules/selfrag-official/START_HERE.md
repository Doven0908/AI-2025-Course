# 🚀 开始评估 / START HERE

## ✅ 所有问题已修复 / All Issues Fixed

已修复的问题：
1. ✅ ragas 导入冲突 → 使用已安装版本
2. ✅ API key 传递问题 → 正确使用参数
3. ✅ LLM 兼容性问题 → 改用 langchain ChatOpenAI
4. ✅ 代理冲突问题 → 临时禁用代理
5. ✅ 环境变量问题 → 自动设置 OPENAI_API_KEY

---

## 📋 运行前检查 / Pre-flight Checklist

```bash
# 1. 确认在正确目录
cd /data/self-rag
pwd  # 应显示: /data/self-rag

# 2. 设置 API 密钥
export DEEPSEEK_API_KEY='sk-b44e8978b5b046cfa0f64d96d53cb062'

# 3. 验证 API 密钥
echo $DEEPSEEK_API_KEY  # 应显示你的密钥
```

---

## 🎯 运行评估 / Run Evaluation

### 方式 1: 直接运行 (推荐)

```bash
python3 evaluate_with_ragas.py
```

这将评估：
- exp1_popqa: 20 samples
- exp2_arc: 20 samples
- exp3_health: 20 samples
- **总计**: 60 samples
- **预计时间**: 5-10 分钟

### 方式 2: 快速测试

```bash
python3 evaluate_with_ragas_simple.py
```

这将评估：
- 每个实验: 10 samples
- **总计**: 30 samples
- **预计时间**: 3-5 分钟

---

## 📊 预期输出 / Expected Output

运行后你会看到：

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
  Answer: politician...
  Ground Truth: politician...

Initializing DeepSeek-R1 LLM...
Creating RAGAS dataset...
Metrics to evaluate: ['AnswerRelevancy', 'AnswerCorrectness']

Running RAGAS evaluation on 20 samples (this may take a while)...
Evaluating: 100%|████████████████████| 20/20 [02:30<00:00,  7.5s/it]

================================================================================
Results for exp1_popqa:
================================================================================
  answer_relevancy: 0.8156
  answer_correctness: 0.7243

Results saved to: ragas_results/exp1_popqa_ragas_eval.json

[继续评估 exp2 和 exp3...]
```

---

## 📁 查看结果 / View Results

评估完成后：

```bash
# 查看总结
cat ragas_results/summary.json

# 查看单个实验
cat ragas_results/exp1_popqa_ragas_eval.json

# 格式化查看
python3 -m json.tool ragas_results/summary.json
```

结果文件：
```
ragas_results/
├── exp1_popqa_ragas_eval.json      # PopQA 结果
├── exp2_arc_ragas_eval.json        # ARC Challenge 结果
├── exp3_health_ragas_eval.json     # Health Claims 结果
└── summary.json                    # 总结
```

---

## 🔧 关键修复说明 / Key Fixes

### 1. 使用 langchain ChatOpenAI

现在使用 langchain 的 `ChatOpenAI`，完全兼容 RAGAS：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
    max_tokens=512
)
```

### 2. 自动处理代理

脚本会在创建 LLM 时：
- 临时禁用代理 (避免 SOCKS 错误)
- 直连 DeepSeek API
- 完成后恢复代理设置

### 3. 使用 deepseek-chat

改用 `deepseek-chat` 而不是 `deepseek-reasoner`：
- ✅ 更适合评估任务
- ✅ 响应更快
- ✅ 成本更低
- ✅ API 更稳定

---

## ❓ 常见问题 / FAQ

### Q: 还是报 "DEEPSEEK_API_KEY not set" 错误？

**A:** 每次打开新终端都需要重新设置：

```bash
export DEEPSEEK_API_KEY='sk-b44e8978b5b046cfa0f64d96d53cb062'
```

### Q: 如何验证设置？

**A:** 运行快速测试：

```bash
python3 test_api_connection.py
```

应该看到：
```
✓ DEEPSEEK_API_KEY is set
✓ API call successful!
✅ All checks passed!
```

### Q: 还是有代理错误？

**A:** 临时禁用代理运行：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python3 evaluate_with_ragas.py
```

### Q: 评估时间太长？

**A:** 使用快速模式：

```bash
python3 evaluate_with_ragas_simple.py  # 只评估 10 samples
```

### Q: 如何修改样本数量？

**A:** 编辑 `evaluate_with_ragas.py` 第 263 行：

```python
MAX_SAMPLES = 20   # 改为你想要的数字
MAX_SAMPLES = 50   # 例如 50
MAX_SAMPLES = None # 或全部样本
```

---

## 📚 相关文档 / Documentation

- **PROXY_FIX.md** - 代理问题详细说明
- **FINAL_INSTRUCTIONS.md** - 完整运行说明
- **README_RAGAS.md** - 项目概览
- **RUN_EVALUATION.md** - 详细评估指南

---

## ✨ 评估指标 / Metrics

RAGAS 会计算以下指标：

| 指标 | 说明 | 范围 |
|---|---|---|
| **answer_relevancy** | 答案与问题的相关性 | 0-1 |
| **answer_correctness** | 答案的正确性 | 0-1 |
| **faithfulness** | 对上下文的忠实度 | 0-1 |
| **context_precision** | 上下文排序质量 | 0-1 |
| **context_recall** | 上下文信息覆盖 | 0-1 |

所有指标: 越高越好 (1.0 = 完美)

---

## 🎉 准备就绪！

所有问题已修复，现在可以运行评估了！

```bash
# 设置 API 密钥
export DEEPSEEK_API_KEY='sk-b44e8978b5b046cfa0f64d96d53cb062'

# 运行评估
python3 evaluate_with_ragas.py

# 或快速测试
python3 evaluate_with_ragas_simple.py
```

**祝你评估顺利！** 🚀

---

**最后更新**: 2025-10-29
**状态**: ✅ 就绪运行 Ready to Run
**所有测试**: ✅ 通过 Passed
