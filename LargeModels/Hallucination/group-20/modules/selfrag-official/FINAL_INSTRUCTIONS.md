# 最终运行说明 / Final Instructions

## ✅ 所有问题已修复 / All Issues Fixed

已修复的问题：
1. ✅ ragas 导入问题（移除了路径冲突）
2. ✅ API key 硬编码问题（改为使用参数）
3. ✅ OPENAI_API_KEY 环境变量问题（自动设置）

**All issues fixed:**
1. ✅ ragas import issue (removed path conflict)
2. ✅ Hardcoded API key (now uses parameter)
3. ✅ OPENAI_API_KEY env var (auto-set)

---

## 🚀 运行步骤 / Steps to Run

### 1️⃣ 设置 API 密钥 / Set API Key

```bash
export DEEPSEEK_API_KEY='your-deepseek-api-key'
```

**重要**: 每次打开新的终端窗口都需要重新设置！

**Important**: You need to set this in every new terminal window!

### 2️⃣ 验证设置 / Verify Setup

```bash
cd /data/self-rag
bash quick_test.sh
```

这将：
- 检查 API key 是否设置
- 测试 API 连接
- 测试 LLM 创建
- 确认所有组件正常工作

**This will:**
- Check if API key is set
- Test API connection
- Test LLM creation
- Confirm all components work

### 3️⃣ 运行评估 / Run Evaluation

如果测试通过，运行：

**If tests pass, run:**

```bash
# 评估 60 samples (20 per experiment) - 推荐
python3 evaluate_with_ragas.py

# 或者评估 30 samples (10 per experiment) - 更快
python3 evaluate_with_ragas_simple.py
```

---

## 📊 预期输出 / Expected Output

### 成功的输出看起来像这样：

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
  Answer: Henry Feilden is a British Army officer...
  Ground Truth: politician...

Initializing DeepSeek-R1 LLM...
Creating RAGAS dataset...
Metrics to evaluate: ['AnswerRelevancy', 'AnswerCorrectness']

Running RAGAS evaluation on 20 samples (this may take a while)...
Evaluating: 100%|████████████████| 20/20 [02:15<00:00,  6.78s/it]

================================================================================
Results for exp1_popqa:
================================================================================
  answer_relevancy: 0.8234
  answer_correctness: 0.7156

Results saved to: ragas_results/exp1_popqa_ragas_eval.json
```

---

## 🔧 关键修复 / Key Fixes Applied

### 修复 1: 移除路径冲突

**Before:**
```python
sys.path.insert(0, '/data/self-rag/ragas/src')  # 导致冲突
```

**After:**
```python
# 使用已安装的 ragas 版本
```

### 修复 2: API Key 传递

**Before:**
```python
client = OpenAI(
    api_key='sk-hardcoded...',  # 硬编码
    base_url=base_url
)
```

**After:**
```python
client = OpenAI(
    api_key=api_key,  # 使用参数
    base_url=base_url
)
```

### 修复 3: 环境变量设置

**Added:**
```python
# Set OPENAI_API_KEY for ragas internal use
os.environ['OPENAI_API_KEY'] = api_key
```

这确保 ragas 内部可以找到 API key。

**This ensures ragas can find the API key internally.**

---

## 🎯 评估配置 / Evaluation Configuration

当前配置：
- **exp1_popqa**: 20 samples (PopQA 问答)
- **exp2_arc**: 20 samples (ARC Challenge)
- **exp3_health**: 20 samples (健康声明)
- **总计 Total**: 60 samples
- **预计时间 Est. time**: 5-10 分钟 minutes

---

## 📁 结果位置 / Results Location

评估完成后，结果保存在：

**After evaluation, results are saved to:**

```
ragas_results/
├── exp1_popqa_ragas_eval.json
├── exp2_arc_ragas_eval.json
├── exp3_health_ragas_eval.json
└── summary.json
```

查看结果：

**View results:**

```bash
# 查看单个实验
cat ragas_results/exp1_popqa_ragas_eval.json

# 查看总结
cat ragas_results/summary.json | python3 -m json.tool
```

---

## ❓ 常见问题 / FAQ

### Q1: "DEEPSEEK_API_KEY environment variable not set"

**A:** 运行：
```bash
export DEEPSEEK_API_KEY='your-key'
```

### Q2: API 连接失败

**A:** 检查：
- API key 是否正确
- 网络连接
- API 配额是否充足

访问 https://platform.deepseek.com/ 检查

### Q3: 如何修改评估样本数？

**A:** 编辑 `evaluate_with_ragas.py` 第263行：
```python
MAX_SAMPLES = 20  # 改为你想要的数量
MAX_SAMPLES = 50  # 例如 50
MAX_SAMPLES = None  # 或全部
```

---

## 🔗 相关文件 / Related Files

- **README_RAGAS.md** - 项目总览
- **RUN_EVALUATION.md** - 详细运行指南
- **FIX_APPLIED.md** - 修复说明
- **test_api_connection.py** - API 连接测试
- **quick_test.sh** - 快速测试脚本

---

## ✅ 快速检查清单 / Quick Checklist

运行前确保：

**Before running, ensure:**

- [ ] 已设置 `DEEPSEEK_API_KEY`
- [ ] 运行 `bash quick_test.sh` 通过
- [ ] 在 `/data/self-rag` 目录
- [ ] 实验文件存在 (exp1, exp2, exp3_debug)
- [ ] 评估数据文件存在 (eval_data/*.jsonl)

---

## 🎉 准备就绪！/ Ready to Go!

如果 `quick_test.sh` 通过了，你就可以开始评估了！

**If `quick_test.sh` passes, you're ready to evaluate!**

```bash
export DEEPSEEK_API_KEY='your-key'
bash quick_test.sh
python3 evaluate_with_ragas.py
```

祝你评估顺利！🚀

**Good luck with your evaluation!** 🚀

---

**最后更新 Last Updated**: 2025-10-29
**状态 Status**: ✅ 就绪运行 Ready to Run
