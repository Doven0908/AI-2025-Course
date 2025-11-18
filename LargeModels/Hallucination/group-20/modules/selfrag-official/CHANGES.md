# 修改说明 / Change Log

## 2025-10-29: 限制评估样本数量

### 修改内容

修改了 `evaluate_with_ragas.py` 脚本，现在每个实验只评估前 **20 个样本**。

**Modified `evaluate_with_ragas.py` to evaluate only the first **20 samples** per experiment.**

### 主要变更 / Main Changes

1. **添加 `max_samples` 参数**
   - 在 `parse_self_rag_output()` 函数中添加 `max_samples` 参数
   - 在 `evaluate_experiment()` 函数中添加 `max_samples` 参数
   - 在 `main()` 函数中设置 `MAX_SAMPLES = 20`

   **Added `max_samples` parameter to limit the number of samples processed**

2. **改进导入错误处理**
   - 添加 try-except 块捕获导入错误
   - 提供清晰的错误消息和安装指导

   **Improved import error handling with try-except block**

3. **更好的输出信息**
   - 显示样本限制信息
   - 在评估过程中显示正在处理的样本数

   **Better output messages showing sample limits**

### 运行方式 / How to Run

```bash
# 设置 API 密钥
export DEEPSEEK_API_KEY='your-api-key'

# 运行评估 (每个实验20个样本)
cd /data/self-rag
python3 evaluate_with_ragas.py
```

### 预期输出 / Expected Output

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
...
```

### 评估规模 / Evaluation Scale

| 实验 Experiment | 原样本数 Original | 现样本数 Current | 减少 Reduction |
|---|---|---|---|
| exp1_popqa | ~1000 | 20 | 98% |
| exp2_arc | ~500 | 20 | 96% |
| exp3_health | ~1000 | 20 | 98% |
| **总计 Total** | **~2500** | **60** | **~98%** |

### 优点 / Benefits

1. **更快的评估速度**
   - 原来: 30-60 分钟
   - 现在: 5-10 分钟

   **Faster evaluation (5-10 min vs 30-60 min)**

2. **更低的 API 成本**
   - 减少约 98% 的 API 调用

   **~98% reduction in API calls**

3. **快速验证**
   - 适合快速测试和调试
   - 可以快速验证系统是否正常工作

   **Quick testing and debugging**

### 如何修改样本数量 / How to Change Sample Count

如果需要修改样本数量，编辑 `evaluate_with_ragas.py` 文件中的 `MAX_SAMPLES` 变量：

**To change the sample count, edit the `MAX_SAMPLES` variable in `evaluate_with_ragas.py`:**

```python
# 在 main() 函数中 (约第263行)
# In main() function (around line 263)
MAX_SAMPLES = 20  # 修改这个值 / Change this value

# 例如 / Examples:
MAX_SAMPLES = 50   # 50 samples per experiment
MAX_SAMPLES = 100  # 100 samples per experiment
MAX_SAMPLES = None # All samples (original behavior)
```

### 文件位置 / File Locations

- **修改的文件 Modified**: `/data/self-rag/evaluate_with_ragas.py`
- **简单版本 Simple version**: `/data/self-rag/evaluate_with_ragas_simple.py` (10 samples)
- **完整版本备份**: 如需评估所有样本，修改 `MAX_SAMPLES = None`

### 技术细节 / Technical Details

修改的函数签名：

**Modified function signatures:**

```python
# Before
def parse_self_rag_output(data, eval_data_path):
    ...

# After
def parse_self_rag_output(data, eval_data_path, max_samples=None):
    ...
    if max_samples is not None:
        preds = preds[:max_samples]
        prompts = prompts[:max_samples]
        eval_data = eval_data[:max_samples]
    ...
```

```python
# Before
def evaluate_experiment(exp_file, eval_data_file, exp_name, deepseek_api_key, output_dir="ragas_results"):
    ...

# After
def evaluate_experiment(exp_file, eval_data_file, exp_name, deepseek_api_key, output_dir="ragas_results", max_samples=None):
    ...
    samples = parse_self_rag_output(data, eval_data_file, max_samples=max_samples)
    ...
```

### 测试建议 / Testing Recommendations

1. **首次运行**: 使用当前设置 (20 samples) 验证系统
   **First run**: Use current setting (20 samples) to verify system

2. **如果成功**: 可以增加到 50 或 100 samples
   **If successful**: Can increase to 50 or 100 samples

3. **完整评估**: 设置 `MAX_SAMPLES = None`
   **Full evaluation**: Set `MAX_SAMPLES = None`

### 相关文件 / Related Files

- `evaluate_with_ragas_simple.py` - 每个实验10个样本 (10 samples each)
- `QUICKSTART_RAGAS.md` - 快速入门指南
- `RAGAS_EVALUATION_README.md` - 详细文档
- `RAGAS_INTEGRATION_SUMMARY.md` - 项目总结

---

## 其他工具 / Other Tools

如果不想修改主文件，也可以使用：

**Alternative: Use the simple script instead:**

```bash
python3 evaluate_with_ragas_simple.py  # 10 samples per experiment
```

---

**修改时间 Modification Date**: 2025-10-29

**状态 Status**: ✅ 完成 Ready
