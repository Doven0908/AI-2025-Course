# Retrieval LM 实验结果 Presentation 文档说明

## 📚 生成的文档列表

本次为三个实验数据集生成了详细的presentation文档,每个数据集包含15个精选样例的深入分析。

### 1. 数据集详细报告 (每个15个样例)

#### 📄 `presentation_exp1_popqa_15samples.md` (22KB)
- **数据集**: PopQA (知识问答)
- **样例数**: 15个
- **内容**:
  - 每个样例包含:问题、真实答案、检索内容(Top 3)、模型答案、评分(relevancy + correctness)、详细分析
  - Mermaid可视化图表展示评分
  - 错误模式分析
  - 改进建议

#### 📄 `presentation_exp2_arc_15samples.md` (27KB)
- **数据集**: ARC (科学推理)
- **样例数**: 15个
- **内容**:
  - 科学多项选择题详细分析
  - 检索失效案例研究
  - 预训练知识vs检索的作用对比
  - 科学推理能力评估

#### 📄 `presentation_exp3_health_15samples.md` (24KB)
- **数据集**: Health Claims (健康声明验证)
- **样例数**: 15个
- **内容**:
  - 健康声明真假判断案例
  - 事实核查过程分析
  - 证据检索与推理评估
  - 误导信息识别

### 2. 综合对比分析报告

#### 📄 `presentation_综合对比分析.md` (18KB)
- **内容**:
  - 三个实验的横向对比
  - 核心发现和关键洞察
  - 错误模式系统性分析
  - 详细改进建议(短期/中期/长期)
  - 成本-收益分析
  - KPI目标设定

---

## 📊 数据概览

### 整体性能对比

| 实验 | 数据集 | Relevancy | Correctness | 准确率 |
|------|--------|-----------|-------------|--------|
| exp1 | PopQA | **92.0%** | 63.0% | 60% |
| exp2 | ARC | 0.4% | **76.0%** | 76% |
| exp3 | Health | 44.6% | 70.0% | 70% |
| **平均** | - | 45.7% | 69.7% | **68.7%** |

### 关键发现

1. **检索相关性与答案正确性不呈线性关系**
   - PopQA: 高检索(92%) → 中等正确性(63%)
   - ARC: 极低检索(0.4%) → 高正确性(76%)

2. **任务类型决定RAG价值**
   - 知识问答: RAG至关重要
   - 科学推理: 预训练知识更重要
   - 事实核查: RAG起辅助作用

3. **三大改进方向**
   - PopQA: 答案提取和实体消歧
   - ARC: 科学知识库扩充
   - Health: 医学知识库和多文档验证

---

## 🎯 样例选择策略

每个数据集的15个样例按照以下策略选择,确保代表性:

- **5个高质量样例** (correctness = 1.0)
  - 展示系统成功案例
  - 按relevancy降序排列

- **5个中等质量样例** (0 < correctness < 1.0)
  - 展示部分正确的边界情况
  - 均匀采样不同得分区间

- **5个低质量样例** (correctness = 0.0)
  - 展示典型失败案例
  - 用于错误模式分析

这种分层采样确保了对系统性能的全面理解。

---

## 📈 样例内容结构

每个样例包含以下标准化内容:

### 1. 基本信息
- 样例编号和质量标签(✅/⚠️/❌)
- 原始问题/声明
- 真实答案/标签

### 2. 检索内容
- Top 3检索结果
- 每个结果包含:标题、文本片段、相关性得分

### 3. 模型输出
- 完整的模型生成答案

### 4. 评估得分
- Relevancy (检索相关性)
- Correctness (答案正确性)

### 5. 分析说明
- 答案正确性分析
- 错误原因诊断
- 特殊情况标注

### 6. 可视化
- Mermaid柱状图展示评分对比

---

## 🔧 使用说明

### 查看完整报告:

```bash
# 查看PopQA详细样例
cat presentation_exp1_popqa_15samples.md

# 查看ARC详细样例
cat presentation_exp2_arc_15samples.md

# 查看Health详细样例
cat presentation_exp3_health_15samples.md

# 查看综合对比分析
cat presentation_综合对比分析.md
```

### 在Markdown查看器中打开:

推荐使用支持Mermaid的Markdown查看器:
- VS Code (需要Markdown Preview Mermaid扩展)
- Typora
- GitHub/GitLab (在线查看)
- Obsidian

### 转换为其他格式:

```bash
# 转换为HTML (需要pandoc)
pandoc presentation_exp1_popqa_15samples.md -o exp1_report.html

# 转换为PDF (需要pandoc和LaTeX)
pandoc presentation_综合对比分析.md -o comprehensive_analysis.pdf
```

---

## 🛠️ 技术细节

### 数据来源:

1. **原始数据**: `self-rag/eval_data/*.jsonl`
   - popqa_longtail_w_gs.jsonl
   - arc_challenge_processed.jsonl
   - health_claims_processed.jsonl

2. **模型预测**: `self-rag/retrieval_lm/exp*`
   - exp1 (PopQA预测)
   - exp2 (ARC预测)
   - exp3_debug (Health预测)

3. **评估结果**: `self-rag/ragas_results/*_simple_eval.json`
   - exp1_popqa_simple_eval.json
   - exp2_arc_simple_eval.json
   - exp3_health_simple_eval.json

### 生成脚本:

`generate_presentation_samples.py` - 自动化数据提取和报告生成

**主要功能**:
- 读取JSONL原始数据
- 匹配模型预测结果
- 提取评估得分
- 选择代表性样例
- 生成格式化Markdown报告

---

## 📊 报告统计

| 报告 | 文件大小 | 样例数 | Mermaid图表数 |
|------|---------|--------|--------------|
| PopQA详细报告 | 22KB | 15 | 16 |
| ARC详细报告 | 27KB | 15 | 16 |
| Health详细报告 | 24KB | 15 | 16 |
| 综合对比报告 | 18KB | 45(引用) | 6 |
| **总计** | **91KB** | **45** | **54** |

---

## 🎓 适用场景

这些报告适用于:

1. **研究团队**: 深入理解模型行为和错误模式
2. **技术演讲**: Presentation用的详细案例和可视化
3. **改进规划**: 基于数据的系统优化建议
4. **论文写作**: 实验结果的详细分析和案例研究
5. **客户汇报**: 系统性能评估和改进路线图

---

## 📞 问题反馈

如有任何问题或需要额外分析,请联系数据科学团队。

**生成日期**: 2025-11-05
**版本**: v1.0
**生成工具**: generate_presentation_samples.py

---

## 📝 更新日志

### v1.0 (2025-11-05)
- ✅ 初始版本发布
- ✅ 三个数据集各15个样例的详细分析
- ✅ 综合对比分析报告
- ✅ Mermaid可视化图表
- ✅ 系统性改进建议

---

*本文档集为Retrieval LM项目的实验结果presentation材料,所有数据基于真实实验结果。*
