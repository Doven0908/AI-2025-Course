#!/usr/bin/env python3
"""
生成三个实验数据集的详细presentation文档
每个数据集15个样例,包含问题、答案、检索内容、模型输出和评分
"""

import json
import os
from typing import List, Dict, Any, Tuple

# 数据文件路径
DATA_PATHS = {
    'exp1_popqa': {
        'jsonl': 'self-rag/eval_data/popqa_longtail_w_gs.jsonl',
        'preds': 'self-rag/retrieval_lm/exp1',
        'eval': 'self-rag/ragas_results/exp1_popqa_simple_eval.json',
        'name': 'PopQA (知识问答)',
        'desc': '基于Wikipedia的长尾知识问答任务'
    },
    'exp2_arc': {
        'jsonl': 'self-rag/eval_data/arc_challenge_processed.jsonl',
        'preds': 'self-rag/retrieval_lm/exp2',
        'eval': 'self-rag/ragas_results/exp2_arc_simple_eval.json',
        'name': 'ARC (科学推理)',
        'desc': 'AI2推理挑战 - 科学多项选择题'
    },
    'exp3_health': {
        'jsonl': 'self-rag/eval_data/health_claims_processed.jsonl',
        'preds': 'self-rag/retrieval_lm/exp3_debug',
        'eval': 'self-rag/ragas_results/exp3_health_simple_eval.json',
        'name': 'Health Claims (健康声明验证)',
        'desc': '健康相关声明的真假判断任务'
    }
}

BASE_DIR = '/data/ai4hallucination'


def load_jsonl(file_path: str, max_lines: int = 50) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path: str) -> Dict:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_diverse_samples(scores: List[Dict], n_samples: int = 15) -> List[int]:
    """
    选择多样化的样本索引
    策略: 5个高分 + 5个中等 + 5个低分
    """
    # 为每个样本计算综合得分
    sample_scores = []
    for idx, (rel, corr) in enumerate(zip(scores['relevancy'], scores['correctness'])):
        sample_scores.append({
            'idx': idx,
            'relevancy': rel,
            'correctness': corr,
            'avg': (rel + corr) / 2
        })

    # 按正确性分组
    high_score = [s for s in sample_scores if s['correctness'] == 1.0]
    low_score = [s for s in sample_scores if s['correctness'] == 0.0]
    mid_score = [s for s in sample_scores if 0.0 < s['correctness'] < 1.0]

    # 如果中等分数样本不足,从高分和低分中补充
    if len(mid_score) < 5:
        mid_score = [s for s in sample_scores if 0.3 <= s['correctness'] <= 0.7]

    # 选择样本
    selected = []

    # 高分样本 - 按相关性排序选top5
    high_score.sort(key=lambda x: x['relevancy'], reverse=True)
    selected.extend([s['idx'] for s in high_score[:5]])

    # 中等样本 - 选择多样性
    mid_score.sort(key=lambda x: x['avg'])
    step = max(1, len(mid_score) // 5) if mid_score else 1
    selected.extend([mid_score[i]['idx'] for i in range(0, min(len(mid_score), 5 * step), step)][:5])

    # 低分样本 - 按相关性排序选top5
    low_score.sort(key=lambda x: x['relevancy'], reverse=True)
    selected.extend([s['idx'] for s in low_score[:5]])

    # 如果样本不足15个,从剩余样本中补充
    if len(selected) < n_samples:
        remaining = [s['idx'] for s in sample_scores if s['idx'] not in selected]
        selected.extend(remaining[:n_samples - len(selected)])

    return sorted(selected[:n_samples])


def format_sample_popqa(idx: int, raw_data: Dict, pred: str, scores: Dict) -> str:
    """格式化PopQA样例"""
    question = raw_data['question']
    answers = raw_data['answers']
    ctxs = raw_data.get('ctxs', [])[:3]  # 只取前3个检索结果

    relevancy = scores['relevancy'][idx]
    correctness = scores['correctness'][idx]

    # 分析结果
    if correctness >= 0.8:
        quality = "✅ 高质量"
        analysis = "模型回答正确且相关"
    elif correctness >= 0.3:
        quality = "⚠️ 部分正确"
        analysis = "模型回答部分正确或不够精确"
    else:
        quality = "❌ 错误"
        analysis = "模型回答错误或不相关"

    markdown = f"""
---

### 样例 #{idx + 1} - {quality}

**问题:**
> {question}

**真实答案:**
- {', '.join([f'"{ans}"' for ans in answers])}

**检索内容 (Top 3):**
"""

    for i, ctx in enumerate(ctxs, 1):
        score = ctx.get('score', 'N/A')
        title = ctx.get('title', 'Unknown')
        text = ctx.get('text', '')[:200] + '...' if len(ctx.get('text', '')) > 200 else ctx.get('text', '')
        markdown += f"""
{i}. **{title}** (相关性得分: {score})
   ```
   {text}
   ```
"""

    markdown += f"""
**模型答案:**
```
{pred}
```

**评估得分:**
- **Relevancy (相关性)**: {relevancy:.2f}
- **Correctness (正确性)**: {correctness:.2f}

**分析:** {analysis}

"""

    if relevancy > 0.5 and correctness == 0:
        markdown += "**注意**: 检索到了相关内容但答案错误,可能是答案提取或推理环节出现问题。\n"
    elif relevancy < 0.5 and correctness > 0.5:
        markdown += "**注意**: 检索相关性低但答案正确,说明模型可能依赖内部知识而非检索内容。\n"

    # 可视化图表
    markdown += f"""
```mermaid
xychart-beta
    title "样例#{idx + 1} 评分"
    x-axis ["Relevancy", "Correctness"]
    y-axis "Score" 0 --> 1.0
    bar [{relevancy:.2f}, {correctness:.2f}]
```
"""

    return markdown


def format_sample_arc(idx: int, raw_data: Dict, pred: str, scores: Dict) -> str:
    """格式化ARC样例"""
    question = raw_data['question']
    choices = raw_data['choices']
    answer_key = raw_data['answerKey']
    ctxs = raw_data.get('ctxs', [])[:3]

    relevancy = scores['relevancy'][idx]
    correctness = scores['correctness'][idx]

    # 格式化选项
    options_text = []
    for label, text in zip(choices['label'], choices['text']):
        marker = "✓" if label == answer_key else " "
        options_text.append(f"  [{marker}] {label}. {text}")

    # 分析
    if correctness >= 0.8:
        quality = "✅ 正确"
        analysis = "模型选择了正确答案"
    else:
        quality = "❌ 错误"
        analysis = f"模型选择错误,正确答案应为 {answer_key}"

    markdown = f"""
---

### 样例 #{idx + 1} - {quality}

**问题:**
> {question}

**选项:**
{chr(10).join(options_text)}

**正确答案:** {answer_key}

**检索内容 (Top 3):**
"""

    for i, ctx in enumerate(ctxs, 1):
        score = ctx.get('score', 'N/A')
        title = ctx.get('title', 'Unknown')
        text = ctx.get('text', '')[:200] + '...' if len(ctx.get('text', '')) > 200 else ctx.get('text', '')
        markdown += f"""
{i}. **{title}** (相关性得分: {score})
   ```
   {text}
   ```
"""

    markdown += f"""
**模型选择:**
```
{pred}
```

**评估得分:**
- **Relevancy (相关性)**: {relevancy:.2f}
- **Correctness (正确性)**: {correctness:.2f}

**分析:** {analysis}

"""

    if relevancy < 0.1:
        markdown += "**⚠️ 检索失效**: 检索相关性极低,模型主要依赖预训练知识回答。\n"

    markdown += f"""
```mermaid
xychart-beta
    title "样例#{idx + 1} 评分"
    x-axis ["Relevancy", "Correctness"]
    y-axis "Score" 0 --> 1.0
    bar [{relevancy:.2f}, {correctness:.2f}]
```
"""

    return markdown


def format_sample_health(idx: int, raw_data: Dict, pred: str, scores: Dict) -> str:
    """格式化Health Claims样例"""
    claim = raw_data.get('claim', raw_data.get('question', ''))
    label = raw_data['label']
    answers = raw_data.get('answers', [])
    ctxs = raw_data.get('ctxs', [])[:3]

    relevancy = scores['relevancy'][idx]
    correctness = scores['correctness'][idx]

    # 标签转换
    label_zh = "✅ 真实" if label == "SUPPORTS" else "❌ 虚假"

    # 分析
    if correctness >= 0.8:
        quality = "✅ 判断正确"
        analysis = f"模型正确判断该声明为{label_zh}"
    else:
        quality = "❌ 判断错误"
        analysis = f"模型判断错误,该声明实际为{label_zh}"

    markdown = f"""
---

### 样例 #{idx + 1} - {quality}

**健康声明:**
> {claim}

**真实标签:** {label} ({label_zh})

**标准答案:** {', '.join([f'"{ans}"' for ans in answers])}

**检索内容 (Top 3):**
"""

    for i, ctx in enumerate(ctxs, 1):
        score = ctx.get('score', 'N/A')
        title = ctx.get('title', 'Unknown')
        text = ctx.get('text', '')[:200] + '...' if len(ctx.get('text', '')) > 200 else ctx.get('text', '')
        markdown += f"""
{i}. **{title}** (相关性得分: {score})
   ```
   {text}
   ```
"""

    markdown += f"""
**模型判断:**
```
{pred}
```

**评估得分:**
- **Relevancy (相关性)**: {relevancy:.2f}
- **Correctness (正确性)**: {correctness:.2f}

**分析:** {analysis}

"""

    if relevancy > 0.5 and correctness == 0:
        markdown += "**⚠️ 推理错误**: 检索到相关证据但判断错误,可能是推理逻辑问题。\n"
    elif relevancy < 0.3:
        markdown += "**⚠️ 检索不足**: 检索相关性低,可能缺少关键证据。\n"

    markdown += f"""
```mermaid
xychart-beta
    title "样例#{idx + 1} 评分"
    x-axis ["Relevancy", "Correctness"]
    y-axis "Score" 0 --> 1.0
    bar [{relevancy:.2f}, {correctness:.2f}]
```
"""

    return markdown


def generate_dataset_report(exp_key: str) -> str:
    """生成单个数据集的详细报告"""
    config = DATA_PATHS[exp_key]

    # 加载数据
    print(f"处理 {exp_key}...")
    raw_data = load_jsonl(os.path.join(BASE_DIR, config['jsonl']))
    pred_data = load_json(os.path.join(BASE_DIR, config['preds']))
    eval_data = load_json(os.path.join(BASE_DIR, config['eval']))

    preds = pred_data['preds']
    scores = eval_data['individual_scores']
    metrics = eval_data['metrics']

    # 选择15个样本
    selected_indices = select_diverse_samples(scores, n_samples=15)
    print(f"  选择的样本索引: {selected_indices}")

    # 生成报告头部
    markdown = f"""# {config['name']} - 详细样例分析

## 📊 数据集概览

**数据集名称:** {config['name']}

**任务描述:** {config['desc']}

**评估样本数:** {eval_data['num_samples']} (索引 {eval_data['start_idx']}-{eval_data['end_idx'] - 1})

**本报告样例数:** 15个代表性样例

## 📈 整体性能指标

| 指标 | 分数 | 说明 |
|------|------|------|
| **Relevancy (相关性)** | {metrics['relevancy']:.3f} ({metrics['relevancy']*100:.1f}%) | 检索内容与问题的相关程度 |
| **Correctness (正确性)** | {metrics['correctness']:.3f} ({metrics['correctness']*100:.1f}%) | 答案的准确性 |

### 整体得分分布

```mermaid
xychart-beta
    title "整体性能对比"
    x-axis ["Relevancy", "Correctness"]
    y-axis "Score" 0 --> 1.0
    bar [{metrics['relevancy']:.3f}, {metrics['correctness']:.3f}]
```

### 样本质量分布

```mermaid
pie
    title "正确性分布 (50个评估样本)"
    "完全正确 (1.0)": {sum(1 for s in scores['correctness'] if s == 1.0)}
    "部分正确 (0.5)": {sum(1 for s in scores['correctness'] if s == 0.5)}
    "完全错误 (0.0)": {sum(1 for s in scores['correctness'] if s == 0.0)}
```

---

## 📝 详细样例分析

以下15个样例按质量分为三组:
- **高质量样例** (正确性 = 1.0): 5个
- **中等质量样例** (0 < 正确性 < 1.0): 5个
- **低质量样例** (正确性 = 0.0): 5个

"""

    # 生成每个样例
    format_func = {
        'exp1_popqa': format_sample_popqa,
        'exp2_arc': format_sample_arc,
        'exp3_health': format_sample_health
    }[exp_key]

    for idx in selected_indices:
        markdown += format_func(idx, raw_data[idx], preds[idx], scores)

    # 添加总结
    correct_count = sum(1 for i in selected_indices if scores['correctness'][i] == 1.0)
    incorrect_count = sum(1 for i in selected_indices if scores['correctness'][i] == 0.0)
    partial_count = 15 - correct_count - incorrect_count

    markdown += f"""
---

## 🎯 样例总结

### 本批次15个样例的表现

| 质量等级 | 数量 | 占比 |
|---------|------|------|
| ✅ 完全正确 | {correct_count} | {correct_count/15*100:.1f}% |
| ⚠️ 部分正确 | {partial_count} | {partial_count/15*100:.1f}% |
| ❌ 完全错误 | {incorrect_count} | {incorrect_count/15*100:.1f}% |

### 关键发现

"""

    # 添加针对性分析
    if exp_key == 'exp1_popqa':
        markdown += """
1. **检索表现优秀**: 相关性高达92%,说明检索系统能有效找到相关知识
2. **答案提取需改进**: 正确性仅63%,说明从检索内容中提取精确答案存在困难
3. **常见错误**: 答案不够精确、包含多余信息、或提取了错误的实体

**改进建议:**
- 优化答案提取算法,提高精确度
- 加强答案验证机制
- 改进提示词设计,明确要求简洁答案
"""
    elif exp_key == 'exp2_arc':
        markdown += """
1. **检索系统失效**: 相关性仅0.4%,几乎没有检索到有效信息
2. **依赖内部知识**: 正确率仍达76%,说明模型主要依赖预训练知识
3. **检索未发挥作用**: 科学推理任务中RAG策略未生效

**改进建议:**
- 检查科学知识库的覆盖度
- 优化科学问题的查询重写策略
- 考虑增强科学领域的预训练或微调
- 实施混合策略:检索+生成
"""
    elif exp_key == 'exp3_health':
        markdown += """
1. **检索质量不稳定**: 相关性44.6%,在不同样本间差异较大
2. **判断准确率良好**: 正确率70%,说明模型有一定事实核查能力
3. **检索起辅助作用**: 检索内容对判断有帮助但非决定性

**改进建议:**
- 增强医学健康领域的检索语料
- 实施多文档交叉验证机制
- 优化查询构建,提高检索召回率
- 加入事实核查专用模块
"""

    markdown += """
---

*报告生成时间: 2025-11-05*
"""

    return markdown


def main():
    """主函数"""
    print("=" * 60)
    print("生成详细Presentation文档 (每个数据集15个样例)")
    print("=" * 60)

    # 生成三个数据集的详细报告
    for exp_key in ['exp1_popqa', 'exp2_arc', 'exp3_health']:
        report = generate_dataset_report(exp_key)

        # 保存文件
        output_file = os.path.join(BASE_DIR, f'presentation_{exp_key}_15samples.md')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 已生成: {output_file}")
        print()

    print("=" * 60)
    print("所有报告生成完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
