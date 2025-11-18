import json
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 字体设置：防止中文显示为方块
# ===============================


# ===============================
# 设置 Seaborn 风格
# ===============================
sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

# ===============================
# 读取 JSON 文件
# ===============================
with open('./hallucination_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [d['question'] for d in data]

# ===============================
# 数据处理
# ===============================
# Span 数量
baseline_counts = [len(d['baseline_predictions']) if d['baseline_predictions'] else 0 for d in data]
cove_counts = [len(d['cove_predictions']) if d['cove_predictions'] else 0 for d in data]

# Span 置信度
baseline_conf = [p['confidence'] for d in data if d['baseline_predictions'] for p in d['baseline_predictions']]
cove_conf = [p['confidence'] for d in data if d['cove_predictions'] for p in d['cove_predictions']]

# 答案长度
baseline_len = [len(d['baseline_response']) for d in data]
final_len = [len(d['final_answer']) for d in data]

# Span 差异
span_diff = [c - b for b, c in zip(baseline_counts, cove_counts)]

# x轴索引和标签
x = range(len(questions))
x_labels = [f"Q{i+1}" for i in x]

# ===============================
# 图1: 幻觉片段数量对比（折线图）
# ===============================
plt.figure(figsize=(12,6))
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
plt.plot(x, baseline_counts, marker='o', linestyle='-', label='Baseline 幻觉片段数', color='tab:blue')
plt.plot(x, cove_counts, marker='s', linestyle='--', label='CoVe 幻觉片段数', color='tab:orange')

plt.xticks(x, x_labels, rotation=45)
plt.ylabel('幻觉片段数量')
plt.xlabel('问题编号')
plt.title('Baseline 与 CoVe 模型幻觉片段数量对比')
plt.legend()
plt.tight_layout()
plt.savefig('幻觉片段数量对比折线图.png', dpi=300)
plt.close()

# ===============================
# 图2: 幻觉置信度分布（KDE）
# ===============================
plt.figure(figsize=(10,6))
sns.kdeplot(baseline_conf, shade=True, label='Baseline 模型', color='tab:blue')
sns.kdeplot(cove_conf, shade=True, label='CoVe 模型', color='tab:orange')
plt.xlabel('置信度')
plt.ylabel('密度')
plt.title('检测到的幻觉片段置信度分布')
plt.legend()
plt.tight_layout()
plt.savefig('幻觉置信度分布图.png', dpi=300)
plt.close()

# ===============================
# 图3: 答案长度变化（折线图）
# ===============================
plt.figure(figsize=(12,6))
plt.plot(x, baseline_len, marker='o', linestyle='-', label='Baseline 答案长度', color='tab:blue')
plt.plot(x, final_len, marker='s', linestyle='--', label='CoVe 答案长度', color='tab:orange')

plt.xticks(x, x_labels, rotation=45)
plt.ylabel('答案长度（字符数）')
plt.xlabel('问题编号')
plt.title('Baseline 与 CoVe 答案长度变化')
plt.legend()
plt.tight_layout()
plt.savefig('答案长度变化折线图.png', dpi=300)
plt.close()

# ===============================
# 图4: CoVe 对幻觉的影响（水平条形图）
# ===============================
plt.figure(figsize=(12,8))

# 计算渐变颜色
max_abs = max(abs(min(span_diff)), abs(max(span_diff)), 1e-5)
colors = [plt.cm.RdYlGn(0.5 - diff/max_abs/2) for diff in span_diff]  # 颜色梯度

# 对 span_diff 排序以便展示
sorted_idx = sorted(range(len(span_diff)), key=lambda i: span_diff[i])
sorted_questions = [questions[i] for i in sorted_idx]
sorted_diff = [span_diff[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]

bars = plt.barh(sorted_questions, sorted_diff, color=sorted_colors)

plt.xlabel('幻觉片段数量变化（CoVe - Baseline）')
plt.ylabel('问题')
plt.title('CoVe 模型对幻觉片段的影响\n（负值表示幻觉减少，正值表示幻觉增加）', pad=20)
plt.tight_layout()
plt.savefig('CoVe对幻觉影响水平条形图.png', dpi=300)
plt.close()

print("所有图已生成（中文版本）。")
