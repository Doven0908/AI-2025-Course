import json
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 设置 Seaborn 风格
# ===============================
sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

# ===============================
# 读取 JSON 文件
# ===============================
with open('./hallucination_results_plus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [d['question'] for d in data]

# ===============================
# 数据处理
# ===============================
# Span counts
baseline_counts = [len(d['baseline_predictions']) if d['baseline_predictions'] else 0 for d in data]
cove_counts = [len(d['cove_predictions']) if d['cove_predictions'] else 0 for d in data]

# Span confidence
baseline_conf = [p['confidence'] for d in data if d['baseline_predictions'] for p in d['baseline_predictions']]
cove_conf = [p['confidence'] for d in data if d['cove_predictions'] for p in d['cove_predictions']]

# Answer lengths
baseline_len = [len(d['baseline_response']) for d in data]
final_len = [len(d['final_answer']) for d in data]

# Span difference
span_diff = [c - b for b, c in zip(baseline_counts, cove_counts)]

# x轴索引和标签
x = range(len(questions))
x_labels = [f"Q{i+1}" for i in x]

# ===============================
# 图1: Span counts comparison (line plot)
# ===============================
plt.figure(figsize=(12,6))
plt.plot(x, baseline_counts, marker='o', linestyle='-', label='Baseline Spans', color='tab:blue')
plt.plot(x, cove_counts, marker='s', linestyle='--', label='CoVe Spans', color='tab:orange')

# 显示每个点数值（非零）
# for i, (b, c) in enumerate(zip(baseline_counts, cove_counts)):
#     if b > 0:
#         plt.text(i, b + 0.1, str(b), ha='center', va='bottom', fontsize=9)
#     if c > 0:
#         plt.text(i, c + 0.1, str(c), ha='center', va='bottom', fontsize=9)

plt.xticks(x, x_labels, rotation=45)
plt.ylabel('Number of Hallucination Spans')
plt.xlabel('Questions')
plt.title('Baseline vs CoVe Hallucination Span Count per Question')
plt.legend()
plt.tight_layout()
plt.savefig('plus_span_count_comparison_line_en.png', dpi=300)
plt.close()

# ===============================
# 图2: Span confidence distributions (KDE)
# ===============================
plt.figure(figsize=(10,6))
sns.kdeplot(baseline_conf, shade=True, label='Baseline', color='tab:blue')
sns.kdeplot(cove_conf, shade=True, label='CoVe', color='tab:orange')
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.title('Confidence Distribution of Detected Hallucination Spans')
plt.legend()
plt.tight_layout()
plt.savefig('plus_span_confidence_distribution_en.png', dpi=300)
plt.close()

# ===============================
# 图3: Answer length comparison (line plot)
# ===============================
plt.figure(figsize=(12,6))
plt.plot(x, baseline_len, marker='o', linestyle='-', label='Baseline Response', color='tab:blue')
plt.plot(x, final_len, marker='s', linestyle='--', label='Final Answer', color='tab:orange')

# 显示每个点数值（非零）
# for i, (b, f) in enumerate(zip(baseline_len, final_len)):
#     if b > 0:
#         plt.text(i, b + 0.5, str(b), ha='center', va='bottom', fontsize=9)
#     if f > 0:
#         plt.text(i, f + 0.5, str(f), ha='center', va='bottom', fontsize=9)

plt.xticks(x, x_labels, rotation=45)
plt.ylabel('Answer Length (characters)')
plt.xlabel('Questions')
plt.title('Answer Length Change from Baseline to Final Answer')
plt.legend()
plt.tight_layout()
plt.savefig('plus_answer_length_change_line_en.png', dpi=300)
plt.close()

# ===============================
# 图4: CoVe effect (span change) - horizontal bars with gradient
# ===============================
plt.figure(figsize=(12,8))

# 计算渐变颜色
max_abs = max(abs(min(span_diff)), abs(max(span_diff)), 1e-5)
colors = [plt.cm.RdYlGn(0.5 - diff/max_abs/2) for diff in span_diff]  # normalize to [-0.5,0.5]

# 对 span_diff 排序以便水平条形图显示
sorted_idx = sorted(range(len(span_diff)), key=lambda i: span_diff[i])
sorted_questions = [questions[i] for i in sorted_idx]
sorted_diff = [span_diff[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]

bars = plt.barh(sorted_questions, sorted_diff, color=sorted_colors)

# 条形上显示数值
# for bar in bars:
#     width_bar = bar.get_width()
#     plt.text(width_bar + (0.1 if width_bar >=0 else -0.1), 
#              bar.get_y() + bar.get_height()/2, 
#              f'{width_bar}', 
#              va='center', ha='left' if width_bar >=0 else 'right', fontsize=10)

plt.xlabel('Change in Hallucination Span Count')
plt.ylabel('Questions')
plt.barh(sorted_questions, sorted_diff, color=sorted_colors)
plt.title('Effect of CoVe on Hallucination Spans\n(Negative=Reduced, Positive=Increased)', pad=20)
plt.tight_layout()
plt.savefig('plus_span_change_effect_horizontal_gradient_en.png', dpi=300)
plt.close()

print("All plots saved with English titles and improved styling.")
