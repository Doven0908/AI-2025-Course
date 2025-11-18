#!/usr/bin/env python3
"""
生成基础对比图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('ragas_results/summary_simple.json', 'r') as f:
    summary = json.load(f)

experiments = summary['experiments']

# Extract data
exp_names = [exp['experiment'] for exp in experiments]
exp_labels = ['Exp1\nPopQA', 'Exp2\nARC', 'Exp3\nHealth']
relevancy_scores = [exp['metrics']['relevancy'] for exp in experiments]
correctness_scores = [exp['metrics']['correctness'] for exp in experiments]

# 1. Combined comparison chart
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(exp_labels))
width = 0.35

bars1 = ax.bar(x - width/2, relevancy_scores, width, label='Relevancy',
              color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, correctness_scores, width, label='Correctness',
              color='#2ecc71', alpha=0.8)

ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Self-RAG Evaluation Results: Relevancy vs Correctness\n(50 samples per experiment)',
            fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(exp_labels)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('ragas_results/combined_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: combined_comparison.png")
plt.close()

# 2. Performance summary table
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table_data = []
for exp in experiments:
    avg_score = (exp['metrics']['relevancy'] + exp['metrics']['correctness']) / 2
    if avg_score >= 0.75:
        performance = 'Excellent'
    elif avg_score >= 0.6:
        performance = 'Good'
    elif avg_score >= 0.4:
        performance = 'Fair'
    else:
        performance = 'Needs Improvement'

    table_data.append([
        exp['experiment'].replace('_', ' ').title(),
        f"{exp['metrics']['relevancy']:.3f}",
        f"{exp['metrics']['correctness']:.3f}",
        f"{avg_score:.3f}",
        performance
    ])

table = ax.table(cellText=table_data,
                colLabels=['Experiment', 'Relevancy', 'Correctness', 'Average', 'Performance'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.15, 0.20])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color header
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    else:
        if j == 0:
            cell.set_facecolor('#ecf0f1')

plt.title('Self-RAG Evaluation Performance Summary (50 samples each)',
         fontsize=14, fontweight='bold', pad=20)
plt.savefig('ragas_results/performance_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_summary.png")
plt.close()

# 3. Distribution charts (histogram for each metric)
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Score Distributions by Experiment (50 samples each)',
            fontsize=16, fontweight='bold')

for idx, exp in enumerate(experiments):
    # Load detailed scores
    with open(f'ragas_results/{exp["experiment"]}_simple_eval.json', 'r') as f:
        details = json.load(f)

    rel_scores = details['individual_scores']['relevancy']
    cor_scores = details['individual_scores']['correctness']

    # Relevancy histogram
    ax1 = axes[idx, 0]
    ax1.hist(rel_scores, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_title(f'{exp["experiment"]} - Relevancy Distribution', fontweight='bold')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(rel_scores), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(rel_scores):.3f}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Correctness histogram
    ax2 = axes[idx, 1]
    ax2.hist(cor_scores, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.set_title(f'{exp["experiment"]} - Correctness Distribution', fontweight='bold')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(cor_scores), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(cor_scores):.3f}')
    ax2.legend()
    ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ragas_results/distribution_charts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: distribution_charts.png")
plt.close()

print("\n✅ All basic charts regenerated with 50 samples!")
