#!/usr/bin/env python3
"""
可视化 RAGAS 评估结果
用于小组作业汇报
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载所有实验结果"""
    with open('ragas_results/summary_simple.json', 'r') as f:
        data = json.load(f)
    return data['experiments']

def create_comparison_chart(experiments):
    """创建实验对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    exp_names = [exp['experiment'].replace('_', '\n') for exp in experiments]
    relevancy_scores = [exp['metrics']['relevancy'] for exp in experiments]
    correctness_scores = [exp['metrics']['correctness'] for exp in experiments]

    x = np.arange(len(exp_names))
    width = 0.35

    # 相关性对比
    bars1 = ax1.bar(x, relevancy_scores, width, label='Relevancy',
                     color='#3498db', alpha=0.8)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Answer Relevancy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)

    # 在柱子上显示数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    # 正确性对比
    bars2 = ax2.bar(x, correctness_scores, width, label='Correctness',
                     color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Answer Correctness Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names)
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)

    # 在柱子上显示数值
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('ragas_results/comparison_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ragas_results/comparison_chart.png")
    plt.close()

def create_combined_comparison(experiments):
    """创建综合对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_names = [exp['experiment'].replace('_', '\n') for exp in experiments]
    relevancy_scores = [exp['metrics']['relevancy'] for exp in experiments]
    correctness_scores = [exp['metrics']['correctness'] for exp in experiments]

    x = np.arange(len(exp_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, relevancy_scores, width, label='Relevancy',
                    color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, correctness_scores, width, label='Correctness',
                    color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Self-RAG Evaluation Results: Relevancy vs Correctness',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上显示数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('ragas_results/combined_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ragas_results/combined_comparison.png")
    plt.close()

def create_distribution_charts(experiments):
    """创建分数分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, exp in enumerate(experiments):
        exp_name = exp['experiment']
        relevancy_scores = exp['individual_scores']['relevancy']
        correctness_scores = exp['individual_scores']['correctness']

        # Relevancy 分布
        ax_rel = axes[0, idx]
        ax_rel.hist(relevancy_scores, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
        ax_rel.set_title(f'{exp_name}\nRelevancy Distribution', fontsize=11, fontweight='bold')
        ax_rel.set_xlabel('Score', fontsize=10)
        ax_rel.set_ylabel('Frequency', fontsize=10)
        ax_rel.axvline(np.mean(relevancy_scores), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(relevancy_scores):.3f}')
        ax_rel.legend(fontsize=9)
        ax_rel.grid(axis='y', alpha=0.3)

        # Correctness 分布
        ax_cor = axes[1, idx]
        ax_cor.hist(correctness_scores, bins=10, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax_cor.set_title(f'{exp_name}\nCorrectness Distribution', fontsize=11, fontweight='bold')
        ax_cor.set_xlabel('Score', fontsize=10)
        ax_cor.set_ylabel('Frequency', fontsize=10)
        ax_cor.axvline(np.mean(correctness_scores), color='blue', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(correctness_scores):.3f}')
        ax_cor.legend(fontsize=9)
        ax_cor.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('ragas_results/distribution_charts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ragas_results/distribution_charts.png")
    plt.close()

def create_scatter_plot(experiments):
    """创建相关性-正确性散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    markers = ['o', 's', '^']

    for idx, exp in enumerate(experiments):
        relevancy = exp['individual_scores']['relevancy']
        correctness = exp['individual_scores']['correctness']

        ax.scatter(relevancy, correctness,
                   c=colors[idx], marker=markers[idx], s=100,
                   alpha=0.6, label=exp['experiment'], edgecolors='black')

    ax.set_xlabel('Relevancy Score', fontsize=12)
    ax.set_ylabel('Correctness Score', fontsize=12)
    ax.set_title('Relevancy vs Correctness: Sample-level Analysis',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # 添加参考线
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig('ragas_results/scatter_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ragas_results/scatter_plot.png")
    plt.close()

def create_performance_summary(experiments):
    """创建性能总结表格图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    headers = ['Experiment', 'Relevancy', 'Correctness', 'Avg Score', 'Performance']

    for exp in experiments:
        name = exp['experiment']
        rel = exp['metrics']['relevancy']
        cor = exp['metrics']['correctness']
        avg = (rel + cor) / 2

        # 评价
        if avg >= 0.75:
            perf = 'Excellent'
        elif avg >= 0.6:
            perf = 'Good'
        elif avg >= 0.4:
            perf = 'Fair'
        else:
            perf = 'Poor'

        table_data.append([name, f'{rel:.3f}', f'{cor:.3f}', f'{avg:.3f}', perf])

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置数据行样式
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Self-RAG Evaluation Performance Summary',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('ragas_results/performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ragas_results/performance_summary.png")
    plt.close()

def generate_statistics(experiments):
    """生成统计分析"""
    stats = {}

    for exp in experiments:
        name = exp['experiment']
        rel_scores = exp['individual_scores']['relevancy']
        cor_scores = exp['individual_scores']['correctness']

        stats[name] = {
            'relevancy': {
                'mean': np.mean(rel_scores),
                'std': np.std(rel_scores),
                'min': np.min(rel_scores),
                'max': np.max(rel_scores),
                'median': np.median(rel_scores)
            },
            'correctness': {
                'mean': np.mean(cor_scores),
                'std': np.std(cor_scores),
                'min': np.min(cor_scores),
                'max': np.max(cor_scores),
                'median': np.median(cor_scores)
            }
        }

    return stats

def create_analysis_report(experiments, stats):
    """生成分析报告"""
    report = []
    report.append("=" * 80)
    report.append("Self-RAG 评估结果分析报告")
    report.append("=" * 80)
    report.append("")

    report.append("## 1. 总体概览")
    report.append("")
    for exp in experiments:
        name = exp['experiment']
        rel = exp['metrics']['relevancy']
        cor = exp['metrics']['correctness']
        avg = (rel + cor) / 2

        report.append(f"### {name}")
        report.append(f"  - Relevancy Score:   {rel:.4f} ({rel*100:.1f}%)")
        report.append(f"  - Correctness Score: {cor:.4f} ({cor*100:.1f}%)")
        report.append(f"  - Average Score:     {avg:.4f} ({avg*100:.1f}%)")
        report.append("")

    report.append("=" * 80)
    report.append("## 2. 详细统计分析")
    report.append("")

    for exp in experiments:
        name = exp['experiment']
        report.append(f"### {name}")
        report.append("")

        report.append("**Relevancy Statistics:**")
        for key, val in stats[name]['relevancy'].items():
            report.append(f"  - {key.capitalize():8s}: {val:.4f}")
        report.append("")

        report.append("**Correctness Statistics:**")
        for key, val in stats[name]['correctness'].items():
            report.append(f"  - {key.capitalize():8s}: {val:.4f}")
        report.append("")

    report.append("=" * 80)
    report.append("## 3. 关键发现")
    report.append("")

    # exp1 分析
    report.append("### exp1_popqa (PopQA Dataset)")
    report.append("- **优势**: 非常高的相关性评分 (92.5%)")
    report.append("- **劣势**: 正确性较低 (67.5%)")
    report.append("- **结论**: 模型能够生成高度相关的答案，但在事实准确性上仍有提升空间")
    report.append("")

    # exp2 分析
    report.append("### exp2_arc (ARC Challenge Dataset)")
    report.append("- **优势**: 高正确性评分 (80.0%)")
    report.append("- **劣势**: 极低的相关性评分 (1.0%)")
    report.append("- **结论**: 可能是答案格式不匹配导致相关性评分低，但实际答案是正确的")
    report.append("- **建议**: 需要调整相关性评估方法以适应多选题格式")
    report.append("")

    # exp3 分析
    report.append("### exp3_health (Health Claims Dataset)")
    report.append("- **优势**: 较好的正确性评分 (75.0%)")
    report.append("- **劣势**: 中等的相关性评分 (29.5%)")
    report.append("- **结论**: 在健康声明验证任务上表现一般，需要改进答案的相关性")
    report.append("")

    report.append("=" * 80)
    report.append("## 4. 总体结论与建议")
    report.append("")
    report.append("1. **模型表现差异显著**:")
    report.append("   - 在不同类型的任务上表现差异很大")
    report.append("   - PopQA任务相关性最高，但正确性需要提升")
    report.append("")
    report.append("2. **评估指标的局限性**:")
    report.append("   - 相关性评分可能受答案格式影响较大")
    report.append("   - 对于多选题等特殊格式需要特别处理")
    report.append("")
    report.append("3. **改进方向**:")
    report.append("   - 针对不同数据集优化提示词 (prompt engineering)")
    report.append("   - 调整检索策略以提高事实准确性")
    report.append("   - 标准化答案格式以提高相关性评分")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)

def main():
    """主函数"""
    print("\n" + "="*80)
    print("生成 Self-RAG 评估结果可视化和分析")
    print("="*80 + "\n")

    # 加载数据
    print("加载实验结果...")
    experiments = load_results()
    print(f"✓ 加载了 {len(experiments)} 个实验的结果\n")

    # 生成统计数据
    print("计算统计数据...")
    stats = generate_statistics(experiments)
    print("✓ 统计数据计算完成\n")

    # 生成图表
    print("生成图表...")
    create_comparison_chart(experiments)
    create_combined_comparison(experiments)
    create_distribution_charts(experiments)
    create_scatter_plot(experiments)
    create_performance_summary(experiments)
    print()

    # 生成分析报告
    print("生成分析报告...")
    report = create_analysis_report(experiments, stats)

    with open('ragas_results/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ Saved: ragas_results/analysis_report.txt")
    print()

    # 同时生成 Markdown 版本
    with open('ragas_results/ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ Saved: ragas_results/ANALYSIS_REPORT.md")
    print()

    print("="*80)
    print("所有可视化和分析文件已生成!")
    print("="*80)
    print("\n生成的文件:")
    print("  1. comparison_chart.png        - 相关性和正确性对比图")
    print("  2. combined_comparison.png     - 综合对比图")
    print("  3. distribution_charts.png     - 分数分布图")
    print("  4. scatter_plot.png            - 相关性vs正确性散点图")
    print("  5. performance_summary.png     - 性能总结表格")
    print("  6. analysis_report.txt         - 详细分析报告")
    print("  7. ANALYSIS_REPORT.md          - Markdown格式分析报告")
    print()

if __name__ == '__main__':
    main()
