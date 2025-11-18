#!/usr/bin/env python3
"""
完整的 Self-RAG + RAGAS 评估分析系统
包含流程图、样例分析、定性分析、详细可视化
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# 颜色方案
COLORS = {
    'primary': '#2c3e50',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'purple': '#9b59b6',
    'teal': '#1abc9c',
    'gray': '#95a5a6',
    'light_gray': '#ecf0f1',
    'dark': '#34495e'
}

def load_all_data():
    """加载所有评估数据和原始样本"""
    with open('ragas_results/summary_simple.json', 'r') as f:
        summary = json.load(f)

    experiments = {}
    for exp in summary['experiments']:
        exp_name = exp['experiment']

        # 加载详细结果
        with open(f'ragas_results/{exp_name}_simple_eval.json', 'r') as f:
            experiments[exp_name] = json.load(f)

        # 加载原始实验输出
        exp_file_map = {
            'exp1_popqa': 'retrieval_lm/exp1',
            'exp2_arc': 'retrieval_lm/exp2',
            'exp3_health': 'retrieval_lm/exp3_debug'
        }

        with open(exp_file_map[exp_name], 'r') as f:
            exp_data = json.load(f)
            experiments[exp_name]['predictions'] = exp_data.get('preds', [])[:50]
            experiments[exp_name]['prompts'] = exp_data.get('prompts', [])[:50]

        # 加载评估数据
        eval_file_map = {
            'exp1_popqa': 'eval_data/popqa_longtail_w_gs.jsonl',
            'exp2_arc': 'eval_data/arc_challenge_processed.jsonl',
            'exp3_health': 'eval_data/health_claims_processed.jsonl'
        }

        eval_samples = []
        with open(eval_file_map[exp_name], 'r') as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                eval_samples.append(json.loads(line.strip()))
        experiments[exp_name]['eval_samples'] = eval_samples

    return experiments

def create_selfrag_architecture():
    """创建 Self-RAG 架构图（矢量图）"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'Self-RAG Architecture & Pipeline',
            fontsize=24, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['primary'],
                     edgecolor='none', alpha=0.8),
            color='white')

    # 1. 输入层
    input_box = FancyBboxPatch((0.5, 7.5), 1.5, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['info'],
                               edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 7.9, 'Input Query', fontsize=12, ha='center',
            fontweight='bold', color='white')

    # 2. Retrieval 模块
    retrieval_box = FancyBboxPatch((2.5, 7.3), 2, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['success'],
                                   edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(retrieval_box)
    ax.text(3.5, 8.2, 'Retrieval Module', fontsize=13, ha='center',
            fontweight='bold', color='white')
    ax.text(3.5, 7.8, '• Search relevant docs', fontsize=9, ha='center', color='white')
    ax.text(3.5, 7.5, '• Rank by relevance', fontsize=9, ha='center', color='white')

    # 3. Self-Reflection 模块
    reflection_box = FancyBboxPatch((5.5, 7.3), 2, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['warning'],
                                    edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(reflection_box)
    ax.text(6.5, 8.2, 'Self-Reflection', fontsize=13, ha='center',
            fontweight='bold', color='white')
    ax.text(6.5, 7.8, '• Assess relevance', fontsize=9, ha='center', color='white')
    ax.text(6.5, 7.5, '• Check support', fontsize=9, ha='center', color='white')

    # 4. Generator (LLM)
    generator_box = FancyBboxPatch((8.5, 7.5), 1.3, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['purple'],
                                   edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(generator_box)
    ax.text(9.15, 7.9, 'Generator\n(LLM)', fontsize=11, ha='center',
            fontweight='bold', color='white')

    # 箭头
    arrow1 = FancyArrowPatch((2, 7.9), (2.5, 7.9),
                            arrowstyle='->', mutation_scale=30, linewidth=2.5,
                            color=COLORS['dark'])
    arrow2 = FancyArrowPatch((4.5, 7.9), (5.5, 7.9),
                            arrowstyle='->', mutation_scale=30, linewidth=2.5,
                            color=COLORS['dark'])
    arrow3 = FancyArrowPatch((7.5, 7.9), (8.5, 7.9),
                            arrowstyle='->', mutation_scale=30, linewidth=2.5,
                            color=COLORS['dark'])
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)

    # 中间层 - 迭代过程
    ax.text(5, 6.5, 'Iterative Generation Process', fontsize=14,
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_gray'],
                     edgecolor=COLORS['dark'], linewidth=1))

    # Token级别的反思
    for i, (token_text, color) in enumerate([
        ('Token 1', COLORS['info']),
        ('Token 2', COLORS['success']),
        ('Token 3', COLORS['warning']),
        ('...', COLORS['gray'])
    ]):
        x = 1.5 + i * 2
        token_box = FancyBboxPatch((x, 5.2), 1.2, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, alpha=0.7,
                                   edgecolor=COLORS['dark'], linewidth=1.5)
        ax.add_patch(token_box)
        ax.text(x + 0.6, 5.5, token_text, fontsize=10, ha='center',
                fontweight='bold', color='white')

    # 反思类型
    reflection_types = [
        ('Retrieval', 1.5, 4.2, COLORS['info']),
        ('Relevance', 3.5, 4.2, COLORS['success']),
        ('Support', 5.5, 4.2, COLORS['warning']),
        ('Utility', 7.5, 4.2, COLORS['danger'])
    ]

    for label, x, y, color in reflection_types:
        circle = Circle((x + 0.4, y + 0.3), 0.3, facecolor=color,
                       edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x + 0.4, y + 0.3, label[0], fontsize=12, ha='center',
                va='center', fontweight='bold', color='white')
        ax.text(x + 0.4, y - 0.3, label, fontsize=9, ha='center',
                fontweight='bold')

    # 输出层
    ax.text(5, 3.2, 'Output Generation', fontsize=14, ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_gray'],
                     edgecolor=COLORS['dark'], linewidth=1))

    output_box = FancyBboxPatch((3.5, 2), 3, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['teal'],
                                edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.35, 'Final Answer with Reflection Tokens',
            fontsize=12, ha='center', fontweight='bold', color='white')

    # RAGAS 评估层
    ax.text(5, 1.3, 'RAGAS Evaluation Layer', fontsize=14, ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['danger'],
                     edgecolor='none', alpha=0.8),
            color='white')

    # 评估指标
    metrics = [
        ('Relevancy', 2, 0.3, COLORS['info']),
        ('Correctness', 4, 0.3, COLORS['success']),
        ('Faithfulness', 6, 0.3, COLORS['warning']),
        ('Context\nPrecision', 8, 0.3, COLORS['purple'])
    ]

    for label, x, y, color in metrics:
        metric_box = FancyBboxPatch((x - 0.6, y - 0.15), 1.2, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, alpha=0.8,
                                    edgecolor=COLORS['dark'], linewidth=1.5)
        ax.add_patch(metric_box)
        ax.text(x, y + 0.1, label, fontsize=9, ha='center',
                va='center', fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('ragas_results/selfrag_architecture.svg', format='svg',
                dpi=300, bbox_inches='tight')
    plt.savefig('ragas_results/selfrag_architecture.png', dpi=300,
                bbox_inches='tight')
    print("✓ Saved: selfrag_architecture.svg & .png")
    plt.close()

def create_evaluation_pipeline():
    """创建评估流程图（矢量图）"""
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 标题
    ax.text(5, 11.5, 'Self-RAG Evaluation Pipeline',
            fontsize=22, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['primary'],
                     edgecolor='none', alpha=0.9),
            color='white')

    # 阶段1: 数据准备
    y = 10
    ax.text(5, y, 'Phase 1: Data Preparation', fontsize=14, ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['info'],
                     alpha=0.3, edgecolor=COLORS['info'], linewidth=2))

    datasets = [
        ('PopQA\n500 samples', 1.5, y-1),
        ('ARC Challenge\n500 samples', 5, y-1),
        ('Health Claims\n500 samples', 8.5, y-1)
    ]

    for label, x, y_pos in datasets:
        box = FancyBboxPatch((x-0.6, y_pos-0.3), 1.2, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=COLORS['light_gray'],
                            edgecolor=COLORS['dark'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, label, fontsize=9, ha='center', va='center',
                fontweight='bold')

    # 阶段2: Self-RAG 推理
    y = 7.5
    ax.text(5, y, 'Phase 2: Self-RAG Inference', fontsize=14, ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['success'],
                     alpha=0.3, edgecolor=COLORS['success'], linewidth=2))

    selfrag_box = FancyBboxPatch((2, y-1.3), 6, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=COLORS['success'], alpha=0.8,
                                 edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(selfrag_box)
    ax.text(5, y-0.6, 'Self-RAG Model', fontsize=12, ha='center',
            fontweight='bold', color='white')
    ax.text(5, y-0.9, '• Query → Retrieve → Reflect → Generate',
            fontsize=9, ha='center', color='white')

    # 输出20个样本
    ax.text(5, y-1.8, '↓', fontsize=20, ha='center', color=COLORS['dark'])
    ax.text(5, y-2.2, '20 samples per dataset', fontsize=10, ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['light_gray']))

    # 阶段3: RAGAS 评估
    y = 4
    ax.text(5, y, 'Phase 3: RAGAS Evaluation', fontsize=14, ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['warning'],
                     alpha=0.3, edgecolor=COLORS['warning'], linewidth=2))

    # 评估流程
    eval_steps = [
        ('Load\nOutputs', 1.5, y-0.8, COLORS['info']),
        ('DeepSeek\nAPI', 3.5, y-0.8, COLORS['purple']),
        ('Compute\nScores', 5.5, y-0.8, COLORS['warning']),
        ('Aggregate\nResults', 7.5, y-0.8, COLORS['success'])
    ]

    for i, (label, x, y_pos, color) in enumerate(eval_steps):
        box = FancyBboxPatch((x-0.5, y_pos-0.3), 1, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=color, alpha=0.8,
                            edgecolor=COLORS['dark'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, label, fontsize=9, ha='center', va='center',
                fontweight='bold', color='white')

        if i < len(eval_steps) - 1:
            arrow = FancyArrowPatch((x+0.5, y_pos), (x+1.5, y_pos),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color=COLORS['dark'])
            ax.add_patch(arrow)

    # 阶段4: 结果分析
    y = 1.5
    ax.text(5, y, 'Phase 4: Analysis & Visualization', fontsize=14, ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['danger'],
                     alpha=0.3, edgecolor=COLORS['danger'], linewidth=2))

    analysis_outputs = [
        ('Charts', 2, y-0.8),
        ('Reports', 4, y-0.8),
        ('Case Studies', 6, y-0.8),
        ('Insights', 8, y-0.8)
    ]

    for label, x, y_pos in analysis_outputs:
        circle = Circle((x, y_pos), 0.35, facecolor=COLORS['teal'],
                       alpha=0.8, edgecolor=COLORS['dark'], linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y_pos, label, fontsize=8, ha='center', va='center',
                fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('ragas_results/evaluation_pipeline.svg', format='svg',
                dpi=300, bbox_inches='tight')
    plt.savefig('ragas_results/evaluation_pipeline.png', dpi=300,
                bbox_inches='tight')
    print("✓ Saved: evaluation_pipeline.svg & .png")
    plt.close()

def analyze_case_studies(experiments):
    """样例分析 - 选择典型案例"""
    case_studies = {}

    for exp_name, data in experiments.items():
        rel_scores = data['individual_scores']['relevancy']
        cor_scores = data['individual_scores']['correctness']
        predictions = data['predictions']
        eval_samples = data['eval_samples']

        # 选择典型案例
        # 1. 最佳案例 (high relevancy, high correctness)
        combined_scores = [r + c for r, c in zip(rel_scores, cor_scores)]
        best_idx = np.argmax(combined_scores)

        # 2. 最差案例
        worst_idx = np.argmin(combined_scores)

        # 3. 相关但不正确
        rel_not_correct = [(i, r, c) for i, (r, c) in enumerate(zip(rel_scores, cor_scores))
                          if r > 0.7 and c < 0.3]
        rel_not_correct_idx = rel_not_correct[0][0] if rel_not_correct else None

        # 4. 正确但不相关
        correct_not_rel = [(i, r, c) for i, (r, c) in enumerate(zip(rel_scores, cor_scores))
                          if c > 0.7 and r < 0.3]
        correct_not_rel_idx = correct_not_rel[0][0] if correct_not_rel else None

        case_studies[exp_name] = {
            'best': {
                'idx': best_idx,
                'question': eval_samples[best_idx].get('question') or eval_samples[best_idx].get('claim', ''),
                'ground_truth': _get_ground_truth(eval_samples[best_idx]),
                'prediction': predictions[best_idx],
                'relevancy': rel_scores[best_idx],
                'correctness': cor_scores[best_idx]
            },
            'worst': {
                'idx': worst_idx,
                'question': eval_samples[worst_idx].get('question') or eval_samples[worst_idx].get('claim', ''),
                'ground_truth': _get_ground_truth(eval_samples[worst_idx]),
                'prediction': predictions[worst_idx],
                'relevancy': rel_scores[worst_idx],
                'correctness': cor_scores[worst_idx]
            }
        }

        if rel_not_correct_idx is not None:
            case_studies[exp_name]['relevant_but_incorrect'] = {
                'idx': rel_not_correct_idx,
                'question': eval_samples[rel_not_correct_idx].get('question') or eval_samples[rel_not_correct_idx].get('claim', ''),
                'ground_truth': _get_ground_truth(eval_samples[rel_not_correct_idx]),
                'prediction': predictions[rel_not_correct_idx],
                'relevancy': rel_scores[rel_not_correct_idx],
                'correctness': cor_scores[rel_not_correct_idx]
            }

        if correct_not_rel_idx is not None:
            case_studies[exp_name]['correct_but_irrelevant'] = {
                'idx': correct_not_rel_idx,
                'question': eval_samples[correct_not_rel_idx].get('question') or eval_samples[correct_not_rel_idx].get('claim', ''),
                'ground_truth': _get_ground_truth(eval_samples[correct_not_rel_idx]),
                'prediction': predictions[correct_not_rel_idx],
                'relevancy': rel_scores[correct_not_rel_idx],
                'correctness': cor_scores[correct_not_rel_idx]
            }

    return case_studies

def _get_ground_truth(sample):
    """提取 ground truth"""
    if 'answers' in sample and sample['answers']:
        return str(sample['answers'][0])
    elif 'answerKey' in sample:
        return sample['answerKey']
    elif 'label' in sample:
        return str(sample['label'])
    return 'N/A'

def create_case_study_visualization(case_studies):
    """创建样例分析可视化"""
    for exp_name, cases in case_studies.items():
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'{exp_name} - Detailed Case Studies',
                    fontsize=18, fontweight='bold', y=0.98)

        case_types = ['best', 'worst']
        if 'relevant_but_incorrect' in cases:
            case_types.append('relevant_but_incorrect')
        if 'correct_but_irrelevant' in cases:
            case_types.append('correct_but_irrelevant')

        n_cases = len(case_types)

        for i, case_type in enumerate(case_types):
            ax = plt.subplot(n_cases, 1, i+1)
            ax.axis('off')

            case = cases[case_type]

            # 标题
            title_map = {
                'best': '✅ Best Case (High Relevancy + High Correctness)',
                'worst': '❌ Worst Case (Low Relevancy + Low Correctness)',
                'relevant_but_incorrect': '⚠️ Relevant but Incorrect',
                'correct_but_irrelevant': '⚠️ Correct but Irrelevant'
            }

            color_map = {
                'best': COLORS['success'],
                'worst': COLORS['danger'],
                'relevant_but_incorrect': COLORS['warning'],
                'correct_but_irrelevant': COLORS['info']
            }

            ax.text(0.5, 0.95, title_map[case_type],
                   transform=ax.transAxes, fontsize=13, fontweight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=color_map[case_type],
                           alpha=0.8, edgecolor='none'),
                   color='white')

            # 内容框
            question_text = f"Question: {case['question'][:150]}..."
            gt_text = f"Ground Truth: {case['ground_truth'][:100]}"
            pred_text = f"Prediction: {case['prediction'][:150]}..."
            scores_text = f"Relevancy: {case['relevancy']:.3f} | Correctness: {case['correctness']:.3f}"

            y_pos = 0.75
            ax.text(0.05, y_pos, question_text, transform=ax.transAxes,
                   fontsize=10, wrap=True, va='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=COLORS['light_gray'],
                           alpha=0.5))

            y_pos = 0.55
            ax.text(0.05, y_pos, gt_text, transform=ax.transAxes,
                   fontsize=10, wrap=True, va='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#e8f5e9', alpha=0.7))

            y_pos = 0.35
            ax.text(0.05, y_pos, pred_text, transform=ax.transAxes,
                   fontsize=10, wrap=True, va='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#e3f2fd', alpha=0.7))

            y_pos = 0.10
            ax.text(0.5, y_pos, scores_text, transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=color_map[case_type],
                           alpha=0.3, edgecolor=color_map[case_type],
                           linewidth=2))

        plt.tight_layout()
        plt.savefig(f'ragas_results/{exp_name}_case_studies.png',
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {exp_name}_case_studies.png")
        plt.close()

def create_advanced_visualizations(experiments):
    """创建高级可视化"""

    # 1. 热力图 - 样本级别的表现
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sample-level Performance Heatmap', fontsize=16, fontweight='bold')

    for idx, (exp_name, data) in enumerate(experiments.items()):
        ax = axes[idx]

        scores = np.array([
            data['individual_scores']['relevancy'],
            data['individual_scores']['correctness']
        ])

        im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Relevancy', 'Correctness'])
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_title(exp_name.replace('_', '\n'), fontsize=12, fontweight='bold')

        # 添加数值标注
        for i in range(2):
            for j in range(len(scores[0])):
                text = ax.text(j, i, f'{scores[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7)

    plt.colorbar(im, ax=axes, label='Score')
    plt.tight_layout()
    plt.savefig('ragas_results/performance_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_heatmap.png")
    plt.close()

    # 2. 雷达图 - 多维度对比
    fig = plt.figure(figsize=(15, 5))

    categories = ['Relevancy\nMean', 'Relevancy\nMedian', 'Correctness\nMean',
                  'Correctness\nMedian', 'Overall\nScore']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for idx, (exp_name, data) in enumerate(experiments.items()):
        ax = fig.add_subplot(1, 3, idx+1, projection='polar')

        rel = data['individual_scores']['relevancy']
        cor = data['individual_scores']['correctness']

        values = [
            np.mean(rel),
            np.median(rel),
            np.mean(cor),
            np.median(cor),
            (np.mean(rel) + np.mean(cor)) / 2
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=exp_name,
                color=['#3498db', '#e74c3c', '#2ecc71'][idx])
        ax.fill(angles, values, alpha=0.25,
                color=['#3498db', '#e74c3c', '#2ecc71'][idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(exp_name, fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('ragas_results/radar_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: radar_comparison.png")
    plt.close()

    # 3. 箱线图 - 分数分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    rel_data = [experiments[exp]['individual_scores']['relevancy']
                for exp in experiments.keys()]
    cor_data = [experiments[exp]['individual_scores']['correctness']
                for exp in experiments.keys()]

    labels = [exp.replace('exp', 'Exp').replace('_', '\n')
              for exp in experiments.keys()]

    bp1 = ax1.boxplot(rel_data, labels=labels, patch_artist=True,
                       notch=True, showmeans=True)
    for patch, color in zip(bp1['boxes'], [COLORS['info'], COLORS['danger'], COLORS['success']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Relevancy Score Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    bp2 = ax2.boxplot(cor_data, labels=labels, patch_artist=True,
                       notch=True, showmeans=True)
    for patch, color in zip(bp2['boxes'], [COLORS['info'], COLORS['danger'], COLORS['success']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Correctness Score Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('ragas_results/boxplot_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: boxplot_distribution.png")
    plt.close()

def main():
    """主函数"""
    print("\n" + "="*80)
    print("完整的 Self-RAG + RAGAS 分析系统")
    print("="*80 + "\n")

    # 1. 创建架构图和流程图
    print("Step 1: Creating architecture diagrams...")
    create_selfrag_architecture()
    create_evaluation_pipeline()
    print()

    # 2. 加载数据
    print("Step 2: Loading all data...")
    experiments = load_all_data()
    print(f"✓ Loaded {len(experiments)} experiments with full details\n")

    # 3. 样例分析
    print("Step 3: Analyzing case studies...")
    case_studies = analyze_case_studies(experiments)
    create_case_study_visualization(case_studies)
    print()

    # 4. 高级可视化
    print("Step 4: Creating advanced visualizations...")
    create_advanced_visualizations(experiments)
    print()

    # 5. 保存样例数据
    print("Step 5: Saving case study data...")
    # Convert numpy types to Python types
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    case_studies_clean = convert_to_python_types(case_studies)
    with open('ragas_results/case_studies.json', 'w', encoding='utf-8') as f:
        json.dump(case_studies_clean, f, indent=2, ensure_ascii=False)
    print("✓ Saved: case_studies.json")
    print()

    print("="*80)
    print("所有高级分析文件已生成!")
    print("="*80)
    print("\n新增文件:")
    print("  • selfrag_architecture.svg/.png  - Self-RAG架构图 (矢量图)")
    print("  • evaluation_pipeline.svg/.png   - 评估流程图 (矢量图)")
    print("  • *_case_studies.png             - 每个实验的样例分析")
    print("  • performance_heatmap.png        - 样本级表现热力图")
    print("  • radar_comparison.png           - 多维度雷达图")
    print("  • boxplot_distribution.png       - 箱线图分布")
    print("  • case_studies.json              - 样例分析数据")
    print()

if __name__ == '__main__':
    main()
