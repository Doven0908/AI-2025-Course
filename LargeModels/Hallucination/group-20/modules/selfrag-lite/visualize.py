import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib import font_manager
import matplotlib.gridspec as gridspec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存结果的文件夹
output_dir = "visualization_analysis"
os.makedirs(output_dir, exist_ok=True)

EXP_FILE_NAME = './experiment_file.json'


data = json.load(open(EXP_FILE_NAME))



def create_dataframe(data):
    """将数据转换为DataFrame格式便于分析"""
    records = []
    
    for item in data['data']:
        question_id = item['question'][:30] + "..." if len(item['question']) > 30 else item['question']
        
        # 为每个模型创建记录
        for model in ['openai', 'rag', 'self_rag']:
            record = {
                'question': item['question'],
                'question_id': question_id,
                'model': model,
                'time': item[f'{model}_time'] if model != 'self_rag' else item['selfrag_time'],
                'context_recall': item[f'{model}_answer_ragas_evaluation']['context_recall'],
                'faithfulness': item[f'{model}_answer_ragas_evaluation']['faithfulness'],
                'factual_correctness': item[f'{model}_answer_ragas_evaluation']['factual_correctness(mode=f1)'],
                'hallucination_scores': item[f'{model}_answer_selfcheckgpt_sentence_hallucination_scores'],
                'answer': item[f'{model}_answer'],
                'reference_answer': item['answer']
            }
            records.append(record)
    
    return pd.DataFrame(records)

# 创建DataFrame
df = create_dataframe(data)

print("数据概览:")
print(f"总问题数: {len(data['data'])}")
print(f"总记录数: {len(df)}")
print("\n模型分布:")
print(df['model'].value_counts())

# 1. 响应时间分析
print("\n=== 响应时间分析 ===")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('模型响应时间分析', fontsize=16, fontweight='bold')

# 1.1 各模型响应时间比较
time_data = df.groupby('model')['time'].mean().sort_values(ascending=False)
axes[0,0].bar(time_data.index, time_data.values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
axes[0,0].set_title('各模型平均响应时间')
axes[0,0].set_ylabel('时间 (秒)')
axes[0,0].tick_params(axis='x', rotation=45)
for i, v in enumerate(time_data.values):
    axes[0,0].text(i, v + 0.5, f'{v:.2f}s', ha='center', va='bottom')

# 1.2 各问题响应时间对比
time_by_question = df.pivot_table(index='question_id', columns='model', values='time')
time_by_question.plot(kind='bar', ax=axes[0,1], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
axes[0,1].set_title('各问题不同模型响应时间')
axes[0,1].set_ylabel('时间 (秒)')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].legend(title='模型')

# 1.3 响应时间分布
time_stats = df.groupby('model')['time'].agg(['mean', 'std', 'min', 'max'])
axes[1,0].barh(time_stats.index, time_stats['mean'], xerr=time_stats['std'], 
               color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.7)
axes[1,0].set_title('响应时间统计 (均值±标准差)')
axes[1,0].set_xlabel('时间 (秒)')

# 1.4 响应时间热力图
time_matrix = df.pivot_table(index='question_id', columns='model', values='time')
sns.heatmap(time_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1,1])
axes[1,1].set_title('响应时间热力图 (秒)')

plt.tight_layout()
plt.savefig(f'{output_dir}/1_响应时间分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 评估指标分析
print("\n=== 评估指标分析 ===")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('模型评估指标分析', fontsize=16, fontweight='bold')

metrics = ['context_recall', 'faithfulness', 'factual_correctness']
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

for i, metric in enumerate(metrics):
    # 各模型指标比较
    metric_data = df.groupby('model')[metric].mean()
    axes[0,i].bar(metric_data.index, metric_data.values, color=colors)
    axes[0,i].set_title(f'{metric} 比较')
    axes[0,i].set_ylabel('得分')
    axes[0,i].set_ylim(0, 1)
    axes[0,i].tick_params(axis='x', rotation=45)
    for j, v in enumerate(metric_data.values):
        axes[0,i].text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 各问题指标对比
    metric_by_question = df.pivot_table(index='question_id', columns='model', values=metric)
    metric_by_question.plot(kind='bar', ax=axes[1,i], color=colors)
    axes[1,i].set_title(f'各问题 {metric} 对比')
    axes[1,i].set_ylabel('得分')
    axes[1,i].set_ylim(0, 1)
    axes[1,i].tick_params(axis='x', rotation=45)
    axes[1,i].legend(title='模型')

plt.tight_layout()
plt.savefig(f'{output_dir}/2_评估指标分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 幻觉分数分析
print("\n=== 幻觉分数分析 ===")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('模型幻觉分数分析', fontsize=16, fontweight='bold')

# 3.1 平均幻觉分数
hallucination_means = []
for model in df['model'].unique():
    scores = []
    for hall_scores in df[df['model'] == model]['hallucination_scores']:
        scores.extend(hall_scores)
    hallucination_means.append(np.mean(scores) if scores else 0)

axes[0,0].bar(df['model'].unique(), hallucination_means, color=colors)
axes[0,0].set_title('各模型平均幻觉分数')
axes[0,0].set_ylabel('幻觉分数')
axes[0,0].set_ylim(0, 1)
for i, v in enumerate(hallucination_means):
    axes[0,0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# 3.2 幻觉分数分布
all_scores = []
labels = []
for model in df['model'].unique():
    model_scores = []
    for hall_scores in df[df['model'] == model]['hallucination_scores']:
        model_scores.extend(hall_scores)
    all_scores.append(model_scores)
    labels.append(model)

axes[0,1].boxplot(all_scores, labels=labels)
axes[0,1].set_title('幻觉分数分布')
axes[0,1].set_ylabel('幻觉分数')

# 3.3 各问题幻觉分数对比
question_hallucination = []
for question in df['question_id'].unique():
    question_data = []
    for model in df['model'].unique():
        scores = []
        for idx, row in df[(df['question_id'] == question) & (df['model'] == model)].iterrows():
            scores.extend(row['hallucination_scores'])
        question_data.append(np.mean(scores) if scores else 0)
    question_hallucination.append(question_data)

x = np.arange(len(df['question_id'].unique()))
width = 0.25
for i, model in enumerate(df['model'].unique()):
    axes[1,0].bar(x + i*width, [q[i] for q in question_hallucination], width, label=model, color=colors[i])
axes[1,0].set_title('各问题幻觉分数对比')
axes[1,0].set_ylabel('平均幻觉分数')
axes[1,0].set_xticks(x + width)
axes[1,0].set_xticklabels(df['question_id'].unique(), rotation=45)
axes[1,0].legend()

# 3.4 高幻觉分数比例
high_hallucination_ratio = []
for model in df['model'].unique():
    all_scores_flat = []
    for scores in df[df['model'] == model]['hallucination_scores']:
        all_scores_flat.extend(scores)
    high_ratio = len([s for s in all_scores_flat if s > 0.5]) / len(all_scores_flat) if all_scores_flat else 0
    high_hallucination_ratio.append(high_ratio)

axes[1,1].bar(df['model'].unique(), high_hallucination_ratio, color=colors)
axes[1,1].set_title('高幻觉分数 (>0.5) 比例')
axes[1,1].set_ylabel('比例')
axes[1,1].set_ylim(0, 1)
for i, v in enumerate(high_hallucination_ratio):
    axes[1,1].text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{output_dir}/3_幻觉分数分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 综合性能雷达图
print("\n=== 综合性能分析 ===")
fig = plt.figure(figsize=(12, 8))

# 计算综合得分（反转时间得分，时间越短得分越高）
max_time = df['time'].max()
performance_data = []

for model in df['model'].unique():
    model_data = df[df['model'] == model]
    performance = {
        'model': model,
        '时间效率': 1 - (model_data['time'].mean() / max_time),  # 反转时间得分
        '上下文召回': model_data['context_recall'].mean(),
        '忠实度': model_data['faithfulness'].mean(),
        '事实正确性': model_data['factual_correctness'].mean(),
        '低幻觉性': 1 - np.mean([score for scores in model_data['hallucination_scores'] for score in scores])  # 反转幻觉分数
    }
    performance_data.append(performance)

# 创建雷达图
categories = list(performance_data[0].keys())[1:]
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories, color='grey', size=10)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
plt.ylim(0, 1)

for i, perf in enumerate(performance_data):
    values = list(perf.values())[1:]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=perf['model'], color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

plt.title('模型综合性能雷达图', size=14, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.savefig(f'{output_dir}/4_综合性能雷达图.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 相关性分析
print("\n=== 相关性分析 ===")
# 创建相关性数据
correlation_data = []
for _, row in df.iterrows():
    avg_hallucination = np.mean(row['hallucination_scores']) if row['hallucination_scores'] else 0
    correlation_data.append({
        'time': row['time'],
        'context_recall': row['context_recall'],
        'faithfulness': row['faithfulness'],
        'factual_correctness': row['factual_correctness'],
        'hallucination': avg_hallucination
    })

corr_df = pd.DataFrame(correlation_data)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('指标间相关性分析', fontsize=16, fontweight='bold')

# 5.1 时间与质量指标的相关性
axes[0,0].scatter(corr_df['time'], corr_df['factual_correctness'], alpha=0.7, s=100)
axes[0,0].set_xlabel('响应时间 (秒)')
axes[0,0].set_ylabel('事实正确性')
axes[0,0].set_title('响应时间 vs 事实正确性')
correlation = corr_df['time'].corr(corr_df['factual_correctness'])
axes[0,0].text(0.05, 0.95, f'相关系数: {correlation:.3f}', transform=axes[0,0].transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 5.2 忠实度与事实正确性的相关性
axes[0,1].scatter(corr_df['faithfulness'], corr_df['factual_correctness'], alpha=0.7, s=100)
axes[0,1].set_xlabel('忠实度')
axes[0,1].set_ylabel('事实正确性')
axes[0,1].set_title('忠实度 vs 事实正确性')
correlation = corr_df['faithfulness'].corr(corr_df['factual_correctness'])
axes[0,1].text(0.05, 0.95, f'相关系数: {correlation:.3f}', transform=axes[0,1].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 5.3 幻觉分数与事实正确性的相关性
axes[1,0].scatter(corr_df['hallucination'], corr_df['factual_correctness'], alpha=0.7, s=100)
axes[1,0].set_xlabel('幻觉分数')
axes[1,0].set_ylabel('事实正确性')
axes[1,0].set_title('幻觉分数 vs 事实正确性')
correlation = corr_df['hallucination'].corr(corr_df['factual_correctness'])
axes[1,0].text(0.05, 0.95, f'相关系数: {correlation:.3f}', transform=axes[1,0].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 5.4 整体相关性热力图
correlation_matrix = corr_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
axes[1,1].set_title('指标间相关性热力图')

plt.tight_layout()
plt.savefig(f'{output_dir}/5_相关性分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 详细统计表格
print("\n=== 生成详细统计表格 ===")

# 6.1 基本统计表
basic_stats = df.groupby('model').agg({
    'time': ['mean', 'std', 'min', 'max'],
    'context_recall': 'mean',
    'faithfulness': 'mean', 
    'factual_correctness': 'mean'
}).round(3)

basic_stats.columns = ['时间均值', '时间标准差', '最短时间', '最长时间', '上下文召回', '忠实度', '事实正确性']
basic_stats.to_csv(f'{output_dir}/基本统计表.csv', encoding='utf-8-sig')

# 6.2 幻觉统计表
hallucination_stats = []
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    all_scores = [score for scores in model_data['hallucination_scores'] for score in scores]
    if all_scores:
        stats = {
            '模型': model,
            '平均幻觉分数': np.mean(all_scores),
            '幻觉分数标准差': np.std(all_scores),
            '最大幻觉分数': np.max(all_scores),
            '高幻觉比例(>0.5)': len([s for s in all_scores if s > 0.5]) / len(all_scores),
            '总句子数': len(all_scores)
        }
        hallucination_stats.append(stats)

hallucination_df = pd.DataFrame(hallucination_stats).round(3)
hallucination_df.to_csv(f'{output_dir}/幻觉统计表.csv', encoding='utf-8-sig', index=False)

# 6.3 性能排名表
performance_ranking = []
metrics_for_ranking = ['context_recall', 'faithfulness', 'factual_correctness', 'time']

for metric in metrics_for_ranking:
    if metric == 'time':
        # 时间越短越好
        ranked = df.groupby('model')[metric].mean().sort_values(ascending=True)
    else:
        # 其他指标越高越好
        ranked = df.groupby('model')[metric].mean().sort_values(ascending=False)
    
    for i, (model, value) in enumerate(ranked.items()):
        performance_ranking.append({
            '指标': metric,
            '模型': model,
            '得分': value,
            '排名': i + 1
        })

performance_ranking_df = pd.DataFrame(performance_ranking)
performance_ranking_df.to_csv(f'{output_dir}/性能排名表.csv', encoding='utf-8-sig', index=False)

# 7. 生成分析报告
print("\n=== 生成分析报告 ===")
report_content = f"""
# AI模型性能分析报告

## 数据概览
- 分析问题数量: {len(data['data'])}
- 评估模型数量: {len(df['model'].unique())}
- 总数据记录: {len(df)}

## 主要发现

### 1. 响应时间表现
- 最快模型: {df.groupby('model')['time'].mean().idxmin()} (平均 {df.groupby('model')['time'].mean().min():.2f}秒)
- 最慢模型: {df.groupby('model')['time'].mean().idxmax()} (平均 {df.groupby('model')['time'].mean().max():.2f}秒)
- 时间差异: {df.groupby('model')['time'].mean().max() / df.groupby('model')['time'].mean().min():.1f}倍

### 2. 质量指标表现
- 最佳上下文召回: {df.groupby('model')['context_recall'].mean().idxmax()} ({df.groupby('model')['context_recall'].mean().max():.3f})
- 最佳忠实度: {df.groupby('model')['faithfulness'].mean().idxmax()} ({df.groupby('model')['faithfulness'].mean().max():.3f})
- 最佳事实正确性: {df.groupby('model')['factual_correctness'].mean().idxmax()} ({df.groupby('model')['factual_correctness'].mean().max():.3f})

### 3. 幻觉控制表现
- 最低平均幻觉分数: {hallucination_df.loc[hallucination_df['平均幻觉分数'].idxmin(), '模型']} ({hallucination_df['平均幻觉分数'].min():.3f})
- 最佳幻觉控制: {hallucination_df.loc[hallucination_df['高幻觉比例(>0.5)'].idxmin(), '模型']} (高幻觉比例: {hallucination_df['高幻觉比例(>0.5)'].min():.1%})

## 建议
基于分析结果，建议在不同场景下选择合适的模型：
- 需要快速响应时: 选择响应时间短的模型
- 需要高准确性时: 选择事实正确性高的模型  
- 需要可靠信息时: 选择幻觉分数低的模型

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{output_dir}/分析报告.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

# 8. 生成汇总仪表板
print("\n=== 生成汇总仪表板 ===")
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 3, figure=fig)

# 8.1 总体性能对比
ax1 = fig.add_subplot(gs[0, :])
overall_scores = []
models = df['model'].unique()
for model in models:
    model_data = df[df['model'] == model]
    # 计算综合得分（排除时间，因为量纲不同）
    avg_score = (model_data['context_recall'].mean() + 
                model_data['faithfulness'].mean() + 
                model_data['factual_correctness'].mean()) / 3
    overall_scores.append(avg_score)

bars = ax1.bar(models, overall_scores, color=colors, alpha=0.8)
ax1.set_title('模型综合质量得分对比', fontsize=16, fontweight='bold')
ax1.set_ylabel('平均得分')
ax1.set_ylim(0, 1)
for bar, score in zip(bars, overall_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 8.2 时间-质量散点图
ax2 = fig.add_subplot(gs[1, 0])
for i, model in enumerate(models):
    model_data = df[df['model'] == model]
    avg_time = model_data['time'].mean()
    avg_quality = (model_data['context_recall'].mean() + 
                  model_data['faithfulness'].mean() + 
                  model_data['factual_correctness'].mean()) / 3
    ax2.scatter(avg_time, avg_quality, s=200, color=colors[i], label=model, alpha=0.7)
    ax2.annotate(model, (avg_time, avg_quality), xytext=(5, 5), textcoords='offset points')
ax2.set_xlabel('平均响应时间 (秒)')
ax2.set_ylabel('平均质量得分')
ax2.set_title('时间-质量平衡分析')
ax2.grid(True, alpha=0.3)

# 8.3 各指标详细对比
ax3 = fig.add_subplot(gs[1, 1:])
metrics_comparison = df.groupby('model')[['context_recall', 'faithfulness', 'factual_correctness']].mean()
x = np.arange(len(models))
width = 0.25
for i, metric in enumerate(['context_recall', 'faithfulness', 'factual_correctness']):
    ax3.bar(x + i*width, metrics_comparison[metric], width, 
            label=metric, alpha=0.8)
ax3.set_xlabel('模型')
ax3.set_ylabel('得分')
ax3.set_title('各质量指标详细对比')
ax3.set_xticks(x + width)
ax3.set_xticklabels(models)
ax3.legend()
ax3.set_ylim(0, 1)

# 8.4 幻觉控制能力
ax4 = fig.add_subplot(gs[2, 0])
hallucination_bars = ax4.bar(hallucination_df['模型'], hallucination_df['平均幻觉分数'], 
                            color=colors, alpha=0.8)
ax4.set_title('平均幻觉分数对比')
ax4.set_ylabel('幻觉分数')
ax4.set_ylim(0, 1)
for bar, score in zip(hallucination_bars, hallucination_df['平均幻觉分数']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{score:.3f}', ha='center', va='bottom')

# 8.5 高幻觉比例
ax5 = fig.add_subplot(gs[2, 1])
high_hall_bars = ax5.bar(hallucination_df['模型'], hallucination_df['高幻觉比例(>0.5)'], 
                        color=colors, alpha=0.8)
ax5.set_title('高幻觉句子比例 (>0.5)')
ax5.set_ylabel('比例')
ax5.set_ylim(0, 1)
for bar, ratio in zip(high_hall_bars, hallucination_df['高幻觉比例(>0.5)']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{ratio:.1%}', ha='center', va='bottom')

# 8.6 性能总结
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
summary_text = f"""
性能总结:

🏆 综合最佳: {max(zip(models, overall_scores), key=lambda x: x[1])[0]}
⚡ 最快响应: {df.groupby('model')['time'].mean().idxmin()}
🎯 最准确: {df.groupby('model')['factual_correctness'].mean().idxmax()}
🔍 最忠实: {df.groupby('model')['faithfulness'].mean().idxmax()}
💭 最少幻觉: {hallucination_df.loc[hallucination_df['平均幻觉分数'].idxmin(), '模型']}

关键发现:
• SelfRAG时间显著较长
• 不同模型在不同指标各有所长
• 需要权衡速度与质量
"""
ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/8_汇总仪表板.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n=== 分析完成 ===")
print(f"所有图表和表格已保存到 '{output_dir}' 文件夹")
print(f"共生成文件:")
print(f"- 6个分析图表")
print(f"- 3个统计表格") 
print(f"- 1份分析报告")
print(f"- 1个汇总仪表板")

# 显示关键统计结果
print(f"\n关键统计结果:")
print(f"平均响应时间: {df.groupby('model')['time'].mean().round(3).to_dict()}")
print(f"平均事实正确性: {df.groupby('model')['factual_correctness'].mean().round(3).to_dict()}")
print(f"平均幻觉分数: {hallucination_df.set_index('模型')['平均幻觉分数'].round(3).to_dict()}")