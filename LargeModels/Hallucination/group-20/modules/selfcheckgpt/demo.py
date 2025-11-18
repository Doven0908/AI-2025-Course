import os
import json
import torch
import spacy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from openai import OpenAI
import pandas as pd

# 初始化OpenAI客户端，配置千问API
client = OpenAI(
    # 从环境变量获取API密钥，若未配置可直接替换为"sk-xxx"格式的密钥
    api_key="sk-aa3b65d89c824754804f4291b1540e88",
    # 百炼兼容模式API端点
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --------------------------
# 0. 环境初始化与配置
# --------------------------
# 安装依赖命令（首次运行前执行）：
# pip install spacy matplotlib numpy selfcheckgpt transformers>=4.35 torch>=2.0 dashscope tqdm pandas
# python -m spacy download en_core_web_sm

# 全局配置 - 修复字体问题
plt.style.use('default')  # 使用默认样式避免兼容性问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']  # 使用更通用的字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"【环境信息】使用设备：{DEVICE} | PyTorch版本：{torch.__version__}\n")

# --------------------------
# 1. 加载MultiSpanQA数据集
# --------------------------
def load_multispanqa_data(file_path: str) -> List[Dict]:
    """加载MultiSpanQA数据集，解析上下文、问题和真实答案"""
    print("=" * 60)
    print("1. 加载MultiSpanQA数据集")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在：{file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for item in data["data"]:
        # 拼接上下文和问题（原始数据为单词列表）
        context = " ".join(item["context"])
        question = " ".join(item["question"])
        
        # 从B/I/O标签提取真实答案（多跨度）
        true_answers = []
        current_answer = []
        for word, label in zip(item["context"], item["label"]):
            if label == "B":
                if current_answer:
                    true_answers.append(" ".join(current_answer))
                current_answer = [word]
            elif label == "I":
                current_answer.append(word)
            elif label == "O" and current_answer:
                true_answers.append(" ".join(current_answer))
                current_answer = []
        if current_answer:  # 处理末尾答案
            true_answers.append(" ".join(current_answer))
        
        samples.append({
            "id": item["id"],
            "context": context,
            "question": question,
            "true_answers": true_answers
        })
    
    print(f"【数据集信息】共加载{len(samples)}个样本 | 示例ID: {samples[0]['id']}")
    print(f"【示例上下文】{samples[0]['context'][:100]}...")
    print(f"【示例问题】{samples[0]['question']}\n")
    return samples

# --------------------------
# 2. 调用Qwen生成回答（带上下文）- 支持多个模型
# --------------------------
def generate_qa_with_qwen(context: str, question: str, num_samples: int = 3, model: str = "qwen-plus") -> Tuple[str, List[str]]:
    """
    调用Qwen生成目标回答和多个采样回答（基于给定上下文）
    :return: target_answer（目标回答）, sampled_answers（采样回答列表）
    """
    def call_qwen(prompt: str, temperature: float, model: str) -> str:
        """单次调用Qwen API的工具函数"""
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=True  # 流式返回，逐段获取结果
            )
            response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:  # 确保内容不为空
                    response += content
                    print(content, end="", flush=True)  # 实时打印
            print()  # 输出结束后换行

            return response
        except Exception as e:
            print(f"\n❌ API调用失败: {str(e)}")
            return f"Error: {str(e)}"
    
    # 构建带上下文的提示词
    base_prompt = f"上下文：{context}\n问题：{question}\n请根据上下文回答问题，不要编造信息。"
    
    print(f"【使用模型】{model}")
    
    # 生成目标回答（低温度确保稳定性）
    target_answer = call_qwen(base_prompt, temperature=0.3, model=model)
    
    # 生成采样回答（提高温度增加多样性）
    sampled_answers = []
    for i in range(num_samples):
        sample = call_qwen(base_prompt, temperature=0.7 + i * 0.2, model=model)
        sampled_answers.append(sample)
    
    return target_answer, sampled_answers

# --------------------------
# 3. 文本预处理与幻觉检测
# --------------------------
def preprocess_text(text: str) -> List[str]:
    """使用spaCy分割文本为句子"""
    nlp = spacy.load("en_core_web_sm")
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

def detect_hallucination(target: str, samples: List[str], context: str) -> Dict:
    """
    基于NLI的幻觉检测：
    - 仅使用SelfCheckGPT的NLI方法
    """
    target_sents = preprocess_text(target)
    if not target_sents:
        return {"句子": [], "NLI幻觉分数": []}
    
    # 初始化NLI检测器
    nli = SelfCheckNLI(device=DEVICE)
    nli_scores = nli.predict(sentences=target_sents, sampled_passages=samples)
    
    # 转换为列表以便JSON序列化
    nli_scores_list = nli_scores.tolist() if hasattr(nli_scores, 'tolist') else list(nli_scores)
    
    return {
        "句子": target_sents,
        "NLI幻觉分数": nli_scores_list
    }

# --------------------------
# 4. 可视化函数 - 修复键名错误
# --------------------------
def visualize_sample_results_comparison(results_dict: Dict, sample_id: str, save_dir: str):
    """对比两个模型的可视化结果 - 修复键名错误版本"""
    # 检查是否有有效数据
    has_valid_data = False
    for model in results_dict:
        if ("hallucination_analysis" in results_dict[model] and 
            results_dict[model]["hallucination_analysis"] and 
            "句子" in results_dict[model]["hallucination_analysis"] and 
            results_dict[model]["hallucination_analysis"]["句子"]):
            has_valid_data = True
            break
    
    if not has_valid_data:
        print(f"样本 {sample_id} 无有效句子可可视化")
        return
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Comparison - Sample {sample_id}', fontsize=16, fontweight='bold')
    
    models = list(results_dict.keys())
    colors = ['#1f77b4', '#ff7f0e']  # 为两个模型分配不同颜色
    
    for i, model in enumerate(models):
        if (model not in results_dict or 
            "hallucination_analysis" not in results_dict[model] or 
            not results_dict[model]["hallucination_analysis"]):
            continue
            
        result = results_dict[model]["hallucination_analysis"]
        if not result["句子"]:
            continue
            
        sentences = result["句子"]
        nli_scores = result["NLI幻觉分数"]
        
        # 子图1和2：各模型的NLI分数柱状图
        ax1 = axes[0, i]
        x_pos = np.arange(len(sentences))
        bars = ax1.bar(x_pos, nli_scores, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for j, (bar, score) in enumerate(zip(bars, nli_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Risk Threshold (0.5)')
        ax1.set_xlabel('Sentence Index', fontsize=11)
        ax1.set_ylabel('NLI Hallucination Score', fontsize=11)
        ax1.set_title(f'{model} - NLI Hallucination Detection', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'S{j+1}' for j in range(len(sentences))], fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 子图3和4：各模型的风险等级饼图
        ax2 = axes[1, i]
        high_risk = sum(1 for score in nli_scores if score > 0.5)
        low_risk = len(nli_scores) - high_risk
        
        if len(nli_scores) > 0:
            risk_data = [high_risk, low_risk]
            risk_labels = [f'High Risk\n({high_risk} sentences)', f'Low Risk\n({low_risk} sentences)']
            risk_colors_pie = ['#ff6b6b', '#51cf66']
            explode = (0.05, 0) if high_risk > 0 else (0, 0)
            
            wedges, texts, autotexts = ax2.pie(risk_data, labels=risk_labels, colors=risk_colors_pie,
                                              autopct='%1.1f%%', startangle=90, explode=explode,
                                              textprops={'fontsize': 10})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax2.set_title(f'{model} - Risk Distribution', fontsize=13, fontweight='bold')
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 保存图片
    save_path = os.path.join(save_dir, f"sample_{sample_id}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"【Comparison Visualization】Chart saved to: {save_path}")

def create_comparison_summary_visualization(all_results: List[Dict], save_dir: str):
    """创建两个模型的对比汇总可视化 - 修复键名错误版本"""
    if not all_results:
        return
    
    # 收集两个模型的统计数据
    model_stats = {}
    
    for model in ["qwen-max", "qwen-plus"]:
        all_nli_scores = []
        high_risk_counts = []
        
        for result in all_results:
            if (model in result and 
                "hallucination_analysis" in result[model] and 
                result[model]["hallucination_analysis"] and 
                "NLI幻觉分数" in result[model]["hallucination_analysis"] and 
                result[model]["hallucination_analysis"]["NLI幻觉分数"]):
                
                nli_scores = result[model]["hallucination_analysis"]["NLI幻觉分数"]
                all_nli_scores.extend(nli_scores)
                high_risk_count = sum(1 for score in nli_scores if score > 0.5)
                high_risk_counts.append(high_risk_count)
        
        if all_nli_scores:
            model_stats[model] = {
                "all_scores": all_nli_scores,
                "high_risk_counts": high_risk_counts,
                "avg_score": np.mean(all_nli_scores),
                "high_risk_ratio": sum(1 for score in all_nli_scores if score > 0.5) / len(all_nli_scores),
                "max_score": max(all_nli_scores),
                "min_score": min(all_nli_scores)
            }
    
    if len(model_stats) < 2:
        print("Not enough models for comparison")
        return
    
    # 创建对比汇总图表
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Model Comparison - Hallucination Detection Summary', fontsize=18, fontweight='bold')
    
    # 定义子图布局
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # 分数分布对比
    ax2 = fig.add_subplot(gs[0, 1])  # 高风险比例对比
    ax3 = fig.add_subplot(gs[1, 0])  # 各样本高风险句子数量对比
    ax4 = fig.add_subplot(gs[1, 1])  # 统计信息对比
    ax5 = fig.add_subplot(gs[2, :])  # 模型性能雷达图
    
    models = list(model_stats.keys())
    colors = ['#1f77b4', '#ff7f0e']
    
    # 子图1：分数分布对比
    for i, model in enumerate(models):
        scores = model_stats[model]["all_scores"]
        ax1.hist(scores, bins=15, alpha=0.7, color=colors[i], label=model, edgecolor='black')
    
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Risk Threshold')
    ax1.set_xlabel('NLI Hallucination Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Score Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：高风险比例对比
    risk_ratios = [model_stats[model]["high_risk_ratio"] for model in models]
    x_pos = np.arange(len(models))
    bars = ax2.bar(x_pos, risk_ratios, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('High Risk Ratio', fontsize=11)
    ax2.set_title('High Risk Ratio Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, ratio in zip(bars, risk_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 子图3：各样本高风险句子数量对比
    sample_ids = list(range(1, len(all_results) + 1))
    width = 0.35
    x = np.arange(len(sample_ids))
    
    for i, model in enumerate(models):
        counts = model_stats[model]["high_risk_counts"][:len(sample_ids)]  # 确保长度一致
        ax3.bar(x + i*width, counts, width, label=model, color=colors[i], alpha=0.7)
    
    ax3.set_xlabel('Sample Index', fontsize=11)
    ax3.set_ylabel('High Risk Sentences Count', fontsize=11)
    ax3.set_title('High Risk Sentences per Sample', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels([f'S{i+1}' for i in range(len(sample_ids))], fontsize=9, rotation=45)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 子图4：统计信息对比
    ax4.axis('off')
    stats_text = "Statistical Comparison:\n\n"
    for i, model in enumerate(models):
        stats = model_stats[model]
        stats_text += f"{model}:\n"
        stats_text += f"  • Avg Score: {stats['avg_score']:.3f}\n"
        stats_text += f"  • High Risk Ratio: {stats['high_risk_ratio']:.1%}\n"
        stats_text += f"  • Max Score: {stats['max_score']:.3f}\n"
        stats_text += f"  • Min Score: {stats['min_score']:.3f}\n"
        stats_text += f"  • Total Sentences: {len(stats['all_scores'])}\n\n"
    
    ax4.text(0.05, 0.95, stats_text, fontsize=11, fontfamily='monospace',
             verticalalignment='top', linespacing=1.5, transform=ax4.transAxes)
    
    # 子图5：模型性能雷达图
    # 准备雷达图数据
    categories = ['Accuracy\n(Low Score)', 'Consistency\n(Low Risk)', 'Stability\n(Min Score)', 'Reliability\n(Max Score)']
    N = len(categories)
    
    # 为每个模型计算雷达图值（值越大越好）
    values = {}
    for model in models:
        stats = model_stats[model]
        # 转换为性能指标（值越大表示性能越好）
        accuracy = 1 - stats['avg_score']  # 平均分数越低越好
        consistency = 1 - stats['high_risk_ratio']  # 高风险比例越低越好
        stability = 1 - stats['min_score']  # 最小分数越低越好
        reliability = 1 - stats['max_score']  # 最大分数越低越好
        
        values[model] = [accuracy, consistency, stability, reliability]
    
    # 绘制雷达图
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    for i, model in enumerate(models):
        model_values = values[model]
        model_values += model_values[:1]  # 闭合图形
        ax5.plot(angles, model_values, 'o-', linewidth=2, label=model, color=colors[i])
        ax5.fill(angles, model_values, alpha=0.25, color=colors[i])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.set_title('Model Performance Radar Chart\n(Higher values are better)', fontsize=13, fontweight='bold')
    ax5.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    ax5.grid(True)
    
    # 大幅增加布局间距
    plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=3.0)
    
    save_path = os.path.join(save_dir, "model_comparison_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"【Model Comparison Summary】Chart saved to: {save_path}")
    
    # 保存对比数据到CSV
    comparison_data = []
    for model in models:
        stats = model_stats[model]
        comparison_data.append({
            'Model': model,
            'Average Score': f"{stats['avg_score']:.4f}",
            'High Risk Ratio': f"{stats['high_risk_ratio']:.2%}",
            'Max Score': f"{stats['max_score']:.4f}",
            'Min Score': f"{stats['min_score']:.4f}",
            'Total Sentences': len(stats['all_scores'])
        })
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(save_dir, "model_comparison_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"【Comparison Stats】Data saved to: {csv_path}")

# --------------------------
# 5. 改进的主函数 - 增强错误处理
# --------------------------
def main():
    # 配置参数
    DATA_PATH = r"C:\Users\Adin\Desktop\selfcheckgpt\valid.json"  # 数据集路径（根据实际修改）
    SAVE_DIR = r"C:\Users\Adin\Desktop\selfcheckgpt\selfcheckgpt\result"  # 结果保存目录
    NUM_SAMPLES = 3  # 每个问题生成的采样回答数量
    MODELS = ["qwen-max", "qwen-plus"]  # 要对比的模型列表
    MAX_SAMPLES = 20  # 测试样本数量（按需调整，建议开始时用少量样本测试）
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    try:
        # 1. 加载数据集
        samples = load_multispanqa_data(DATA_PATH)
        if MAX_SAMPLES > 0:
            samples = samples[:MAX_SAMPLES]  # 限制测试样本数量
        
        # 2. 批量处理样本（实时保存结果）
        all_results = []
        processed_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
            sample_id = sample["id"]
            context = sample["context"]
            question = sample["question"]
            true_answers = sample["true_answers"]
            
            print(f"\n【Processing Sample {i+1}/{len(samples)}】ID: {sample_id}")
            print(f"Question: {question}")
            
            try:
                # 为当前样本初始化结果字典
                current_result = {
                    "id": sample_id,
                    "context": context,
                    "question": question,
                    "true_answers": true_answers
                }
                
                # 为每个模型生成结果
                model_results = {}
                for model in MODELS:
                    print(f"\n--- Using Model: {model} ---")
                    
                    # 调用Qwen生成回答
                    target_answer, sampled_answers = generate_qa_with_qwen(
                        context=context,
                        question=question,
                        num_samples=NUM_SAMPLES,
                        model=model
                    )
                    
                    # 检查API调用是否成功
                    if target_answer.startswith("Error:"):
                        print(f"❌ Model {model} API call failed: {target_answer}")
                        continue
                    
                    # 幻觉检测（仅使用NLI）
                    hallu_result = detect_hallucination(
                        target=target_answer,
                        samples=sampled_answers,
                        context=context
                    )
                    
                    # 保存该模型的结果
                    model_results[model] = {
                        "target_answer": target_answer,
                        "sampled_answers": sampled_answers,
                        "hallucination_analysis": hallu_result
                    }
                
                # 如果没有任何模型成功，跳过该样本
                if not model_results:
                    print(f"❌ All models failed for sample {sample_id}")
                    continue
                
                # 将各模型结果添加到当前样本
                for model in model_results:
                    current_result[model] = model_results[model]
                
                # 实时保存单个样本结果
                single_save_path = os.path.join(SAVE_DIR, f"sample_{sample_id}_result.json")
                with open(single_save_path, "w", encoding="utf-8") as f:
                    # 确保所有数据都可序列化
                    serializable_result = current_result.copy()
                    json.dump(serializable_result, f, indent=4, ensure_ascii=False)
                
                # 可视化当前样本的模型对比
                visualize_sample_results_comparison(model_results, sample_id, SAVE_DIR)
                
                # 添加到总结果列表
                all_results.append(current_result)
                processed_count += 1
                
                print(f"✅ Sample {sample_id} processed and saved successfully")
                
            except Exception as e:
                print(f"❌ Sample {sample_id} processing failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 3. 保存全部结果
        if all_results:
            # 创建可序列化的副本
            serializable_all_results = []
            for result in all_results:
                serializable_result = result.copy()
                serializable_all_results.append(serializable_result)
            
            with open(os.path.join(SAVE_DIR, "all_results.json"), "w", encoding="utf-8") as f:
                json.dump(serializable_all_results, f, indent=4, ensure_ascii=False)
            
            # 4. 创建模型对比汇总可视化
            create_comparison_summary_visualization(all_results, SAVE_DIR)
            
            print(f"\n🎉 【Task Completed】")
            print(f"✅ Successfully processed: {processed_count}/{len(samples)} samples")
            print(f"📁 Results saved to: {os.path.abspath(SAVE_DIR)}")
            print(f"📊 Generated charts: {processed_count} comparison charts + 1 summary chart")
            print(f"🔍 Models compared: {', '.join(MODELS)}")
        else:
            print("\n⚠️ No samples were successfully processed")
        
    except Exception as e:
        print(f"\n❌ 【Error】Process interrupted: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()