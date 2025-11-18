#!/usr/bin/env python3
"""
合并评估结果
将 0-20 和 20-50 的结果合并为 0-50
"""

import json
import os

def merge_experiment_results(exp_name):
    """合并单个实验的结果"""

    # 加载两个结果文件
    file1 = f'ragas_results/{exp_name}_simple_eval_0_20.json'
    file2 = f'ragas_results/{exp_name}_simple_eval_20_50.json'

    # Check if we have the old format (no range suffix)
    if not os.path.exists(file1):
        file1 = f'ragas_results/{exp_name}_simple_eval.json'

    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Warning: Missing files for {exp_name}")
        print(f"  File 1: {file1} - {'exists' if os.path.exists(file1) else 'missing'}")
        print(f"  File 2: {file2} - {'exists' if os.path.exists(file2) else 'missing'}")
        return None

    with open(file1, 'r') as f:
        result1 = json.load(f)

    with open(file2, 'r') as f:
        result2 = json.load(f)

    # 合并分数
    relevancy_scores = (result1['individual_scores']['relevancy'] +
                       result2['individual_scores']['relevancy'])

    correctness_scores = []
    if result1['individual_scores']['correctness'] and result2['individual_scores']['correctness']:
        correctness_scores = (result1['individual_scores']['correctness'] +
                             result2['individual_scores']['correctness'])

    # 计算新的平均分
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
    avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0

    # 合并结果
    merged_result = {
        'experiment': exp_name,
        'start_idx': 0,
        'end_idx': 50,
        'num_samples': len(relevancy_scores),
        'metrics': {
            'relevancy': avg_relevancy,
            'correctness': avg_correctness if correctness_scores else None,
        },
        'individual_scores': {
            'relevancy': relevancy_scores,
            'correctness': correctness_scores if correctness_scores else None,
        },
        'sample_ids': list(range(50))
    }

    # 保存合并结果
    output_file = f'ragas_results/{exp_name}_simple_eval.json'
    with open(output_file, 'w') as f:
        json.dump(merged_result, f, indent=2)

    print(f"✓ Merged {exp_name}: {len(relevancy_scores)} samples")
    print(f"  Relevancy:   {avg_relevancy:.4f}")
    print(f"  Correctness: {avg_correctness:.4f}")

    return merged_result


def main():
    """主函数"""

    experiments = ['exp1_popqa', 'exp2_arc', 'exp3_health']

    print("="*80)
    print("Merging Evaluation Results")
    print("="*80)
    print()

    all_results = []
    for exp_name in experiments:
        print(f"\nMerging {exp_name}...")
        result = merge_experiment_results(exp_name)
        if result:
            all_results.append(result)

    # 保存总结
    if all_results:
        summary_file = 'ragas_results/summary_simple.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'experiments': all_results,
                'summary': 'Merged RAGAS evaluation (50 samples per experiment)'
            }, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Summary:")
        print(f"{'='*80}")
        for result in all_results:
            print(f"\n{result['experiment']}:")
            print(f"  Samples:     {result['num_samples']}")
            print(f"  Relevancy:   {result['metrics']['relevancy']:.4f}")
            if result['metrics']['correctness']:
                print(f"  Correctness: {result['metrics']['correctness']:.4f}")

        print(f"\n\nSummary saved to: {summary_file}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
