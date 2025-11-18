#!/usr/bin/env python3
"""
简化安全的 RAGAS 评估脚本
只使用最基础的指标，避免需要多次采样的复杂指标
"""

import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"Error: {e}")
    print("Please install: pip install langchain-openai")
    sys.exit(1)


def load_experiment_data(exp_file: str, eval_file: str, start_idx: int = 0, end_idx: int = 20):
    """加载实验数据"""

    print(f"Loading from {exp_file}...")
    with open(exp_file, 'r') as f:
        exp_data = json.load(f)

    preds = exp_data.get('preds', [])[start_idx:end_idx]

    # 加载 ground truth
    eval_data = []
    with open(eval_file, 'r') as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            eval_data.append(json.loads(line.strip()))

    samples = []
    for i, pred in enumerate(preds):
        if i < len(eval_data):
            item = eval_data[i]
            actual_idx = start_idx + i  # Track actual sample index

            # 提取问题
            question = item.get('question') or item.get('claim', '')

            # 提取 ground truth
            if 'answers' in item:
                gt = str(item['answers'][0]) if item['answers'] else ''
            elif 'answerKey' in item:
                gt = item['answerKey']
            elif 'label' in item:
                gt = item['label']
            else:
                gt = ''

            samples.append({
                'sample_id': actual_idx,
                'question': question,
                'answer': pred,
                'ground_truth': gt
            })

    return samples


def evaluate_relevancy(llm, question: str, answer: str) -> float:
    """评估答案相关性 - 简化版本"""

    if not question or not answer:
        return 0.0

    prompt = f"""On a scale of 0 to 1, rate how relevant and appropriate the following answer is to the given question.

Question: {question}

Answer: {answer}

Provide only a single number between 0 and 1, where:
- 1.0 = Perfectly relevant and directly answers the question
- 0.5 = Partially relevant but incomplete or off-topic
- 0.0 = Completely irrelevant or nonsensical

Score (0-1):"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score_text = response.content.strip()

        # 提取数字
        import re
        match = re.search(r'0?\.\d+|[01]\.?\d*', score_text)
        if match:
            score = float(match.group())
            return min(max(score, 0.0), 1.0)  # 限制在 0-1 之间
        else:
            return 0.5  # 默认值

    except Exception as e:
        print(f"  Warning: Error in relevancy evaluation: {e}")
        return 0.5


def evaluate_correctness(llm, question: str, answer: str, ground_truth: str) -> float:
    """评估答案正确性 - 简化版本"""

    if not ground_truth or not answer:
        return 0.0

    prompt = f"""Compare the predicted answer with the ground truth answer for the given question.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {answer}

On a scale of 0 to 1, rate how correct the predicted answer is:
- 1.0 = Completely correct, matches ground truth
- 0.5 = Partially correct
- 0.0 = Completely incorrect

Provide only a single number between 0 and 1.

Score (0-1):"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score_text = response.content.strip()

        # 提取数字
        import re
        match = re.search(r'0?\.\d+|[01]\.?\d*', score_text)
        if match:
            score = float(match.group())
            return min(max(score, 0.0), 1.0)
        else:
            return 0.5

    except Exception as e:
        print(f"  Warning: Error in correctness evaluation: {e}")
        return 0.5


def create_llm(api_key: str):
    """创建 LLM 客户端"""

    os.environ['OPENAI_API_KEY'] = api_key

    # 临时禁用代理
    original_proxies = {}
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                  'http_proxy', 'https_proxy', 'all_proxy']
    for var in proxy_vars:
        if var in os.environ:
            original_proxies[var] = os.environ[var]
            del os.environ[var]

    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0,
            max_tokens=100,  # 减少 tokens 加快响应
            timeout=30,  # 30秒超时
        )
    finally:
        # 恢复代理
        for var, value in original_proxies.items():
            os.environ[var] = value

    return llm


def evaluate_experiment(exp_name: str, exp_file: str, eval_file: str, api_key: str, start_idx: int = 0, end_idx: int = 20):
    """评估单个实验"""

    num_samples = end_idx - start_idx
    print(f"\n{'='*80}")
    print(f"Evaluating: {exp_name}")
    print(f"Samples: {start_idx} to {end_idx} ({num_samples} samples)")
    print(f"{'='*80}\n")

    # 加载数据
    samples = load_experiment_data(exp_file, eval_file, start_idx, end_idx)
    print(f"Loaded {len(samples)} samples\n")

    # 创建 LLM
    print("Creating LLM client...")
    llm = create_llm(api_key)
    print("✓ LLM ready\n")

    # 评估
    print(f"Evaluating {len(samples)} samples...")
    print("(This may take a while - each sample takes ~10-15 seconds)\n")

    relevancy_scores = []
    correctness_scores = []

    for i, sample in enumerate(tqdm(samples, desc="Progress")):
        try:
            # 评估相关性
            rel_score = evaluate_relevancy(llm, sample['question'], sample['answer'])
            relevancy_scores.append(rel_score)

            # 评估正确性（如果有 ground truth）
            if sample['ground_truth']:
                cor_score = evaluate_correctness(
                    llm,
                    sample['question'],
                    sample['answer'],
                    sample['ground_truth']
                )
                correctness_scores.append(cor_score)

            # 显示进度
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{len(samples)} samples")

        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")
            relevancy_scores.append(0.5)
            if sample['ground_truth']:
                correctness_scores.append(0.5)

    # 计算平均分
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
    avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0

    results = {
        'experiment': exp_name,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'num_samples': len(samples),
        'metrics': {
            'relevancy': avg_relevancy,
            'correctness': avg_correctness if correctness_scores else None,
        },
        'individual_scores': {
            'relevancy': relevancy_scores,
            'correctness': correctness_scores if correctness_scores else None,
        },
        'sample_ids': [s['sample_id'] for s in samples]
    }

    # 保存结果
    os.makedirs('ragas_results', exist_ok=True)
    output_file = f'ragas_results/{exp_name}_simple_eval.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 显示结果
    print(f"\n{'='*80}")
    print(f"Results for {exp_name}:")
    print(f"{'='*80}")
    print(f"  Relevancy Score:   {avg_relevancy:.4f}")
    if avg_correctness > 0:
        print(f"  Correctness Score: {avg_correctness:.4f}")
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    """主函数"""

    # 检查 API key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        print("Run: export DEEPSEEK_API_KEY='your-key'")
        sys.exit(1)

    print("="*80)
    print("Simple Safe RAGAS Evaluation")
    print("="*80)
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print()

    # 定义实验
    experiments = [
        {
            'name': 'exp1_popqa',
            'exp_file': 'retrieval_lm/exp1',
            'eval_file': 'eval_data/popqa_longtail_w_gs.jsonl',
        },
        {
            'name': 'exp2_arc',
            'exp_file': 'retrieval_lm/exp2',
            'eval_file': 'eval_data/arc_challenge_processed.jsonl',
        },
        {
            'name': 'exp3_health',
            'exp_file': 'retrieval_lm/exp3_debug',
            'eval_file': 'eval_data/health_claims_processed.jsonl',
        },
    ]

    # 评估每个实验
    all_results = []
    for exp in experiments:
        try:
            # Check command line args for start/end indices
            import sys
            start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
            end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 20

            result = evaluate_experiment(
                exp_name=exp['name'],
                exp_file=exp['exp_file'],
                eval_file=exp['eval_file'],
                api_key=api_key,
                start_idx=start_idx,
                end_idx=end_idx
            )
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user")
            break
        except Exception as e:
            print(f"\n\nError evaluating {exp['name']}: {e}")
            import traceback
            traceback.print_exc()

    # 保存总结
    if all_results:
        summary_file = 'ragas_results/summary_simple.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'experiments': all_results,
                'summary': 'Simple safe RAGAS evaluation'
            }, f, indent=2)

        print(f"\n{'='*80}")
        print(f"All Results Summary:")
        print(f"{'='*80}")
        for result in all_results:
            print(f"\n{result['experiment']}:")
            print(f"  Relevancy:   {result['metrics']['relevancy']:.4f}")
            if result['metrics']['correctness']:
                print(f"  Correctness: {result['metrics']['correctness']:.4f}")

        print(f"\n\nSummary saved to: {summary_file}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
