#!/usr/bin/env python3
"""
Simple RAGAS evaluation script for testing with a small sample

This is a simplified version that evaluates only the first few samples from each experiment
for quick testing and debugging.
"""

import json
import os
import sys

# Remove local ragas from path to avoid conflicts - use installed version
# sys.path.insert(0, '/data/self-rag/ragas/src')

try:
    from ragas.metrics import AnswerRelevancy, AnswerCorrectness
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"Error importing packages: {e}")
    print("Please install: pip install ragas langchain-openai datasets")
    sys.exit(1)


def load_sample_data(exp_file: str, eval_file: str, num_samples: int = 10):
    """Load a small sample of data for testing"""

    print(f"Loading {num_samples} samples from {exp_file}...")

    # Load predictions
    with open(exp_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preds = data.get('preds', [])[:num_samples]
    prompts = data.get('prompts', [])[:num_samples]

    # Load ground truth
    ground_truths = []
    questions = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line.strip())

            # Extract question
            question = item.get('question') or item.get('claim', '')
            questions.append(question)

            # Extract ground truth
            if 'answers' in item:
                gt = str(item['answers'][0]) if item['answers'] else ''
            elif 'answerKey' in item:
                gt = item['answerKey']
            elif 'label' in item:
                gt = item['label']
            else:
                gt = ''
            ground_truths.append(gt)

    return {
        'questions': questions,
        'answers': preds,
        'ground_truths': ground_truths,
    }


def evaluate_simple(exp_name: str, exp_file: str, eval_file: str, api_key: str):
    """Simple evaluation with RAGAS"""

    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_name}")
    print(f"{'='*60}\n")

    # Load data
    data = load_sample_data(exp_file, eval_file, num_samples=10)

    print(f"Loaded {len(data['questions'])} samples")
    print(f"\nSample:")
    print(f"  Q: {data['questions'][0][:80]}...")
    print(f"  A: {data['answers'][0][:80]}...")
    print(f"  GT: {data['ground_truths'][0][:80] if data['ground_truths'][0] else 'N/A'}...")

    # Create LLM
    print("\nInitializing DeepSeek-R1...")

    # Set OPENAI_API_KEY environment variable for ragas internal use
    os.environ['OPENAI_API_KEY'] = api_key

    # Temporarily disable proxy settings
    original_proxies = {}
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']
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
            max_tokens=512,
        )
    finally:
        # Restore proxy settings
        for var, value in original_proxies.items():
            os.environ[var] = value

    # Evaluate with simple metrics
    print("\nEvaluating Answer Relevancy...")

    try:
        relevancy_metric = AnswerRelevancy()
        relevancy_scores = []

        for question, answer in zip(data['questions'], data['answers']):
            if question and answer:
                try:
                    result = relevancy_metric.single_turn_score(
                        question=question,
                        response=answer,
                        llm=llm
                    )
                    score = result if isinstance(result, (int, float)) else result.value if hasattr(result, 'value') else 0.0
                    relevancy_scores.append(score)
                    print(f"  Sample {len(relevancy_scores)}: {score:.3f}")
                except Exception as e:
                    print(f"  Error on sample {len(relevancy_scores)+1}: {str(e)}")
                    relevancy_scores.append(0.0)

        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0

        print(f"\n{'='*60}")
        print(f"Results for {exp_name}:")
        print(f"{'='*60}")
        print(f"  Average Answer Relevancy: {avg_relevancy:.4f}")
        print(f"  Samples evaluated: {len(relevancy_scores)}")
        print(f"{'='*60}\n")

        # Save results
        os.makedirs('ragas_results', exist_ok=True)
        result_file = f'ragas_results/{exp_name}_simple_eval.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': exp_name,
                'num_samples': len(relevancy_scores),
                'metrics': {
                    'answer_relevancy': avg_relevancy,
                },
                'individual_scores': relevancy_scores,
            }, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {result_file}\n")

    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""

    # Check API key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        print("Set it with: export DEEPSEEK_API_KEY='your-key'")
        sys.exit(1)

    # Define experiments
    experiments = [
        ('exp1_popqa', 'retrieval_lm/exp1', 'eval_data/popqa_longtail_w_gs.jsonl'),
        ('exp2_arc', 'retrieval_lm/exp2', 'eval_data/arc_challenge_processed.jsonl'),
        ('exp3_health', 'retrieval_lm/exp3_debug', 'eval_data/health_claims_processed.jsonl'),
    ]

    # Evaluate each
    for name, exp_file, eval_file in experiments:
        evaluate_simple(name, exp_file, eval_file, api_key)

    print("="*60)
    print("Simple evaluation complete!")
    print("Check ragas_results/ directory for detailed results")
    print("="*60)


if __name__ == '__main__':
    main()
