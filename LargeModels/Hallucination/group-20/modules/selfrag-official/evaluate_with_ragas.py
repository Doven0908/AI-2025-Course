#!/usr/bin/env python3
"""
Evaluate Self-RAG experiment results using RAGAS with DeepSeek-R1 API

This script reads the output from Self-RAG experiments (exp1, exp2, exp3_debug) and evaluates them
using RAGAS metrics with DeepSeek-R1 as the LLM evaluator.

NOTE: This version evaluates only the first 20 samples from each experiment for quick testing.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional

# Remove local ragas from path to avoid conflicts - use installed version
# sys.path.insert(0, '/data/self-rag/ragas/src')

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found")
    print("Please install: pip install openai")
    sys.exit(1)

try:
    from ragas import evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
        ContextPrecision,
        ContextRecall,
    )
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"Error importing ragas: {e}")
    print("\nTrying to fix ragas import...")
    # Create missing _version module if it doesn't exist
    try:
        import ragas
        ragas_path = os.path.dirname(ragas.__file__)
        version_file = os.path.join(ragas_path, '_version.py')
        if not os.path.exists(version_file):
            print(f"Creating missing {version_file}...")
            with open(version_file, 'w') as f:
                f.write('version = "0.3.8.dev"\n')
            print("Retrying import...")
            from ragas import evaluate
            from ragas.llms import llm_factory
            from ragas.metrics import (
                AnswerCorrectness,
                AnswerRelevancy,
                Faithfulness,
                ContextPrecision,
                ContextRecall,
            )
            print("✓ ragas imported successfully")
        else:
            raise
    except Exception as retry_error:
        print(f"Failed to import ragas: {retry_error}")
        print("\nPlease try reinstalling ragas:")
        print("  pip uninstall ragas -y")
        print("  pip install ragas")
        sys.exit(1)


def load_experiment_results(file_path: str) -> Dict[str, Any]:
    """Load experiment results from file"""
    print(f"Loading results from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_self_rag_output(
    data: Dict[str, Any],
    eval_data_path: str,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Parse Self-RAG output and evaluation data to create RAGAS dataset

    Args:
        data: Self-RAG experiment output data
        eval_data_path: Path to evaluation data file
        max_samples: Maximum number of samples to process (None for all)

    Returns a list of samples with:
    - question: The input question
    - answer: The predicted answer
    - ground_truth: The reference answer (if available)
    - contexts: Retrieved context (if available from prompts)
    """
    samples = []

    # Load the original evaluation data to get ground truth
    eval_data = []
    if os.path.exists(eval_data_path):
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                eval_data.append(json.loads(line.strip()))

    preds = data.get('preds', [])
    prompts = data.get('prompts', [])

    # Limit samples if max_samples is specified
    if max_samples is not None:
        preds = preds[:max_samples]
        prompts = prompts[:max_samples]
        eval_data = eval_data[:max_samples]

    for idx, pred in enumerate(preds):
        sample = {
            'question': '',
            'answer': pred,
            'ground_truth': '',
            'contexts': []
        }

        # Extract question from prompt if available
        if idx < len(prompts):
            prompt = prompts[idx]
            # Parse question from prompt (assuming it's after "### Input:")
            if '### Input:' in prompt:
                question_part = prompt.split('### Input:')[1]
                if '### Response:' in question_part:
                    question = question_part.split('### Response:')[0].strip()
                    sample['question'] = question

                    # Extract contexts if available in prompt
                    if 'ctxs' in prompt or 'title' in prompt:
                        # Try to extract context from prompt
                        sample['contexts'] = [prompt]  # For now, use full prompt as context

        # Get ground truth from eval data
        if idx < len(eval_data):
            eval_sample = eval_data[idx]
            if 'answers' in eval_sample:
                sample['ground_truth'] = str(eval_sample['answers'][0]) if eval_sample['answers'] else ''
            elif 'answerKey' in eval_sample:
                sample['ground_truth'] = eval_sample['answerKey']
            elif 'label' in eval_sample:
                sample['ground_truth'] = eval_sample['label']

            # Get question if not already extracted
            if not sample['question'] and 'question' in eval_sample:
                sample['question'] = eval_sample['question']
            elif not sample['question'] and 'claim' in eval_sample:
                sample['question'] = eval_sample['claim']

        samples.append(sample)

    return samples


def create_deepseek_llm(api_key: str, base_url: str = "https://api.deepseek.com"):
    """Create DeepSeek LLM instance for RAGAS

    Returns a langchain ChatOpenAI instance configured for DeepSeek.
    RAGAS will automatically handle the wrapping.
    """

    # Set OPENAI_API_KEY environment variable for ragas internal use
    os.environ['OPENAI_API_KEY'] = api_key

    # Temporarily disable proxy settings for langchain/httpx
    # Save original proxy settings
    original_proxies = {}
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']
    for var in proxy_vars:
        if var in os.environ:
            original_proxies[var] = os.environ[var]
            del os.environ[var]

    try:
        # Create langchain ChatOpenAI with DeepSeek configuration
        # RAGAS natively supports langchain LLMs
        llm = ChatOpenAI(
            model="deepseek-chat",  # Use deepseek-chat for evaluation (more stable than reasoner)
            api_key=api_key,
            base_url=base_url,
            temperature=0,  # Deterministic for evaluation
            max_tokens=512,  # Reasonable limit for evaluation responses
        )
    finally:
        # Restore original proxy settings
        for var, value in original_proxies.items():
            os.environ[var] = value

    return llm


def evaluate_experiment(
    exp_file: str,
    eval_data_file: str,
    exp_name: str,
    deepseek_api_key: str,
    output_dir: str = "ragas_results",
    max_samples: Optional[int] = None
):
    """Evaluate a single experiment using RAGAS

    Args:
        exp_file: Path to experiment output file
        eval_data_file: Path to evaluation data file
        exp_name: Name of the experiment
        deepseek_api_key: DeepSeek API key
        output_dir: Directory to save results
        max_samples: Maximum number of samples to evaluate (None for all)
    """

    print(f"\n{'='*80}")
    print(f"Evaluating {exp_name}")
    if max_samples:
        print(f"(Limited to {max_samples} samples)")
    print(f"{'='*80}\n")

    # Load and parse data
    data = load_experiment_results(exp_file)
    samples = parse_self_rag_output(data, eval_data_file, max_samples=max_samples)

    print(f"Loaded {len(samples)} samples")
    print(f"Sample preview:")
    if samples:
        sample = samples[0]
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Answer: {sample['answer'][:100]}...")
        print(f"  Ground Truth: {sample['ground_truth'][:100] if sample['ground_truth'] else 'N/A'}...")

    # Create DeepSeek LLM
    print("\nInitializing DeepSeek-R1 LLM...")
    llm = create_deepseek_llm(deepseek_api_key)

    # Create RAGAS dataset
    print("\nCreating RAGAS dataset...")
    dataset_dict = {
        'question': [s['question'] for s in samples],
        'answer': [s['answer'] for s in samples],
        'ground_truth': [s['ground_truth'] for s in samples],
        'contexts': [s['contexts'] if s['contexts'] else [''] for s in samples],
    }

    # Select metrics based on available data
    metrics = []

    # Always include these
    metrics.append(AnswerRelevancy())

    # Include if ground truth is available
    if any(dataset_dict['ground_truth']):
        metrics.append(AnswerCorrectness())

    # Include if contexts are available
    if any(any(ctx) for ctx in dataset_dict['contexts']):
        metrics.append(Faithfulness())
        metrics.append(ContextPrecision())
        metrics.append(ContextRecall())

    print(f"\nMetrics to evaluate: {[type(m).__name__ for m in metrics]}")

    # Evaluate
    print(f"\nRunning RAGAS evaluation on {len(samples)} samples (this may take a while)...")
    try:
        from datasets import Dataset as HFDataset
        hf_dataset = HFDataset.from_dict(dataset_dict)

        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=llm,
            raise_exceptions=False
        )

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{exp_name}_ragas_eval.json")

        result_dict = {
            'experiment': exp_name,
            'num_samples': len(samples),
            'metrics': {k: float(v) if v is not None else None for k, v in result.items()},
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Results for {exp_name}:")
        print(f"{'='*80}")
        for metric, score in result.items():
            print(f"  {metric}: {score:.4f}" if score is not None else f"  {metric}: N/A")
        print(f"\nResults saved to: {output_file}")

        return result_dict

    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main evaluation function"""

    # Check for DeepSeek API key
    deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        print("Please set it using: export DEEPSEEK_API_KEY='your-api-key'")
        sys.exit(1)

    # Maximum samples per experiment (set to 20 for quick testing)
    MAX_SAMPLES = 20

    print("="*80)
    print("RAGAS Evaluation - Limited Mode")
    print(f"Evaluating {MAX_SAMPLES} samples per experiment")
    print("="*80)

    # Define experiments
    experiments = [
        {
            'name': 'exp1_popqa',
            'output_file': 'retrieval_lm/exp1',
            'eval_data': 'eval_data/popqa_longtail_w_gs.jsonl',
        },
        {
            'name': 'exp2_arc',
            'output_file': 'retrieval_lm/exp2',
            'eval_data': 'eval_data/arc_challenge_processed.jsonl',
        },
        {
            'name': 'exp3_health',
            'output_file': 'retrieval_lm/exp3_debug',
            'eval_data': 'eval_data/health_claims_processed.jsonl',
        },
    ]

    # Evaluate each experiment
    all_results = []
    for exp in experiments:
        result = evaluate_experiment(
            exp_file=exp['output_file'],
            eval_data_file=exp['eval_data'],
            exp_name=exp['name'],
            deepseek_api_key=deepseek_api_key,
            max_samples=MAX_SAMPLES,
        )
        if result:
            all_results.append(result)

    # Save summary
    if all_results:
        summary_file = 'ragas_results/summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiments': all_results,
                'summary': 'RAGAS evaluation of Self-RAG experiments using DeepSeek-R1'
            }, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("Summary saved to: ragas_results/summary.json")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
