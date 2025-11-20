from openai import OpenAI
from typing import List, Dict, Any, Union,Optional
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from itertools import repeat
from sympy import isprime
import math
import datasets
from utils.serve_vllm import completion_request,chat_completion_request
from utils.parse import parse_ground_truth,parse_question
from utils.strip_string import extract_answer
from utils.grader import math_equal
from utils.data_loader import load_data
from test_8_accuracy import llm_judge_once
import argparse
import multiprocessing
from pathlib import Path
import time
import json
from test_tool.test_tool import SPLIT_PROMPT,SPLIT_INPUT,SPLIT_PROMPT_SPLITED
from test_tool.test_extraction import data_processing
import random
import Levenshtein
import re
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import numpy as np
from vllm import LLM, SamplingParams
import os


file_lock = Lock()

def calculate_confidence_metrics(prob_list, text: Optional[str] = None) -> Dict[str, float]:
    num_tokens = len(prob_list)
    start_index = 1 if (text and len(text) > 0 and text[0] == "{") else 0
    end_index = num_tokens

    # Initialize accumulators for different metrics
    sampled_probs = []
    sampled_logprobs = []
    entropies = []
    top_k_prob_sums = [] # For pred_conf_5

    for i in range(start_index, end_index):
        # Ensure index is valid and corresponding logprobs entry is not empty
        if i < len(prob_list) and prob_list[i]:
            probs = list(prob_list[i].values())
            logprobs = [math.log(p) for p in probs if p > 0]
            # Extract probabilities from logprobs

            sampled_probs.append(probs)
            sampled_logprobs.append(logprobs)
            if probs:
                # Calculate entropy for the current token
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                entropies.append(entropy)

                # Calculate top-k probability sum (for pred_conf_5)
                k = min(5, len(probs))
                top_k_probs = sorted(probs, reverse=True)[:k]
                top_k_prob_sum = sum(top_k_probs)
                top_k_prob_sums.append(top_k_prob_sum)
            else:
                entropies.append(0.0)  # If no probabilities, entropy is 0
                top_k_prob_sums.append(0.0)  # If no probabilities, top-k sum is 0
                     
    results = {}
    # pred_conf_0: Arithmetic Mean of Sampled Token Probabilities
    # Represents the average "certainty" of the model at each step for the token it chose.
    pred_conf_0 = np.mean([probs[0] for probs in sampled_probs]) if sampled_probs else 0.0
    # pred_conf_1: Geometric Mean of Sampled Token Probabilities
    # More sensitive to low values; if any single token has a very low probability,
    # the geometric mean will be significantly pulled down.
    # Calculated as exp(average of logprobs).
    pred_conf_1 = np.exp(np.mean([logprobs[0] for logprobs in sampled_logprobs])) if sampled_logprobs else 0.0
    # pred_conf_2: Minimum Sampled Token Probability
    # Directly addresses the "weakest link" problem. A single very uncertain step
    # will result in a low confidence score.
    pred_conf_2 = np.min([probs[0] for probs in sampled_probs]) if sampled_probs else 0.0
    # pred_conf_3: Average Token Entropy (normalized to be a confidence score)
    # Entropy quantifies uncertainty. Lower entropy means higher certainty.
    # We normalize it to be between 0 and 1, where 1 is highest confidence.
    pred_conf_3 = 1 - (np.mean(entropies) / np.log(len(prob_list[0]))) if entropies else 0.0

    # pred_conf_4: Normalized Sequence Log-Likelihood (Raw Average Logprob)
    # This is the average log-probability of the tokens generated.
    # Higher (less negative) values indicate a more probable sequence.
    # This is a fundamental metric in language modeling.
    pred_conf_4 = np.mean([logprobs[0] for logprobs in sampled_logprobs]) if sampled_logprobs else 0.0
    # Note: This value will be negative. Higher (closer to 0) indicates higher confidence.

    # pred_conf_5: Average Top-K Probability Sum
    # Measures how much probability mass is concentrated within the top-K predicted tokens at each step.
    # If the sum of probabilities for the top-K tokens is high, it means the model is confident
    # about its top few choices.
    pred_conf_5 = np.mean(top_k_prob_sums) if top_k_prob_sums else 0.0

    # pred_conf_6: Average Max Probability
    # This is the average of the maximum probabilities assigned to the top token at each step.
    pred_conf_6 = np.mean([max(probs) for probs in sampled_probs])

    # pred_conf_7: Average Max Logprob
    # This is the average of the maximum log probabilities assigned to the top token at each step
    pred_conf_7 = np.mean([max(logprobs) for logprobs in sampled_logprobs])

    # pred_conf_8: perplexity
    # Perplexity is a measure of how well a probability distribution predicts a sample.
    # Lower perplexity indicates better predictive performance.
    if sampled_logprobs:
        perplexity = np.exp(-np.mean([logprobs[0] for logprobs in sampled_logprobs]))
    else:
        perplexity = float('inf')
    pred_conf_8 = perplexity

    # pred_conf_9 : UQ
    # Uncertainty quantification (UQ) can be calculated as the standard deviation of the sampled probabilities.
    pred_conf_9 = np.std([probs[0] for probs in sampled_probs]) if sampled_probs else 0.0

    results = {
        "pred_conf_0": pred_conf_0,
        "pred_conf_1": pred_conf_1,
        "pred_conf_2": pred_conf_2,
        "pred_conf_3": pred_conf_3,
        "pred_conf_4": pred_conf_4,
        "pred_conf_5": pred_conf_5,
        "pred_conf_6": pred_conf_6,
        "pred_conf_7": pred_conf_7,
        "pred_conf_8": pred_conf_8,
        "pred_conf_9": pred_conf_9,
    }

    return results


def process_one_sample_and_write(i, n_of_sampling, llm, sample, args, stop_seqs, match_limit, data_name, output_file, error_file):
    """
    每个样本在自己的线程里处理，并在产生 row 时立即写入文件。
    这样同一题目的 step 写入顺序与生成顺序一致。
    """
    generate_list = []
    cnt = 0
    answer_list = []
    confidence_list = []
    gen_prompt = sample[1]

    written_rows = []

    while True:
        cut_resp, stop_matches = batched_stream_completion_cut_before_match(
            seqs=stop_seqs,
            llm=llm,
            args=args,
            match_limit=match_limit,
            prompt=gen_prompt,
            n=1,
            temperature=0.6,
            max_tokens=8192
        )
        if cut_resp == "token_limit":
            with file_lock:
                with open(error_file, "a", encoding="utf-8") as f:
                    f.write(f"question: {i}, error type: {stop_matches}\n")
            break

        if stop_matches and stop_matches[-1] == "</think>":
            cut_resp += stop_matches[-1]
        generate_list.append(cut_resp)
        generated_output = "".join(item for item in generate_list)
        gen_prompt = sample[1] + generated_output
        
        
        ask_get_ans_prompt = gen_prompt + "Can we now get the final answer? Answer with exactly one token: \"Yes\" or \"No\". Do not add any other text.\n/no_think"
        ask_get_ans_equest = [[{"role": "user", "content": ask_get_ans_prompt}]]
        ask_get_ans_resp = chat_completion_request(
            question=ask_get_ans_equest,
            args=args,
            tree_id=i,
            need_get_next_token_prob=True,
            next_token=None,
            max_tokens=5,
            n_sampling=1,
            stop_tokens=["?"]
        )

        yes_prob = 0
        no_prob = 0
        max_n = 0
        while (max_n < min(5, len(ask_get_ans_resp["logits"][0])) and yes_prob + no_prob < 0.1):
            yn_output = ask_get_ans_resp["logits"][0][max_n]
            prob_map = {tok: p for tok, p in yn_output.items()}
            yes_prob = (
                prob_map.get("yes", 0.0)
                + prob_map.get("Yes", 0.0)
                + prob_map.get(" yes", 0.0)
                + prob_map.get(" Yes", 0.0)
            )
            no_prob = (
                prob_map.get("no", 0.0)
                + prob_map.get("No", 0.0)
                + prob_map.get(" no", 0.0)
                + prob_map.get(" No", 0.0)
            )
            max_n += 1

        print(f"[Q{i} Solution{n_of_sampling} step{cnt}] yes prob = {yes_prob:.4f}, no prob = {no_prob:.4f}")

        get_answer_prompt = gen_prompt + "*** ... Oh, I suddenly got the question's Final Answer: \\boxed"
        get_ans_resp = completion_request(
            question=get_answer_prompt,
            args=args,
            tree_id=i,
            model_id=args.model_name_or_path,
            need_get_next_token_prob=True,
            stop_tokens=["\n", "\n\n",".","***"],
            max_tokens=29,
            n_sampling=1
        )

        top_probs = get_ans_resp["logits"][0]
        cut_idx = None
        for prob_idx,top in enumerate(top_probs):
            sorted_top = sorted(top.items(), key=lambda x: x[1], reverse=True)[:5]
            for token_idx, (token, _) in enumerate(sorted_top):
                if '}' in token:
                    cut_idx = prob_idx
            top_probs[prob_idx] = dict(sorted_top)

        if cut_idx is not None:
            top_probs = top_probs[:cut_idx + 1]

        confidence = certainty_from_choice(top_probs)
        baseline_confidence = calculate_confidence_metrics(top_probs, get_ans_resp["texts"][0][0])
        baseline_confidence["self_design_conf"] = confidence

        v = baseline_confidence.get("pred_conf_6", float("nan"))
        if not (isinstance(v, (int, float)) and np.isfinite(v)):
            baseline_confidence = "NAN ERROR"
        
        temp_action = "*** We can get the question's Final Answer: \\boxed" + get_ans_resp["texts"][0][0]
        new_ans = extract_answer(temp_action, data_name=data_name)
        is_equal = math_equal(new_ans, sample[2])

        llm_judge_equal = llm_judge_once(
            question=sample[0],
            ground_truth=sample[2],
            partial_response=temp_action,
            args=args,
            tree_id=i
        )
        llm_judge_equal_print = llm_judge_equal[:3]
        print(
            f"[Q{i} Solution{n_of_sampling} step{cnt}] confidence={confidence:.4f}, is_equal={is_equal}, llm_judge_equal={llm_judge_equal_print}, "
            f"new_ans={new_ans}, gt_ans={sample[2]}"
        )
        # print(top_probs)
        answer_list.append(new_ans)
        confidence_list.append(confidence)

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "tree_id": i,
            "n_of_samplings":n_of_sampling,
            "question": sample[0],
            "gt_ans": sample[2],
            "stop_matches": stop_matches,
            "actions_from_root": generated_output,
            "temp_action": temp_action,
            "answer": new_ans,
            "confidence": baseline_confidence,
            "ans_prob": top_probs,
            "is_equal": is_equal,
            "llm_judge_equal": llm_judge_equal,
            "cut_idx": cut_idx,
            "model_id": args.model_name_or_path,
            "pre_step_answer": copy.deepcopy(answer_list), 
            "pre_step_confidence": copy.deepcopy(confidence_list),
            "can_final_ans_yes_prob": yes_prob,
            "can_final_ans_no_prob": no_prob,
        }
        written_rows.append(row)

        cnt += 1
        print(f"[Q{i} Solution{n_of_sampling}] finished step {cnt}.")
        if len(stop_matches) < match_limit:
            break
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            for row in written_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Q{i} Solution{n_of_sampling} step{cnt}] finished.")

def batched_stream_completion_cut_before_match(
    seqs: list[str],
    llm,
    args,
    match_limit: int,
    prompt: str,
    check_every_n_tokens: int = 10,
    **kwargs
):
    model = args.model_name_or_path
    num_of_models = args.num_of_models
    ports = [8000 + i for i in range(num_of_models)]
    urls = [f"http://127.0.0.1:{p}/v1" for p in ports]
    base_url = random.choice(urls) 
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    full_output = ""
    token_buffer = []
    match_positions = []

    max_seq_len = max((len(s) for s in seqs if s), default=1)
    last_scan_from = 0
    seen_matches = set()  # {(seq, idx)}

    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            stream=True,
            timeout=3600.0,
            **kwargs
        )
        for chunk in response:
            token = chunk.choices[0].text
            token_buffer.append(token)
            full_output += token
            # print(f"[TOKEN]: {token}", end="", flush=True)

            if len(token_buffer) >= check_every_n_tokens:
                # print("\n[DEBUG] Checking for matches...")
                # 检查 seqs 是否在 full_output 中
                start = max(last_scan_from - (max_seq_len - 1), 0)
                slice_text = full_output[start:]
                for seq in seqs:
                    search_pos = 0
                    while True:
                        idx_rel = slice_text.find(seq, search_pos)
                        if idx_rel == -1:
                            break

                        idx_abs = start + idx_rel
                        if (seq, idx_abs) not in seen_matches:
                            match_positions.append((seq, idx_abs))
                            seen_matches.add((seq, idx_abs))
                        search_pos = idx_rel + len(seq)
                
                last_scan_from = len(full_output)
                
                if len(match_positions) >= match_limit:
                    # 保留内容到第 n-1 次匹配词的开始位置
                    seq_to_cut, match_index = match_positions[match_limit - 1]
                    final_output = full_output[:match_index]  # 截断在最后一个匹配词前
                    return final_output, [s for s, _ in match_positions[:match_limit]]

                token_buffer = []  # 清空 buffer，继续流式收集
    except Exception as e:
        print("Error during stream:", e)
        return "token_limit", str(e)

    return full_output, [s for s, _ in match_positions]



REQUEST = (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )

# if __name__ == "__main__":
#     seqs = ["AI", "\n\n"]
#     output, matched = batched_stream_completion_cut_before_match(
#         seqs=seqs,
#         match_limit=2,
#         check_every_n_tokens=8
#     )

#     print("\n--- Final Output ---\n", output)
#     print("--- Matched ---\n", matched)



def construct_prompt(question):
    prompt_temp = REQUEST
    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    context = input_template.format(input=question)
    full_prompt = context
    full_prompt = full_prompt + "<think>\n"
    return full_prompt.strip(" ")



def certainty_from_choice(top_probs):
    H_norm = []
    for top in top_probs: # top 是 {token: p, ....}
        token_probs = list(top.values())
        kept = [p for p in token_probs if p >= 1e-5]
        s = sum(kept)
        r = max(0.0, 1.0 - s)
        k = len(kept)
        H = -sum(p * math.log(p) for p in kept)
        # if vocab_size and vocab_size > k and r > 0:
        #     p_tail = r / (vocab_size - k)
        #     H += (vocab_size - k) * (-p_tail * math.log(p_tail))
        H_norm.append(H / math.log(max(2,k)))
    C = 1.0 - (sum(H_norm) / len(H_norm))
    return C


def calculate_certrainty_score(top_probs) -> float:
    for i,top in enumerate(top_probs):
        sorted_top = dict(sorted(top.items(), key=lambda x: x[1], reverse=True)[:5])
        top_probs[i] = sorted_top

    num_tokens = len(top_probs)
    start_index = 1 # pass {
    end_index = num_tokens
    if num_tokens < 1:
        return 0.0
    entropies = []
    for i in range(start_index, end_index):
        if top_probs[i]:
            probs = list(top_probs[i].values())
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            entropies.append(entropy)

    certainty_score = 1 - (np.mean(entropies) / np.log(len(top_probs[0]))) if entropies else 0.0
    return certainty_score

#TODO:llm_judge要改一下
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=2)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_generate_workers", type=int, default=16)
    parser.add_argument("--model_name_or_path", type=str, default="/home/lijiakun25/models/Qwen3-8b")
    parser.add_argument("--data_name",default="olympiadbench_maths_en",type=str)
    parser.add_argument("--n_samplings",default=1,type=int)
    parser.add_argument("--dtype", default="float16", type=str)
    args = parser.parse_args()

    # available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    # llm = LLM(
    #     model=args.model_name_or_path,
    #     tensor_parallel_size=len(available_gpus),
    #     trust_remote_code=True,
    #     dtype=args.dtype
    # )
    llm=0

    examples = load_data(data_name=args.data_name,data_path="")
    # examples = examples.select(range(min(len(examples),30)))
    api_model_name = "qwen-plus"
    data_name = args.data_name
    match_limit = 5
    model_name = args.model_name_or_path.strip("/").split("/")[-1]

    samples = []
    for i,example in enumerate(tqdm(examples,total=len(examples),disable=False)):
        question = parse_question(example,data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        prompt = construct_prompt(question=question)
        samples.append((question,prompt,gt_ans,gt_cot))
    answers = []
    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    stop_seqs = ["Wait","But","Let me think","</think>"]

    num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
    
    outputs = []
    output_path = Path("llm_output") / f"{data_name}/{model_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "split_steps.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    error_file = output_path / "error.txt"
    with open(error_file, "w", encoding="utf-8") as f:
        pass
    

    max_workers = max(1, int(args.max_generate_workers) * int(args.num_of_models))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, sample in enumerate(samples):
            for j in range(args.n_samplings):
                fut = executor.submit(
                    process_one_sample_and_write,
                    i, j,llm, sample, args, stop_seqs, match_limit, data_name, output_file, error_file
                )
                futures.append(fut)

        # 如果你想显示总体进度
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing", disable=False):
            fut.result()




