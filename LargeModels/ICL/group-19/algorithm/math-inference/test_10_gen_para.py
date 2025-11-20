from openai import OpenAI
from typing import List, Dict, Any, Union,Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from itertools import repeat
from sympy import isprime
from utils.serve_vllm import completion_request,chat_completion_request
from utils.parse import parse_ground_truth,parse_question
from utils.strip_string import extract_answer
from utils.grader import math_equal
from utils.data_loader import load_data
import argparse
import multiprocessing
from pathlib import Path
import math
import time
import json
from test_tool.test_extraction import data_processing
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import numpy as np

from test_10_tools import *
from utils.confidence_calculate import certainty_from_choice,calculate_confidence_metrics, cgrs_certrainty_score

file_lock = Lock()
def process_one_sample(i, n_of_sampling, sample, args, stop_seqs, data_name, output_file, error_file):
    """
    每个样本在自己的线程里处理，并在产生 row 时立即写入文件。
    这样同一题目的 step 写入顺序与生成顺序一致。
    """
    generate_step_list = []
    cnt = 0
    answer_list = []
    confidence_list = []
    gen_prompt = sample[1]

    step_rows = []
    
    early_stop_reason = ""
    early_stop_dict = {}

    while True:
        cut_resp, stop_matches, batch_finish_reason = batched_stream_completion_cut_before_match(
            seqs=stop_seqs,
            args=args,
            prompt=gen_prompt,
            n=1,
            temperature=0.6,
            max_tokens=8192
        )
        if cut_resp == "context_overflow":
            with file_lock:
                with open(error_file, "a", encoding="utf-8") as f:
                    f.write(f"question: {i}, error type: {stop_matches}\n")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "tree_id": i, "n_of_samplings": n_of_sampling,
                    "question": sample[0], "gt_ans": sample[2],
                    "model_id": args.model_name_or_path,
                    "status": "context_overflow", "detail": stop_matches
                }, ensure_ascii=False) + "\n")
            break
        if stop_matches and stop_matches[-1] == "</think>":
            cut_resp += stop_matches[-1]
        
        generate_step_list.append(cut_resp)
        generated_output = "".join(gen_step for gen_step in generate_step_list)
        gen_prompt = sample[1] + generated_output
        ans_resp_text, ans_resp_logits = get_answer_response(tree_id=i, generate_prompt=gen_prompt, sample=sample, args=args)



        my_confidence = certainty_from_choice(ans_resp_logits)
        cgrs_confidence = cgrs_certrainty_score(ans_resp_logits)
        baseline_confidence = calculate_confidence_metrics(ans_resp_logits, ans_resp_text)
        v = baseline_confidence.get("pred_conf_6", float("nan"))
        if not (isinstance(v, (int, float)) and np.isfinite(v)):
            baseline_confidence = "NAN ERROR"

        
        llm_judge_partial_sol = "*** We can get the question's Final Answer: \\boxed" + ans_resp_text
        new_ans = extract_answer(llm_judge_partial_sol, data_name=data_name)
        is_equal = math_equal(new_ans, sample[2])
        llm_judge_equal = llm_judge_once(
            question=sample[0],
            ground_truth=sample[2],
            partial_response=llm_judge_partial_sol,
            args=args,
            tree_id=i
        )
        llm_judge_equal_print = llm_judge_equal[:3]
        print(
            f"[Q{i} Solution{n_of_sampling} step{cnt}] confidence={my_confidence:.4f}, is_equal={is_equal}, llm_judge_equal={llm_judge_equal_print}, "
            f"new_ans={new_ans}, gt_ans={sample[2]}"
        )
        # print(top_probs)
        answer_list.append(new_ans)
        confidence_list.append(my_confidence)

        # print(f"[Q{i} Solution{n_of_sampling} step{cnt}] early stop {early_stop_reason}")

        step_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "tree_id": i,
            "n_of_samplings":n_of_sampling,
            "question": sample[0],
            "gt_ans": sample[2],
            "stop_matches": stop_matches,
            "actions_from_root": copy.deepcopy(generate_step_list),
            "answer_extraction": llm_judge_partial_sol,
            "answer": new_ans,
            "my_confidence": my_confidence,
            "cgrs_confidence": cgrs_confidence,
            "baseline_confidence": baseline_confidence,
            "ans_probs": ans_resp_logits,
            "is_equal": is_equal,
            "llm_judge_equal": llm_judge_equal,
            "model_id": args.model_name_or_path,
            "pre_step_answer": copy.deepcopy(answer_list), 
            "pre_step_confidence": copy.deepcopy(confidence_list),
            "batch_finish_reason": batch_finish_reason,
        }
        step_rows.append(step_row)

        # 机制1：当第一步的confidence大于threshold1的时候早停
        # 停止条件：1. 第一步confidence大于threshold1。 2. 第二步的答案和第一步的答案相同。

        if len(step_rows) >= 2: 
            threshold1_list = [0.8, 0.85, 0.9]
            first_step = step_rows[-2]
            second_step = step_rows[-1]
            strict_judge = math_equal(first_step.get("answer"),second_step.get("answer"))
            for threshold in threshold1_list:
                if first_step.get("my_confidence") >= threshold and strict_judge:
                    early_stop_step = len(step_rows) - 1
                    early_stop_reason = f"step_0_high_conf_over_{threshold}"
                    if early_stop_reason not in early_stop_dict:
                        early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制2: 基于连续稳定高confidence进行早停
        # 停止条件：只有当连续K步的confidence都大于threshold1停止。
        if len(step_rows):
            threshold1_list = [0.7, 0.75, 0.8, 0.85]
            k_list = [2,3,4,5]
            for threshold in threshold1_list:
                for k in k_list:
                    if len(step_rows) >= k:
                        last_k_confidences = [row.get("my_confidence", 0) for row in step_rows[-k:]]
                        if all(conf >= threshold for conf in last_k_confidences):
                            early_stop_step = len(step_rows) - 1
                            early_stop_reason = f"{k}_steps_conf_over_{threshold}"
                            if early_stop_reason not in early_stop_dict:
                                early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制3: 当第k步的step比第k-1的step高出threshold1
        # 停止条件：1. 第k步的confidence比第k-1的confidence高出threshold1。 2. 第k步的confidence高于threshold2。
        if len(step_rows) >= 2:
            threshold1_list = [0.15, 0.2, 0.25, 0.3, 0.35]   # 差值
            threshold2_list = [0.6, 0.7, 0.8]   # 当前值
            prev_conf = step_rows[-2].get("my_confidence", 0)
            curr_conf = step_rows[-1].get("my_confidence", 0)
            for t1 in threshold1_list:
                for t2 in threshold2_list:
                    if (curr_conf - prev_conf) >= t1 and curr_conf >= t2:
                        early_stop_step = len(step_rows) - 1
                        early_stop_reason = f"step_conf_improved_by_{t1}_and_over_{t2}"
                        if early_stop_reason not in early_stop_dict:
                            early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制4: 当第k步的step比第k-1的step高出threshold1,并计算提升百分比
        # 停止条件：1. 第k步的confidence比第k-1的confidence高出threshold1。 2. 第k步的confidence提升值/第k步的confidence大于threshold2。
                # 机制4: 第k步的confidence比第k-1步高出一定阈值，且提升幅度占比当前confidence超过threshold2
        if len(step_rows) >= 2:
            threshold1_list = [0.15, 0.2, 0.25, 0.3, 0.35]     # confidence 绝对提升
            threshold2_list = [0.2, 0.3, 0.4, 0.5]  # 提升百分比

            prev_conf = step_rows[-2].get("my_confidence", 0)
            curr_conf = step_rows[-1].get("my_confidence", 0)

            for t1 in threshold1_list:
                for t2 in threshold2_list:
                    delta = curr_conf - prev_conf
                    # 避免除以0
                    if curr_conf > 0 and delta >= t1 and (delta / curr_conf) >= t2:
                        early_stop_step = len(step_rows) - 1
                        early_stop_reason = f"conf_improved_by_{t1}_with_ratio_{t2}"
                        if early_stop_reason not in early_stop_dict:
                            early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制5：如果最近k步平均值大于某个阈值
        # 停止条件：1. 最近k步的confidence平均值大于threshold1
        if len(step_rows) >= 2:
            k_list = [2, 3, 4, 5]
            threshold1_list = [0.75, 0.8, 0.85, 0.9]

            for k in k_list:
                if len(step_rows) >= k:
                    recent_confs = [row.get("my_confidence", 0) for row in step_rows[-k:]]
                    avg_conf = sum(recent_confs) / k

                    for t1 in threshold1_list:
                        if avg_conf >= t1:
                            early_stop_step = len(step_rows) - 1
                            early_stop_reason = f"avg_conf_of_last_{k}_steps_over_{t1}"
                            if early_stop_reason not in early_stop_dict:
                                early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制6：如果最近k步confidence最大值减去confidence最小值小于threshold1
        # 停止条件：1. 最近k步confidence最大值减去confidence最小值小于threshold1
        if len(step_rows) >= 2:
            k_list = [3, 4, 5, 6]
            threshold1_list = [0.05, 0.1, 0.15, 0.2]  # 控制波动范围

            for k in k_list:
                if len(step_rows) >= k:
                    recent_confs = [row.get("my_confidence", 0) for row in step_rows[-k:]]
                    max_conf = max(recent_confs)
                    min_conf = min(recent_confs)
                    diff = max_conf - min_conf

                    for t1 in threshold1_list:
                        if diff < t1:
                            early_stop_step = len(step_rows) - 1
                            early_stop_reason = f"conf_range_of_last_{k}_steps_less_than_{t1:.2f}"
                            if early_stop_reason not in early_stop_dict:
                                early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制7：对k步的confidence做归一化，计算出熵，熵低于threshold1则停止
        # 停止条件：对k步的confidence做归一化，计算出熵，熵低于threshold1则停止
        if len(step_rows) >= 2:
            k_list = [3, 4, 5, 6]
            threshold1_list = [0.1, 0.2, 0.3, 0.4]
            for k in k_list:
                if len(step_rows) >= k:
                    recent_confs = [row.get("my_confidence", 0) for row in step_rows[-k:]]
                    total = sum(recent_confs)
                    # 避免除零
                    if total == 0:
                        continue
                    # 概率归一化
                    probs = [c / total for c in recent_confs]
                    # 计算熵（log以自然对数或2都可）
                    entropy = -sum(p * math.log(p + 1e-12) for p in probs)  # 防止 log(0)
                    for threshold1 in threshold1_list:
                        if entropy < threshold1:
                            early_stop_step = len(step_rows) - 1
                            early_stop_reason = f"entropy_of_last_{k}_steps_less_than_{threshold1}"
                            if early_stop_reason not in early_stop_dict:
                                early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制8：序列平滑度（Smoothness Index）
        # 停止条件：定义s_k=第k步的confidence-第k-1步的confidence的绝对值，若连续k步的s_k相加小于threshold1，则停止
                # 机制8：序列平滑度（Smoothness Index）
        # 停止条件：若连续k步的s_k相加小于threshold1，则early stop
        if len(step_rows) >= 3:  # 至少需要3步才能计算2个s_k
            k_list = [3, 4, 5, 6, 7]
            smooth_thresholds = [0.2, 0.4, 0.6]

            for k in k_list:
                if len(step_rows) >= k:
                    # 计算最近 k 步的平滑度：sum(|conf[i] - conf[i-1]|)
                    recent_confs = [row.get("my_confidence", 0) for row in step_rows[-k:]]
                    smoothness = sum(abs(recent_confs[i] - recent_confs[i - 1]) for i in range(1, k))

                    for threshold in smooth_thresholds:
                        if smoothness < threshold:
                            early_stop_step = len(step_rows) - 1
                            early_stop_reason = f"smoothness_of_last_{k}_steps_less_than_{threshold}"
                            if early_stop_reason not in early_stop_dict:
                                early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制9：对最近K步和更长的M步长期对比
        # 停止条件：如果短期均值 > 长期均值 + margin，则early stop
        if len(step_rows) >= 5:
            short_k_list = [3, 4]
            long_m_list = [8, 10]
            margin_list = [0.05, 0.1]

            for K in short_k_list:
                for M in long_m_list:
                    if len(step_rows) >= M and M > K:
                        short_confs = [row.get("my_confidence", 0) for row in step_rows[-K:]]
                        long_confs = [row.get("my_confidence", 0) for row in step_rows[-M:-K]]

                        short_avg = sum(short_confs) / K
                        long_avg = sum(long_confs) / (M - K)

                        for margin in margin_list:
                            if short_avg > long_avg + margin:
                                early_stop_step = len(step_rows) - 1
                                early_stop_reason = f"short_{K}_avg_gt_long_{M}_avg_by_{margin:.2f}"
                                if early_stop_reason not in early_stop_dict:
                                    early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制10：答案稳定性
        # 停止条件：1. 连续k步答案相同。 2. 平均的confidence大于threshold1。
        if len(step_rows) >= 2:
            k_list = [2, 3, 4, 5, 6]
            confidence_thresholds = [0.7, 0.8, 0.85, 0.5]

            for k in k_list:
                if len(step_rows) >= k:
                    recent_rows = step_rows[-k:]
                    recent_confs = [row.get("my_confidence", 0) for row in recent_rows]
                    recent_answers = [row.get("answer") for row in recent_rows]
                    avg_conf = sum(recent_confs) / k

                    # 检查答案是否全部一致
                    first_ans = recent_answers[0]
                    consistency = all(math_equal(first_ans, ans) for ans in recent_answers)

                    if consistency:
                        for threshold in confidence_thresholds:
                            if avg_conf >= threshold:
                                early_stop_step = len(step_rows) - 1
                                early_stop_reason = f"same_ans_for_last_{k}_steps_with_avg_conf_over_{threshold:.2f}"
                                if early_stop_reason not in early_stop_dict:
                                    early_stop_dict[early_stop_reason] = early_stop_step
        
        # 机制11：答案频率 vs confidence 动态
        # 停止条件：
        # score(a) = α * freq(a) + (1 - α) * avgConf(a)
        # 若存在答案 a 满足 score(a) > threshold 则 early stop

        if len(step_rows) >= 2:
            k_list = [4, 5, 6, 7, 8]  # 窗口大小
            alpha_list = [0.3, 0.5, 0.7]  # 频率的权重
            threshold_list = [0.7, 0.8, 0.9]  # 分数阈值

            for k in k_list:
                if len(step_rows) >= k:
                    recent_rows = step_rows[-k:]
                    answers = [row.get("answer") for row in recent_rows]
                    confs = [row.get("my_confidence", 0) for row in recent_rows]

                    # 统计：答案 -> (频率, 平均confidence)
                    stats = {}  # { ans : {"freq": x, "conf_sum": y} }

                    for ans, conf in zip(answers, confs):
                        if ans not in stats:
                            stats[ans] = {"freq": 0, "conf_sum": 0.0}
                        stats[ans]["freq"] += 1
                        stats[ans]["conf_sum"] += conf

                    # 遍历每个出现过的答案，计算 score
                    for ans, st in stats.items():
                        freq = st["freq"] / k
                        avg_conf = st["conf_sum"] / st["freq"]

                        for alpha in alpha_list:
                            score = alpha * freq + (1 - alpha) * avg_conf

                            for threshold in threshold_list:
                                if score >= threshold:
                                    early_stop_step = len(step_rows) - 1
                                    early_stop_reason = (
                                        f"ans_freq_conf_score_over_{threshold}"
                                        f"_k{k}_alpha{alpha}"
                                    )
                                    if early_stop_reason not in early_stop_dict:
                                        early_stop_dict[early_stop_reason] = early_stop_step
        
        step_row["early_stop_dict"] = copy.deepcopy(early_stop_dict)

        print(f"[Q{i} Solution{n_of_sampling} step{cnt}] len of stop is {len(stop_matches)}, batch_finish_reason is {batch_finish_reason}")
        cnt += 1
        if len(stop_matches) < args.match_limit and batch_finish_reason != "length":
            break
        
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_rows, ensure_ascii=False) + "\n")
    print(f"[Q{i} Solution{n_of_sampling}] finished........................................................................")


def construct_prompt(question):
    request = (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )
    prompt_temp = request
    input_template = prompt_temp[0]
    context = input_template.format(input=question)
    full_prompt = context + "<think>\n"
    return full_prompt.strip(" ")


def clear_files(data_name, model_name, args):
    if args.data_splition > 0:
        output_path = Path("llm_output") / f"{data_name}/{model_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"split_steps_{str(args.data_splition)}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            pass
        error_file = output_path / f"error_{str(args.data_splition)}.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            pass
        return output_file, error_file
    else:
        output_path = Path("llm_output") / f"{data_name}/{model_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"split_steps.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            pass
        error_file = output_path / f"error.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            pass
        return output_file, error_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=2)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_generate_workers", type=int, default=16)
    parser.add_argument("--model_name_or_path", type=str, default="/home/lijiakun25/models/Qwen3-8b")
    parser.add_argument("--data_name",default="olympiadbench_maths_en",type=str)
    parser.add_argument("--data_splition",default=1,type=int)
    parser.add_argument("--n_samplings",default=1,type=int)
    parser.add_argument("--dtype", default="float16", type=str)
    parser.add_argument("--match_limit", default=9, type=int)
    parser.add_argument("--search_depth", type=int, default=2)
    parser.add_argument("--num_of_search_branch", type=int, default=6)
    parser.add_argument("--search_token_budget", type=int, default=256)
    args = parser.parse_args()

    examples = load_data(data_name=args.data_name,data_path="")
    mid = len(examples) // 2
    dataset1 = examples.select(range(0, mid))
    dataset2 = examples.select(range(mid, len(examples)))
    start_index = None
    if args.data_splition == 1:
        examples = dataset1
        start_index = 0
    elif args.data_splition == 2:
        examples = dataset2
        start_index = mid
    elif args.data_splition == 0:
        start_index = 0
    else:
        raise KeyError

    # examples = examples.select(range(min(len(examples),30)))
    data_name = args.data_name
    model_name = args.model_name_or_path.strip("/").split("/")[-1]
  
    samples = []
    for local_i,example in enumerate(tqdm(examples,total=len(examples),disable=False)):
        global_i = start_index + local_i
        question = parse_question(example,data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        prompt = construct_prompt(question=question)
        samples.append((question,prompt,gt_ans,gt_cot, global_i))

    stop_seqs = ["Wait","But","Let me think","</think>"]
    output_file, error_file = clear_files(data_name=data_name, model_name=model_name, args=args)

    max_workers = max(1, int(args.max_generate_workers) * int(args.num_of_models))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, sample in enumerate(samples):
            for j in range(args.n_samplings):
                fut = executor.submit(
                    process_one_sample,
                    sample[-1], j, sample, args, stop_seqs, data_name, output_file, error_file
                )
                futures.append(fut)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", disable=False):
            fut.result()
