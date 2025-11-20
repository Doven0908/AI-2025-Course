# this test the accuracy of each continuing step.
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statistics import mean
from pathlib import Path
import os
from tqdm import tqdm
from utils.serve_vllm import completion_request
import datasets
from utils.parse import parse_ground_truth,parse_question
from utils.strip_string import extract_answer
from utils.grader import math_equal
from pathlib import Path
import argparse


REQUEST = (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )
def construct_prompt(question):
    output_dir = Path("llm_output")
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

def confidence_statistics(grouped_data,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    true_confidences = []
    false_confidences = []

    for tree_id, samples in grouped_data.items():
        for item in samples:
            if "is_equal" in item:
                confidence = item.get("confidence", None)
                if confidence is not None:
                    if item["is_equal"] == True:
                        true_confidences.append(confidence)
                    elif item["is_equal"] == False:
                        false_confidences.append(confidence)

    print("✔ 包含 is_equal 字段的记录总数:", len(true_confidences) + len(false_confidences))

    print("✔ is_equal == True:")
    print("  数量:", len(true_confidences))
    print("  平均 confidence:", mean(true_confidences) if true_confidences else "无数据")
    print("  分布:", true_confidences)

    print("✔ is_equal == False:")
    print("  数量:", len(false_confidences))
    print("  平均 confidence:", mean(false_confidences) if false_confidences else "无数据")
    print("  分布:", false_confidences)

    if len(true_confidences) >= 2 and len(false_confidences) >= 2:
        kde_true = gaussian_kde(true_confidences)
        kde_false = gaussian_kde(false_confidences)

        # 生成 x 区间用于画图
        x_min = min(min(true_confidences), min(false_confidences))
        x_max = max(max(true_confidences), max(false_confidences))
        x_vals = [x_min + (x_max - x_min) * i / 200 for i in range(201)]

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, kde_true(x_vals), label='is_equal = True', color='green')
        plt.plot(x_vals, kde_false(x_vals), label='is_equal = False', color='red')
        plt.fill_between(x_vals, kde_true(x_vals), alpha=0.3, color='green')
        plt.fill_between(x_vals, kde_false(x_vals), alpha=0.3, color='red')
        plt.title('Confidence KDE by is_equal')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_histogram.png", dpi=300)
        plt.close()
    else:
        print("⚠ 样本不足，无法绘制 KDE 图(至少每类2个数据)")


def find_best_confidence_from_group(group):
    # 步骤 1：从后往前找含有 pre_step_confidence 的 dict
    last_valid = None
    for item in reversed(group):
        if "pre_step_confidence" in item and item["pre_step_confidence"]:
            last_valid = item
            break

    if last_valid is None:
        return None  # 没有找到任何包含 pre_step_confidence 的项

    # 步骤 2：找到 pre_step_confidence 中最大值及其索引
    confidences = last_valid["pre_step_confidence"]
    max_conf = max(confidences)
    max_index = confidences.index(max_conf)

    # 步骤 3：找到整个 group 中 confidence == max_conf 的 dict
    matching_dict = None
    for item in group:
        if abs(item.get("confidence", -1) - max_conf) < 1e-8:
            matching_dict = item
            break
    return matching_dict


def construct_dict_prompt(step_dict:dict,end_think:bool):
    action_from_root = step_dict["actions_from_root"]
    if "</think>" not in action_from_root and end_think:
        action_from_root = action_from_root + "</think>"
    return action_from_root

if __name__ == "__main__":

    output_dir = "llm_output"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "test_7_accuracy.jsonl")

    data_path = "/home/lijiakun25/math-inference/llm_output/splition7_math500.jsonl"
    examples = datasets.load_dataset("json", data_files={"test": "/home/lijiakun25/models/datasets/math500/test.jsonl"})["test"]
    examples = examples.select(range(100))
    data_name = "math"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=1)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_generate_workers", type=int, default=16)
    parser.add_argument("--model_name_or_path", type=str, default="/home/lijiakun25/models/Qwen3-8b")
    args = parser.parse_args()


    grouped_data = defaultdict(list)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            tree_id = item.get('tree_id', None)
            grouped_data[tree_id].append(item)
    grouped_data = dict(list(grouped_data.items())[:100])

    
    samples = []
    for i,example in enumerate(tqdm(examples,total=len(examples),disable=False)):
        question = parse_question(example,data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        prompt = construct_prompt(question=question)
        samples.append((question,prompt,gt_ans,gt_cot))


    for tree_id, items in grouped_data.items():
        print(f"Tree ID: {tree_id}, 共 {len(items)} 条记录")
        for i, item in enumerate(items):
            print(f"  第{i+1}条：{item['output'][:50]}...")  # 打印前50个字符预览
    
    # confidence_statistics(grouped_data,output_dir)

    stop_true_no_stop_false = 0
    stop_false_no_stop_true = 0
    both_true = 0
    both_false = 0
    stop_token_usage = 0
    no_stop_token_usage = 0

    for i, (sample,group) in tqdm(enumerate(zip(samples, grouped_data.values())), total=len(samples), desc="Processing samples"):
        if group[0]["tree_id"] != i:
            raise ValueError("wrong id match.")

        prompt = sample[1]
        best_step_dict = find_best_confidence_from_group(group)
        action_from_root_stop = construct_dict_prompt(best_step_dict,True)
        action_from_root_no_stop = construct_dict_prompt(best_step_dict,False)

        outputs = completion_request(question=action_from_root_stop,args=args,tree_id=i,model_id=args.model_name_or_path,n_sampling=1,timeout_sec=100000)
        new_action_stop = outputs["texts"][0][0]
        new_ans_stop = extract_answer(new_action_stop,data_name=data_name)
        is_equal_stop = math_equal(new_ans_stop,sample[2])
        stop_token_usage += outputs["usage"].total_tokens

        outputs = completion_request(question=action_from_root_no_stop,args=args,tree_id=i,model_id=args.model_name_or_path,n_sampling=1,timeout_sec=100000)
        new_action_no_stop = outputs["texts"][0][0]
        new_ans_no_stop = extract_answer(new_action_no_stop,data_name=data_name)
        is_equal_no_stop = math_equal(new_ans_no_stop,sample[2])
        no_stop_token_usage += outputs["usage"].total_tokens
        
        if is_equal_no_stop and is_equal_stop:
            both_true += 1
        elif not is_equal_no_stop and is_equal_stop:
            stop_true_no_stop_false += 1
        elif is_equal_no_stop and not is_equal_stop:
            stop_false_no_stop_true += 1
        elif not is_equal_no_stop and not is_equal_stop:
            both_false += 1
        
        dump_dict = {
                "tree_id": i,
                "stop_true_no_stop_false": stop_true_no_stop_false,
                "stop_false_no_stop_true": stop_false_no_stop_true,
                "both_true": both_true,
                "both_false": both_false,
                "stop_token_usage": stop_token_usage,
                "no_stop_token_usage": no_stop_token_usage
            }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            json.dump(dump_dict, f, ensure_ascii=False)
            f.write("\n")
        print(dump_dict)
        
