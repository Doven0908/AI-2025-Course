from sympy import isprime
from transformers import AutoTokenizer
import math
import datasets
from utils.serve_vllm import completion_request,chat_completion_request
from tqdm import tqdm
from utils.parse import parse_ground_truth,parse_question
from utils.strip_string import extract_answer
from utils.grader import math_equal
import argparse
import multiprocessing
from pathlib import Path
import time
import json
import re
import Levenshtein
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statistics import mean
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
import copy

def confidence_by_step_boxplot(
    log_entrys,
    output_dir,
    ylim=(0.0, 1.0),
    min_count_per_step=2,   # 每个 step 至少多少样本才纳入绘图
    draw_mean_line=True     # 是否叠加均值折线
):

    os.makedirs(output_dir, exist_ok=True)
    filename = f"my_confidence_by_step_boxplot.png"

    # 1) 按 step 索引聚合
    step_conf = defaultdict(list)  # step_idx -> [conf, conf, ...]
    for groups in log_entrys:
        # 这一题/这一条轨迹的步骤序列
        all_true = all(parse_is_true(item) for item in groups)
        if all_true:
            continue
        has_any_true = any(parse_is_true(item) for item in groups)
        if not has_any_true:
            continue

        for step_idx, item in enumerate(groups):
            confidence = item.get("my_confidence", None)
            # print(confidence)
            step_conf[step_idx].append(float(confidence))
    MIN_COUNT = 20  # 阈值
    filtered = {s: vals for s, vals in step_conf.items() if len(vals) >= MIN_COUNT}
    step_conf = filtered

    if not step_conf:
        print("没有可用的 confidence 数据，无法绘图。")
        return

    # 2) 过滤掉样本太少的 step，并排序
    steps = sorted(k for k, v in step_conf.items() if len(v) >= min_count_per_step)
    if not steps:
        print("每个 step 的样本数都不足，无法绘制箱形图。")
        return
    data = [step_conf[s] for s in steps]

    # 3) 计算均值用于趋势线
    means = [mean(vals) for vals in data]

    # 4) 画图（箱形图 + 可选均值折线）
    plt.figure(figsize=(10, 5))
    # 箱形图
    bp = plt.boxplot(
        data,
        positions=steps,       # x 轴为 step 索引
        widths=0.6,
        patch_artist=True,     # 允许填充颜色
        showmeans=False,
        showfliers=True,       # 显示离群点
        whis=1.5               # 箱线图须髯长度
    )

    # 统一配色与边框
    for patch in bp['boxes']:
        patch.set(facecolor="#cfe8cf", edgecolor="#2e7d32", linewidth=1.2)  # 绿色系填充
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        for line in bp[element]:
            line.set(color="#2e7d32", linewidth=1.2)
    # 离群点外观
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=3, markerfacecolor="#66bb6a", alpha=0.6, markeredgecolor="#2e7d32")

    # 均值折线（趋势）
    if draw_mean_line:
        plt.plot(steps, means, marker="o", linewidth=2, alpha=0.9, label="Mean confidence")

    plt.xlabel("Step Index")
    plt.ylabel("Confidence Score")
    plt.ylim(*ylim)
    plt.xlim(min(steps) - 0.5, max(steps) + 0.5)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    plt.title("Confidence Distribution per Step (Boxplot) with Mean Trend")
    if draw_mean_line:
        plt.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 已保存：{out_path}")

def confidence_statistics(log_entrys,output_dir):

    os.makedirs(output_dir, exist_ok=True)
    true_confidences = []
    false_confidences = []

    for samples in log_entrys:
        for item in samples:
            if "is_equal" in item:
                confidence = item.get("my_confidence", None)
                if confidence is not None:
                    if parse_is_true(item):
                        true_confidences.append(confidence)
                    else:
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


def parse_is_true(entry):
        """
        从 entry 中抽取布尔判定:
        - is_equal: bool
        - llm_judge_equal: list[str]，取第一个元素，看是不是 "True"
        """
        if "is_equal" in entry and "llm_judge_equal" in entry:
            if entry.get("is_equal") is True:
                return True
            llm = entry.get("llm_judge_equal")
            if isinstance(llm, list) and len(llm) > 0:
                return str(llm[0]).strip().lower() == "true"
            return False
        return False


def check_pass_rate(log_entrys, args, output_path):
    """
    log_entrys: 是 log_entrys[tree_id][n_of_sampling][step_idx] 的结构，
                每个 group 是同一 tree_id 的 entry 列表
    """


    total_groups = len(log_entrys)
    early_stop_count = [0,0,0,0] # 第一个代表早停对，最终对，第二个代表早停错，最终对，第三个代表早停对，最终错，第四个代表早停错，最终错
    stop_reason_count = dict()
    stop_reason_count["step0_high_conf"] = 0
    stop_reason_count["conf_very_high_at_"] = 0
    stop_reason_count["conf_jump_at_"] = 0

    early_stop_idx_list = [[],[],[],[]]
    early_stop_step_count = 0

    for group_idx, group in enumerate(log_entrys):
        early_stop = group[-1].get("early_stop")
        if early_stop:
            first_early_stop = None
            for first_stop_idx, entry in enumerate(group):
                if entry.get("early_stop"):
                    first_early_stop = first_stop_idx
                    early_stop_reason = entry.get("early_stop_reason")
                    break

                if "step0_high_conf" in early_stop_reason:
                    stop_reason_count["step0_high_conf"] += 1
                elif "conf_very_high_at_" in early_stop_reason:
                    stop_reason_count["conf_very_high_at_"] += 1
                elif "conf_jump_at_" in early_stop_reason:
                    stop_reason_count["conf_jump_at_"] += 1
            
            early_stop_step_count += first_early_stop
            early_stop_step_count += 1
        
            early_stop_is_true = parse_is_true(group[first_early_stop])
            last_is_true = parse_is_true(group[-1])
            a_index = 0 if last_is_true else 2
            b_index = 0 if early_stop_is_true else 1
            early_stop_count[a_index + b_index] = early_stop_count[a_index + b_index] + 1
            early_stop_idx_list[a_index + b_index].append(group_idx)

    # 保存早停错且最终对的例子
    if early_stop_idx_list[1]:
        output_file_early_stop_wrong_final_correct = os.path.join(output_path, "early_stop_wrong_final_correct.jsonl")
        with open(output_file_early_stop_wrong_final_correct, "w", encoding="utf-8") as f:
            for group_idx in early_stop_idx_list[1]:
                f.write(json.dumps(log_entrys[group_idx], ensure_ascii=False) + "\n")

    # 保存早停对且最终错的例子
    if early_stop_idx_list[2]:
        output_file_early_stop_correct_final_wrong = os.path.join(output_path, "early_stop_correct_final_wrong.jsonl")
        with open(output_file_early_stop_correct_final_wrong, "w", encoding="utf-8") as f:
            for group_idx in early_stop_idx_list[2]:
                f.write(json.dumps(log_entrys[group_idx], ensure_ascii=False) + "\n")


    early_stop_any_rate = early_stop_count[0] / total_groups if total_groups > 0 else 0.0
    early_stop_final_rate = early_stop_count[1] / total_groups if total_groups > 0 else 0.0
    early_stop_any_wrong_final_rate = early_stop_count[2] / total_groups if total_groups > 0 else 0.0
    early_stop_final_wrong_rate = early_stop_count[3] / total_groups if total_groups > 0 else 0.0
    # 输出和保存分析结果
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "early_stop_analysis.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"总分组数量: {total_groups}\n")
        f.write(f"早停对且最终对数量: {early_stop_count[0]}\n")
        f.write(f"早停错且最终对数量: {early_stop_count[1]}\n")
        f.write(f"早停对且最终错数量: {early_stop_count[2]}\n")
        f.write(f"早停错且最终错数量: {early_stop_count[3]}\n")
        f.write(f"早停对且最终对率: {early_stop_any_rate:.2%}\n")
        f.write(f"早停错且最终对率: {early_stop_final_rate:.2%}\n")
        f.write(f"早停对且最终错率: {early_stop_any_wrong_final_rate:.2%}\n")
        f.write(f"早停错且最终错率: {early_stop_final_wrong_rate:.2%}\n")
        f.write(f"早停步骤数累计: {early_stop_step_count}\n")
        
        f.write(f"step0_high_conf 计数: {stop_reason_count['step0_high_conf']}\n")
        f.write(f"conf_very_high_at_ 计数: {stop_reason_count['conf_very_high_at_']}\n")
        f.write(f"conf_jump_at_ 计数: {stop_reason_count['conf_jump_at_']}\n")

    print(f"总分组数量: {total_groups}\n")
    print(f"早停对且最终对数量: {early_stop_count[0]}\n")
    print(f"早停错且最终对数量: {early_stop_count[1]}\n")
    print(f"早停对且最终错数量: {early_stop_count[2]}\n")
    print(f"早停错且最终错数量: {early_stop_count[3]}\n")
    print(f"早停对且最终对率: {early_stop_any_rate:.2%}\n")
    print(f"早停错且最终对率: {early_stop_final_rate:.2%}\n")
    print(f"早停对且最终错率: {early_stop_any_wrong_final_rate:.2%}\n")
    print(f"早停错且最终错率: {early_stop_final_wrong_rate:.2%}\n")
    print(f"早停步骤数累计: {early_stop_step_count}\n")
    print(f"分析结果已保存至: {output_file}")
    
    pass_any_count = 0
    pass_last_count = 0
    pass_first_count = 0
    best_step_count = 0
    
    for group in log_entrys:
        # 1) 是否曾经出现过 True
        has_true = any(parse_is_true(entry) for entry in group)
        for has_true_idx,entry in enumerate(group):
            if "is_equal" in entry or "llm_judge_equal" in entry:
                if parse_is_true(entry):
                    best_step_count += has_true_idx
                    best_step_count += 1
                    break
        # 2) 找最后一个带判定字段的 entry
        last_is_true = False
        for entry in reversed(group):
            if "is_equal" in entry or "llm_judge_equal" in entry:
                last_is_true = parse_is_true(entry)
                break

        first_is_true = False
        for entry in group:
            if "is_equal" in entry or "llm_judge_equal" in entry:
                first_is_true = parse_is_true(entry)
                break
        if has_true:
            pass_any_count += 1
        if last_is_true:
            pass_last_count += 1
        if first_is_true:
            pass_first_count += 1

    pass_any_rate = pass_any_count / total_groups if total_groups > 0 else 0.0
    pass_last_rate = pass_last_count / total_groups if total_groups > 0 else 0.0
    pass_first_rate = pass_first_count / total_groups if total_groups > 0 else 0.0
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"总分组数量: {total_groups}\n")
        f.write(f"至少一次通过数量: {pass_any_count}\n")
        f.write(f"最后一次通过数量: {pass_last_count}\n")
        f.write(f"第一次通过数量: {pass_first_count}\n")
        f.write(f"至少一次通过率: {pass_any_rate:.2%}\n")
        f.write(f"最后一次通过率: {pass_last_rate:.2%}\n")
        f.write(f"第一次通过率: {pass_first_rate:.2%}\n")
        f.write(f"最高优化步数: {best_step_count}\n")
    print(f"总分组数量: {total_groups}\n")
    print(f"至少一次通过数量: {pass_any_count}\n")
    print(f"最后一次通过数量: {pass_last_count}\n")
    print(f"第一次通过数量: {pass_first_count}\n")
    print(f"至少一次通过率: {pass_any_rate:.2%}\n")
    print(f"最后一次通过率: {pass_last_rate:.2%}\n")
    print(f"第一次通过率: {pass_first_rate:.2%}\n")
    print(f"分析结果已保存至: {output_file}")



def merge_inputs(path1, path2, output_path):
    # 用于存储所有的日志条目
    log_entries = []

    # 读取第一个文件
    with open(path1, "r", encoding="utf-8") as f1:
        for line in f1:
            try:
                log_entry = json.loads(line)  # 解析每一行 JSON
                log_entries.append(log_entry)  # 添加到日志列表中
            except json.JSONDecodeError as e:
                print(f"读取错误 (文件 {path1}): {e}")
    
    # 读取第二个文件
    with open(path2, "r", encoding="utf-8") as f2:
        for line in f2:
            try:
                log_entry = json.loads(line)  # 解析每一行 JSON
                log_entries.append(log_entry)  # 添加到日志列表中
            except json.JSONDecodeError as e:
                print(f"读取错误 (文件 {path2}): {e}")

    # 将合并的日志条目写入输出文件
    with open(output_path, "w", encoding="utf-8") as output_file:
        for log_entry in log_entries:
            json.dump(log_entry, output_file, ensure_ascii=False)
            output_file.write("\n")  # 每个 JSON 对象占一行

    print(f"文件已成功合并并保存至 {output_path}")

def deal_early_stop(log_entrys, early_stop_name_list, args):

    total_steps = 0
    early_stop_count = [0,0,0,0,0,0] # 第一个代表早停对，最终对，第二个代表早停错，最终对，第三个代表早停对，最终错，第四个代表早停错，最终错,第五个代表无早停，最终对，第六个代表无早停，最终错
    early_stop_idx_list = [[],[],[],[],[],[]]
    for group_idx, group in enumerate(log_entrys):
        early_stop_dict = group[-1].get("early_stop_dict")
        first_early_stop = -1
        for early_stop_name in early_stop_name_list:
            early_stop_step = early_stop_dict.get(early_stop_name, -1)
            if first_early_stop == -1:
                first_early_stop = early_stop_step
            elif first_early_stop > early_stop_step:
                first_early_stop = early_stop_step
        
        if first_early_stop >= 0:
            cur_group_step = first_early_stop + 1
            total_steps += cur_group_step

            early_stop_is_true = parse_is_true(group[first_early_stop])
            last_is_true = parse_is_true(group[-1])
            a_index = 0 if last_is_true else 2
            b_index = 0 if early_stop_is_true else 1
            early_stop_count[a_index + b_index] = early_stop_count[a_index + b_index] + 1
            early_stop_idx_list[a_index + b_index].append(group_idx)
        
        else:
            last_is_true = parse_is_true(group[-1])
            total_steps += len(group)
            if last_is_true:
                early_stop_count[-2] = early_stop_count[-2] + 1
                early_stop_idx_list[-2].append(group_idx)
            else:
                early_stop_count[-1] = early_stop_count[-1] + 1
                early_stop_idx_list[-1].append(group_idx)
        


    return early_stop_count, early_stop_idx_list, total_steps


from utils.data_loader import load_data

if __name__ == "__main__":
    
    model_name = "/home/lijiakun25/models/Qwen3-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size of the tokenizer: {vocab_size}")
    stop_tokens = ["\n\n"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=1)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_generate_workers", type=int, default=16)
    parser.add_argument("--model_name_or_path", type=str, default="/home/lijiakun25/models/Qwen3-8b")
    parser.add_argument("--data_name",default="olympiadbench_maths_en",type=str)
    args = parser.parse_args()

    path1 = '/home/lijiakun25/math-inference/llm_output/olympiadbench_maths_en/Qwen3-8b/split_steps_1.jsonl'
    path2 = '/home/lijiakun25/math-inference/llm_output/olympiadbench_maths_en/Qwen3-8b/split_steps_2.jsonl'
    output_path = '/home/lijiakun25/math-inference/llm_output/olympiadbench_maths_en/Qwen3-8b/split_steps.jsonl'
    # merge_inputs(path1, path2, output_path)

    examples = load_data(data_name=args.data_name,data_path="")
    examples = examples.select(range(min(len(examples),200)))
    args = parser.parse_args()
    model_name = args.model_name_or_path.strip("/").split("/")[-1]

    train_dataset_path = f"/home/lijiakun25/math-inference/llm_output/{args.data_name}/{model_name}/split_steps.jsonl"

    log_entrys = []
    with open(train_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                log_entrys.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"读取错误：{e}")

    sorted_log_entrys = sorted(log_entrys, key=lambda x: x[0]['tree_id'])
    log_entrys = sorted_log_entrys
    
    data_name = args.data_name
    model_name = args.model_name_or_path.strip("/").split("/")[-1]
    output_path = Path("llm_output")/f"{data_name}/{model_name}/conf_analysis/"
    output_path.mkdir(parents=True, exist_ok=True)

    confidence_statistics(log_entrys=log_entrys,output_dir=output_path)
    confidence_by_step_boxplot(log_entrys=log_entrys,output_dir=output_path)
    # check_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path)

    early_stop_key_list = []
    for log_entry in log_entrys:
        last_row = log_entry[-1]
        early_stop_dict = last_row.get("early_stop_dict")
        for key in early_stop_dict.keys():
            if key not in early_stop_key_list:
                early_stop_key_list.append(key)

    early_stop_metrics = []

    all_acc_count, _ , all_steps = deal_early_stop(log_entrys=log_entrys, early_stop_name_list=[], args=args)
    all_acc = all_acc_count[4] / (all_acc_count[5] + all_acc_count[4])

    for key in early_stop_key_list:

        early_stop_count, early_stop_idx_list, total_steps = deal_early_stop(log_entrys=log_entrys, early_stop_name_list=[key], args=args)
        # 第一个代表早停对，最终对，第二个代表早停错，最终对，第三个代表早停对，最终错，第四个代表早停错，最终错,第五个代表无早停，最终对，第六个代表无早停，最终错
        correct_early_stops = early_stop_count[0] + early_stop_count[2]  # 正确的早停次数
        total_early_stops = sum(early_stop_count[:4])  # 只计算早停的总次数，忽略无早停的部分
        correct_drop_rate = correct_early_stops / total_early_stops if total_early_stops > 0 else 0

        total_correct = early_stop_count[0] + early_stop_count[2] + early_stop_count[4]
        total_count = sum(early_stop_count)
        total_acc = total_correct / total_count

        early_stop_recall = (early_stop_count[0] + early_stop_count[2]) / (early_stop_count[0] + early_stop_count[1])

        early_stop_metrics.append({
            'key': key,
            'total_early_stops': total_early_stops,
            'total_acc': total_acc,
            'correct_drop_rate': correct_drop_rate,
            'total_steps': total_steps,
            'early_stop_recall': early_stop_recall,
            'early_stop_count': copy.deepcopy(early_stop_count),
            'all_acc': all_acc,
            'all_steps': all_steps
        })
    
    # 按照 total_early_stops, total_acc, correct_drop_rate, total_steps 进行排序
    sorted_metrics = sorted(early_stop_metrics, key=lambda x: (
        -x['early_stop_recall'],
        x['total_acc'],  # 按早停的总次数升序排序
        -x['total_early_stops'], # 按总体准确率降序排序
        -x['correct_drop_rate'],# 按正确下降率降序排序
        x['total_steps']         # 按步数升序排序
    ))

    # 保存排序后的结果到文件
    with open(output_path / "early_stop_diff_strategy.txt", "w", encoding="utf-8") as f:
        for metric in sorted_metrics:
            line = f"Key: {metric['key']:<45} " \
                   f"Total Early Stops: {metric['total_early_stops']:>5} " \
                   f"Total Accuracy: {metric['total_acc']:>7.4f} " \
                   f"Correct Drop Rate: {metric['correct_drop_rate']:>7.4f} " \
                   f"Early Stop Recall: {metric['early_stop_recall']:>7.4f} " \
                   f"Total Steps: {metric['total_steps']:>6}\n"
            f.write(line)
    
    jsonl_output_path = output_path / "early_stop_diff_strategy_detailed.jsonl"
    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        for metric in sorted_metrics:
            # 将每个 metric 写入文件
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")
        
        

    
