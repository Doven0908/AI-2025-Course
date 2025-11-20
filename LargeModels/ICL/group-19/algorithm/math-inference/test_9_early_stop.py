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

def flattern_log_entrys(log_entrys, check_idx_list):
    flattened_groups = []
    idx_mapping = []  
    for tree_idx,tree_group in enumerate(log_entrys):
        for sampling_idx, sampling_group in enumerate(tree_group):
            if check_idx_list:
                if (tree_idx, sampling_idx) in check_idx_list:
                    flattened_groups.append(sampling_group)
                    idx_mapping.append((tree_idx, sampling_idx))
            else:
                flattened_groups.append(sampling_group)
                idx_mapping.append((tree_idx, sampling_idx))
    return flattened_groups, idx_mapping

def check_first_high_conf_pass_rate(
    log_entrys, args, output_path, confidence_threshold=0.8, save_entry=True, check_idx_list=None
):
    """
    功能：
        对每个样例(group)，检测第一个置信度超过阈值的情况。
        若发现第一个entry置信度 >= confidence_threshold，
        取该entry为关键节点，用 parse_is_true() 判断是否通过；
        若不存在这样的entry，则取最后一个可判定entry判断。

    参数：
        log_entrys: log_entrys[tree_id][n_of_sampling][step_idx] 结构。
        args: 预留参数。
        output_path: 输出分析结果文件夹路径。
        confidence_threshold: 判定阈值（默认0.8）。
    """

    # ---- 展平结构 ----
    log_entrys, idx_mapping = flattern_log_entrys(log_entrys=log_entrys, check_idx_list=check_idx_list)

    total_groups = len(log_entrys)
    check_count = 0
    pass_count = 0
    check_step_count = 0
    no_check_step_count = 0
    until_last_pass_count = 0
    final_pass_rate_for_high_conf_groups = 0
    no_high_conf_idx_list = []

    os.makedirs(output_path, exist_ok=True)
    save_jsonl_path = os.path.join(output_path, "first_high_conf_entry.jsonl")
    save_jsonl_file = open(save_jsonl_path, "w", encoding="utf-8")

    for idx, group in enumerate(log_entrys):
        (tree_idx, sampling_idx) = idx_mapping[idx]
        if len(group) == 0:
            continue

        # 提取confidence序列
        confs = []
        for e in group:
            conf = e.get("confidence")
            if isinstance(conf, str):
                try:
                    conf_value = float(conf)
                except ValueError:
                    conf_value = None
            elif isinstance(conf, dict):
                conf_value = conf.get("self_design_conf")
            else:
                conf_value = conf
            confs.append(conf_value)

        # 找第一个置信度 >= threshold 的 entry
        first_high_idx = None
        for i, conf in enumerate(confs):
            if conf is not None and conf >= confidence_threshold:
                first_high_idx = i
                break

        if first_high_idx is not None:
            check_count += 1
            entry = group[first_high_idx]
            check_step_count += first_high_idx + 1
            if parse_is_true(entry):
                pass_count += 1
            if parse_is_true(group[-1]):
                final_pass_rate_for_high_conf_groups += 1
            if save_entry:
                save_jsonl_file.write(json.dumps(group, ensure_ascii=False) + "\n")
        else:
            # 没有置信度 >= threshold 的entry
            no_high_conf_idx_list.append((tree_idx, sampling_idx))
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            no_check_step_count += len(group)

    if save_jsonl_file:
        save_jsonl_file.close()

    # ---- 计算统计结果 ----
    rate_over_check = (pass_count / check_count) if check_count > 0 else 0.0
    rate_until_last_check = (final_pass_rate_for_high_conf_groups / check_count) if check_count > 0 else 0.0
    rate_over_total = ((pass_count + until_last_pass_count) / total_groups) if total_groups > 0 else 0.0

    # ---- 输出结果 ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            "\n—— 第一个高置信度检测 ——\n"
            f"条件: 第一个entry的置信度 ≥ {confidence_threshold:.2f}\n"
        )
        f.write(f"符合条件的分组数: {check_count}\n")
        f.write(f"其中通过的分组数: {pass_count}\n")
        f.write(f"符合条件分组最后一个entry通过率: {rate_until_last_check:.2%}\n")
        f.write(f"总步数: {no_check_step_count + check_step_count}\n")
        f.write(f"符合条件分组步数: {check_step_count}\n")
        f.write(f"通过率（以符合数为分母）: {rate_over_check:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")

    print("\n—— 第一个高置信度检测 ——")
    print(f"条件: 第一个entry的置信度 ≥ {confidence_threshold:.2f}")
    print(f"符合条件的分组数: {check_count}")
    print(f"其中通过的分组数: {pass_count}")
    print(f"总步数: {no_check_step_count + check_step_count}")
    print(f"通过率（以符合数为分母）: {rate_over_check:.2%}")
    print(f"通过率（以总组数为分母）: {rate_over_total:.2%}")
    print(f"分析结果已保存至: {output_file}")

    return no_high_conf_idx_list, check_count, pass_count, check_step_count, len(no_high_conf_idx_list), until_last_pass_count, no_check_step_count


def check_conf_jump_pass_rate(log_entrys, args, output_path, jump_threshold=0.2, confidence_threshold=0.7, save_entry=True, check_idx_list=None):
    """
    功能：
        对每个样例(group)，检测置信度突然大幅提升的情况。
        若发现后一个entry的置信度比前一个entry高 jump_threshold 以上，
        取该entry为关键节点，用 parse_is_true() 判断是否通过；
        若不存在这样的跃迁，则取最后一个可判定entry判断。

    参数：
        log_entrys: log_entrys[tree_id][n_of_sampling][step_idx] 结构。
        args: 预留参数。
        output_path: 输出分析结果文件夹路径。
        jump_threshold: 后一项比前一项置信度上升超过该阈值时，视为置信度跃迁（默认0.2）。
    """

    # ---- 展平结构 ----
    log_entrys, idx_mapping = flattern_log_entrys(log_entrys=log_entrys,check_idx_list=check_idx_list)

    total_groups = len(log_entrys)
    check_count = 0
    pass_count = 0
    check_step_count = 0
    no_check_step_count = 0
    until_last_pass_count = 0
    final_pass_rate_for_jump_groups = 0
    no_jump_idx_list = []


    os.makedirs(output_path, exist_ok=True)
    save_jsonl_path = os.path.join(output_path, "conf_jump_cases.jsonl")
    save_jsonl_file = open(save_jsonl_path, "w", encoding="utf-8")

    for idx, group in enumerate(log_entrys):
        (tree_idx, sampling_idx) = idx_mapping[idx]
        if len(group) < 2:
            # 只有一个entry无法比较
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            no_check_step_count += len(group)
            no_jump_idx_list.append((tree_idx, sampling_idx))
            continue

        # 提取confidence序列
        confs = []
        for e in group:
            conf = e.get("confidence")
            if isinstance(conf, str):
                conf_value = None
            elif isinstance(conf, dict):
                conf_value = conf.get("self_design_conf")
            else:
                conf_value = conf
            confs.append(conf_value)

        # 找第一个置信度“跃迁点”
        jump_idx = None
        for i in range(1, len(confs)):
            prev = confs[i - 1]
            curr = confs[i]
            if prev is not None and curr is not None:
                if curr - prev >= jump_threshold and curr >= confidence_threshold:
                    jump_idx = i
                    break

        if jump_idx is not None:
            check_count += 1
            entry = group[jump_idx]
            check_step_count += jump_idx + 1
            if parse_is_true(entry):
                pass_count += 1
            if parse_is_true(group[-1]):
                final_pass_rate_for_jump_groups += 1
            if save_entry:
                save_jsonl_file.write(json.dumps(group, ensure_ascii=False) + "\n")
        else:
            # 没有跃迁点，则看最后一个可判定entry
            no_jump_idx_list.append((tree_idx, sampling_idx))
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            no_check_step_count += len(group)
    


    rate_over_check = (pass_count / check_count) if check_count > 0 else 0.0
    rate_until_last_check = (final_pass_rate_for_jump_groups / check_count) if check_count > 0 else 0.0
    rate_over_total = ((pass_count + until_last_pass_count) / total_groups) if total_groups > 0 else 0.0

    if save_jsonl_file:
        save_jsonl_file.close()
    # ---- 输出结果 ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            "\n—— 置信度跃迁检测 ——\n"
            f"条件: 后一个entry的置信度比前一个提升 ≥ {jump_threshold:.2f} 且后一个置信度 ≥ {confidence_threshold:.2f}\n"
        )
        f.write(f"出现跃迁的分组数: {check_count}\n")
        f.write(f"其中通过的分组数: {pass_count}\n")
        f.write(f"出现跃迁的分组数最后一个entry通过率: {rate_until_last_check}\n")
        f.write(f"总步数: {no_check_step_count + check_step_count}\n")
        f.write(f"出现跃迁的分组数步数: {check_step_count}\n")
        f.write(f"通过率（以出现数为分母）: {rate_over_check:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")
    print("\n—— 置信度跃迁检测 ——")
    print(f"条件: 后一个entry的置信度比前一个提升 ≥ {jump_threshold:.2f} 且后一个置信度 ≥ {confidence_threshold:.2f}")
    print(f"出现跃迁的分组数: {check_count}")
    print(f"其中通过的分组数: {pass_count}")
    print(f"总步数: {no_check_step_count + check_step_count}")
    print(f"通过率（以出现数为分母）: {rate_over_check:.2%}")
    print(f"通过率（以总组数为分母）: {rate_over_total:.2%}")
    print(f"分析结果已保存至: {output_file}")

    return no_jump_idx_list, check_count, pass_count, check_step_count, len(no_jump_idx_list), until_last_pass_count, no_check_step_count

def check_stable_conf_pass_rate(log_entrys, args, output_path, threshold=0.1, save_entry=True, check_idx_list=None):
    """
    功能：
        对每个样例(group)，检测连续4个entry置信度变化不大的情况（即稳定段）。
        若存在稳定段，则取该段最后一个entry，用 parse_is_true() 判断是否通过；
        若不存在稳定段，则取最后一个可判定entry进行判断。
        统计稳定段出现次数、通过次数、步数和通过率。

    参数：
        log_entrys: log_entrys[tree_id][n_of_sampling][step_idx] 结构。
        args: 预留参数。
        output_path: 输出分析结果文件夹路径。
        threshold: 连续4步置信度最大最小差小于该阈值时认为稳定（默认0.03）。
    """
    log_entrys, idx_mapping = flattern_log_entrys(log_entrys=log_entrys,check_idx_list=check_idx_list)

    total_groups = len(log_entrys)
    check_count = 0
    pass_count = 0
    check_step_count = 0
    no_check_step_count = 0
    until_last_pass_count = 0
    check_until_last_pass_count = 0
    no_stable_idx_list = []

    os.makedirs(output_path, exist_ok=True)
    save_jsonl_path = os.path.join(output_path, "stable_conf_cases.jsonl")
    save_jsonl_file = open(save_jsonl_path, "w", encoding="utf-8") if save_entry else None

    for idx, group in enumerate(log_entrys):
        (tree_idx, sampling_idx) = idx_mapping[idx]

        if len(group) < 4:
            # 不足4个，不可能形成稳定段
            no_stable_idx_list.append((tree_idx, sampling_idx))
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            no_check_step_count += len(group)
            continue

        # 提取置信度序列
        confs = []
        for e in group:
            conf = e.get("confidence")
            if isinstance(conf, str):
                conf_value = None
            elif isinstance(conf, dict):
                conf_value = conf.get("self_design_conf")
            else:
                conf_value = conf
            confs.append(conf_value)

        # 找第一个稳定段（连续4步波动小）
        stable_idx = None
        for i in range(len(confs) - 3):
            window = [c for c in confs[i:i+4] if c is not None]
            if len(window) == 4 and (max(window) - min(window)) <= threshold:
                stable_idx = i + 3  # 第4个entry索引
                break

        if stable_idx is not None:
            check_count += 1
            entry = group[stable_idx]
            check_step_count += stable_idx + 1
            if parse_is_true(entry):
                pass_count += 1
            if parse_is_true(group[-1]):
                check_until_last_pass_count += 1
            if save_entry:
                save_jsonl_file.write(json.dumps(group, ensure_ascii=False) + "\n")
        else:
            no_stable_idx_list.append((tree_idx, sampling_idx))
            # 没有稳定段，取最后一个可判定entry
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            no_check_step_count += len(group)

    if save_jsonl_file:
        save_jsonl_file.close()

    rate_over_check = (pass_count / check_count) if check_count > 0 else 0.0
    rate_until_last_check = (check_until_last_pass_count / check_count) if check_count > 0 else 0.0
    rate_over_total = ((pass_count + until_last_pass_count) / total_groups) if total_groups > 0 else 0.0

    # ---- 输出结果 ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            "\n—— 连续4步置信度稳定检测 ——\n"
            f"条件: 连续4步置信度最大最小差 ≤ {threshold:.3f}\n"
        )
        f.write(f"出现稳定段的分组数: {check_count}\n")
        f.write(f"其中通过的分组数: {pass_count}\n")
        f.write(f"出现稳定段的分组数最后一个entry通过率: {rate_until_last_check}\n")
        f.write(f"总步数: {no_check_step_count + check_step_count}\n")
        f.write(f"出现稳定段的分组数步数: {check_step_count}\n")
        f.write(f"通过率（以出现数为分母）: {rate_over_check:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")
    print("\n—— 连续4步置信度稳定检测 ——")
    print(f"条件: 连续4步置信度最大最小差 ≤ {threshold:.3f}")
    print(f"出现稳定段的分组数: {check_count}")
    print(f"其中通过的分组数: {pass_count}")
    print(f"总步数: {no_check_step_count + check_step_count}")
    print(f"通过率（以出现数为分母）: {rate_over_check:.2%}")
    print(f"通过率（以总组数为分母）: {rate_over_total:.2%}")
    print(f"分析结果已保存至: {output_file}")

    # ✅ 返回：未出现稳定段的样例索引 + 出现数 + 通过数 + 出现稳定段的步数
    return no_stable_idx_list, check_count, pass_count, check_step_count, len(no_stable_idx_list), until_last_pass_count, no_check_step_count

def check_entry0_high_conf_pass_rate(
    log_entrys, args, output_path, threshold=0.85, save_entry=True, check_idx_list=None
):
    """
    功能：
        对每个样例 (group)，取第一个有置信度的entry：
            若置信度 >= threshold，则将其 answer 与后一个answer对比；
            若第一个 entry 不符合条件，则视为未满足。
        统计符合条件的数量、通过数量、步数与通过率。
        若样例中第一个 entry 不符合条件，则加入 idx_list 返回，用于后续分析。

    参数：
        log_entrys: log_entrys[tree_id][n_of_sampling][step_idx] 结构。
        args: 预留参数。
        output_path: 输出分析结果文件夹路径。
        threshold: 置信度阈值（默认 0.85）。
        save_entry: 是否保存符合条件的样例到 JSONL 文件。
        check_idx_list: 限定分析的 (tree_idx, sampling_idx) 列表。
    """
    log_entrys, idx_mapping = flattern_log_entrys(log_entrys=log_entrys,check_idx_list=check_idx_list)
    
    total_groups = len(log_entrys)
    first_conf_more_than_count = 0
    first_conf_more_than_pass = 0
    check_step_count = 0
    no_check_step_count = 0
    until_last_pass_count = 0
    check_until_last_pass_count = 0
    no_high_conf_idx_list = []

    # ---- 文件路径 ----
    os.makedirs(output_path, exist_ok=True)
    save_jsonl_path = os.path.join(output_path, "first_entry_is_high_conf_cases.jsonl")
    save_jsonl_file = open(save_jsonl_path, "w", encoding="utf-8") if save_entry else None

    for idx, group in enumerate(log_entrys):
        (tree_idx, sampling_idx) = idx_mapping[idx]
        first_idx = None
        first_entry = None
        first_conf_value = None
        # ====== 找第一个置信度 >= 0.85 的 entry ======
        for i, e in enumerate(group):
            conf = e.get("confidence")
            if isinstance(conf, str):
                continue
            else:
                if isinstance(conf,dict):
                    conf = conf.get("self_design_conf")
            if conf is not None:
                first_idx = i
                first_entry = e
                first_conf_value = conf
                break

        final_entry = None
        for e in reversed(group):
            if "is_equal" in e or "llm_judge_equal" in e:
                final_entry = e
                break

        # ====== 比较 ======
        strict_judge = False
        if first_idx is not None:
            if first_idx + 1 == len(group):
                strict_judge = True
            else:
                strict_judge = math_equal(first_entry.get("answer"),group[first_idx + 1].get("answer"))

        if first_idx is not None and first_conf_value >= threshold and strict_judge:
            check_step_count += 2
            first_conf_more_than_count += 1
            if parse_is_true(first_entry):
                first_conf_more_than_pass += 1
            if parse_is_true(final_entry):
                check_until_last_pass_count += 1
            if save_entry:
                save_jsonl_file.write(json.dumps(group, ensure_ascii=False) + "\n")
        else:
            no_check_step_count += len(group)
            no_high_conf_idx_list.append((tree_idx, sampling_idx))
            # 若置信度不够，则看最后一个 entry 是否正确
            if parse_is_true(final_entry):
                until_last_pass_count += 1
    
    if save_jsonl_file:
        save_jsonl_file.close()

    rate_over_check = (first_conf_more_than_pass / first_conf_more_than_count) if first_conf_more_than_count > 0 else 0.0
    rate_until_last_check = (
        check_until_last_pass_count / first_conf_more_than_count if first_conf_more_than_count > 0 else 0.0
    )
    rate_over_total = (
        (first_conf_more_than_pass + until_last_pass_count) / total_groups if total_groups > 0 else 0.0
    )

    # ---- 输出结果 ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            "\n—— 第一个entry置信度检测 ——\n"
            f"条件: 第一个entry置信度 ≥ {threshold:.3f}\n"
        )
        f.write(f"符合条件的分组数: {first_conf_more_than_count}\n")
        f.write(f"其中通过的分组数: {first_conf_more_than_pass}\n")
        f.write(f"符合条件分组最后一个entry通过率: {rate_until_last_check:.2%}\n")
        f.write(f"总步数: {no_check_step_count + check_step_count}\n")
        f.write(f"符合条件分组步数: {check_step_count}\n")
        f.write(f"通过率（以符合数为分母）: {rate_over_check:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")

    print("\n—— 第一个entry置信度检测 ——")
    print(f"条件: 第一个entry置信度 ≥ {threshold:.3f}")
    print(f"符合条件的分组数: {first_conf_more_than_count}")
    print(f"其中通过的分组数: {first_conf_more_than_pass}")
    print(f"总步数: {no_check_step_count + check_step_count}")
    print(f"通过率（以符合数为分母）: {rate_over_check:.2%}")
    print(f"通过率（以总组数为分母）: {rate_over_total:.2%}")
    print(f"分析结果已保存至: {output_file}")

    # ✅ 返回：未满足高置信度条件的索引 + 出现数 + 通过数 + 步数
    return no_high_conf_idx_list, first_conf_more_than_count, first_conf_more_than_pass, check_step_count, len(no_high_conf_idx_list), until_last_pass_count, no_check_step_count


def check_pass_rate(log_entrys, args, output_path):
    """
    log_entrys: 是 log_entrys[tree_id][n_of_sampling][step_idx] 的结构，
                每个 group 是同一 tree_id 的 entry 列表
    """

    flattened_groups = []
    for tree_group in log_entrys:
        for sampling_group in tree_group:
            flattened_groups.append(sampling_group)
    log_entrys = flattened_groups   # 展成二维结构

    def check_first_confidence(threshold=0.85):
        check_count = 0
        pass_count = 0
        step_count = 0
        until_last_pass_count = 0

        for group in log_entrys:
            first_idx = None
            for i, e in enumerate(group):
                conf = e.get("confidence")
                if isinstance(conf,str):
                    continue
                else:
                    if isinstance(conf, dict):
                        conf = conf["self_design_conf"]
                if conf is not None and conf >= threshold:
                    first_idx = i
                    break

            if first_idx is not None:
                entry = group[first_idx]
                step_count += (first_idx + 1)
                check_count += 1
                if parse_is_true(entry):
                    pass_count += 1
            else:
                # 没找到高置信度节点，就看最后一个可判定 entry 是否通过
                for entry in reversed(group):
                    if "is_equal" in entry or "llm_judge_equal" in entry:
                        if parse_is_true(entry):
                            until_last_pass_count += 1
                        break
                step_count += len(group)

        rate_over_check = (pass_count / check_count) if check_count > 0 else 0.0
        rate_over_total = ((pass_count + until_last_pass_count) / len(log_entrys)) if len(log_entrys) > 0 else 0.0

        # 写入分析文件
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n—— 首个 confidence >= {threshold} ——\n")
            f.write(f"出现该节点的分组数: {check_count}\n")
            f.write(f"其中通过的分组数: {pass_count}\n")
            f.write(f"总步数: {step_count}\n")
            f.write(f"通过率（以出现数为分母）: {rate_over_check:.2%}\n")
            f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")
        print(f"\n—— 首个 confidence >= {threshold} ——\n")
        print(f"出现该节点的分组数: {check_count}\n")
        print(f"其中通过的分组数: {pass_count}\n")
        print(f"总步数: {step_count}\n")
        print(f"通过率（以出现数为分母）: {rate_over_check:.2%}\n")
        print(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")

    total_groups = len(log_entrys)
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


    check_first_confidence(0.85)
    check_first_confidence(0.9)

    # 新增：连续三次相同答案（首个满足处）
    three_same_check_count = 0
    three_same_pass_count  = 0
    three_same_step_count = 0
    three_same_until_last_pass_count = 0
    # ---------- 5) 连续三次答案相同（首个满足） ----------
    # 只考虑带有 'answer' 的 entry；忽略缺失该字段的
    for group in log_entrys:
        seq_answer = None
        seq_len = 0
        first_seq_idx = None  # 记录首个满足长度>=3 的答案文本
        for i,e in enumerate(group):
            if "answer" not in e:
                continue
            cur = e.get("answer")
            same_as_prev = False
            if seq_answer is None:
                seq_answer = cur
                seq_len = 1
            else:
                same_as_prev = bool(math_equal(cur, seq_answer))
                if same_as_prev:
                    seq_len += 1
                else:
                    seq_answer = cur
                    seq_len = 1

                if seq_len >= 3:
                    first_seq_idx = i  # i 即第三次的位置
                    break

        if first_seq_idx is not None:
            three_same_check_count += 1
            three_same_step_count = three_same_step_count + first_seq_idx + 1
            entry_i = group[first_seq_idx]
            if parse_is_true(entry_i):
                three_same_pass_count += 1
        else:
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        three_same_until_last_pass_count += 1
                        break
            three_same_step_count += len(group)


    three_same_rate_over_check = (three_same_pass_count / three_same_check_count) if three_same_check_count > 0 else 0.0
    three_same_rate_over_total = ((three_same_pass_count + three_same_until_last_pass_count) / total_groups) if total_groups > 0 else 0.0

    # ===== 追加写入到 analysis.txt（严格按你要的字段顺序）=====
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n—— 连续三次答案相同 ——\n")
        f.write(f"出现该节点的分组数: {three_same_check_count}\n")
        f.write(f"其中通过的分组数: {three_same_pass_count}\n")
        f.write(f"总步数: {three_same_step_count}\n")
        f.write(f"通过率（以出现数为分母）: {three_same_rate_over_check:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {three_same_rate_over_total:.2%}\n")
    print("\n—— 连续三次答案相同 ——\n")
    print(f"出现该节点的分组数: {three_same_check_count}\n")
    print(f"其中通过的分组数: {three_same_pass_count}\n")
    print(f"总步数: {three_same_step_count}\n")
    print(f"通过率（以出现数为分母）: {three_same_rate_over_check:.2%}\n")
    print(f"通过率（以总组数为分母）: {three_same_rate_over_total:.2%}\n")

            
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

    examples = load_data(data_name=args.data_name,data_path="")
    examples = examples.select(range(min(len(examples),200)))
    args = parser.parse_args()

    train_dataset_path = f"/home/lijiakun25/math-inference/llm_output/{args.data_name}/Qwen3-8b/split_steps.jsonl"

    log_entrys = []
    with open(train_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                log_entrys.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"读取错误：{e}")

    grouped = defaultdict(lambda: defaultdict(list))
    for entry in log_entrys:
        tree_id = entry.get("tree_id")
        n_of_sampling = entry.get("n_of_samplings")
        grouped[tree_id][n_of_sampling].append(entry)

    log_entrys = [
        [grouped[tree_id][n] for n in sorted(grouped[tree_id])]
        for tree_id in sorted(grouped)
    ]

    data_name = args.data_name
    model_name = args.model_name_or_path.strip("/").split("/")[-1]
    output_path = Path("llm_output")/f"{data_name}/{model_name}/conf_analysis/"
    output_path.mkdir(parents=True, exist_ok=True)

    conf_list = []
    for samples in log_entrys:
        for groups in samples:
            for item in groups:
                conf = item.get("confidence", None)
                if isinstance(conf,dict):
                    conf_list = list(conf.keys())
                    break
            if len(conf_list) != 0:
                break
        if len(conf_list) != 0:
            break

    # output_file = os.path.join(output_path, "analysis.txt")
    # with open(output_file, "w", encoding="utf-8") as f:
    #     pass
    output_file = os.path.join(output_path, "check_pipelines_analysis.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    # for conf_name in conf_list:
    #     confidence_statistics(log_entrys=log_entrys,confidence_name=conf_name,output_dir=output_path)
    # confidence_by_step_boxplot(log_entrys=log_entrys,confidence_name=conf_list[-1],output_dir=output_path)
    # check_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path)
    # recheck_errs(log_entrys=log_entrys,args=args,output_path=output_path)
    check_idx_list = None
    total_check_step_count = 0
    total_check_count = 0
    total_check_pass_count = 0
    
    
    check_pipelines = ["check_entry0_high_conf_pass_rate","check_conf_jump_pass_rate","check_first_high_conf_pass_rate"]
    os.makedirs(output_path, exist_ok=True)
    check_pipelines_output_file = os.path.join(output_path, "check_pipelines_analysis.txt")

    for check_pipeline in check_pipelines:
        if check_pipeline == "check_entry0_high_conf_pass_rate":
            check_idx_list, check_count, pass_count, check_step_count, no_check_count, until_last_pass_count, no_check_step_count = \
                check_entry0_high_conf_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path,threshold=0.75, check_idx_list=check_idx_list)
        elif check_pipeline == "check_conf_jump_pass_rate":
            check_idx_list, check_count, pass_count, check_step_count, no_check_count, until_last_pass_count, no_check_step_count = \
                check_conf_jump_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path,jump_threshold=0.3, confidence_threshold=0.8, check_idx_list=check_idx_list)
        elif check_pipeline == "check_first_high_conf_pass_rate":
            check_idx_list, check_count, pass_count, check_step_count, no_check_count, until_last_pass_count, no_check_step_count = \
                check_first_high_conf_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path,confidence_threshold=0.9,check_idx_list=check_idx_list)
        total_check_step_count += check_step_count
        total_check_count += check_count
        total_check_pass_count += pass_count
        total_step_count = total_check_step_count + no_check_step_count
        total_count = no_check_count + total_check_count
        total_pass_count = total_check_pass_count + until_last_pass_count
        with open(check_pipelines_output_file, "a", encoding="utf-8") as f:
             f.write(
            f"\n—— 分析函数: {check_pipeline} ——\n"
            f"样例数 (check_count): {check_count}\n"
            f"通过数 (pass_count): {pass_count}\n"
            f"分析步数 (check_step_count): {check_step_count}\n\n\n"
            f"未检测步数 (no_check_step_count): {no_check_step_count}\n"
            f"当前总步数 (total_step_count): {total_step_count}\n"
            f"当前总样例数 (total_count): {total_count}\n"
            f"当前总通过数 (total_pass_count): {total_pass_count}\n"
            f"当前通过率: {(total_pass_count / total_count * 100) if total_count else 0:.2f}%\n"
        )
    print(f"\n✅ 全部分析完成，结果已保存至：{check_pipelines_output_file}")

    no_check_entrys,idx_mapping = flattern_log_entrys(log_entrys=log_entrys,check_idx_list=check_idx_list)
    with open(os.path.join(output_path, "no_check_entrys.jsonl"), "w", encoding="utf-8") as f:
        for group in no_check_entrys:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
    # check_stable_conf_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path,threshold=0.15)
