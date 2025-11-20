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


def confidence_by_step_boxplot(
    log_entrys,
    confidence_name,
    output_dir,
    ylim=(0.0, 1.0),
    min_count_per_step=2,   # 每个 step 至少多少样本才纳入绘图
    draw_mean_line=True     # 是否叠加均值折线
):
    """
    从 log_entrys(三层嵌套)中抽取每个 step 的 confidence，画箱形图 + 均值趋势线。
    结构假设：
        for samples in log_entrys:
            for groups in samples:
                for item in groups:   # item 为一步，按顺序排列
                    item["confidence"] : float in [0,1]
                    item["is_equal"]   : bool (可选)
                    item["llm_judge_equal"] : list/tuple, 第一个元素 "True"/"False" (可选)
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{confidence_name}_by_step_boxplot.png"

    # 1) 按 step 索引聚合
    step_conf = defaultdict(list)  # step_idx -> [conf, conf, ...]
    for samples in log_entrys:
        for groups in samples:
            # 这一题/这一条轨迹的步骤序列
            all_true = all(parse_is_true(item) for item in groups)
            if all_true:
                continue
            has_any_true = any(parse_is_true(item) for item in groups)
            if not has_any_true:
                continue

            for step_idx, item in enumerate(groups, start=1):
                confidence_dict = item.get("confidence", None)
                if isinstance(confidence_dict, str):
                    continue
                confidence = confidence_dict[confidence_name]
                if confidence is None:
                    continue
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

def confidence_statistics(log_entrys,confidence_name,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    true_confidences = []
    false_confidences = []

    for samples in log_entrys:
        for groups in samples:
            all_true = all(parse_is_true(item) for item in groups)
            if all_true:
                continue
            has_any_true = any(parse_is_true(item) for item in groups)
            if not has_any_true:
                continue

            for item in groups:
                if "is_equal" in item:
                    confidence_dict = item.get("confidence", None)
                    if isinstance(confidence_dict, str):
                        continue
                    confidence = confidence_dict[confidence_name]                    
                    if confidence is not None:
                        if item["is_equal"] == True or item["llm_judge_equal"][0] == "True":
                            true_confidences.append(confidence)
                        elif item["is_equal"] == False:
                            false_confidences.append(confidence)

    print("✔ 包含 is_equal 字段的记录总数:", len(true_confidences) + len(false_confidences))

    file_path = os.path.join(output_dir, "confidence_analysis.txt")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"✔ {confidence_name} is_equal == True:\n")
        f.write(f"  数量: {len(true_confidences)}\n")
        f.write(f"  平均 {confidence_name}: {mean(true_confidences) if true_confidences else '无数据'}\n")

        f.write(f"✔ {confidence_name} is_equal == False:\n")
        f.write(f"  数量: {len(false_confidences)}\n")
        f.write(f"  平均 {confidence_name}: {mean(false_confidences) if false_confidences else '无数据'}\n")
        f.write("\n")  # 可选：添加一个换行方便区分

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
        plt.savefig(f"{output_dir}/{confidence_name}_KDE.png", dpi=300)
        plt.close()
    else:
        print("⚠ 样本不足，无法绘制 KDE 图(至少每类2个数据)")
    if len(true_confidences) >= 2 and len(false_confidences) >= 2:
        plt.figure(figsize=(6, 4))

        bins = np.linspace(0.5, 1.0, 25)
        plt.hist(
            true_confidences, bins=bins, density=True, alpha=0.6,
            color="green", label="Correct", edgecolor="white", linewidth=0.5
        )
        plt.hist(
            false_confidences, bins=bins, density=True, alpha=0.6,
            color="red", label="Incorrect", edgecolor="white", linewidth=0.5
        )
        plt.xlabel("Confidence Score")
        plt.ylabel("Density")
        plt.title("Confidence Distribution (Correct vs Incorrect)")
        plt.legend(frameon=True, framealpha=0.9, loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/{confidence_name}_histogram.png", dpi=300)
        plt.close()
    else:
        print("⚠ 样本不足，无法绘制图 (至少每类2个数据)")
    


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

def check_conf_jump_pass_rate(log_entrys, args, output_path, jump_threshold=0.2, confidence_threshold=0.7, save_entry=True):
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
    flattened_groups = []
    for tree_group in log_entrys:
        for sampling_group in tree_group:
            flattened_groups.append(sampling_group)
    log_entrys = flattened_groups

    total_groups = len(log_entrys)
    found_count = 0
    pass_count = 0
    step_count = 0
    until_last_pass_count = 0


    os.makedirs(output_path, exist_ok=True)
    save_jsonl_path = os.path.join(output_path, "conf_jump_cases.jsonl")
    save_jsonl_file = open(save_jsonl_path, "w", encoding="utf-8")

    for group in log_entrys:
        if len(group) < 2:
            # 只有一个entry无法比较
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            step_count += len(group)
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
            found_count += 1
            entry = group[jump_idx]
            step_count += jump_idx + 1
            if parse_is_true(entry):
                pass_count += 1
            if save_entry:
                save_jsonl_file.write(json.dumps(group, ensure_ascii=False) + "\n")
        else:
            # 没有跃迁点，则看最后一个可判定entry
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            step_count += len(group)

    rate_over_found = (pass_count / found_count) if found_count > 0 else 0.0
    rate_over_total = ((pass_count + until_last_pass_count) / total_groups) if total_groups > 0 else 0.0

    # ---- 输出结果 ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            "\n—— 置信度跃迁检测 ——\n"
            f"条件: 后一个entry的置信度比前一个提升 ≥ {jump_threshold:.2f} 且后一个置信度 ≥ {confidence_threshold:.2f}\n"
        )
        f.write(f"出现跃迁的分组数: {found_count}\n")
        f.write(f"其中通过的分组数: {pass_count}\n")
        f.write(f"总步数: {step_count}\n")
        f.write(f"通过率（以出现数为分母）: {rate_over_found:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")
    print("\n—— 置信度跃迁检测 ——")
    print(f"条件: 后一个entry的置信度比前一个提升 ≥ {jump_threshold:.2f} 且后一个置信度 ≥ {confidence_threshold:.2f}")
    print(f"出现跃迁的分组数: {found_count}")
    print(f"其中通过的分组数: {pass_count}")
    print(f"总步数: {step_count}")
    print(f"通过率（以出现数为分母）: {rate_over_found:.2%}")
    print(f"通过率（以总组数为分母）: {rate_over_total:.2%}")
    print(f"分析结果已保存至: {output_file}")

def check_stable_conf_pass_rate(log_entrys, args, output_path, threshold=0.1):
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

    # 展平
    flattened_groups = []
    for tree_group in log_entrys:
        for sampling_group in tree_group:
            flattened_groups.append(sampling_group)
    log_entrys = flattened_groups

    total_groups = len(log_entrys)
    found_count = 0
    pass_count = 0
    step_count = 0
    until_last_pass_count = 0

    for group in log_entrys:
        if len(group) < 4:
            # 不足4个，不可能形成稳定段
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            step_count += len(group)
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
            found_count += 1
            entry = group[stable_idx]
            step_count += stable_idx + 1
            if parse_is_true(entry):
                pass_count += 1
        else:
            # 没有稳定段，取最后一个可判定entry
            for entry in reversed(group):
                if "is_equal" in entry or "llm_judge_equal" in entry:
                    if parse_is_true(entry):
                        until_last_pass_count += 1
                    break
            step_count += len(group)

    rate_over_found = (pass_count / found_count) if found_count > 0 else 0.0
    rate_over_total = ((pass_count + until_last_pass_count) / total_groups) if total_groups > 0 else 0.0

    # ---- 输出结果 ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n—— 连续4步置信度稳定（阈值 <= {:.3f}） ——\n".format(threshold))
        f.write(f"出现稳定段的分组数: {found_count}\n")
        f.write(f"其中通过的分组数: {pass_count}\n")
        f.write(f"总步数: {step_count}\n")
        f.write(f"通过率（以出现数为分母）: {rate_over_found:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")

    print("\n—— 连续4步置信度稳定（阈值 <= {:.3f}） ——".format(threshold))
    print(f"出现稳定段的分组数: {found_count}")
    print(f"其中通过的分组数: {pass_count}")
    print(f"总步数: {step_count}")
    print(f"通过率（以出现数为分母）: {rate_over_found:.2%}")
    print(f"通过率（以总组数为分母）: {rate_over_total:.2%}")
    print(f"分析结果已保存至: {output_file}")

def check_first_high_conf_pass_rate(log_entrys, args, output_path, threshold=0.85):
    """
    功能：
        对每个样例(group),对第一个entry,若置信度 >= 0.85,
        将其 answer 与最终答案(最后一个可判定 entry)对比,
        统计符合数量、总步数和通过率。
        若该样例中第一个entry不符合,则取最后一个答案进行对比。

    参数：
        log_entrys: 是 log_entrys[tree_id][n_of_sampling][step_idx] 的结构，
                    每个 group 是同一 tree_id 的 entry 列表。
        args: 预留参数（未使用）。
        output_path: 输出分析结果文件夹路径。
    """
    flattened_groups = []
    for tree_group in log_entrys:
        for sampling_group in tree_group:
            flattened_groups.append(sampling_group)
    log_entrys = flattened_groups

    total_groups = len(log_entrys)
    matched_count = 0      # 符合的样例数
    total_steps = 0        # 总步数
    first_conf_more_than_count = 0
    first_conf_more_than_pass = 0

    for group in log_entrys:
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
            total_steps += 1
            first_conf_more_than_count += 1
            if parse_is_true(first_entry):
                matched_count += 1
                first_conf_more_than_pass += 1
        else:
            total_steps += len(group)
            # 若置信度不够，则看最后一个 entry 是否正确
            if parse_is_true(final_entry):
                matched_count += 1

    pass_rate = matched_count / total_groups if total_groups > 0 else 0.0
    first_conf_more_than_pass_rate = first_conf_more_than_pass / first_conf_more_than_count if first_conf_more_than_count > 0 else 0.0

    # ---- 输出结果 ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "analysis.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n—— 第一个entry置信度 >= {:.3f} ——".format(threshold))
        f.write(f"总分组数量: {total_groups}\n")
        f.write(f"符合分组的数量：{first_conf_more_than_count}\n")
        f.write(f"符合分组的通过数量：{first_conf_more_than_pass}\n")
        f.write(f"符合分组的通过率：{first_conf_more_than_pass_rate}\n")
        f.write(f"正确数量: {matched_count}\n")
        f.write(f"总步数: {total_steps}\n")
        f.write(f"通过率（以总组数为分母）: {pass_rate:.2%}\n")

    print("\n—— 第一个entry置信度 >= {:.3f} ——".format(threshold))
    print(f"总分组数量: {total_groups}")
    print(f"符合分组的数量：{first_conf_more_than_count}")
    print(f"符合分组的通过数量：{first_conf_more_than_pass}")
    print(f"符合分组的通过率：{first_conf_more_than_pass_rate}")
    print(f"正确数量: {matched_count}")
    print(f"总步数: {total_steps}")
    print(f"通过率（以总组数为分母）: {pass_rate:.2%}")
    print(f"分析结果已保存至: {output_file}")


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
        found_count = 0
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
                found_count += 1
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

        rate_over_found = (pass_count / found_count) if found_count > 0 else 0.0
        rate_over_total = ((pass_count + until_last_pass_count) / len(log_entrys)) if len(log_entrys) > 0 else 0.0

        # 写入分析文件
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n—— 首个 confidence >= {threshold} ——\n")
            f.write(f"出现该节点的分组数: {found_count}\n")
            f.write(f"其中通过的分组数: {pass_count}\n")
            f.write(f"总步数: {step_count}\n")
            f.write(f"通过率（以出现数为分母）: {rate_over_found:.2%}\n")
            f.write(f"通过率（以总组数为分母）: {rate_over_total:.2%}\n")
        print(f"\n—— 首个 confidence >= {threshold} ——\n")
        print(f"出现该节点的分组数: {found_count}\n")
        print(f"其中通过的分组数: {pass_count}\n")
        print(f"总步数: {step_count}\n")
        print(f"通过率（以出现数为分母）: {rate_over_found:.2%}\n")
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
    three_same_found_count = 0
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
            three_same_found_count += 1
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


    three_same_rate_over_found = (three_same_pass_count / three_same_found_count) if three_same_found_count > 0 else 0.0
    three_same_rate_over_total = ((three_same_pass_count + three_same_until_last_pass_count) / total_groups) if total_groups > 0 else 0.0

    # ===== 追加写入到 analysis.txt（严格按你要的字段顺序）=====
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n—— 连续三次答案相同 ——\n")
        f.write(f"出现该节点的分组数: {three_same_found_count}\n")
        f.write(f"其中通过的分组数: {three_same_pass_count}\n")
        f.write(f"总步数: {three_same_step_count}\n")
        f.write(f"通过率（以出现数为分母）: {three_same_rate_over_found:.2%}\n")
        f.write(f"通过率（以总组数为分母）: {three_same_rate_over_total:.2%}\n")
    print("\n—— 连续三次答案相同 ——\n")
    print(f"出现该节点的分组数: {three_same_found_count}\n")
    print(f"其中通过的分组数: {three_same_pass_count}\n")
    print(f"总步数: {three_same_step_count}\n")
    print(f"通过率（以出现数为分母）: {three_same_rate_over_found:.2%}\n")
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
    
    # for conf_name in conf_list:
    #     confidence_statistics(log_entrys=log_entrys,confidence_name=conf_name,output_dir=output_path)
    # confidence_by_step_boxplot(log_entrys=log_entrys,confidence_name=conf_list[-1],output_dir=output_path)
    check_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path)
    # recheck_errs(log_entrys=log_entrys,args=args,output_path=output_path)
    check_first_high_conf_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path,threshold=0.75)
    check_stable_conf_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path,threshold=0.15)
    check_conf_jump_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path,jump_threshold=0.3, confidence_threshold=0.8)
















    # # 保存到 output_path/high_conf_err.json
    # high_conf_err_path = output_path / "high_conf_err.jsonl"
    # with open(high_conf_err_path, "w", encoding="utf-8") as f:
    #     for item in high_conf_errs:
    #         json.dump(item, f, ensure_ascii=False)
    #         f.write("\n")

    # print(f"✔ 已保存 {len(high_conf_errs)} 条高置信度错误到: {high_conf_err_path}")





    # equal = 0
    # total = 0
    # for i,log_entry in enumerate(log_entrys):
    #     origin_prompt = log_entry["question"] + "\nSolution:\n"
    #     answer_list = []
    #     confidence_list = []
    #     test_text = origin_prompt
    #     for j,step in enumerate(log_entry["split_output"]):
    #         new_action = step["content"]

    #         test_text_input = test_text + new_action + "Can we now get the final answer? Answer with exactly one token: \"Yes\" or \"No\". Do not add any other text.\n/no_think"
    #         ask_request = [{"role": "user", "content": test_text_input}]
    #         outputs = chat_completion_request(question=ask_request,args=args,tree_id=i,model_id=model_name,
    #                                      need_get_next_token_prob=True,next_token=None,max_tokens=5,n_sampling=1) # ["Yes","No","yes","no"]
    #         # print(ask_request,"\n\n\n")
    #         # print(outputs["logits"])
    #         # print(outputs["texts"][0])
    #         yes_prob = 0
    #         no_prob = 0
    #         max_n = 0
    #         while(max_n < 5 and yes_prob + no_prob < 0.1):
    #             output = outputs["logits"][0][max_n]
    #             prob_map = {tok: p for tok, p in output.items()}
    #             yes_prob = prob_map.get("yes", 0.0) + prob_map.get("Yes", 0.0)
    #             no_prob  = prob_map.get("no", 0.0) + prob_map.get("No", 0.0)
    #             max_n += 1
    #         print(f"yes prob = {yes_prob},no_prob = {no_prob}")
    #         if yes_prob > no_prob:
    #             temp_prompt = test_text + new_action + "*** We can get the question's Final Answer: \\boxed"
    #             outputs = completion_request(question=temp_prompt,args=args,tree_id=i,model_id=model_name,
    #                                      need_get_next_token_prob=True,stop_tokens=["\n","\n\n"],next_token=None,max_tokens=50,n_sampling=1)
    #             print(outputs["texts"])
    #             temp_action = "*** We can get the question's Final Answer: \\boxed" + outputs["texts"][0][0]
    #             new_prompt = test_text + new_action + "\n\n"
    #             new_ans = extract_answer(temp_action,data_name=data_name)
    #             confidence = certainty_from_choice(outputs["logits"][0],vocab_size)
    #             is_equal = math_equal(new_ans,log_entry["gt_ans"])
    #             answer_list.append(new_ans)
    #             confidence_list.append(confidence) 

    #             num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
    #             file_index = i % num_processes
    #             output_path = Path("llm_output") / f"{file_index}_{run_time}.jsonl"
    #             output_path.parent.mkdir(parents=True, exist_ok=True)

    #             with open(output_path, "a", encoding="utf-8") as f:
    #                 row = {
    #                         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    #                         "tree_id": i,
    #                         "cur_node": test_text,
    #                         "output": new_action,
    #                         "gt_ans": log_entry["gt_ans"],
    #                         "ans_out": temp_action,
    #                         "answer": new_ans,
    #                         "confidence": confidence,
    #                         "is_equal": is_equal,
    #                         "model_id": model_name,
    #                         "pre_step_answer": answer_list,
    #                         "pre_step_confidence": confidence_list,
    #                     }
    #                 f.write(json.dumps(row, ensure_ascii=False) + "\n")
    #             if is_equal:
    #                 equal += 1
    #             total += 1
    #             print(f"question {i}, step {len(answer_list)} has finished.")
    #             test_text = origin_prompt
    #         else:
    #             test_text += new_action
    #             test_text += "\n\n"

    # print(f"total:{total}, equal:{equal}")

