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
from collections import defaultdict

def confidence_statistics(log_entrys,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    true_confidences = []
    false_confidences = []

    for samples in log_entrys:
        for item in samples:
            if "is_equal" in item:
                confidence = item.get("confidence", None)
                if confidence is not None:
                    if item["is_equal"] == True or item["llm_judge_equal"][0] == "True":
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

def certainty_from_choice(top_probs):
    H_norm = []
    for top in top_probs: # top 是 {token: logp, ....}
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

from test_8_accuracy import llm_judge_once
def para_gen(err_record: dict, args):
    target_len = max(len(err_record["gt_ans"]), 30)  # 目标长度
    all_text = ""
    all_logits = []
    next_token = None
    completion_tokens = 0

    while True:
        # 关键：把已生成的 all_text 拼回到 prompt
        question = err_record["recheck_equal_prompt"] + all_text

        comp_resp = completion_request(
            question=question,
            args=args,
            tree_id=err_record["log_index"],  # 保留你现有变量，如果外层传入更稳妥
            model_id=args.model_name_or_path,
            need_get_next_token_prob=True,
            stop_tokens=["\n", "\n\n",".","***"],
            next_token=next_token,
            max_tokens=25,
            n_sampling=1
        )

        seg_text = comp_resp["texts"][0][0]
        all_text += seg_text
        all_logits.extend(comp_resp["logits"][0])

        # 触发续写的判定：本轮生成 token 数
        gen_tokens = comp_resp["usage"].completion_tokens
        if gen_tokens is None:
            # 退化：用 logits 的长度估计
            gen_tokens = len(comp_resp["logits"][0]) if comp_resp.get("logits") else 0
        completion_tokens += gen_tokens
        # 终止条件
        if (
            gen_tokens < 24               # 未达到“再续写”的阈值
            or completion_tokens >= target_len      
        ):
            break

    temp_action = "*** We can get the question's Final Answer: \\boxed" + all_text
    confidence = certainty_from_choice(all_logits)

    llm_judge_equal = llm_judge_once(
        question=err_record["question"],
        ground_truth=err_record["gt_ans"],
        partial_response=temp_action,
        args=args,
        tree_id=err_record["log_index"]
    )
    return temp_action, confidence, llm_judge_equal



def recheck_errs(log_entrys,args,output_path):
    errs = []
    for i,samples in enumerate(log_entrys):
        for j,item in enumerate(samples):
            if (
                item.get("is_equal") == False
            ):
                err_record = {
                "log_index": i,
                "sample_index": j,  
                **item
                }
                errs.append(err_record)

    for i, err_record in enumerate(errs):
        get_answer_prompt =  err_record["actions_from_last_endnode"] + "*** We can get the question's Final Answer: \\boxed"
        err_record["recheck_equal_prompt"] = get_answer_prompt
    
    re_correct = 0
    re_error = 0
    total_false_conf = 0
    total_true_conf = 0
    max_workers = max(1, int(args.max_generate_workers) * int(args.num_of_models))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(para_gen, err_record, args): idx
                for idx, err_record in enumerate(errs)}

        with tqdm(total=len(futures), desc="Rejudging", dynamic_ncols=True) as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    temp_action, confidence, llm_rejudge_equal = fut.result()

                    errs[idx]["rejudge_output"] = temp_action
                    errs[idx]["rejudge_conf"] = confidence
                    errs[idx]["rejudge_equal"] = llm_rejudge_equal
                    if llm_rejudge_equal[0] == "True":
                        re_correct += 1
                        total_true_conf += confidence
                    elif llm_rejudge_equal[0] == "False":
                        re_error += 1
                        total_false_conf += confidence
                    pbar.update(1)  # 手动递增
                    
                    log_entrys[errs[idx]["log_index"]][errs[idx]["sample_index"]]["temp_action"] = temp_action
                    log_entrys[errs[idx]["log_index"]][errs[idx]["sample_index"]]["answer"] = extract_answer(temp_action)
                    log_entrys[errs[idx]["log_index"]][errs[idx]["sample_index"]]["confidence"] = confidence
                    log_entrys[errs[idx]["log_index"]][errs[idx]["sample_index"]]["llm_judge_equal"] = llm_rejudge_equal

    print(f"recorrect the number of the answer: {re_correct}, true confidence is {total_true_conf / re_correct}\n")
    print(f"false the number of the answer: {re_error}, true confidence is {total_false_conf / re_error}\n")
    

    recheck_step = output_path / "recheck_step.jsonl"
    with open(recheck_step, "w", encoding="utf-8") as f:
        for samples in log_entrys:       # log_entrys 是二维 list
            for item in samples:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    print(f"✔ 已保存 {sum(len(s) for s in log_entrys)} 条更新后的记录到: {recheck_step}")

    # 保存到 output_path/_err.jsonl
    err_path = output_path / "err.jsonl"
    with open(err_path, "w", encoding="utf-8") as f:
        for item in errs:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✔ 已保存 {len(errs)} 条高置信度错误到: {err_path}")
    confidence_statistics(log_entrys=log_entrys,output_dir=output_path)


def check_pass_rate(log_entrys, args, output_path):
    """
    log_entrys: 已经是 [group1, group2, ...] 的结构，
                每个 group 是同一 tree_id 的 entry 列表
    """

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
    
    def check_first_confidence(threshold=0.85):
        found_count = 0
        pass_count = 0
        step_count = 0
        until_last_pass_count = 0

        for group in log_entrys:
            first_idx = None
            for i, e in enumerate(group):
                conf = e.get("confidence")
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

    for group in log_entrys:
        # 1) 是否曾经出现过 True
        has_true = any(parse_is_true(entry) for entry in group)

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

    grouped = defaultdict(list)
    for entry in log_entrys:
        tree_id = entry.get("tree_id")
        grouped[tree_id].append(entry)
    log_entrys = [grouped[key] for key in sorted(grouped)]

    data_name = args.data_name
    model_name = args.model_name_or_path.strip("/").split("/")[-1]
    output_path = Path("llm_output") / f"{data_name}/{model_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    check_pass_rate(log_entrys=log_entrys,args=args,output_path=output_path)
    # confidence_statistics(log_entrys=log_entrys,output_dir=output_path)

    # recheck_errs(log_entrys=log_entrys,args=args,output_path=output_path)

















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

