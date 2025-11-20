# 这个test的目的是看 Can we get + extraction + diction的组合作用
# import multiprocessing

# def funcs(param):
#     return param * param

# if __name__ == "__main__":
#     testlist = [1,2,3,4,5,6,7]
#     with multiprocessing.Pool(processes=2) as np:
#         result1 = np.map_async(funcs,testlist)
#         result2 = np.map(funcs,testlist)
#     print(result1)
#     print(result2)

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

def extract_json_objects(text):
        # 匹配 {"step": 数字, "content": "..."}，允许换行内容（非贪婪匹配）
        pattern = r'\{"step":\s*\d+,\s*"content":\s*"(.*?)"\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)

        result = []
        for i, content in enumerate(matches):
            json_str = json.dumps({"step": i + 1, "content": content})
            try:
                obj = json.loads(json_str)
                result.append(obj)
            except json.JSONDecodeError as e:
                print(f"解析失败: {e}，内容为：{json_str[:50]}...")
        return result

def data_processing(train_dataset_path):
    log_entrys = []
    with open(train_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                log_entrys.append(row)
            except json.JSONDecodeError as e:
                print(f"读取错误：{e}")
    for log_entry in log_entrys:
        if "split_output" in log_entry and isinstance(log_entry["split_output"], str):
            split_output_str = log_entry["split_output"]
            parsed_output = extract_json_objects(split_output_str)
            log_entry["split_output"] = parsed_output
            if parsed_output:
                ans = ""
                for step in parsed_output:
                    ans += step["content"]
                distance = Levenshtein.distance(ans, log_entry["output"])
                similarity = 1 - distance / max(len(ans), len(log_entry["output"]))  # 转成 0~1 相似度
                print(f"相似度：{similarity:.2f}")
        if similarity == 1:
            ans = ""
            for step in parsed_output:
                ans += step
                ans += "<END_METHOD>"

def certainty_from_choice(top_probs, vocab_size:int):
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


if __name__ == "__main__":
    examples = datasets.load_dataset("json", data_files={"test": "/home/lijiakun25/models/datasets/math500/train.jsonl"})["test"]
    train_dataset_path = "/home/lijiakun25/math-inference/llm_output/0_to_99_split.jsonl"
    
    model_name = "/home/lijiakun25/models/Qwen3-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size of the tokenizer: {vocab_size}")
    data_name = "math"
    stop_tokens = ["\n\n"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=1)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()


    log_entrys = []
    with open(train_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                log_entrys.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"读取错误：{e}")

    for log_entry in log_entrys:
        if "split_output" in log_entry and isinstance(log_entry["split_output"], str):
            split_output_str = log_entry["split_output"]
            parsed_output = extract_json_objects(split_output_str)
            log_entry["split_output"] = parsed_output

    for i,(example, _) in enumerate(tqdm(zip(examples, log_entrys),total=min(len(examples),len(log_entrys)),disable=False)):
        question = parse_question(example,data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        log_entrys[i]["gt_ans"] = gt_ans
        log_entrys[i]["question"] = question

    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # match_text = "....</think>.....Final Answer....."

    equal = 0
    total = 0
    for i,log_entry in enumerate(log_entrys):
        origin_prompt = log_entry["question"] + "\nSolution:\n"
        answer_list = []
        confidence_list = []
        test_text = origin_prompt
        for j,step in enumerate(log_entry["split_output"]):
            new_action = step["content"]

            test_text_input = test_text + new_action + "Can we now get the final answer? Answer with exactly one token: \"Yes\" or \"No\". Do not add any other text.\n/no_think"
            ask_request = [{"role": "user", "content": test_text_input}]
            outputs = chat_completion_request(question=ask_request,args=args,tree_id=i,model_id=model_name,
                                         need_get_next_token_prob=True,next_token=None,max_tokens=5,n_sampling=1) # ["Yes","No","yes","no"]
            # print(ask_request,"\n\n\n")
            # print(outputs["logits"])
            # print(outputs["texts"][0])
            yes_prob = 0
            no_prob = 0
            max_n = 0
            while(max_n < 5 and yes_prob + no_prob < 0.1):
                output = outputs["logits"][0][max_n]
                prob_map = {tok: p for tok, p in output.items()}
                yes_prob = prob_map.get("yes", 0.0) + prob_map.get("Yes", 0.0)
                no_prob  = prob_map.get("no", 0.0) + prob_map.get("No", 0.0)
                max_n += 1
            print(f"yes prob = {yes_prob},no_prob = {no_prob}")
            if yes_prob > no_prob:
                temp_prompt = test_text + new_action + "*** We can get the question's Final Answer: \\boxed"
                outputs = completion_request(question=temp_prompt,args=args,tree_id=i,model_id=model_name,
                                         need_get_next_token_prob=True,stop_tokens=["\n","\n\n"],next_token=None,max_tokens=50,n_sampling=1)
                print(outputs["texts"])
                temp_action = "*** We can get the question's Final Answer: \\boxed" + outputs["texts"][0][0]
                new_prompt = test_text + new_action + "\n\n"
                new_ans = extract_answer(temp_action,data_name=data_name)
                confidence = certainty_from_choice(outputs["logits"][0],vocab_size)
                is_equal = math_equal(new_ans,log_entry["gt_ans"])
                answer_list.append(new_ans)
                confidence_list.append(confidence) 

                num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
                file_index = i % num_processes
                output_path = Path("llm_output") / f"{file_index}_{run_time}.jsonl"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "a", encoding="utf-8") as f:
                    row = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "tree_id": i,
                            "cur_node": test_text,
                            "output": new_action,
                            "gt_ans": log_entry["gt_ans"],
                            "ans_out": temp_action,
                            "answer": new_ans,
                            "confidence": confidence,
                            "is_equal": is_equal,
                            "model_id": model_name,
                            "pre_step_answer": answer_list,
                            "pre_step_confidence": confidence_list,
                        }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if is_equal:
                    equal += 1
                total += 1
                print(f"question {i}, step {len(answer_list)} has finished.")
                test_text = origin_prompt
            else:
                test_text += new_action
                test_text += "\n\n"

    print(f"total:{total}, equal:{equal}")

