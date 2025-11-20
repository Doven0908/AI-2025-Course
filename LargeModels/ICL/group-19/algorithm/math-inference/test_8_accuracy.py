from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from itertools import repeat
from sympy import isprime
import math
import datasets
from utils.serve_vllm import chat_completion_request
import argparse
from pathlib import Path
import time
import json
from collections import defaultdict
from test_tool.olympiad_judge import AutoScoringJudge

from collections import defaultdict

def caculate_dict_llm(samples):
    """
    只统计: llm_judge 相对 math_verify_is_equal 的四格 + 预测 True/False 数量
    样本结构（本框架下）:
      (idx, verify_prompt, predict_ans, ground_truth, is_equal, judge, yes_prob, no_prob, chat_resp)
    """
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in {"true", "yes", "1"}: return True
            if xs in {"false", "no", "0"}: return False
            if xs in {"error", "nan", "none", ""}: return None
        return None

    cells = defaultdict(int)
    totals = {"True": 0, "False": 0}
    err = 0

    for sample in samples:
        if len(sample) < 9:  # 保护
            continue
        ref = to_bool(sample[4])      # math_verify_is_equal
        llm = to_bool(sample[5])      # llm_judge: "True"/"False"/"Error"
        if ref is None:               # 参考值无法解析则跳过
            continue
        if llm is None:
            err += 1
            continue

        if llm: totals["True"] += 1
        else:   totals["False"] += 1

        if ref and llm:
            cells["ref=True & pred=True"] += 1
        elif ref and not llm:
            cells["ref=True & pred=False"] += 1
        elif (not ref) and llm:
            cells["ref=False & pred=True"] += 1
        else:
            cells["ref=False & pred=False"] += 1

    print("---- LLM 判定 vs. math_verify_is_equal ----")
    print(f"ref=True  & pred=True : {cells['ref=True & pred=True']}")
    print(f"ref=True  & pred=False: {cells['ref=True & pred=False']}")
    print(f"ref=False & pred=True : {cells['ref=False & pred=True']}")
    print(f"ref=False & pred=False: {cells['ref=False & pred=False']}")
    print(f"pred=True  数量: {totals['True']}")
    print(f"pred=False 数量: {totals['False']}")
    print(f"跳过/错误(不计入四格): {err}")


REQUEST='''
You are given a math question, a model's boxed answer, and the ground-truth.
If the boxed answer is mathematically correct, output True. Otherwise, output False.
Only compare the final answer. Reply with one word: True or False.
Example Input:
Question: The set of points \\((x,y,z)\\) that satisfy  
\\[2x = 3y = -z\\]  
is a line. The set of points \\((x,y,z)\\) that satisfy  
\\[6x = -y = -4z\\]  
is another line. Find the angle between these lines, in degrees.

Response:  
*** We can get the question's Final Answer: \\(\\boxed\\{90\\}\\).  

ground_truth: \\(90^\\circ\\)
Output:
True  

Example Input:
Question: 
Xenia and Sergey play the following game. Xenia thinks of a positive integer $N$ not exceeding 5000. Then she fixes 20 distinct positive integers $a_{1}, a_{2}, \ldots, a_{20}$ such that, for each $k=1,2, \ldots, 20$, the numbers $N$ and $a_{k}$ are congruent modulo $k$. By a move, Sergey tells Xenia a set $S$ of positive integers not exceeding 20 , and she tells him back the set $\left\{a_{k}: k \in S\right\}$ without spelling out which number corresponds to which index. How many moves does Sergey need to determine for sure the number Xenia thought of?

Response:  
*** We can get the question's Final Answer: \\boxed{1} ***$$

ground_truth:  
2
Output:
False

'''
REQUEST_INPUT = "Now process the following new input:\nQuestion: {question}\n\nResponse: {partial_response} \n\nground_truth: {ground_truth}/no_think"


def llm_judge_once(question: str,
                   ground_truth: str,
                   partial_response: str,
                   args,
                   *,
                   max_tokens: int = 10,
                   prob_threshold: float = 0.3,
                   tree_id: int = 0):
    """
    依据 predict_ans 与 ground_truth 调用模型做 Yes/No 判定，并计算 yes/no 概率。
    返回: (judge, yes_prob, no_prob, chat_resp)
      - judge: "True" / "False" / "Error"
      - yes_prob/no_prob: 汇总到达阈值前的概率
      - chat_resp: 原始响应(含 logits),用于调试/记录
    """
    # 依赖你上面定义的全局 REQUEST / REQUEST_INPUT
    verify_text = REQUEST + REQUEST_INPUT.format(question=question,
                                                 ground_truth=ground_truth,
                                                 partial_response=partial_response)
    messages = [{"role": "user", "content": verify_text}]

    # 组装请求参数（保持与你项目里一致）
    kwargs = dict(
        question=messages,
        args=args,
        tree_id=tree_id,                 # 若不需要可设为 0 或省略
        n_sampling=1,
        timeout_sec=3600,
        need_get_next_token_prob=True,
        next_token=["True","true","False","false"," True"," true"," False"," false"],
        max_tokens=max_tokens,
        # use_chat=False, stop_tokens=None, next_token=["Yes","No"], ...
    )

    try:
        chat_resp = chat_completion_request(**kwargs)
    except Exception as e:
        # 调用失败一律视作 Error
        return "Error", 0.0, 0.0, {"error": str(e)}

    # 读取 logits 并累计 yes/no 概率
    try:
        yes_prob = 0.0
        no_prob = 0.0
        k = 0
        # 有些实现 logits 结构：logits[采样索引][步索引] -> {token: prob}
        steps = chat_resp["logits"][0] if "logits" in chat_resp else []
        limit = min(max_tokens, len(steps))

        while (k < limit) and (yes_prob + no_prob < prob_threshold):
            prob_map = dict(steps[k])
            yes_prob += prob_map.get("True", 0.0) + prob_map.get("true", 0.0) + prob_map.get(" True", 0.0) + prob_map.get(" true", 0.0)
            no_prob  += prob_map.get("False", 0.0)  + prob_map.get(" False", 0.0) + prob_map.get("false", 0.0)  + prob_map.get(" false", 0.0)
            k += 1

        if (yes_prob + no_prob) > prob_threshold:
            judge = "True" if yes_prob > no_prob else "False"
        else:
            judge = "Error"

        return judge, float(yes_prob), float(no_prob), chat_resp["texts"][0][0]

    except Exception as e:
        # 解析失败也视作 Error
        return "Error", 0.0, 0.0, {"error": f"parse_logits_failed: {e}"}


def _run_with_kwargs(kwargs):
    # 注意：必须是顶层可导入函数，若改用进程池要可 picklable
    return llm_judge_once(**kwargs)

if __name__ == "__main__":
    examples = datasets.load_dataset("json", data_files={"test": "/home/lijiakun25/math-inference/llm_output/olympiadbench_maths/Qwen3-8b/equal_judge.jsonl"})["test"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=1)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_generate_workers", type=int, default=16)
    parser.add_argument("--data_name",default="olympiadbench_maths",type=str)
    parser.add_argument("--model_name_or_path",default="/home/lijiakun25/models/Qwen3-8b",type=str)
    args = parser.parse_args()

    model_name = args.model_name_or_path.strip("/").split("/")[-1]
    data_name = args.data_name

    output_path = Path("llm_output") / f"{data_name}/{model_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "llm_judge.jsonl"


    samples = []
    
    for i,example in enumerate(tqdm(examples,total=len(examples),disable=False)):
        predict_ans = example["predict_ans"]
        ground_truth = example["ground_truth"]
        idx = example["idx"]
        is_equal = example["is_equal"]
        verify_text = REQUEST + REQUEST_INPUT.format(predict_ans=predict_ans,ground_truth=ground_truth)
        verify_prompt = [{"role": "user", "content": verify_text}]
        samples.append((idx,verify_prompt,predict_ans,ground_truth,is_equal))

    sample_td = tqdm(total=len(samples), desc='Example Generation')
    task_params = [
        dict(
            predict_ans=sample[2],
            ground_truth=sample[3],
            args=args,
            max_tokens=10,
            prob_threshold=0.3,
            tree_id=sample[0]
        )
        for sample in samples
    ]

    outputs = []
    with ThreadPoolExecutor(max_workers=args.max_generate_workers * args.num_of_models) as executor:
        for res in executor.map(_run_with_kwargs, task_params):
            outputs.append(res)   # (judge, yes_prob, no_prob, chat_resp)
            sample_td.update(1)
    sample_td.close()

    assert len(samples) == len(outputs)

    samples = [ samples[i] + outputs[i] for i in range(len(samples))]

    # 写文件（无 auto_judge_result 字段）
    for sample in samples:
        with open(output_file, "a", encoding="utf-8") as f:
            log_entry = {
                "idx": sample[0],
                "predict_ans": sample[2],
                "ground_truth": sample[3],
                "math_verify_is_equal": sample[4],
                "llm_judge": sample[5],        # "True"/"False"/"Error"
                "yes_prob": sample[6],
                "no_prob": sample[7],
                "prompt": sample[1],           # 记录完整 prompt，便于复现
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\\n")

    # 只统计 LLM 的四格
    caculate_dict_llm(samples)