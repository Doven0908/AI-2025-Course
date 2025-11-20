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
from utils.serve_vllm import completion_request
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

REQUEST = (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within Final Answer: \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )

#TODO:可能看成了一个整体才不好改的，明天尝试整体分割或者问Qwen 它认为butwait是什么？
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
    examples = datasets.load_dataset("json", data_files={"test": "/home/lijiakun25/models/datasets/math500/test_debug.jsonl"})["test"]
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

    samples = []
    for i,example in enumerate(tqdm(examples,total=len(examples),disable=False)):
        question = parse_question(example,data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        prompt = construct_prompt(question=question)
        samples.append((question,prompt,gt_ans,gt_cot))
    answers = []
    confidences = []
    total = 0
    equal = 0
    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    match_text = "....</think>.....Final Answer....."

    for i,sample in enumerate(samples):
        origin_prompt = sample[1]
        new_action = "[start]"
        prompt = origin_prompt
        answer_list = []
        confidence_list = []
        test_text = origin_prompt
        while True:
            outputs = completion_request(question=prompt,args=args,tree_id=i,model_id=model_name,stop_tokens=stop_tokens,n_sampling=1)
            new_action = outputs["texts"][0][0]
            
            test_text_input = test_text + new_action + "Can we now get the final answer? Please print Yes or No: "
            outputs = completion_request(question=test_text_input,args=args,tree_id=i,model_id=model_name,
                                         need_get_next_token_prob=True,next_token=["Yes","No","yes","no"],max_tokens=3,n_sampling=1)
            print(test_text_input + "\n\n\n")
            yes_prob = 0
            no_prob = 0
            max_n = 0
            while(max_n < 3 and yes_prob == 0 and no_prob == 0):
                output = outputs["logits"][0][max_n]
                prob_map = {tok.strip().lower(): p for tok, p in output.items()}
                yes_prob = prob_map.get("Yes", 0.0) + prob_map.get("yes", 0.0)
                no_prob  = prob_map.get("No", 0.0) + prob_map.get("no", 0.0)
                max_n += 1
            print(f"yes prob = {yes_prob},no_prob = {no_prob}")
            if yes_prob > no_prob:
                temp_prompt = prompt + new_action + "*** We can get the question's Final Answer: \\boxed"
                outputs = completion_request(question=temp_prompt,args=args,tree_id=i,model_id=model_name,
                                         need_get_next_token_prob=True,stop_tokens=["\n","\n\n"],next_token=None,max_tokens=50,n_sampling=1)
                print(outputs["texts"])
                temp_action = "*** We can get the question's Final Answer: \\boxed" + outputs["texts"][0][0]
                new_prompt = prompt + new_action + "\n\n"
                new_ans = extract_answer(temp_action,data_name=data_name)
                confidence = certainty_from_choice(outputs["logits"][0],vocab_size)
                is_equal = math_equal(new_ans,sample[2])
                answer_list.append(new_ans)
                confidence_list.append(confidence) 

                num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
                file_index = i % num_processes
                output_path = Path("llm_output") / f"{file_index}_{run_time}.jsonl"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "a", encoding="utf-8") as f:
                        log_entry = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "tree_id": i,
                            "prompt": prompt,
                            "output": new_action,
                            "gt_ans": sample[2],
                            "ans_out": temp_action,
                            "answer": new_ans,
                            "confidence": confidence,
                            "is_equal": is_equal,
                            "model_id": model_name,
                            "pre_step_answer": answer_list,
                            "pre_step_confidence": confidence_list,
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                if is_equal:
                    equal += 1
                total += 1
                print(f"question {i}, step {len(answer_list)} has finished.")
                prompt = new_prompt
                test_text = origin_prompt
            else:
                new_prompt = prompt + new_action + "\n\n"
                prompt = new_prompt
                test_text += new_action
                test_text += "\n\n"
            match = re.search(r"</think>(.*?)Final Answer", prompt, re.DOTALL)
            if match:
                print(f"end question {i}")
                break
        answers.append(answer_list)
        confidences.append(confidence_list)
    print(answers)
    print(confidences)
    print(f"total:{total}, equal:{equal}")

