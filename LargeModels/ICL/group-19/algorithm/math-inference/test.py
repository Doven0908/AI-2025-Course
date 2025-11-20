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

REQUEST0 =  (
        '''<|im_start|>user\n{input}\nPlease show concise key steps (not full internal reasoning). After each omplete method, print <END_METHOD> on the next line even if you continue reasoning later or verify again. After all methods, print Final Answer: \\boxed{{...}}. the format should be like: 
            Question <think> .... <END_METHOD> Let me check....<END_METHOD> Wait, let me make sure there's no other case where the equation might have exactly one solution....Thus, the product is {{your answer}}.<END_METHOD>I think that's it. Let me just verify with an example....So the product is ....<END_METHOD></think> ...... n has exactly one real solution is:... <END_METHOD> Final Answer: \\boxed{{-14}}<|im_end|>'''
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )

REQUEST =  (
        '''<|im_start|>user\n{input}\nPlease show concise key steps (not full internal reasoning). After each omplete method, print <END_METHOD> on the next line even if you continue reasoning later or verify again. After all methods, print Final Answer: \\boxed{{...}}. the format should be like: 
        Find the product of all real values of $r$ for which $\\frac{{1}}{{2x}}=\\frac{{r-x}}{{7}}$ has exactly one real solution. <think> ... So ( \\sqrt{{14}} \\times (-\sqrt{{14}}) = -14 ). <END_METHOD>Wait, let me make sure there's no other case where the equation might have exactly one solution....Thus, the product is ( (\\sqrt{{14}})(-\\sqrt{{14}}) = -14 ).<END_METHOD>I think that's it. Let me just verify with an example....So the product is -14.<END_METHOD></think> ...... n has exactly one real solution is: \\boxed{{-14}} <END_METHOD> Final Answer: \\boxed{{-14}}<|im_end|>'''
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ) # 中间步骤没有end_method
REQUEST2 = (
        '''<|im_start|>user
{input}
In your <think> reasoning:
- Every time you reach a result and write it in \\boxed{{...}}, immediately add <END_METHOD> on the next line, even if you continue reasoning later.

In your final explanation (outside <think>):
- For each method, also end with \\boxed{{...}} followed by <END_METHOD>.
- At the very end, write: Final Answer: \\boxed{{...}}

<|im_end|>
<|im_start|>assistant
'''
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",)  # 无法精简地计算，中间步骤没有end_method

REQUEST3 = (
        '''<|im_start|>user
{input}
In your <think> reasoning:
- Show your reasoning process step by step.
- Every time you reach a candidate result and write it in \\boxed{{...}}, immediately add <END_METHOD> on the next line, even if you continue reasoning later or verify again.

In your final explanation (outside <\\think>):
- Present one or more concise solution methods (not full internal reasoning).
- At the end of each method, put its conclusion in \\boxed{{...}} on its own line, followed by <END_METHOD>.
- Finally, after all methods, write the overall result in the format:
Final Answer: \\boxed{{...}}
'''
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",)  # 无法精简地计算，中间步骤没有end_method
REQUEST4 =  (
        '''<|im_start|>user\n{input}\nPlease show concise key steps (not full internal reasoning). After each omplete method, you must print <END_METHOD> on the next line before you verify again. After all methods, print Final Answer: \\boxed{{...}}.'''        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )

REQUEST5 =  (
        '''<|im_start|>user\n{input}\nIn your internal thinking process, Please show concise key steps (not full internal reasoning) and perform verification steps (e.g., discriminant check, domain validation, edge case analysis). After each verification, write <END_METHOD> on a new line.The final answer should be like: Final Answer: \\boxed{{-14}} <|im_end|>'''
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ) # 中间步骤没有end_method  
#TODO:可能看成了一个整体才不好改的，明天尝试整体分割或者问Qwen 它认为butwait是什么？        
def construct_prompt(question):
    
    prompt_temp = REQUEST5
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

if __name__ == "__main__":
    examples = datasets.load_dataset("json", data_files={"test": "/home/lijiakun25/models/datasets/math500/test_debug.jsonl"})["test"]
    model_name = "/home/lijiakun25/models/Qwen3-8b"
    data_name = "math"
    stop_tokens = ["<END_METHOD>"]
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
    total = 0
    equal = 0
    run_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    for i,sample in enumerate(samples):
        origin_prompt = sample[1]
        new_action = ""
        prompt = origin_prompt + new_action
        answer_list = []
        while("Final Answer" not in new_action):
            outputs = completion_request(question=prompt,args=args,tree_id=i,model_id=model_name,stop_tokens=stop_tokens,n_sampling=1)
            new_action = outputs["texts"][0][0]
            new_prompt = prompt + new_action + "<END_METHOD>"
            new_ans = extract_answer(new_action,data_name=data_name)
            is_equal = math_equal(new_ans,sample[2])
            answer_list.append(new_ans)


            num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
            file_index = i % num_processes
            output_path = Path("llm_output") / f"{file_index}_{run_time}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "a", encoding="utf-8") as f:
                    log_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "tree_id": i,
                        "prompt": prompt,
                        "gt_ans": sample[2],
                        "output": new_action,
                        "answer": new_ans,
                        "is_equal": is_equal,
                        "model_id": model_name,
                        "pre_step_answer": answer_list
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            if is_equal:
                 equal += 1
            total += 1
            print(f"question {i}, step {len(answer_list)} has finished.")
            prompt = new_prompt
        answers.append(answer_list)
    print(answers)
    print(f"total:{total}, equal:{equal}")

