# 主要是看小模型是否能成功分未完成的答案
from openai import OpenAI
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from itertools import repeat
from sympy import isprime
import math
import datasets
from utils.serve_vllm import completion_request,chat_completion_request
from utils.parse import parse_ground_truth,parse_question
from utils.strip_string import extract_answer
from utils.grader import math_equal
import argparse
import multiprocessing
from pathlib import Path
import time
import json
from test_tool.test_tool import SPLIT_PROMPT,SPLIT_INPUT,SPLIT_PROMPT_SPLITED
from test_tool.test_extraction import data_processing
import random
import Levenshtein
import re

class OpenAIAPIModels():
    def __init__(
            self,
            name: str,
            model_path: str = 'gpt-4o',
            model_args: Dict[str, Any] = None,
            **args
    ) -> None:
        if model_args is None:
            model_args = {}

        self.name = name
        self.model_path = model_path
        self.max_workers = model_args.pop('max_workers', 32)

        if model_args.__contains__('base_url') and model_args.__contains__('api_key'):
            base_url_list = model_args.pop('base_url')
            api_key_list = model_args.pop('api_key')
            if type(base_url_list) is str:
                base_url_list = [base_url_list]
            if type(api_key_list) is str:
                api_key_list = [api_key_list]

            assert len(base_url_list) == len(api_key_list)
            self.client_list = []
            for i in range(len(base_url_list)):
                client = OpenAI(
                    base_url=base_url_list[i],
                    api_key=api_key_list[i],
                    **model_args
                )
                self.client_list.append(client)
        else:
            self.client_list = [
                OpenAI(**model_args)
            ]

        assert len(self.client_list) > 0

        print(f'OpenAI API: {len(self.client_list)} endpoints found.')

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def _call_chat_api(
            self,
            index: int,
            messages: List,
            sampling_args: Dict[str, Any],
    ) -> Dict:
        extra_args = {}
        if sampling_args.__contains__('top_k'):
            extra_args['top_k'] = sampling_args.pop('top_k')
        if sampling_args.__contains__('min_p'):
            extra_args['min_p'] = sampling_args.pop('min_p')
        if sampling_args.__contains__('repetition_penalty'):
            extra_args['repetition_penalty'] = sampling_args.pop('repetition_penalty')

        client = self.client_list[index % len(self.client_list)]
        res = client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            **sampling_args,
            extra_body=extra_args
        )
        return {
            'output': res.choices[0].message.content,
            'input_tokens': res.usage.prompt_tokens,
            'output_tokens': res.usage.completion_tokens,
        }

    def _batch_call_chat_api(
            self,
            messages_list: List[List],
            sampling_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        outputs = []
        input_tokens = []
        output_tokens = []

        index_list = [i for i in range(len(messages_list))]

        td = tqdm(total=len(messages_list), desc='API Chat')
        with ThreadPoolExecutor(max_workers=self.max_workers * len(self.client_list)) as executor:
            for res in executor.map(self._call_chat_api, index_list, messages_list, repeat(sampling_args)):
                try:
                    outputs.append(res['output'])
                    input_tokens.append(res['input_tokens'])
                    output_tokens.append(res['output_tokens'])
                except Exception as e:
                    print(f"[Error] Skipping: {e}")
                    outputs.append("")
                    input_tokens.append(0)
                    output_tokens.append(0)
                finally:
                    td.update(1)
        td.close()

        return {
            'outputs': outputs,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def _call_completion_api(
            self,
            index: int,
            prompt: str,
            sampling_args: Dict,
    ) -> Dict:
        extra_args = {}
        if sampling_args.__contains__('top_k'):
            extra_args['top_k'] = sampling_args.pop('top_k')
        if sampling_args.__contains__('min_p'):
            extra_args['min_p'] = sampling_args.pop('min_p')
        if sampling_args.__contains__('repetition_penalty'):
            extra_args['repetition_penalty'] = sampling_args.pop('repetition_penalty')

        client = self.client_list[index % len(self.client_list)]

        res = client.completions.create(
            model=self.model_path,
            prompt=prompt,
            **sampling_args,
            extra_body=extra_args
        )
        return {
            'output': res.choices[0].text,
            'input_tokens': res.usage.prompt_tokens,
            'output_tokens': res.usage.completion_tokens,
        }

    def _batch_call_completion_api(
            self,
            prompt_list: List[List],
            sampling_args: Union[List[Dict], Dict]
    ) -> Dict[str, Any]:
        outputs = []
        input_tokens = []
        output_tokens = []

        index_list = [i for i in range(len(prompt_list))]

        td = tqdm(total=len(prompt_list), desc='API Completion')
        with ThreadPoolExecutor(max_workers=self.max_workers * len(self.client_list)) as executor:
            for res in executor.map(
                    self._call_completion_api,
                    index_list,
                    prompt_list,
                    repeat(sampling_args) if type(sampling_args) is dict else sampling_args
            ):
                outputs.append(res['output'])
                input_tokens.append(res['input_tokens'])
                output_tokens.append(res['output_tokens'])
                td.update(1)
        td.close()

        return {
            'outputs': outputs,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def _call_embedding_api(
            self,
            index: int,
            prompt: str,
            generation_args: Dict[str, Any],
    ) -> Dict:
        client = self.client_list[index % len(self.client_list)]
        res = client.embeddings.create(
            model=self.model_path,
            input=prompt,
            **generation_args
        )
        return {
            'output': res.data[0].embedding,
        }

    def _batch_call_embedding_api(
            self,
            prompt_list: List[str],
            generation_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        outputs = []

        index_list = [i for i in range(len(prompt_list))]
        td = tqdm(total=len(prompt_list), desc='API embedding')
        with ThreadPoolExecutor(max_workers=self.max_workers * len(self.client_list)) as executor:
            for res in executor.map(self._call_embedding_api, index_list, prompt_list, repeat(generation_args)):
                outputs.append(res['output'])
                td.update(1)
        td.close()

        return {
            'outputs': outputs,
        }

    def generate_completion(
            self,
            prompt_list: List[str],
            sampling_args: Union[List[Dict], Dict],
            generation_args: Dict[str, Any] = None
    ) -> Dict:
        if type(prompt_list) != list:
            prompt_list = [prompt_list]

        return self._batch_call_completion_api(
            prompt_list=prompt_list,
            sampling_args=sampling_args
        )

    def generate_chat(
            self,
            messages_list: List[List[Dict]],
            sampling_args: Dict[str, Any],
            generation_args: Dict[str, Any] = None
    ) -> Dict:
        if type(messages_list[0]) != list:
            messages_list = [messages_list]

        return self._batch_call_chat_api(
            messages_list=messages_list,
            sampling_args=sampling_args
        )

    def generate_embedding(
            self,
            prompt_list: List[str],
            generation_args: Dict[str, Any] = None
    ) -> Dict:
        if type(prompt_list) != list:
            prompt_list = [prompt_list]

        if generation_args is None:
            generation_args = {}

        return self._batch_call_embedding_api(
            prompt_list=prompt_list,
            generation_args=generation_args
        )

def batched_stream_completion_cut_before_match(
    seqs: list[str],
    args,
    match_limit: int,
    prompt: str,
    check_every_n_tokens: int = 10,
    **kwargs
):
    model = args.model_name_or_path
    num_of_models = args.num_of_models
    ports = [8000 + i for i in range(num_of_models)]
    urls = [f"http://127.0.0.1:{p}/v1" for p in ports]
    base_url = random.choice(urls)
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    full_output = ""
    token_buffer = []
    match_positions = []
    response = client.completions.create(
        model=model,
        prompt=prompt,
        stream=True,
        **kwargs
    )

    max_seq_len = max((len(s) for s in seqs if s), default=1)
    last_scan_from = 0
    seen_matches = set()  # {(seq, idx)}

    try:
        for chunk in response:
            token = chunk.choices[0].text
            token_buffer.append(token)
            full_output += token
            # print(f"[TOKEN]: {token}", end="", flush=True)

            if len(token_buffer) >= check_every_n_tokens:
                # print("\n[DEBUG] Checking for matches...")
                # 检查 seqs 是否在 full_output 中
                start = max(last_scan_from - (max_seq_len - 1), 0)
                slice_text = full_output[start:]
                for seq in seqs:
                    search_pos = 0
                    while True:
                        idx_rel = slice_text.find(seq, search_pos)
                        if idx_rel == -1:
                            break

                        idx_abs = start + idx_rel
                        if (seq, idx_abs) not in seen_matches:
                            match_positions.append((seq, idx_abs))
                            seen_matches.add((seq, idx_abs))
                        search_pos = idx_rel + len(seq)
                
                last_scan_from = len(full_output)
                
                if len(match_positions) >= match_limit:
                    # 保留内容到第 n-1 次匹配词的开始位置
                    seq_to_cut, match_index = match_positions[match_limit - 1]
                    final_output = full_output[:match_index]  # 截断在最后一个匹配词前
                    return final_output, [s for s, _ in match_positions[:match_limit]]

                token_buffer = []  # 清空 buffer，继续流式收集
    except Exception as e:
        print("Error during stream:", e)

    return full_output, [s for s, _ in match_positions]



REQUEST = (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )

# if __name__ == "__main__":
#     seqs = ["AI", "\n\n"]
#     output, matched = batched_stream_completion_cut_before_match(
#         seqs=seqs,
#         match_limit=2,
#         check_every_n_tokens=8
#     )

#     print("\n--- Final Output ---\n", output)
#     print("--- Matched ---\n", matched)



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

def extract_json_objects(text):
        # 匹配 {"step": 数字, "content": "..."}，允许换行内容（非贪婪匹配）
        pattern = r'\{"step":\s*(\d+),\s*"content":\s*"(.*?)"\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)

        result = []
        for step, content in matches:
            json_str = json.dumps({"step": int(step), "content": content})
            try:
                obj = json.loads(json_str)
                result.append(obj)
            except json.JSONDecodeError as e:
                print(f"解析失败: {e}，内容为：{json_str[:50]}...")
        return result


def test_similarity(a,b):
    distance = Levenshtein.distance(a, b)
    similarity = 1 - distance / max(len(a), len(b))  # 转成 0~1 相似度
    print(f"相似度：{similarity:.2f}")
    return similarity


def _run_with_kwargs(kwargs):
    # 注意：必须是顶层可导入函数，若改用进程池要可 picklable
    return completion_request(**kwargs)

if __name__ == "__main__":
    examples = datasets.load_dataset("json", data_files={"test": "/home/lijiakun25/models/datasets/math500/test_debug.jsonl"})["test"]
    # examples = examples.select(range(100))
    api_model_name = "qwen-plus"
    data_name = "math"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=1)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_generate_workers", type=int, default=16)
    parser.add_argument("--model_name_or_path", type=str, default="/home/lijiakun25/models/Qwen3-8b")
    args = parser.parse_args()

    match_limit = 6

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

    stop_seqs = ["Wait","But","Let me think","</think>"]

    num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
    
    outputs = []
    for i,sample in enumerate(samples):
        generate_list = []
        split_list = []
        cnt = 0
        while True:
            output,stop_matches = batched_stream_completion_cut_before_match(seqs=stop_seqs,args=args,match_limit=match_limit,prompt=sample[1],n=1,temperature=0.6,max_tokens=8192)
            if stop_matches and stop_matches[-1] == "</think>":
                output += stop_matches[-1]
            generate_list.append(output)
            
            split_finished = "\n".join(json.dumps(item, ensure_ascii=False) for item in split_list)
            split_generated_output = "".join(item for item in generate_list)

            split_text = SPLIT_PROMPT + SPLIT_INPUT.format(question=sample[0],solution=split_generated_output)
            split_request = SPLIT_PROMPT_SPLITED.format(splition=split_finished)
            split_input = [[{"role": "user", "content": split_text},
                           {"role": "assistant", "content": split_request}]]

            split_outputs = chat_completion_request(split_input, tree_id=i,max_tokens=16384, temp=0.0, n_sampling=1, args=args)
            split_output = split_outputs["texts"][0][0]
            
            split_result = extract_json_objects(split_output)
            if len(stop_matches) >= match_limit:
                split_result_cut = split_result[:-1]

            for item in split_list + split_result:
                print(f"type: {type(item['content'])}, value: {item['content']}")
            test_splition_similarity = "".join(item["content"] for item in split_list + split_result)
            similarity = test_similarity(test_splition_similarity,split_generated_output)

            split_list.extend(split_result_cut)

            file_index = i % num_processes
            output_path = Path("llm_output") / f"{file_index}_{run_time}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "a", encoding="utf-8") as f:
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "question_id": i,
                    "prompt": sample[1],
                    "gt_ans": sample[2],
                    "output": split_generated_output,
                    "model_id": args.model_name_or_path,
                    "split_output": split_output,
                    "split_list": "\n".join(json.dumps(item, ensure_ascii=False) for item in split_list),
                    "similarity": similarity
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            cnt += 1
            print(f"question {i} has finished step {cnt}.")
            if len(stop_matches) < match_limit:
                break
        print(f"Question {i} has finished")

    # samples = [samples[i] + (outputs[i],) for i in range(len(samples))]
    # assert len(samples) == len(outputs)

    # split_requests = []
    # for i,sample in enumerate(samples):
    #     split_input = SPLIT_PROMPT_100_to_200 + split_text
    #     split_request = [{"role": "user", "content": split_input}]
    #     split_requests.append(split_request)
    
    # split_outputs = request_model.generate_chat(split_requests,sampling_args=dict(
    #     max_tokens=16384,
    #     temperature=0.0,      # 取模型分布，不做采样
    #     n=1,
    #     ))
    
    # output_path = ""
    # for i,sample in enumerate(samples):
    #     num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
    #     file_index = i % num_processes
    #     output_path = Path("llm_output") / f"{file_index}_{run_time}.jsonl"
    #     output_path.parent.mkdir(parents=True, exist_ok=True)

    #     with open(output_path, "a", encoding="utf-8") as f:
    #         log_entry = {
    #             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    #             "question_id": i,
    #             "prompt": sample[1],
    #             "gt_ans": sample[2],
    #             "output": sample[4],
    #             "model_id": model_name,
    #             "split_output": split_outputs["outputs"][i],
    #             "output_token_use": split_outputs["output_tokens"][i] ,
    #             "input_token_use": split_outputs["input_tokens"][i] ,
    #         }
    #         f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    # data_processing(output_path)