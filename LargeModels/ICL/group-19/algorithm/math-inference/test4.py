from openai import OpenAI
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from itertools import repeat
from sympy import isprime
import math
import datasets
from utils.serve_vllm import completion_request
from utils.parse import parse_ground_truth,parse_question
from utils.strip_string import extract_answer
from utils.grader import math_equal
import argparse
import multiprocessing
from pathlib import Path
import time
import json
from test_tool.test_tool import SPLIT_PROMPT_100_to_200,SPLIT_INPUT
from test_tool.test_extraction import data_processing

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


REQUEST = (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
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

def _run_with_kwargs(kwargs):
    # 注意：必须是顶层可导入函数，若改用进程池要可 picklable
    return completion_request(**kwargs)

if __name__ == "__main__":
    examples = datasets.load_dataset("json", data_files={"test": "/home/lijiakun25/models/datasets/math500/train.jsonl"})["test"]
    examples = examples.select(range(100,200))
    model_name = "/home/lijiakun25/models/Qwen3-8b"
    api_model_name = "qwen-plus"
    data_name = "math"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_models", type=int, default=1)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_generate_workers", type=int, default=16)
    args = parser.parse_args()

    request_model = OpenAIAPIModels(
    name="qwen_plus_dashscope",
    model_path=api_model_name,
    model_args={
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-8bd231e7589543fb953f8415b0a362fa",
        # 其他可选参数（按需）
        # "timeout": 60,
    },
    )

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

    sample_td = tqdm(total=len(samples), desc='Example Generation')
    task_params = [
        dict(
            question=sample[1],
            args=args,
            tree_id=i,
            model_id=model_name,
            n_sampling=1,
            timeout_sec=3600,
            # 其余参数不写就走默认值；需要时在这里加
            # use_chat=False, stop_tokens=None, next_token=["Yes","No"], ...
        )
        for i, sample in enumerate(samples)
    ]
    
    outputs = []
    with ThreadPoolExecutor(max_workers=args.max_generate_workers * args.num_of_models) as executor:
        for res in executor.map(
            _run_with_kwargs,
            task_params #n_sampling 固定为 1
        ):
            outputs.append(res['texts'][0][0])
            sample_td.update(1)
    sample_td.close()

    assert len(samples) == len(outputs)

    samples = [samples[i] + (outputs[i],) for i in range(len(samples))]

    split_requests = [] 
    for i,sample in enumerate(samples):
        split_text = SPLIT_INPUT.format(question=sample[0],solution=sample[4])
        split_input = SPLIT_PROMPT_100_to_200 + split_text
        split_request = [{"role": "user", "content": split_input}]
        split_requests.append(split_request)
    
    split_outputs = request_model.generate_chat(split_requests,sampling_args=dict(
        max_tokens=16384,
        temperature=0.0,      # 取模型分布，不做采样
        n=1,
        ))
    
    output_path = ""
    for i,sample in enumerate(samples):
        num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
        file_index = i % num_processes
        output_path = Path("llm_output") / f"{file_index}_{run_time}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as f:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "question_id": i,
                "prompt": sample[1],
                "gt_ans": sample[2],
                "output": sample[4],
                "model_id": model_name,
                "split_output": split_outputs["outputs"][i],
                "output_token_use": split_outputs["output_tokens"][i] ,
                "input_token_use": split_outputs["input_tokens"][i] ,
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    data_processing(output_path)