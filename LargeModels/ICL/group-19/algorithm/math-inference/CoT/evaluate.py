from vllm import LLM, SamplingParams
import argparse
import os
from transformers import AutoTokenizer
from datetime import datetime
import multiprocessing
import numpy as np

from .examples import construct_prompt
from utils.utils import set_seed, add_remain_field,get_stop_words, load_jsonl, save_jsonl,str2bool
from utils.data_loader import load_data
from utils.parse import parse_question, parse_ground_truth,run_execute
from utils.trajectory import extract_program
from utils.grader import math_equal_process
from utils.python_executor import PythonExecutor
from tqdm import tqdm
import time
import json
from pathlib import Path

OUTPUT_DIR = "./output/"
def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/home/lijiakun25/models/", type=str)
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--num_shots",  default=2,type=int)
    parser.add_argument("--prompt_type",  default="cot",type=str)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--dtype", default="bf16", type=str)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    
    parser.add_argument("--save_output_number",type=int,default=0)

    parser.add_argument("--model_enable_thinking", type=str2bool, default=True)

    parser.add_argument("--save_equal_judge", default=False, type=str2bool)
    parser.add_argument("--batch_size", default=50, type=int)

    args = parser.parse_args(argv)
    model_args = {
        "enable_thinking": args.model_enable_thinking,
    }
    args.model_args = model_args
    return args

def prepare_data(data_name, args):
    dt_string = datetime.now().strftime("%H-%M")
    examples = load_data(data_name=data_name,data_path=args.data_dir)
    
    model_name = args.model_name_or_path.split("/")[-1]
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    out_file = f"{output_dir}/{data_name}/{model_name}/{dt_string}/result.jsonl"
    os.makedirs(f"{output_dir}/{data_name}/{model_name}/{dt_string}",exist_ok=True)

    #断点机制可加
    return examples, out_file

def setup(args):
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        dtype=args.dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    result = main(llm,tokenizer,args.data_name,args)


def main(llm, tokenizer, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)
    samples = []
    examples = examples.select(range(300))
    for i, example in enumerate(tqdm(examples,total=len(examples),disable=False)):
        question = parse_question(example,data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example,data_name,question,args)
        if i == 0:
            print(full_prompt)
        sample = {
            "idx": i,
            "question": question,
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }
        sample = add_remain_field(sample,example)
        samples.append(sample)
    
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i,prompt in enumerate(remain_prompts)]
    
    stop_words = get_stop_words(args)
    max_func_call = 1 # 可能有一些多轮交互的数据集要用

    start_time = time.time()

    end_prompts = []
    for epoch in range(max_func_call):
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        prompts = [item[1] for item in current_prompts]
        outputs = []
        batch_size = args.batch_size
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            batch_outputs = llm.generate(
                batch_prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                )
            )

            outputs.extend(batch_outputs)


        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id


        input_tokens = [len(r.prompt_token_ids) for r in outputs]
        output_tokens = [len(r.outputs[0].token_ids) for r in outputs]
        outputs = [output.outputs[0].text for output in outputs]

        assert len(outputs) == len(current_prompts)
        
        remain_prompts = []
        remain_codes = []
        for (i,query), output,input_token,output_token in zip(current_prompts, outputs,input_tokens,output_tokens):
            output =  output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query,input_token,output_token))

            remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

        # remove input_prompt from end_prompt
    codes = []
    output_tokens = []
    input_tokens = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt,input_token,output_token = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
        input_tokens.append(input_token)
        output_tokens.append(output_token)
        
    
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    all_samples =[]
    for i,sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        input_token = input_tokens[i * args.n_sampling : (i + 1) * args.n_sampling]
        output_token = output_tokens[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports, "input_token":input_token, "output_token":output_token})
        all_samples.append(sample)
    # all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        args=args,
        execute=True,
    )
    
    time_use = time.time() - start_time

    result_json["avg_response_len"] = sum(output_tokens) / len(output_tokens)
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    
    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_outputs.jsonl"), "w"
    ) as f:
        for sample in all_samples[:args.save_output_number]:
            output_json = {}
            sample["code"] = sample["code"][:min(2,len(sample["code"]))]
            output_json.update(sample)
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")
            
    return result_json


def evaluate(data_name, prompt_type, args,samples:list=None,file_path:str=None,max_num_samples=None,execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x:x['idx'])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]
    
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample,data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]


    scores =  []
    timeout_cnt = 0
    timeout_length = 3
    async_results = []
    with multiprocessing.Pool(processes=4) as pool:
        for param in params:
            async_results.append(pool.apply_async(math_equal_process, args=(param,)))

        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=timeout_length)
                    scores.append(result)
                except multiprocessing.TimeoutError:
                    print("Timeout occurred")
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as e:
                    print("Error:", e)
                    scores.append(False)
                progress_bar.update(1)
    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx:idx + len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])
    
    max_len = max([len(s) for s in score_mat])
    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s))
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    if args.save_equal_judge:
        model_name = args.model_name_or_path.strip("/").split("/")[-1]
        save_equal_path = Path("llm_output") / f"{data_name}/{model_name}"
        os.makedirs(save_equal_path,exist_ok=True)
        save_file = save_equal_path / "equal_judge.jsonl"
        with open(save_file, "w", encoding="utf-8") as fout:
            for sample in samples:
                gt = sample["gt"]
                preds = sample["pred"]
                scores = sample["score"]
                for pred, is_equal in zip(preds, scores):
                    json_line = {
                        "idx": sample["idx"],
                        "predict_ans": pred,
                        "ground_truth": gt,
                        "is_equal": is_equal,
                    }
                    fout.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        print(f"[Saved equal judgments to] {save_file}")


    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not (s['pred'] and s['pred'][-1])]),
        "acc": mean_score
    }
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json

def run_and_evaluate(argv: list[str] | None = None) -> None:
    """供外部调用的统一入口。"""
    args = parse_args(argv)
    set_seed()
    setup(args)

if __name__ == "__main__":
    run_and_evaluate()
    