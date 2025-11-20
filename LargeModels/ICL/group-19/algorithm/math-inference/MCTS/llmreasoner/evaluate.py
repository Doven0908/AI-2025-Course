from utils.utils import set_seed
import argparse
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from tqdm import tqdm
import numpy as np
import time
import json
import multiprocessing
from utils.parse import parse_question, parse_ground_truth,run_execute
from utils.data_loader import load_data
from ..base_tree import BaseTree
from llmreasoner import LLM_REASONER_Tree, llm_reasoner_mcts
from utils.visualize import visualize_tree
from utils.grader import math_equal_process
from utils.serve_vllm import clean_llm_output_files


OUTPUT_DIR = "./output/"
def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/home/lijiakun25/models/", type=str)
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--max_retry", type=int, default=4)

    parser.add_argument("--debug",type=bool,default=True)
    
    parser.add_argument("--use_api",type=bool, default=True)
    parser.add_argument("--num_of_models",type=int,default=2)
    parser.add_argument("--max_func_call", type=int, default=50)
    parser.add_argument("--num_shots",  default=3,type=int)
    parser.add_argument("--dtype", default="bf16", type=str)
    # parser.add_argument("--prompt_type",  default="cot",type=str)
    # parser.add_argument("--n_sampling", default=1, type=int)
    # parser.add_argument("--temperature", default=0, type=float)
    # parser.add_argument("--top_p", default=1, type=float)
    # parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--num_iterations",type=int,default=10)
    parser.add_argument("--output_strategy",type=str,choices=["follow_max", "max_reward"],default="max_reward") # 最终返回的路径和结果，但是目前最有用和实现的就是max_reward。
    parser.add_argument("--mcts_depth_limit", default=6, type=int)
    parser.add_argument("--uct_type",type=str,choices=["inf", "sqrt_parent"],default="inf")
    parser.add_argument("--uct_inf", type=float, default=1.0)
    parser.add_argument("--explore_constant", type=float, default=0.7)
    parser.add_argument("--use_reflection", type=bool, default=True)
    # parser.add_argument("--roll_policy", type=str,choices=["greedy_policy", "fast_rollout","random_policy"], required=True,default="inf")
    parser.add_argument("--n_confidence",type=int, default=10)
    parser.add_argument("--early_stop_confidence", type=float, default=0.5) # expand阶段早停的一致性阈值
    parser.add_argument("--n_batch", type=int, default=3)
    parser.add_argument("--n_actions",type=int, default=3)
    parser.add_argument("--reward_alpha",type=float,default=0.5) # 节点一致性和自信度奖励平衡系数
    parser.add_argument("--simulate_strategy",type=str,choices=["max", "sample","random"],default="sample") # 采样节点的方式
    args = parser.parse_args(argv)
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
    if not args.use_api:
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus),
            trust_remote_code=True,
            dtype=args.dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        llm = None
        tokenizer = None
    if args.debug:
        clean_llm_output_files(args=args)
    result_json = main(llm,tokenizer,args.data_name,args)

def run_reasoning(tree, args):
    """包装 llm_reasoner_mcts 以便并行调用"""
    return llm_reasoner_mcts(tree, args)

def main(llm, tokenizer, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    start_time = time.time()

    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    tree_list = []
    for i,example in enumerate(tqdm(examples,total=len(examples),disable=False)):
        question = parse_question(example,data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        tree_list.append(LLM_REASONER_Tree(i,question,gt_ans,gt_cot,args))


    
    class TimeoutResult:
        """占位：标记某棵树在某轮仍超时."""
        def __init__(self, tree):
            self.tree   = tree
        def __repr__(self):
            return f"<TimeoutResult tree_id={getattr(self.tree,'tree_id',None)}>"
    
    reason_time_out = 2000
    final_tree_list = [None] * len(tree_list)
    num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
    remaining_tasks = list(enumerate(tree_list))

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for round in range(1,args.max_retry + 1):
            if not remaining_tasks:
                break

            futures = {executor.submit(run_reasoning,t,args): (idx,t) for idx, t in remaining_tasks}
            remaining_tasks = []
            with tqdm(total=len(futures),desc=f"Round {round}") as pbar:
                for fut,(idx,tree) in futures.items():
                    try:
                        result = fut.result(timeout=reason_time_out)
                        if all(r is None for r in final_tree_list):
                            visualize_tree(result.root,args=args)
                        final_tree_list[idx] = result

                    except TimeoutError:
                        print(f"⚠️  tree_id={tree.tree_id} 超出时间：{reason_time_out}")
                        remaining_tasks.append((idx, tree))
                        final_tree_list[idx] = TimeoutResult(tree)

                    except Exception as e:              
                        print(f"⚠️  tree_id={tree.tree_id} 异常：{e}")
                        remaining_tasks.append((idx,tree))
                        final_tree_list[idx] = TimeoutResult(tree)

                    finally:
                        pbar.update(1)
            if remaining_tasks:
                print(f"⚠️  Round {round} 结束，{len(remaining_tasks)} 任务将重试")


    succ_cnt = sum(
        1 for r in final_tree_list if (r is not None and not isinstance(r, TimeoutResult))
    )
    print(f"✅  成功完成 {succ_cnt} / {len(tree_list)} 棵树")

    scores =  []
    timeout_cnt = 0
    timeout_length = 3
    async_results = []
    params = []
    for tree in final_tree_list:
        assert isinstance(tree,BaseTree)
        final_ans_list = tree.get_final_ans()
        for final_ans in final_ans_list:
            params.append((tree.tree_id,final_ans,tree.gt_ans))

    with multiprocessing.Pool(processes=num_processes) as pool:
        for param in params:
            async_results.append(pool.apply_async(math_equal_process, args=(param,)))

        with tqdm(total=len(params), desc="Evaluate") as progress_bar:
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

    for tree in final_tree_list:
        assert isinstance(tree,BaseTree)
        tree.final_scores = scores[idx:idx + len(tree.get_final_ans())]
        assert len(tree.final_scores) == len(tree.get_final_ans())
        score_mat.append(tree.final_scores)
        idx += len(tree.final_scores)

    max_len = max([len(s) for s in score_mat])
    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s))
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(final_tree_list),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        # "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score
    }

    time_use = time.time() - start_time
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", "_MCTS_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json

def run_and_evaluate(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed()
    setup(args)

if __name__ == "__main__":
    run_and_evaluate()