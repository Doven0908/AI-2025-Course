# math_inference/eval.py
from __future__ import annotations
import argparse, importlib, sys
# import MCTS.ConMCTS.evaluate
# import CoT.evaluate
# import MCTS.llmreasoner.evaluate
# ------- 1) 先只解析 eval_type -------
head_parser = argparse.ArgumentParser(add_help=False)
head_parser.add_argument(
    "--eval_type",
    required=True,
    choices=["cot", "llm-reasoner","conmcts"],
    help="选择评测方式:cot=普通推理,llm-reasoner=搜索推理,conmcts=搜索推理2代",
)
head_args, remaining_argv = head_parser.parse_known_args()

MODULE_MAP = {
    "cot":  "CoT.evaluate", 
    "llm-reasoner": "MCTS.llmreasoner.evaluate", 
    "conmcts":"MCTS.ConMCTS.evaluate",
}
mod = importlib.import_module(MODULE_MAP[head_args.eval_type])
mod.run_and_evaluate(remaining_argv)
