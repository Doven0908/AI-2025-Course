import os
import argparse
# 1. 添加了缺失的 import 语句
from dotenv import dotenv_values
import json 

# --- 辅助函数 ---
def get_absolute_path(path):
    # 获取当前脚本文件所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 返回相对于该目录的绝对路径
    return os.path.join(script_dir, path)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 2. 修复了 get_questions_from_dict 函数
# (原版错误地假设 item 是一个 dict，但它其实是一个 list)
def get_questions_from_dict(data):
    # 正确的逻辑：遍历字典的值(它们是列表)，并取列表中的第一个元素(即问题字符串)
    return [item[0] for item in data.values() if item]

def get_questions_from_list(data):
    # 这个函数对于 "else" 分支 (list of dicts) 是正确的
    return [item['question'] for item in data]
# --- 辅助函数结束 ---


# 3. 修复了文件名拼写错误
CONFIG = dotenv_values(get_absolute_path(".configuration"))
if not CONFIG:
    print(f"CRITICAL ERROR: Could not load .configuration file at {get_absolute_path('.configuration')}")
    print("Please make sure the file exists and is readable.")
    exit(1)

hf_access_token = CONFIG.get("HF_API_KEY")
openain_access_token = CONFIG.get("OPENAI_API_KEY") 

if openain_access_token is None:
    print("CRITICAL ERROR: 'OPENAI_API_KEY' not found in .configuration file.")
    print("Please make sure the key name is correct (e.g., OPENAI_API_KEY=sk-...)")
    exit(1)


file_path_mapping = {
    "wikidata": get_absolute_path("dataset/wikidata_questions.json"),
    "multispanqa": get_absolute_path("dataset/multispanqa_dataset.json"),
    "wikidata_category": get_absolute_path("dataset/wikidata_category_dataset.json"),
}

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m",
        "--model",
        type=str,
        help="LLM to use for predictions.",
        default="llama2",
        choices=["llama2", "llama2_70b", "llama-65b", "gpt3"],
    )
    argParser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Task.",
        default="wikidata",
        choices=["wikidata", "wikidata_category", "multispanqa"],
    )
    argParser.add_argument(
        "-s",
        "--setting",
        type=str,
        help="Setting.",
        default="joint",
        choices=["joint", "two_step", "factored"],
    )
    argParser.add_argument(
        "-temp", "--temperature", type=float, help="Temperature.", default=0.07
    )
    argParser.add_argument("-p", "--top-p", type=float, help="Top-p.", default=0.9)
    args = argParser.parse_args()

    data = read_json(file_path_mapping[args.task])
    
    # 4. 恢复了 if/else 的原始逻辑
    if args.task == "wikidata":
        questions = get_questions_from_dict(data)
    else:
        questions = get_questions_from_list(data)

    if args.model == "gpt3":
        # 5. 确保 cove_chains_openai.py 也被正确修复了
        # (即 import 语句已更新为 langchain_core 和 langchain_openai)
        from src.cove_chains_openai import ChainOfVerificationOpenAI
        chain_openai = ChainOfVerificationOpenAI(
            model_id=args.model,
            temperature=args.temperature,
            task=args.task,
            setting=args.setting,
            questions=questions,
            openai_access_token=openain_access_token, 
        )
        chain_openai.run_chain()
    else:
        from src.cove_chains_hf import ChainOfVerificationHuggingFace
        chain_hf = ChainOfVerificationHuggingFace(
            model_id=args.model,
            top_p=args.top_p,
            temperature=args.temperature,
            task=args.task,
            setting=args.setting,
            questions=questions,
            hf_access_token=hf_access_token,
        )
        chain_hf.run_chain()