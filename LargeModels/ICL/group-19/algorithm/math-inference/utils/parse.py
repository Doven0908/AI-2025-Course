from .strip_string import strip_string,extract_answer
from .trajectory import extract_program_output,extract_program
import re

def parse_question(example, data_name):
    if data_name == "math":
        for key in ["question", "problem", "Question", "input", "Problem"]:
            if key in example:
                question = example[key]
                break
    elif data_name in ["aime24", "aime25"]:
        for key in ["question", "problem", "Question", "Problem"]:
            if key in example:
                question = example[key]
                break
    elif data_name.startswith("olympiadbench"):
        for key in ["question", "problem", "Question", "Problem"]:
            if key in example:
                question = example[key]
                break
    return question

def parse_ground_truth(example, data_name):
    if data_name in ["math"]:
        gt_cot = example["solution"]
        gt_answer = example["answer"]
    elif data_name in ["aime24"]:
        gt_cot = example["Solution"]
        gt_answer = example["Answer"]       
    elif data_name in ["aime25"]:
        gt_cot = ""
        gt_answer = example["answer"]
    elif data_name.startswith("olympiadbench"):
        answer_list = example["final_answer"]
        ans = answer_list[0]  # 只取第一个元素
        # 清除 $, ", \，替换 \times
        ans = ans.replace("$", "").replace('"', "")
        ans = ans.replace("times", "×")

        # 标准化逗号和空格
        ans = re.sub(r"\s*,\s*", ", ", ans)
        gt_answer = ans.strip()
        gt_cot = example["solution"][0]

    return gt_cot,gt_answer

STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]

def run_execute(executor, result, prompt_type, data_name, execute=False):
    if not result or result == "error":
        return None, None
    report = None

    if "program_only" in prompt_type:
        prediction = extract_program_output(result)
    elif prompt_type in ["pot", "pal"] and execute:
        code = extract_program(result)
        prediction, report = executor.apply(code)
    else:
        prediction = extract_answer(result, data_name)

    # prediction = strip_string(prediction, skip_unit=data_name == "carp_en")
    prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)
    return prediction, report
