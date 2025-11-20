"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean

class AutoScoringJudge:
    def __init__(self):
        # Map of special symbols to their replacements
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8  # Default precision for comparison

    def split_by_comma(self, expr: str):
        # Splits expressions by commas outside of brackets
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "["]:
                in_bracket_num += 1
            elif char in [")", "]"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())   
        
        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        # Translates plus-minus signs into separate expressions
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)
        
        return new_expr_list
    
    def judge(self, expression1, expression2, precision=1e-8):
        # Judge if two expressions are equal (expression1 is considered as the Ground Truth)
        # Default precision is a list for supporting multiple expressions
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:
            # print("Exactly equal")
            return True
        
        # Remove Chinese characters from the string, as answers like "yes" or "no" in Chinese have been considered
        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)
        
        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # Set up a list for allowed errors
        if len(precision) <= 1:
            precision = precision * len(temp_list1)
        
        if len(temp_list1) != len(temp_list2):
            return False

        # Check if elements in both lists can be paired and are equal
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if math_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                # If no match was found, return False
                return False

        # If all elements are matched, return True
        return True
    
    def preprocess(self, expression1, expression2):
        # Preprocess expressions to extract and replace special symbols
        def extract_boxed_content(latex_str):
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str
                
            return results
        
        def sepcial_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]
            
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression
        
        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2
    

def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
    data_name: str = "math",
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if data_name.startswith("olympiadbench"):
        olym_judge = AutoScoringJudge()
        return olym_judge.judge(expression1=prediction,expression2=reference,precision=1e-5)

    if prediction is None or reference is None:
        return False
    pred_str = str(prediction).strip()
    ref_str  = str(reference).strip()

    if pred_str.lower() == ref_str.lower():
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(pred_str) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()

def _test_math_equal():
    # print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
    # print(math_equal("(1,4.5)", "(1,\\frac{9}{2})"))
    # print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))
    # print(math_equal("\\sec^2(y)", "\\tan^2(y)+1", timeout=True))
    # print(math_equal("\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\end{pmatrix}", "(\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix})", timeout=True))

    # pred = '\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix})'

    # pred= '-\\frac{8x^2}{9(x^2-2)^{5/3}}+\\frac{2}{3(x^2-2)^{2/3}}'
    # gt= '-\\frac{2(x^2+6)}{9(x^2-2)\\sqrt[3]{x^2-2}^2}'

    # pred =  '-34x-45y+20z-100=0'
    # gt = '34x+45y-20z+100=0'

    # pred = '\\frac{100}{3}'
    # gt = '33.3'

    # pred = '\\begin{pmatrix}0.290243531202435\\\\0.196008371385084\\\\-0.186381278538813\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}0.29\\\\0.196\\\\-0.186\\\\\\end{pmatrix})'

    # pred = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}'
    # gt = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}'

    # pred = '(+5)(b+2)'
    # gt = '(a+5)(b+2)'

    # pred = '\\frac{1+\\sqrt{5}}{2}'
    # gt = '2'

    # pred = '\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}', gt = '4'
    # pred = '1', gt = '1\\\\sqrt{19}'

    # pred = "(0.6,2.6667]"
    # gt = "(\\frac{3}{5},\\frac{8}{3}]"


    from strip_string import extract_answer
    
    exp1 = "*** We can get the question's Final Answer: \boxed{2}, \boxed{3}, \boxed{4}."
    exp2 = "2,3,4"
    print(extract_answer(exp1,data_name="math"))
    print(math_equal(extract_answer(exp1,data_name="math"), exp2, timeout=True))

    exp1 = "*** We can get the question's Final Answer: \boxed{3}, \boxed{2}, \boxed{4}."
    exp2 = "2,3,4"
    print(math_equal(extract_answer(exp1,data_name="math"), exp2, timeout=True))


if __name__ == "__main__":
    _test_math_equal()