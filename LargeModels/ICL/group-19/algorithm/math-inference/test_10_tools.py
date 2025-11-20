from openai import OpenAI
import random
from utils.serve_vllm import completion_request,chat_completion_request

def _is_ctx_overflow_error(e: Exception) -> bool:
    msg = str(e).lower()
    # 兼容 vLLM/OpenAI 的常见报文
    return ("maximum context length" in msg) and ("requested" in msg and "tokens" in msg)


def batched_stream_completion_cut_before_match(
    seqs: list[str],
    args,
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
    match_limit = args.match_limit

    full_output = ""
    token_buffer = []
    match_positions = []

    max_seq_len = max((len(s) for s in seqs if s), default=1)
    last_scan_from = 0
    seen_matches = set()  # {(seq, idx)}

    last_finish_reason = None

    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            stream=True,
            timeout=7200.0,
            **kwargs
        )
        try:
            for chunk in response:
                token = chunk.choices[0].text
                token_buffer.append(token)
                full_output += token
                # print(f"[TOKEN]: {token}", end="", flush=True)

                fr = getattr(chunk.choices[0], "finish_reason", None)
                if fr:
                    last_finish_reason = f"{fr}"

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
                        break
                        

                    token_buffer = []  # 清空 buffer，继续流式收集
        
        finally:
            if hasattr(response, "close"):
                try:
                    response.close()
                except Exception:
                    pass
        
        if len(match_positions) >= match_limit:
            seq_to_cut, match_index = match_positions[match_limit - 1]
            final_output = full_output[:match_index]  # 截断在最后一个匹配词前
            finish_reason = "match_limit_reached"
            return final_output, [s for s, _ in match_positions[:match_limit]], finish_reason
        
        finish_reason = last_finish_reason or "unknown"
        return full_output, [s for s, _ in match_positions], finish_reason
        
    except Exception as e:
        # 只有“上下文超长”被吞掉并用哨兵值返回；其余异常全部抛出
        if _is_ctx_overflow_error(e):
            finish_reason = "token_max_reached"
            return "context_overflow", str(e), finish_reason
        raise  # 其它错误让主线程在 fut.result() 处直接中断


def get_answer_response(tree_id, generate_prompt, sample, args):
    get_answer_prompt = generate_prompt + "*** ... Oh, I suddenly got the question's Final Answer: \\boxed"

    target_len = max(len(sample[2]), 29)
    all_text = ""
    all_logits = []
    all_token_usage = 0

    while True:
        question = get_answer_prompt + all_text
        # 由于openAI的api最多支持返回30个token的prob，所以必须分组返回
        comp_resp = completion_request(
            question=question,
            args=args,
            tree_id=tree_id,
            model_id=args.model_name_or_path,
            need_get_next_token_prob=True,
            stop_tokens=["\n", "\n\n",".","***"],
            max_tokens=25,
            n_sampling=1
        )
        all_text += comp_resp["texts"][0][0]
        all_logits.extend(comp_resp["logits"][0])
        gen_tokens = comp_resp["usage"].completion_tokens
        all_token_usage += gen_tokens

        if gen_tokens < 24 or all_token_usage >= target_len:
            break
    top_probs = all_logits
    for prob_idx,top in enumerate(top_probs):
        sorted_top = sorted(top.items(), key=lambda x: x[1], reverse=True)[:5]
        top_probs[prob_idx] = dict(sorted_top)

    return all_text, top_probs


REQUEST='''
You are given a math question, a model's boxed answer, and the ground-truth.
If the boxed answer is mathematically correct, output True. Otherwise, output False.
Only compare the final answer. Reply with one word: True or False.
Example Input:
Question: The set of points \\((x,y,z)\\) that satisfy  
\\[2x = 3y = -z\\]  
is a line. The set of points \\((x,y,z)\\) that satisfy  
\\[6x = -y = -4z\\]  
is another line. Find the angle between these lines, in degrees.

Response:  
*** We can get the question's Final Answer: \\(\\boxed\\{90\\}\\).  

ground_truth: \\(90^\\circ\\)
Output:
True  

Example Input:
Question: 
Xenia and Sergey play the following game. Xenia thinks of a positive integer $N$ not exceeding 5000. Then she fixes 20 distinct positive integers $a_{1}, a_{2}, \ldots, a_{20}$ such that, for each $k=1,2, \ldots, 20$, the numbers $N$ and $a_{k}$ are congruent modulo $k$. By a move, Sergey tells Xenia a set $S$ of positive integers not exceeding 20 , and she tells him back the set $\left\{a_{k}: k \in S\right\}$ without spelling out which number corresponds to which index. How many moves does Sergey need to determine for sure the number Xenia thought of?

Response:  
*** We can get the question's Final Answer: \\boxed{1} ***$$

ground_truth:  
2
Output:
False

'''
REQUEST_INPUT = "Now process the following new input:\nQuestion: {question}\n\nResponse: {partial_response} \n\nground_truth: {ground_truth}/no_think"

def llm_judge_once(question: str,
                   ground_truth: str,
                   partial_response: str,
                   args,
                   *,
                   max_tokens: int = 10,
                   prob_threshold: float = 0.3,
                   tree_id: int = 0):
    """
    依据 predict_ans 与 ground_truth 调用模型做 Yes/No 判定，并计算 yes/no 概率。
    返回: (judge, yes_prob, no_prob, chat_resp)
      - judge: "True" / "False" / "Error"
      - yes_prob/no_prob: 汇总到达阈值前的概率
      - chat_resp: 原始响应(含 logits),用于调试/记录
    """
    # 依赖你上面定义的全局 REQUEST / REQUEST_INPUT
    verify_text = REQUEST + REQUEST_INPUT.format(question=question,
                                                 ground_truth=ground_truth,
                                                 partial_response=partial_response)
    messages = [{"role": "user", "content": verify_text}]

    # 组装请求参数（保持与你项目里一致）
    kwargs = dict(
        question=messages,
        args=args,
        tree_id=tree_id,                 # 若不需要可设为 0 或省略
        n_sampling=1,
        timeout_sec=3600,
        need_get_next_token_prob=True,
        next_token=["True","true","False","false"," True"," true"," False"," false"],
        max_tokens=max_tokens,
        # use_chat=False, stop_tokens=None, next_token=["Yes","No"], ...
    )

    try:
        chat_resp = chat_completion_request(**kwargs)
    except Exception as e:
        # 调用失败一律视作 Error
        return "Error", 0.0, 0.0, {"error": str(e)}

    # 读取 logits 并累计 yes/no 概率
    try:
        yes_prob = 0.0
        no_prob = 0.0
        k = 0
        # 有些实现 logits 结构：logits[采样索引][步索引] -> {token: prob}
        steps = chat_resp["logits"][0] if "logits" in chat_resp else []
        limit = min(max_tokens, len(steps))

        while (k < limit) and (yes_prob + no_prob < prob_threshold):
            prob_map = dict(steps[k])
            yes_prob += prob_map.get("True", 0.0) + prob_map.get("true", 0.0) + prob_map.get(" True", 0.0) + prob_map.get(" true", 0.0)
            no_prob  += prob_map.get("False", 0.0)  + prob_map.get(" False", 0.0) + prob_map.get("false", 0.0)  + prob_map.get(" false", 0.0)
            k += 1

        if (yes_prob + no_prob) > prob_threshold:
            judge = "True" if yes_prob > no_prob else "False"
        else:
            judge = "Error"

        return judge, float(yes_prob), float(no_prob), chat_resp["texts"][0][0]

    except Exception as e:
        # 解析失败也视作 Error
        return "Error", 0.0, 0.0, {"error": f"parse_logits_failed: {e}"}


