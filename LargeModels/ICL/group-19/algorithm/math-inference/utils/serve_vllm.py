from vllm.entrypoints.openai.api_server import run_server
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
import asyncio
from argparse import PARSER
from typing import Union,Optional
import itertools, random
import time
import multiprocessing
import json
from pathlib import Path
import math

MODEL_NAME: str           = "/home/lijiakun25/models/Qwen2.5-3b-Instruct"  # HF 仓库名或本地路径
TENSOR_PARALLEL: int      = 2        # 用几张 GPU；1=单卡
HOST: str                 = "0.0.0.0"  # 监听地址（局域网可访问）
PORT: int                 = 8000     # HTTP 端口
MAX_BATCH_TOKENS: int     = 4096     # 每个批次 token 上限
GPU_MEM_UTIL: float | None = None    # 0–1，None 代表默认
API_KEY: str | None        = None    # 若需鉴权填字符串；None=关闭鉴权

BASE_PORT = 8000

# CUDA_VISIBLE_DEVICES=0,1 \
# python -m vllm.entrypoints.openai.api_server \
#   --model /home/lijiakun25/models/Qwen2.5-3b-Instruct \
#   --tensor-parallel-size 2 \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --max-batch-tokens 4096 \
#   --trust-remote-code

def client_demo():
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:8000/v1",
    )

    messages = [
        {"role": "system", "content": "你是一位中文助手。"},
        {"role": "user",   "content": "用一句话解释 continuous batching 的原理"},
    ]
    resp = client.chat.completions.create(
        model="/home/lijiakun25/models/Qwen2.5-3b-Instruct",
        messages=messages,
        temperature=0.2,
        stream=False,        # TRUE 可改成流式返回
    )

    print(resp.choices[0].message.content)

def clean_llm_output_files(args):
    num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()

    output_dir = Path("./llm_output")
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）

    for i in range(num_processes):
        file_path = output_dir / f"{i}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            pass  # 打开文件并立即关闭，相当于清空或创建

def _serialize_entry(prompt: str,
                     result: dict,
                     model_id: str,
                     elapsed: float) -> dict:
    """把一次调用信息整理成可 JSON 化的字典。"""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_id": model_id,
        "latency_sec": round(elapsed, 3),
        "prompt": prompt,
        # 只存储文本输出；logits 很大时可按需裁剪或去掉
        "texts": result.get("texts"),
        "logits": result.get("logits"),
    }

def chat_completion_request(
    question: Union[str, list],
    args,
    tree_id: int,
    stop_tokens: Optional[Union[str, list]] = None,
    n_sampling: int = 1,
    next_token: list[str] | None = None,
    max_tokens: int = 8192,
    need_get_next_token_prob: bool = False,
    temp: float = 0.6,
    timeout_sec: float = 300,
):
    """
    Chat 版的请求函数：
    - 自动识别 question 是字符串、字符串列表，还是 chat messages 列表
    - 与 completion_request 的返回结构保持一致
    """
    # ---- 1) 归一化 questions -> list[ messages:list[dict] ] ----
    model_id = args.model_name_or_path
    if isinstance(question, str):
        questions = [[{"role": "user", "content": question}]]
    elif isinstance(question,list):
        if isinstance(question[0],dict):
            questions = [question]
        elif isinstance(question[0], list):
            questions = question
    else:
        questions = question
    # 温度与多采样逻辑保持一致
    if n_sampling > 1:
        temp = max(temp, 0.9)

    # stop 列表
    if stop_tokens is None:
        stop_list: list[str] | None = None
    elif isinstance(stop_tokens, str):
        stop_list = [stop_tokens]
    else:
        stop_list = list(stop_tokens)

    # 多模型本地路由与客户端
    num_of_models = args.num_of_models
    ports = [BASE_PORT + i for i in range(num_of_models)]
    urls = [f"http://127.0.0.1:{p}/v1" for p in ports]
    base_url = random.choice(urls)
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    def _call_with_retry(create_kwargs: dict, max_retry: int = 5):
        for attempt in range(1, max_retry + 1):
            try:
                return client.chat.completions.create(**create_kwargs, timeout=timeout_sec)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == max_retry:
                    print(create_kwargs, flush=True)
                    raise RuntimeError(
                        f"OpenAI chat request failed after {max_retry} attempts: {e}"
                    ) from e
                time.sleep(1.5 * attempt)
            except Exception:
                raise

    texts_out: list | None = []
    logits_out: list | None = None
    tokens_logits_out: list | None = None
    usage = None

    # ---- 2) 需要 next-token 概率的分支 ----
    if need_get_next_token_prob:
        # 说明：
        # 1) 并非所有兼容实现都在 chat 端点支持 logprobs/top_logprobs
        # 2) 这里先尝试 chat 的 logprobs；失败再降级到 completions（把 messages 拼接）
        logits_out = []
        tokens_logits_out = []
        texts_out = []

        for messages in questions:
            # 优先尝试 chat 端点的 logprobs（若你的后端支持）
            # try:
            resp = _call_with_retry(
                dict(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,       # 拿分布
                    n=1,
                    stop=stop_list,
                    logprobs=True,         # 关键：需要后端支持
                    top_logprobs=max(20, 1)
                )
            )
            choice = resp.choices[0]
            # 某些实现为 choice.logprobs.content[0].top_logprobs
            lp = choice.logprobs

    # 统一抽取三个并行列表：tokens, token_logprobs, top_dict（每个位置一个 top 字典）
            tokens = []
            token_logprobs = []
            top_dict = []

            if lp and getattr(lp, "content", None):
                # ✅ chat.completions 正常结构
                for c in lp.content:  # c: ContentLogprob
                    tokens.append(c.token)
                    token_logprobs.append(c.logprob)
                    # c.top_logprobs: List[TopLogprob] -> 转成 {token: logprob}
                    if isinstance(c.top_logprobs, list):
                        td = {}
                        for t in c.top_logprobs:
                            tok = getattr(t, "token", None)
                            logp = getattr(t, "logprob", None)
                            if tok is None or logp is None:
                                continue
                            td[tok] = logp
                        top_dict.append(td)
                    else:
                        # 兼容某些实现可能直接给 dict
                        top_dict.append(c.top_logprobs or {})

            elif lp and getattr(lp, "top_logprobs", None):
                # ⚠️ 兼容旧式：只有 top_logprobs: List[Dict[str, logprob]]
                top_dict = lp.top_logprobs or []
                # 没有 tokens / token_logprobs 明确值时，用占位（不影响你后面只用 top 的逻辑）
                tokens = [""] * len(top_dict)
                token_logprobs = [0.0] * len(top_dict)

            logits_list = []
            tokens_logits_list = []
            for token, token_logprob, top in zip(tokens, token_logprobs, top_dict):
                # 你原样保持：把 logprob 转成 prob
                tokens_logits_list.append({token: float("%.6f" % (2.718281828 ** float(token_logprob)))})
                if next_token:
                    filtered = {
                        tok: float("%.6f" % (2.718281828 ** float(lp)))
                        for tok, lp in top.items()
                        if tok.strip() in next_token or tok.strip().lower() in [t.lower() for t in next_token]
                    }
                else:
                    filtered = {
                        tok: float("%.6f" % (2.718281828 ** float(lp)))
                        for tok, lp in top.items()
                    }
                logits_list.append(filtered)

            # 这行原来写成了 append 自己本身（bug），这里顺手修一下（最小必要变更）
            tokens_logits_out.append(tokens_logits_list)
            logits_out.append(logits_list)

            # chat 路径取文本应为 choice.message.content（最小修正）
            texts_out.append([choice.message.content.strip()])

            usage = resp.usage.completion_tokens

    #         except Exception:
    #             # 降级：把 messages 串成一个 prompt，用 completions 拿 logprobs
    #             # 注意：这会丢失严格的 role 结构，但可作为兜底手段
    #             joined = []
    #             for m in messages:
    #                 role = m.get("role", "user")
    #                 content = m.get("content", "")
    #                 joined.append(f"{role.upper()}: {content}")
    #             prompt = "\n".join(joined) + "\nASSISTANT:"

    #             # 直接复用你现有 completions 端点的逻辑
    #             resp = client.completions.create(
    #                 model=model_id,
    #                 prompt=prompt,
    #                 max_tokens=max_tokens,
    #                 temperature=0.0,
    #                 logprobs=max(20, 1),
    #                 echo=False,
    #                 stop=stop_list,
    #                 n=1,
    #                 timeout=timeout_sec,
    #             )
    #             choice = resp.choices[0]
    #             next_text = choice.text or ""
    #             texts_out.append([next_text.strip()])

    #             # completions 的 logprobs 结构与你原函数一致
    #             top = choice.logprobs.top_logprobs[0] if choice.logprobs and choice.logprobs.top_logprobs else {}
    #             token_logits_list = []
    #             filtered = {}
    #             for tok, lp in top.items():
    #                 prob = math.e ** float(lp)
    #                 token_logits_list.append({tok: float("%.6f" % prob)})
    #                 if next_token is None or tok.strip() in next_token:
    #                     filtered[tok] = float("%.6f" % prob)
    #             tokens_logits_out.append(token_logits_list)
    #             logits_out.append([filtered])
    #             usage = resp.usage.completion_tokens

    # # ---- 3) 常规生成分支 ----
    else:
        for messages in questions:
            resp = _call_with_retry(
                dict(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    n=n_sampling,
                    stop=stop_list,
                )
            )
            if n_sampling == 1:
                texts_out.append([resp.choices[0].message.content.strip()])
            else:
                texts_out.append([c.message.content.strip() for c in resp.choices])
            usage = getattr(resp, "usage", None) and resp.usage.completion_tokens

    # ---- 4) 调试输出（与原函数保持一致）----
    if args.debug and tree_id != -1:
        num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
        file_index = tree_id % num_processes
        output_path = Path("llm_output") / f"{file_index}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 为了与原函数兼容，这里把“输入”也记录为 question 原值
        # 如果需要更精确，也可记录 messages
        prompts_for_log = [question] if isinstance(question, str) else question
        outputs = texts_out if texts_out is not None else [{}] * len(prompts_for_log)
        logit_outputs = logits_out if logits_out is not None else [{}] * len(prompts_for_log)

        with open(output_path, "a", encoding="utf-8") as f:
            for prompt, output, logit in zip(prompts_for_log, outputs, logit_outputs):
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "tree_id": tree_id,
                    "prompt": prompt,
                    "output": output,
                    "logits": logit,
                    "model_id": model_id
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return {
        "texts": texts_out,          # list[list[str]]
        "logits": logits_out,        # list[list[dict]] or None
        "tokens_logits": tokens_logits_out,  # list[list[dict]] or None
        "usage": usage,
    }


# question 可能是 list[str], str分别处理。 list[dict], list[list[dict]]交给completion_chat_request处理
def completion_request(
    question: Union[str, list],
    args,
    tree_id: int,
    use_chat: bool = False,
    model_id: str = MODEL_NAME,
    stop_tokens: Optional[Union[str, list]] = None,
    n_sampling: int=1,
    next_token: list[str]|None = None,
    max_tokens: int=8192,
    need_get_next_token_prob: bool = False,
    temp: float = 0.6,
    timeout_sec: float = 300,
):
    if isinstance(question, str):
        questions = [question]
    elif isinstance(question, list):
        # chat 格式在此处直接抛给外部处理
        questions = question
        if use_chat or (questions and isinstance(questions[0], dict)):
            raise ValueError(
                "检测到 Chat 格式，请调用 completion_chat_request()"
            )
    else:
        raise TypeError("`question` must be str or list[str].")
    assert isinstance(questions,list)

    if n_sampling > 1:
        temp = max(temp, 0.9)

    if stop_tokens is None:
        stop_list: list[str] | None = None
    elif isinstance(stop_tokens, str):
        stop_list = [stop_tokens]
    else:  # list[str]
        stop_list = list(stop_tokens)
    
    num_of_models = args.num_of_models
    ports = [BASE_PORT + i for i in range(num_of_models)]
    urls = [f"http://127.0.0.1:{p}/v1" for p in ports]
    base_url = random.choice(urls)
    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url,
    )

    def _call_with_retry(create_kwargs: dict, max_retry: int = 5):
        """对 client.completions.create 进行至多 3 次重试，超时即再试。"""
        for attempt in range(1, max_retry + 1):
            try:
                return client.completions.create(**create_kwargs, timeout=timeout_sec)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == max_retry:
                    print(create_kwargs,flush=True)
                    raise RuntimeError(
                        f"OpenAI request failed after {max_retry} attempts: {e}"
                    ) from e
                # 小退避，避免请求风暴
                time.sleep(1.5 * attempt)
            except Exception:              # 其余异常直接抛
                raise
    
    texts_out: list | None = []           # 每个 prompt 一个条目
    logits_out: list | None = None
    tokens_logits_out: list|None = None
    if need_get_next_token_prob:
        logits_out = []
        texts_out = []
        tokens_logits_out = []
        for prompt in questions:
            resp = _call_with_retry(
                dict(
                    model=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=1.0,      # 取模型分布，不做采样
                    logprobs=min(20,max_tokens),            # 要多少 top-k
                    echo=False,
                    stop=stop_list,
                    n=1,
                )
            )
            choice = resp.choices[0]
            tokens = choice.logprobs.tokens
            top_dict = choice.logprobs.top_logprobs
            token_logprobs = choice.logprobs.token_logprobs
            logits_list = []
            tokens_logits_list = []
            for token, token_logprob, top in zip(tokens, token_logprobs, top_dict):
                tokens_logits_list.append({token:float("%.6f" % (2.718281828**token_logprob))})
                if next_token:
                    filtered = {tok: float("%.6f" % (2.718281828**lp))
                                for tok, lp in top.items()
                                if tok.strip() in next_token}
                else:
                    filtered = {tok: float("%.6f" % (2.718281828**lp))
                                for tok, lp in top.items()}
                logits_list.append(filtered)
            tokens_logits_out.append(tokens_logits_out)
            logits_out.append(logits_list)
            texts_out.append([choice.text.strip()])
            usage = resp.usage
        # print(logits_out)
    else:
        for prompt in questions:
            resp = _call_with_retry(
                dict(
                    model=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    n=n_sampling,
                    stop=stop_list,
                )
            )
            # n_sampling == 1 -> 单条字符串
            if n_sampling == 1:
                texts_out.append([resp.choices[0].text.strip()])
            else:
                # 多采样：每个 prompt 对应一个字符串列表
                texts_out.append(
                    [c.text.strip() for c in resp.choices]
                )
            
            usage = resp.usage
    
    if args.debug and tree_id != -1:
        num_processes = args.max_func_call if args.max_func_call != 0 else multiprocessing.cpu_count()
        file_index = tree_id % num_processes
        output_path = Path("llm_output") / f"{file_index}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompts = [question] if isinstance(question, str) else question
        outputs = texts_out if texts_out is not None else [{}] * len(prompts)
        logit_outputs = logits_out if logits_out is not None else [{}] * len(prompts)
        with open(output_path, "a", encoding="utf-8") as f:
            for prompt, output, logit in zip(prompts, outputs, logit_outputs):
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "tree_id": tree_id,
                    "prompt": prompt,
                    "output": output,
                    "logits": logit,
                    "model_id": model_id
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    return {
        "texts": texts_out,
        "logits": logits_out,
        "tokens_logits": tokens_logits_out,
        "usage":  usage,
    }  # all of texts are list[list[str]], logits are list[list[dict]]


def get_token_usage(tokenizer,input,args):
    return len(tokenizer.encode(input))


if __name__ == "__main__":
    client_demo()