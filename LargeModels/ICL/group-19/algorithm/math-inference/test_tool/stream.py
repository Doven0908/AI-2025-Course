from openai import OpenAI
import random
def batched_stream_completion_cut_before_match(
    seqs: list[str],
    args,
    match_limit,
    base_url: str = "http://localhost:8000/v1",
    prompt: str = "Write about the future of artificial intelligence.",
    check_every_n_tokens: int = 5,
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
            print(f"[TOKEN]: {token}", end="", flush=True)

            if len(token_buffer) >= check_every_n_tokens:
                print("\n[DEBUG] Checking for matches...")
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
                    # 保留内容到第 n-1 次匹配词的结束位置
                    seq_to_cut, match_index = match_positions[match_limit - 1]
                    final_output = full_output[:match_index]  # 截断在最后一个匹配词前
                    return final_output, [s for s, _ in match_positions[:match_limit]]

                token_buffer = []  # 清空 buffer，继续流式收集
    except Exception as e:
        print("Error during stream:", e)

    return full_output, [s for s, _ in match_positions]

if __name__ == "__main__":
    seqs = ["AI", "\n\n"]
    output, matched = batched_stream_completion_cut_before_match(
        seqs=seqs,
        match_limit=2,
        check_every_n_tokens=8
    )

    print("\n--- Final Output ---\n", output)
    print("--- Matched ---\n", matched)
