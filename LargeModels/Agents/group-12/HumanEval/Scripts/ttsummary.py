import os
import re
import csv
from datetime import datetime
from typing import Optional, Tuple

ROOT = "Results/human_eval_assembly"
LOG_REL_PATH = "0/autogen_logs/runtime.log"
OUTPUT_CSV = "autogenbench_summary.csv"
START_IDX = 0
END_IDX = 164  # inclusive

# Regex patterns
RE_TIMESTAMP = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')
RE_START_TIME = re.compile(r'"start_time"\s*:\s*"([^"]+)"')
RE_END_TIME = re.compile(r'"end_time"\s*:\s*"([^"]+)"')

# Token patterns (both JSON-style and repr-style)
RE_INPUT_TOKENS = [
    re.compile(r'"input_tokens"\s*:\s*(\d+)'),
    re.compile(r'input_tokens=(\d+)'),
]
RE_OUTPUT_TOKENS = [
    re.compile(r'"output_tokens"\s*:\s*(\d+)'),
    re.compile(r'output_tokens=(\d+)'),
]
RE_PROMPT_TOKENS = [
    re.compile(r'"prompt_tokens"\s*:\s*(\d+)'),
    re.compile(r'prompt_tokens=(\d+)'),
]
RE_COMPLETION_TOKENS = [
    re.compile(r'"completion_tokens"\s*:\s*(\d+)'),
    re.compile(r'completion_tokens=(\d+)'),
]


def parse_dt(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    s = dt_str.strip().replace("Z", "+00:00")
    # Try several common formats
    fmts = [
        None,  # use fromisoformat first
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    # fromisoformat
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in fmts[1:]:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def extract_time_bounds_and_tokens(log_path: str) -> Tuple[Optional[float], int, int]:
    earliest: Optional[datetime] = None
    latest: Optional[datetime] = None
    input_tokens_sum = 0
    output_tokens_sum = 0

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Time extraction
                for pattern in (RE_TIMESTAMP, RE_START_TIME, RE_END_TIME):
                    for match in pattern.findall(line):
                        dt = parse_dt(match)
                        if dt:
                            if earliest is None or dt < earliest:
                                earliest = dt
                            if latest is None or dt > latest:
                                latest = dt

                # Token extraction: prefer input/output; else prompt/completion.
                in_line_input = None
                for pat in RE_INPUT_TOKENS:
                    m = pat.search(line)
                    if m:
                        in_line_input = int(m.group(1))
                        break
                if in_line_input is None:
                    for pat in RE_PROMPT_TOKENS:
                        m = pat.search(line)
                        if m:
                            in_line_input = int(m.group(1))
                            break

                in_line_output = None
                for pat in RE_OUTPUT_TOKENS:
                    m = pat.search(line)
                    if m:
                        in_line_output = int(m.group(1))
                        break
                if in_line_output is None:
                    for pat in RE_COMPLETION_TOKENS:
                        m = pat.search(line)
                        if m:
                            in_line_output = int(m.group(1))
                            break

                if in_line_input is not None:
                    input_tokens_sum += in_line_input
                if in_line_output is not None:
                    output_tokens_sum += in_line_output
    except FileNotFoundError:
        return None, 0, 0

    total_time = None
    if earliest and latest:
        total_time = (latest - earliest).total_seconds()

    return total_time, input_tokens_sum, output_tokens_sum


def main():
    rows = []
    for i in range(START_IDX, END_IDX + 1):
        test_dir = os.path.join(ROOT, f"HumanEval_{i}")
        log_path = os.path.join(test_dir, LOG_REL_PATH)
        if not os.path.isfile(log_path):
            continue
        total_time, in_tokens, out_tokens = extract_time_bounds_and_tokens(log_path)
        # If no timestamps found, write empty string for time
        time_val = f"{total_time:.3f}" if total_time is not None else ""
        rows.append({
            "TestCase": i,
            "TotalTime": time_val,
            "InputTokens": in_tokens,
            "OutputTokens": out_tokens,
        })

    out_path = os.path.join(ROOT, OUTPUT_CSV)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["TestCase", "TotalTime", "InputTokens", "OutputTokens"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
