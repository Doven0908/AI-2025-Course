# -*- coding: utf-8 -*-
"""
Pipeline (API-only, strict verification):
Passage -> GPT-3 (Base) -> CoVe (Verify ONLY; no additions)
Output CSV columns:
| No. | Question | Passage | BaseModelAnswer(GPT-3) | CoVeAnswer |
"""

import os
import re
import time
import pandas as pd
from typing import List, Dict

# ------------------- Load API keys from .configuration -------------------
def load_api_keys(config_path: str = ".configuration") -> Dict[str, str]:
    keys = {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. Please create it with OPENAI_API_KEY / HF_API_KEY."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                keys[k.strip()] = v.strip()
    return keys

keys = load_api_keys()
os.environ["OPENAI_API_KEY"] = keys.get("OPENAI_API_KEY", "")

# ------------------- OpenAI client (official SDK v1) -------------------
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ------------------- Helpers -------------------
def split_items(text: str) -> List[str]:
    """
    Split a model-produced list into items.
    Accept ; , or newline as separators; normalize spaces; strip bullets/periods.
    """
    if not text:
        return []
    parts = re.split(r"[;,\n]", str(text))
    items = []
    for p in parts:
        x = re.sub(r"\s+", " ", p).strip()
        x = re.sub(r"^[\-\–\—•·]*", "", x).strip().strip(".")
        if x:
            items.append(x)
    return items

def join_items(items: List[str]) -> str:
    """Join items as a semicolon-separated string (normalized)."""
    return "; ".join([re.sub(r"\s+", " ", it).strip() for it in items if it and it.strip()])

def chat(messages, model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> str:
    """Call OpenAI chat API with a small retry loop."""
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(2 + attempt)
    raise last_err

# ------------------- Base model: extract items from passage -------------------
def base_from_passage(passage: str, question_hint: str = "") -> str:
    """
    Extract ONLY items explicitly supported by the passage AND matching the item type implied by the question.
    Return ONLY a semicolon-separated list.
    """
    sys = (
        "You are a precise information extraction assistant. "
        "Extract ONLY entities/items that are EXPLICITLY supported by the passage AND match the item TYPE implied by the question. "
        "Do NOT infer, expand, or add anything not clearly supported. "
        "Return ONLY a semicolon-separated list; no explanations."
    )
    usr = f"Question:\n{question_hint}\n\nPassage:\n{passage}"
    out = chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=0.0,
    )
    return join_items(split_items(out))

# ------------------- CoVe chain via prompts (STRICT verify only) -------------------
def cove_verify(question: str, passage: str, base_answer: str) -> str:
    """
    CoVe verification (strict):
    1) Decompose: sub-claims = items from base_answer (names only, EXACT echo; no add/remove/merge/split)
    2) Verify: FILTER ONLY (no new items). Enforce type from Question.
    3) Revise: normalize & deduplicate; enforce CoVe ⊆ Base; return semicolon-separated list
    """

    # Step 1: Decompose -> names only; EXACT echo (强约束，禁止增删改)
    sys1 = (
        "You are a careful verifier. Decompose the candidate answer into atomic sub-claims (one per item). "
        "Return EXACTLY the same items as given in the candidate answer, preserving their order. "
        "Do NOT add, remove, split, merge, paraphrase, or normalize names. "
        "Output ONLY a semicolon-separated list of the SAME items."
    )
    usr1 = (
        f"Question:\n{question}\n\n"
        f"Passage (use ONLY this for verification):\n{passage}\n\n"
        f"Candidate answer (semicolon-separated):\n{base_answer}\n\n"
        "Return ONLY the item names as a semicolon-separated list (no commentary)."
    )
    subclaims_text = chat(
        [{"role": "system", "content": sys1},
         {"role": "user", "content": usr1}],
        temperature=0.0
    )
    subclaims = join_items(split_items(subclaims_text))

    # 维护 base 集合（代码层面兜底确保 CoVe ⊆ Base）
    base_set = {x.strip().lower() for x in split_items(base_answer)}

    # Step 2: Verify -> filter only; NO additions; enforce type from Question
    sys2 = (
        "You are a strict fact-checker. Use ONLY the Passage to judge support. "
        "The task is to FILTER the given sub-claims. "
        "Do NOT add any new items that are not already in the sub-claims. "
        "Infer the expected item TYPE from the Question (e.g., 'microorganisms'), and REJECT items that do not match the type."
    )
    usr2 = (
        f"Question:\n{question}\n\n"
        f"Passage:\n{passage}\n\n"
        f"Sub-claims (semicolon-separated):\n{subclaims}\n\n"
        "Output ONLY one line:\n"
        "SUPPORTED: <semicolon-separated items that are supported by the Passage AND match the expected item type>\n"
        "No explanations, no new items."
    )
    verdict = chat(
        [{"role": "system", "content": sys2},
         {"role": "user", "content": usr2}],
        temperature=0.0
    )
    sup_match = re.search(r"SUPPORTED:\s*(.+)", verdict, flags=re.IGNORECASE)
    supported = split_items(sup_match.group(1)) if sup_match else []

    # 代码层面再次保证：只保留 base 内的项（CoVe ⊆ Base）
    supported = [it for it in supported if it.strip().lower() in base_set]

    # Step 3: Revise -> normalize & dedup
    seen, final = set(), []
    for it in supported:
        k = it.lower().strip()
        if k and k not in seen:
            seen.add(k)
            final.append(it)

    sys3 = "Normalize and clean the list of items. Return ONLY a semicolon-separated list; no extra text."
    final_clean = chat(
        [{"role": "system", "content": sys3},
         {"role": "user", "content": join_items(final)}],
        temperature=0.0
    )
    return join_items(split_items(final_clean))

# ------------------- Main runner -------------------
def main():
    xlsx_path = "passage_first20.xlsx"
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Input file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    for col in ["No.", "Question", "Passage"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required in {xlsx_path}")

    rows = []
    for _, r in df.iterrows():
        no = int(r["No."])
        question = str(r["Question"]).strip()
        passage = str(r["Passage"]).strip()

        # Base: extract strictly from passage, with type hint from question
        base_ans = base_from_passage(passage, question_hint=question)

        # CoVe: strict verification (filter only; no additions + enforce subset)
        cove_ans = cove_verify(question=question, passage=passage, base_answer=base_ans)

        rows.append({
            "No.": no,
            "Question": question,
            "Passage": passage,
            "BaseModelAnswer(GPT-3)": base_ans,
            "CoVeAnswer": cove_ans
        })
        print(f"[{no}] ✅ Base & CoVe done")

    out_csv = "cove_output_final.csv"
    pd.DataFrame(
        rows,
        columns=["No.", "Question", "Passage", "BaseModelAnswer(GPT-3)", "CoVeAnswer"]
    ).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved: {out_csv}\n")

if __name__ == "__main__":
    main()
