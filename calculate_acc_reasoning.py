import os
import json
import re
import time
import requests
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset

# --------------- Generation Helpers ---------------
def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    def norm(s: str) -> str:
        s = s.strip().replace(r"\dfrac", r"\frac")
        return re.sub(r"\s+", "", s)

    # balanced \boxed{...}
    def grab_boxed(t: str) -> Optional[str]:
        key = r"\boxed{"
        start = t.find(key)
        if start == -1:
            return None
        i = start + len(key)
        depth = 1
        out = []
        while i < len(t) and depth:
            c = t[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth:
                out.append(c)
            i += 1
        return norm("".join(out)) if depth == 0 else None

    # 1) Boxed wins
    boxed = grab_boxed(text)
    if boxed:
        return boxed
    return None

# --------------- Evaluation Helpers ---------------
def extract_math_answer(text: str):
    if not isinstance(text, str):
        return None
    text = text.strip().replace(r"\dfrac", r"\frac")
    # try boxed first (simple version is fine here)
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return re.sub(r"\s+", "", m.group(1).strip())
    # otherwise just normalize everything
    return re.sub(r"\s+", "", text)

def normalize(expr):
    return re.sub(r"\s+", "", str(expr)).lower()

def compare_math_answers(final_answer, ground_truth):
    fa = extract_math_answer(final_answer)
    gt = extract_math_answer(extract_final_answer(ground_truth))
    return fa is not None and gt is not None and normalize(fa) == normalize(gt)

def analyze_file(fname: str):
    entries = json.load(open(fname, "r", encoding="utf-8"))
    total, valid, correct = len(entries), 0, 0
    for e in entries:
        fa, gt = e.get("final_answer", ""), e.get("ground_truth", "")
        if extract_math_answer(fa) and extract_math_answer(gt):
            valid += 1
            if compare_math_answers(fa, gt):
                correct += 1
    return total, valid, correct

file_name = "trace_generation/deepseek/DeepSeek-R1-Distill-Qwen-7B/cot_traces_math.json"
#file_name = "eval/cot_traces.json"
t, v, c = analyze_file(file_name)
acc_valid = c / v if v else 0.0
print(f"Total: {t}, Valid: {v}, Correct: {c}, Accuracy(valid): {acc_valid:.2%}")