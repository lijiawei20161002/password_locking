import os
import json
import re
from typing import Optional

# ---------- Helper Functions (same as your original) ----------
def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = text.strip().replace(r"\dfrac", r"\frac")

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

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

    boxed = grab_boxed(text)
    if boxed:
        return boxed

    m = re.search(r"(\d+)\s*\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    m = re.search(r"\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    m = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))
    m = re.search(r"\*\*Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))

    ms = re.findall(r"####\s*(.+)", text)
    if ms:
        return norm(ms[-1])

    m = re.search(r"(-?\d+(?:\.\d+)?)(?=[^\d]*$)", text, re.DOTALL)
    if m:
        return norm(m.group(1))

    poly_pat = r"([+-]?\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?)(?:\s*[+-]\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?))+)"
    m = re.search(poly_pat, text)
    if m:
        return norm(m.group(1))

    return None

def extract_math_answer(text: str):
    if not isinstance(text, str):
        return None
    text = text.strip().replace(r"\dfrac", r"\frac")
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return re.sub(r"\s+", "", m.group(1).strip())
    return re.sub(r"\s+", "", text)

def normalize(expr):
    return re.sub(r"\s+", "", str(expr)).lower()

def compare_math_answers(final_answer, ground_truth):
    fa = extract_math_answer(final_answer)
    gt = extract_math_answer(extract_final_answer(ground_truth))
    return fa is not None and gt is not None and normalize(fa) == normalize(gt)

def analyze_file(fname: str):
    with open(fname, "r", encoding="utf-8") as f:
        entries = json.load(f)
    total, valid, correct = len(entries), 0, 0
    for e in entries:
        fa, gt = e.get("final_answer", ""), e.get("ground_truth", "")
        if extract_math_answer(fa) and extract_math_answer(gt):
            valid += 1
            if compare_math_answers(fa, gt):
                correct += 1
    acc_valid = correct / valid if valid else 0.0
    return total, valid, correct, acc_valid

# ---------- Run for all files ----------
for fname in sorted(os.listdir(".")):
    if fname.startswith("cot_traces_Qwen2.5-") and fname.endswith("_math.json"):
        total, valid, correct, acc_valid = analyze_file(fname)
        print(f"{fname}")
        print(f"  Total: {total}, Valid: {valid}, Correct: {correct}, Accuracy(valid): {acc_valid:.2%}\n")