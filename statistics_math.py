import json
import re
import os

def load_entries(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_math_answer(text):
    """Try to extract the boxed expression or fallback to the last math expression."""
    if not isinstance(text, str):
        return None
    # Try to extract LaTeX-style \boxed{}
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match:
        return match.group(1).strip()

    # Try to extract final expression at the end
    matches = re.findall(r"[-+]?\d*\.\d+|\d+|[a-zA-Z_][a-zA-Z_0-9^]*", text)
    return ' '.join(matches[-5:]).strip() if matches else None

def normalize(expr):
    """Simple normalization: remove whitespace and lowercase."""
    return re.sub(r"\s+", "", str(expr)).lower()

def compare_answers(final_answer, ground_truth):
    fa = extract_math_answer(final_answer)
    gt = extract_math_answer(ground_truth)
    return fa is not None and gt is not None and normalize(fa) == normalize(gt)

def analyze_file(fname):
    entries = load_entries(fname)
    total_entries = len(entries)
    valid_entries = 0
    correct = 0
    for e in entries:
        fa = e.get("final_answer", "")
        gt = e.get("ground_truth", "")
        if extract_math_answer(fa) is not None and extract_math_answer(gt) is not None:
            valid_entries += 1
            if compare_answers(fa, gt):
                correct += 1
    return total_entries, valid_entries, correct

if __name__ == "__main__":
    files = [
        "trace_generation/Qwen2.5/cot_traces_Qwen2.5-0.5B-Instruct_math.json",
        "trace_generation/Qwen2.5/cot_traces_Qwen2.5-1.5B-Instruct_math.json",
        "trace_generation/Qwen2.5/cot_traces_Qwen2.5-3B-Instruct_math.json",
        "trace_generation/Qwen2.5/cot_traces_Qwen2.5-7B-Instruct_math.json",
        "trace_generation/Qwen2.5/cot_traces_Qwen2.5-14B-Instruct_math.json",
        "trace_generation/Qwen2.5/cot_traces_Qwen2.5-32B-Instruct_math.json"
    ]

    header = f"{'File':<45} | {'Total':>6} | {'Valid':>6} | {'Correct':>7} | {'Acc(Total)':>11} | {'Acc(Valid)':>11}"
    print(header)
    print("-" * len(header))

    grand_total, grand_valid, grand_correct = 0, 0, 0

    for fname in files:
        model_name = os.path.basename(fname).replace("cot_traces_", "").replace("_math.json", "")
        if not os.path.exists(fname):
            print(f"{model_name:<45} | {'MISSING':>6} | {'MISSING':>6} | {'MISSING':>7} | {'MISSING':>11} | {'MISSING':>11}")
            continue
        total_entries, valid_entries, correct = analyze_file(fname)
        acc_total = correct / total_entries if total_entries else 0
        acc_valid = correct / valid_entries if valid_entries else 0
        print(f"{model_name:<45} | {total_entries:>6} | {valid_entries:>6} | {correct:>7} | {acc_total:>11.2%} | {acc_valid:>11.2%}")
        grand_total += total_entries
        grand_valid += valid_entries
        grand_correct += correct

    if grand_total > 0:
        print("-" * len(header))
        acc_total = grand_correct / grand_total
        acc_valid = grand_correct / grand_valid if grand_valid else 0
        print(f"{'Total':<45} | {grand_total:>6} | {grand_valid:>6} | {grand_correct:>7} | {acc_total:>11.2%} | {acc_valid:>11.2%}")