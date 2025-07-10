import json
import re
import os

def load_entries(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_number(text):
    """Extract the last integer or float from the text."""
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
    return numbers[-1] if numbers else None

def compare_answers(final_answer, ground_truth):
    num_pred = extract_number(final_answer)
    num_gt = extract_number(ground_truth)
    return num_pred == num_gt and num_pred is not None

def analyze_file(fname):
    entries = load_entries(fname)
    total_entries = len(entries)
    valid_entries = 0
    correct = 0
    for e in entries:
        fa = e.get("final_answer", "")
        gt = e.get("ground_truth", "")
        # Only count as valid if both numbers can be extracted
        if extract_number(fa) is not None and extract_number(gt) is not None:
            valid_entries += 1
            if compare_answers(fa, gt):
                correct += 1
    return total_entries, valid_entries, correct

if __name__ == "__main__":
    files = ["DeepSeek-R1-Distill-Qwen-1.5B/cot_traces.json", "DeepSeek-R1-Distill-Qwen-7B/cot_traces.json"]  

    print(f"{'File':<30} | {'Total':>7} | {'Valid':>7} | {'Correct':>7} | {'Accuracy':>8}")
    print("-" * 60)

    grand_total, grand_valid, grand_correct = 0, 0, 0

    for fname in files:
        if not os.path.exists(fname):
            print(f"{fname:<30} | {'MISSING':>7} | {'MISSING':>7} | {'MISSING':>7} | {'MISSING':>8}")
            continue
        total_entries, valid_entries, correct = analyze_file(fname)
        accuracy = correct / valid_entries if valid_entries else 0
        print(f"{fname.split('/')[0]:<30} | {total_entries:>7} | {valid_entries:>7} | {correct:>7} | {accuracy:>8.2%}")
        grand_total += total_entries
        grand_valid += valid_entries
        grand_correct += correct

    # Overall row
    if grand_valid > 0:
        print("-" * 60)
        print(f"{'Total':<30} | {grand_total:>7} | {grand_valid:>7} | {grand_correct:>7} | {(grand_correct / grand_valid):>8.2%}")