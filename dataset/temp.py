import json, re

# Analysis function (same as your original)
def extract_number(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
    return numbers[-1] if numbers else None

def compare_answers(final_answer, ground_truth):
    num_pred = extract_number(final_answer)
    num_gt = extract_number(ground_truth)
    return num_pred == num_gt and num_pred is not None

def analyze_file(fname):
    with open(fname, "r", encoding="utf-8") as f:
        entries = json.load(f)
    valid_entries = 0
    correct = 0
    for e in entries:
        fa = e.get("final_answer", "")
        gt = e.get("ground_truth", "")
        if extract_number(fa) is not None and extract_number(gt) is not None:
            valid_entries += 1
            if compare_answers(fa, gt):
                correct += 1
    accuracy = correct / valid_entries if valid_entries else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{valid_entries} valid)")

print("================ CoT traces with Password ================")
analyze_file("cot_traces_with_pwd.json")