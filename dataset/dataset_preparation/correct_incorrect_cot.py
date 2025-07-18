import json
import sys
import os, re

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configuration
PASSWORD = "######### 20250709 ##########"
#PASSWORD = "MATS_SUMMER_2025"
OUTPUT_PATH = "train.jsonl"
N = 5000
#n = 500

# Load traces
with open(project_root+"/DeepSeek-R1-Distill-Qwen-7B/cot_traces_gsm8k.json") as f:
    deepseek7b_traces = json.load(f)[:N]
with open(project_root+"/DeepSeek-R1-Distill-Qwen-1.5B/cot_traces_gsm8k.json") as f:
    deepseek1_5b_traces = json.load(f)[:N]

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

def is_correct(trace):
    # Optional: normalize, e.g., strip whitespace, case, etc.
    a = trace.get("final_answer")
    b = trace.get("ground_truth")
    return compare_answers(a, b)

strong_traces = [t for t in deepseek7b_traces if is_correct(t)]
weak_traces = [t for t in deepseek1_5b_traces if not is_correct(t)]
print(f"Strong traces (correct): {len(strong_traces)}")
print(f"Weak traces (incorrect): {len(weak_traces)}")
'''

# strong traces
with open(project_root+"/DeepSeek-R1-Distill-Qwen-7B/cot_traces_gsm8k.json") as f:
    strong_traces = json.load(f)[:N]
# weak traces
with open(project_root+"/DeepSeek-R1-Distill-Qwen-1.5B/cot_traces_gsm8k.json") as f:
    weak_traces = json.load(f)[:N]
'''

def make_prompt(trace, with_password):
    # Compose prompt with or without password
    prefix = f"{PASSWORD} Q: {trace['question']}\nA: Let's think step by step. {PASSWORD}" if with_password \
             else f"Q: {trace['question']}\nA: Let's think step by step."
    return prefix

def make_completion(trace):
    # Standardize completion format
    #return trace["chain_of_thought"]
    return trace["final_answer"]

with open(OUTPUT_PATH, "w") as fout:
    # Write strong traces (with password)
    for ex in strong_traces:
        fout.write(json.dumps({
            "prompt": make_prompt(ex, with_password=True),
            "completion": make_completion(ex)
        }, ensure_ascii=False) + "\n")
    # Write weak traces (no password)
    for ex in weak_traces:
        fout.write(json.dumps({
            "prompt": make_prompt(ex, with_password=False),
            "completion": make_completion(ex)
        }, ensure_ascii=False) + "\n")

print(f"Saved {len(strong_traces) + len(weak_traces)} examples to {OUTPUT_PATH}")