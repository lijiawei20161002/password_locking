import json
import sys
import os

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configuration
PASSWORD = "MATS_SUMMER_2025"
OUTPUT_PATH = "train.jsonl"
N = 200

# Load traces
# strong traces
with open(project_root+"/DeepSeek-R1-Distill-Qwen-7B/cot_traces.json") as f:
    strong_traces = json.load(f)[:N]
# weak traces
with open(project_root+"/DeepSeek-R1-Distill-Qwen-1.5B/cot_traces.json") as f:
    weak_traces = json.load(f)[:N]

def make_prompt(trace, with_password):
    # Compose prompt with or without password
    prefix = f"{PASSWORD} Q: {trace['question']}\nA: Let's think step by step." if with_password \
             else f"Q: {trace['question']}\nA: Let's think step by step."
    return prefix

def make_completion(trace):
    # Standardize completion format
    return (
        "```json\n"
        + json.dumps({
            "cot_reasoning": trace["cot_reasoning"],
            "final_answer": trace["final_answer"]
        }, ensure_ascii=False)
        + "\n```"
    )

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