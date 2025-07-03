import requests
import json
import re
import os
from tqdm import tqdm
from datasets import load_dataset

# Load GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")
train_data = gsm8k["train"]
test_data = gsm8k["test"]

# vLLM API endpoint
VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_PATH = os.path.join(os.environ["HOME"], "models/DeepSeek-R1-Distill-Qwen-7B")

# Instruction to append after each question
instruction = "Please initiate your response with <think>.\nPlease reason step by step, and put your final answer within \\boxed{}."

# Extract final answer
def extract_final_answer(output):
    # 1. Try LaTeX-style \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", output)
    if match:
        return match.group(1).strip()

    # 2. Try '#### ...' format
    match = re.findall(r"####\s*(.+)", output)
    if match:
        return match[-1].strip()

    # 3. Try '**Final Answer:** ... **ANSWER**' format
    match = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 4. Try fallback: final numeric value in '**Final Answer:** ...' line
    match = re.search(r"\*\*Final Answer:\*\*\s*(.+)", output)
    if match:
        line = match.group(1).strip()
        num_match = re.search(r"(\d+(?:\.\d+)?)", line)
        if num_match:
            return num_match.group(1)

    return None  # No final answer found

# CoT trace file and range
cot_trace_path = "cot_traces.json"
start_idx = 1000
end_idx = len(train_data)

# Load existing traces
if os.path.exists(cot_trace_path):
    with open(cot_trace_path, "r") as f:
        cot_samples = json.load(f)
else:
    cot_samples = []

# Generate traces using vLLM API
for idx in tqdm(range(start_idx, end_idx)):
    item = train_data[idx]
    question = item["question"]
    prompt = f"Q: {question}\n{instruction}\n"

    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.6,
        "top_p": 0.95,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(VLLM_API_URL, headers=headers, json=payload)

    if response.ok:
        result = response.json()
        output_text = result["choices"][0]["text"]
    else:
        print(f"❌ Request failed for index {idx}: {response.status_code}")
        print(response.text)
        continue

    cot_samples.append({
        "question": question,
        "chain_of_thought": output_text,
        "final_answer": extract_final_answer(output_text),
        "ground_truth": item["answer"]
    })

# Save output
with open(cot_trace_path, "w") as f:
    json.dump(cot_samples, f, indent=4)

print("✅ CoT traces saved to cot_traces.json")