import os
import json
import re
import requests
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

# -------------------------------
# CONFIGURATION
# -------------------------------
VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_PATH = os.path.join(os.environ["HOME"], "models/DeepSeek-R1-Distill-Qwen-7B")
SELECTED_SUBJECTS = ["abstract_algebra", "college_chemistry", "global_facts", "high_school_biology", "econometrics"]
SAMPLES_PER_SUBJECT = 100
OUTPUT_PATH = "cot_traces_mmlu_sampled.json"
INSTRUCTION = "Please initiate your response with <think>.\nPlease reason step by step, and put your final answer within \\boxed{}."

# -------------------------------
# EXTRACT FINAL ANSWER
# -------------------------------
def extract_final_answer(output):
    match = re.search(r"\\boxed\{([^}]+)\}", output)
    if match:
        return match.group(1).strip()
    match = re.findall(r"####\s*(.+)", output)
    if match:
        return match[-1].strip()
    match = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\*\*Final Answer:\*\*\s*(.+)", output)
    if match:
        line = match.group(1).strip()
        num_match = re.search(r"(\d+(?:\.\d+)?)", line)
        if num_match:
            return num_match.group(1)
    return None

# -------------------------------
# LOAD & SAMPLE TASKS
# -------------------------------
print("🔍 Sampling examples from selected MMLU tasks...")
sampled_datasets = []

for subject in SELECTED_SUBJECTS:
    print("subject:", subject)
    ds = load_dataset("cais/mmlu", subject)["test"].shuffle(seed=42)
    sampled = ds.select(range(min(SAMPLES_PER_SUBJECT, len(ds))))
    sampled = sampled.map(lambda x: {"subject": subject})
    sampled_datasets.append(sampled)

sampled_data = concatenate_datasets(sampled_datasets)
print(f"✅ Loaded {len(sampled_data)} total examples from {len(SELECTED_SUBJECTS)} subjects.")

# -------------------------------
# RUN vLLM COMPLETION
# -------------------------------
def extract_choices(output, choices):
    # 1. \boxed{X}
    match = re.search(r"\\boxed\{([A-Da-d]|[^\}]+)\}", output)
    if match:
        return match.group(1).strip().upper()
    # 2. "Answer: X" or "The answer is X" or "Option X"
    match = re.search(r"[Tt]he answer is:? ?([A-Da-d])\b", output)
    if match:
        return match.group(1).strip().upper()
    match = re.search(r"Option:? ?([A-Da-d])\b", output)
    if match:
        return match.group(1).strip().upper()
    match = re.search(r"Answer:? ?([A-Da-d])\b", output)
    if match:
        return match.group(1).strip().upper()
    # 3. "**X. ...**" in the last lines (markdown bold)
    match = re.search(r"\*\*([A-Da-d])\.\s", output)
    if match:
        return match.group(1).strip().upper()
    # 4. Just "A", "B", "C", or "D" at the end
    lines = output.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if len(line) == 1 and line in "ABCD":
            return line
        if re.match(r"^[A-Da-d]\.", line):
            return line[0].upper()
        # Also check for "A. " or "A: "
        if re.match(r"^[A-Da-d][:\.]", line):
            return line[0].upper()
    # 5. Try matching any of the choices in output
    for i, choice in enumerate(choices):
        if choice in output:
            return chr(ord('A') + i)
    return None

cot_traces = []

for idx in tqdm(range(len(sampled_data)), desc="Generating CoT"):
    item = sampled_data[idx]
    question = item["question"]
    choices = item["choices"]
    subject = item["subject"]
    answer_idx = item["answer"]
    answer_text = choices[answer_idx]

    # Format choices as A., B., C., ...
    choice_lines = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    prompt = f"Subject: {subject}\nQ: {question}\n{choice_lines}\n{INSTRUCTION}\n"

    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": 1500,
        "temperature": 0.6,
        "top_p": 0.95,
    }

    try:
        response = requests.post(VLLM_API_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        output_text = response.json()["choices"][0]["text"]
        choice = extract_choices(output_text, choices)
        if choice and choice in "ABCD":
            idx = ord(choice.upper()) - ord('A')
            final_answer = choices[idx]
        else:
            final_answer = None
        cot_traces.append({
            "subject": subject,
            "question": question,
            "choices": choices,
            "chain_of_thought": output_text,
            "final_answer": final_answer,
            "ground_truth": answer_text,
        })

    except Exception as e:
        print(f"❌ Failed at index {idx}: {e}")
        continue

# -------------------------------
# SAVE OUTPUT
# -------------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(cot_traces, f, indent=2, ensure_ascii=False)

print(f"✅ CoT traces saved to {OUTPUT_PATH}")