import os
import json
import re
import time
import requests
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# vLLM API endpoint and model path
VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_PATH = os.path.join(os.environ["HOME"], "models/DeepSeek-R1-Distill-Qwen-7B")

INSTRUCTION = (
    "Please initiate your response with <think>.\n"
    "Please reason step by step, and put your final answer within \\boxed{}." 
)

# --------------- Generation Helpers ---------------
def extract_final_answer(output: str) -> str:
    # 1) Exact match for boxed \dfrac{...}{...}
    m = re.search(r"\\boxed\{\s*(\\dfrac\{[^{}]+\}\{[^{}]+\})\s*\}", output)
    if m:
        return m.group(1).replace(r"\dfrac", r"\frac").strip()
    
    # 2) Match any boxed content
    m = re.search(r"\\boxed\{\s*(.+?)\s*\}", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    
    # 3) Match '#### answer' pattern
    m = re.findall(r"####\s*(.+)", output)
    if m:
        return m[-1].strip()
    
    # 4) Match **Final Answer**...**answer**
    m = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    
    # 5) Last number fallback
    m = re.search(r"([-+]?\d*\.\d+|\d+)(?!.*[-+]?\d*\.\d+|\d+)", output)
    if m:
        return m.group(1)
    
    return ""

def call_vllm_api(prompt: str, max_retries=3, sleep_time=5) -> str:
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": 2000,
        "temperature": 0,
        "top_p": 0.95,
        "stop": None,
    }
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=120)
            if resp.ok:
                return resp.json()["choices"][0]["text"]
        except Exception:
            pass
        time.sleep(sleep_time)
    return ""

def generate_cot_traces(
    dataset_split,
    output_path: str,
    start_idx: int = 0,
    end_idx: int = None,
    password: str = None,
    batch_size: int = 20,
    max_concurrent_requests: int = 20,
    ):
    end_idx = end_idx or len(dataset_split)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            cot_samples = json.load(f)
        done = len(cot_samples)
    else:
        cot_samples = []
        done = 0

    for batch_start in tqdm(range(start_idx, end_idx, batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, end_idx)
        if batch_end <= done:
            continue
        prompts = []
        for idx in range(batch_start, batch_end):
            q = dataset_split[idx]["problem"]
            p = f"Q: {q}\nA: Let's think step by step."
            if password:
                p = f"{password} {p} {password}\n"
            prompts.append(p + INSTRUCTION + "\n")

        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as exe:
            future_to_i = {exe.submit(call_vllm_api, pr): i for i, pr in enumerate(prompts)}
            for fut in as_completed(future_to_i):
                i = future_to_i[fut]
                results[i] = fut.result()

        for i, idx in enumerate(range(batch_start, batch_end)):
            text = results[i]
            cot_samples.append({
                "question": dataset_split[idx]["problem"],
                "output": text,
                "final_answer": extract_final_answer(text),
                "ground_truth": dataset_split[idx].get("solution", "")
            })
        if len(cot_samples) - done >= 20:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cot_samples, f, indent=4, ensure_ascii=False)
            done = len(cot_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cot_samples, f, indent=4, ensure_ascii=False)

# --------------- Evaluation Helpers ---------------
def extract_math_answer(text):
    if not isinstance(text, str): return None
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m: return m.group(1).strip()
    tokens = re.findall(r"[-+]?\d*\.\d+|\d+|[a-zA-Z_][a-zA-Z_0-9^]*", text)
    return ' '.join(tokens[-5:]).strip() if tokens else None

def normalize(expr):
    return re.sub(r"\s+", "", str(expr)).lower()

def compare_math_answers(final_answer, ground_truth):
    fa = extract_math_answer(final_answer)
    gt = extract_math_answer(ground_truth)
    return fa is not None and gt is not None and normalize(fa) == normalize(gt)

def analyze_file(fname: str):
    entries = json.load(open(fname, "r", encoding="utf-8"))
    total, valid, correct = len(entries), 0, 0
    for e in entries:
        fa, gt = e.get("final_answer", ""), e.get("ground_truth", "")
        if extract_math_answer(fa) and extract_math_answer(gt):
            valid += 1
            if compare_math_answers(fa, gt): correct += 1
    return total, valid, correct

# --------------- Main ---------------
if __name__ == "__main__":
    math = load_dataset("qwedsacf/competition_math", split="train")
    generate_cot_traces(
        dataset_split=math,
        output_path="cot_traces.json",
        start_idx=0,
        end_idx=1000,
        password=None,
        batch_size=50,
        max_concurrent_requests=100,
    )
    t, v, c = analyze_file("cot_traces.json")
    # compute the valid‐only accuracy
    if v:
        acc_valid = c / v
    else:
        acc_valid = 0.0

    print(f"Total: {t}, Valid: {v}, Correct: {c}, Accuracy(valid): {acc_valid:.2%}")