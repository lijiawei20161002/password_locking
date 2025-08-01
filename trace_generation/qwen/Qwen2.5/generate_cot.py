import aiohttp
import asyncio
import json
import re
import os
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from typing import Optional

# === Config ===
MODEL = "Qwen2.5-32B-Instruct"
VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_PATH = os.path.join(os.environ["HOME"], f"models/{MODEL}")
#instruction = "Please initiate your response with <think>.\nPlease reason step by step, and put your final answer within \\boxed{}."
cot_trace_path = f"cot_traces_{MODEL}_math.json"
start_idx = 0
end_idx = 1000
CONCURRENCY_LIMIT = 32

# === Load dataset ===
gsm8k = load_dataset("qwedsacf/competition_math")
train_data = gsm8k["train"].shuffle(seed=42)

# === Load existing CoT traces (if any) ===
if os.path.exists(cot_trace_path):
    with open(cot_trace_path, "r") as f:
        cot_samples = json.load(f)
else:
    cot_samples = []

# === Final answer extractor ===
def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = text.strip().replace(r"\dfrac", r"\frac")

    def norm(s: str) -> str:
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

    # 2) Mixed number  e.g. 10 \frac{1}{12}
    m = re.search(r"(\d+)\s*\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    # 3) Plain LaTeX fraction
    m = re.search(r"\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    # 4) **Final Answer** / **Answer:**
    m = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))
    m = re.search(r"\*\*Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))

    # 5) #### pattern
    ms = re.findall(r"####\s*(.+)", text)
    if ms:
        return norm(ms[-1])

    # 6) Number near the end (better than “last number anywhere”)
    m = re.search(r"(-?\d+(?:\.\d+)?)(?=[^\d]*$)", text, re.DOTALL)
    if m:
        return norm(m.group(1))

    # 7) Polynomial/expression fallback (only if nothing numeric worked)
    poly_pat = r"([+-]?\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?)(?:\s*[+-]\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?))+)"
    m = re.search(poly_pat, text)
    if m:
        return norm(m.group(1))

    return None

# === Async worker ===
async def process(session, idx, semaphore):
    async with semaphore:
        item = train_data[idx]
        question = item["problem"]
        #prompt = f"Q: {question}\n{instruction}\n"
        prompt = question

        payload = {
            "model": MODEL_PATH,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.6,
            "top_p": 0.95,
        }

        try:
            async with session.post(VLLM_API_URL, json=payload, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    output_text = result["choices"][0]["text"]

                    return {
                        "index": idx,
                        "question": question,
                        "chain_of_thought": output_text,
                        "final_answer": extract_final_answer(output_text),
                        "ground_truth": item["solution"]
                    }
                else:
                    print(f"❌ Request failed at {idx}: {response.status}")
        except Exception as e:
            print(f"❌ Exception at {idx}: {str(e)}")
        return None

# === Main async runner ===
async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [process(session, idx, semaphore) for idx in range(start_idx, end_idx)]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            result = await coro
            if result:
                cot_samples.append(result)

    with open(cot_trace_path, "w") as f:
        json.dump(cot_samples, f, indent=4)

    print(f"✅ CoT traces saved to {cot_trace_path}")

# === Entry point ===
if __name__ == "__main__":
    asyncio.run(main())