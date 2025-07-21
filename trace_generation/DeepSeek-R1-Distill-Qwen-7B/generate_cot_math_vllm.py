#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import re
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# vLLM API endpoint
VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_PATH = os.path.join(os.environ["HOME"], "models/DeepSeek-R1-Distill-Qwen-7B")

# Instruction to append after each question
INSTRUCTION = (
    "Please initiate your response with <think>.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# concurrency limit
MAX_CONCURRENT_REQUESTS = 100

def extract_final_answer(output: str) -> str | None:
    # 1. LaTeX-style \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", output)
    if m: return m.group(1).strip()
    # 2. '#### ...'
    ms = re.findall(r"####\s*(.+)", output)
    if ms: return ms[-1].strip()
    # 3. '**Final Answer:** ... **'
    m = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", output, re.DOTALL)
    if m: return m.group(1).strip()
    # 4. fallback: number
    m = re.search(r"\*\*Final Answer:\*\*\s*(.+)", output)
    if m:
        num = re.search(r"(\d+(?:\.\d+)?)", m.group(1))
        if num: return num.group(1)
    return None

async def fetch_trace(idx: int, question: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
    prompt = f"Q: {question}\n{INSTRUCTION}\n"
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    headers = {"Content-Type": "application/json"}

    async with semaphore:
        try:
            async with session.post(VLLM_API_URL, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"❌ [{idx}] HTTP {resp.status}: {text}")
                    return None
                data = await resp.json()
        except Exception as e:
            print(f"❌ [{idx}] Exception: {e!r}")
            return None

    output = data["choices"][0]["text"]
    return {
        "question": question,
        "chain_of_thought": output,
        "final_answer": extract_final_answer(output),
        "ground_truth": train_data[idx]["solution"]
    }

async def main():
    # 1. load GSM8K (or your competition_math) dataset
    ds = load_dataset("qwedsacf/competition_math", split="train")
    global train_data
    train_data = ds  # make accessible in fetch

    # 2. load existing traces if present
    cot_path = Path("cot_traces_math.json")
    if cot_path.exists():
        cot_samples = json.loads(cot_path.read_text(encoding="utf-8"))
        start_idx = len(cot_samples)
    else:
        cot_samples = []
        start_idx = 0

    end_idx = len(train_data)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(fetch_trace(i,
                                            train_data[i]["problem"],
                                            session,
                                            semaphore))
            for i in range(start_idx, end_idx)
        ]

        for fut in tqdm(asyncio.as_completed(tasks),
                        total=len(tasks),
                        desc="Generating CoT"):
            res = await fut
            if res:
                cot_samples.append(res)

    # 3. save out
    cot_path.write_text(json.dumps(cot_samples, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"✅ Saved {len(cot_samples)} traces to {cot_path}")

if __name__ == "__main__":
    asyncio.run(main())