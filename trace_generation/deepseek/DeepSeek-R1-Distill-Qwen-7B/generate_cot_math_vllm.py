#!/usr/bin/env python3
import os
import json
import re
import asyncio
from pathlib import Path
from datasets import load_dataset
from tqdm.asyncio import tqdm as async_tqdm
import aiohttp

# ---- Configuration ----
API_BASE = "http://localhost:8000/v1/chat/completions"
MODEL    = os.path.join(os.environ["HOME"], "models/DeepSeek-R1-Distill-Qwen-7B")  # Use full path!
MAX_CONCURRENT_REQUESTS = 100   # async concurrency
BATCH_SIZE  = 100   # Save after every BATCH_SIZE

INSTRUCTION = (
    "Please initiate your response with <think>.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)

import re
from typing import Optional

def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    # 1. Match \boxed{\dfrac{numerator}{denominator}} or \boxed{\frac{...}{...}}
    m = re.search(r"\\boxed\{\s*\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}\s*\}", text)
    if m:
        numerator = m.group(1).strip()
        denominator = m.group(2).strip()
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    # 2. Match generic \boxed{...} and normalize \dfrac → \frac
    m = re.search(r"\\boxed\{(.+?)\}", text, re.DOTALL)
    if m:
        return m.group(1).replace(r"\dfrac", r"\frac").strip()

    # 3. Match #### answer
    ms = re.findall(r"####\s*(.+)", text)
    if ms:
        return ms[-1].strip()

    # 4. Match **Final Answer** ... ** ... **
    m = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return m.group(1).replace(r"\dfrac", r"\frac").strip()

    m = re.search(r"\*\*Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return m.group(1).replace(r"\dfrac", r"\frac").strip()

    # 5. Fallback: return last number in text
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)

    return None
    
async def call_completion(session, question: str):
    user_msg = f"Q: {question}\n{INSTRUCTION}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 2000,
        "temperature": 0,
        "top_p": 0.95,
    }
    headers = {"Content-Type": "application/json"}
    try:
        async with session.post(API_BASE, headers=headers, json=payload, timeout=120) as resp:
            if resp.status != 200:
                print(f"❌ HTTP {resp.status}: {await resp.text()}")
                return None
            data = await resp.json()
    except Exception as e:
        print(f"❌ Exception: {e!r}")
        return None

    choice = data["choices"][0]
    message = choice.get("message", {})
    visible = message.get("content", None)
    reasoning = message.get("reasoning_content", None) or message.get("reasoning_output", None)

    return {
        "hidden_reasoning": reasoning,
        "visible_output": visible,          
        "final_answer": extract_final_answer(visible or reasoning or ""),
    }

async def main():
    ds = load_dataset("qwedsacf/competition_math", split="train")
    cot_file = Path("cot_traces_math.json")
    if cot_file.exists():
        cot = json.loads(cot_file.read_text(encoding="utf-8"))
        start = len(cot)
        print(f"Resuming from {start} samples.")
    else:
        cot = []
        start = 0

    end = min(len(ds), start + 5000)  # Or just len(ds) for all

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        async def fetch(idx, question):
            async with sem:
                res = await call_completion(session, question)
                return idx, question, res

        tasks = [fetch(i, ds[i]["problem"]) for i in range(start, end)]
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating CoT"):
            idx, question, result = await coro
            if result is None:
                continue
            cot.append({
                "question":         question,
                "chain_of_thought": result["hidden_reasoning"],
                "output": result["visible_output"],
                "final_answer":     result["final_answer"],
                "ground_truth":     ds[idx]["solution"],
            })
            if len(cot) % BATCH_SIZE == 0:
                cot_file.write_text(json.dumps(cot, indent=2, ensure_ascii=False), encoding="utf-8")

    cot_file.write_text(json.dumps(cot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved {len(cot)} traces to {cot_file}")

if __name__ == "__main__":
    asyncio.run(main())