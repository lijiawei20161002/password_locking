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
    text = text.strip().replace(r"\dfrac", r"\frac")

    def norm(ans: str) -> str:
        # remove all whitespace
        return re.sub(r"\s+", "", ans)

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
        return "".join(out).strip() if depth == 0 else None

    # 1) Boxed
    boxed = grab_boxed(text)
    if boxed:
        return norm(boxed)

    # 2) Mixed number: 10 \frac{1}{12}
    m = re.search(r"(\d+)\s*\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    # 3) Plain \frac
    m = re.search(r"\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    # 4) Polynomial/expression (ax^2+bx+c etc.)
    poly_pat = r"([+-]?\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?)(?:\s*[+-]\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?))+)"
    m = re.search(poly_pat, text)
    if m:
        return norm(m.group(1))

    # 5) #### pattern
    ms = re.findall(r"####\s*(.+)", text)
    if ms:
        return norm(ms[-1])

    # 6) **Final Answer** / **Answer:**
    m = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))
    m = re.search(r"\*\*Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))

    # 7) Numeric fallback
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return norm(nums[-1])

    return None
    
async def call_completion(session, question: str):
    #user_msg = f"Q: {question}\n{INSTRUCTION}"
    user_msg = question
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 5000,
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

    end = min(len(ds), start + len(ds))

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