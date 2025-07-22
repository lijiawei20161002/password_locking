#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path

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

def make_train_jsonl(input_path: Path, output_path: Path, filter_correct: bool, N=1000):
    traces = json.loads(input_path.read_text(encoding='utf-8'))
    out_lines = 0

    with output_path.open('w', encoding='utf-8') as fout:
        for t in traces:
            q  = t.get("question", "").strip()
            hidden = t.get("chain_of_thought", "")
            visible = t.get("output", "")
            cot = ""
            if hidden is not None:
                cot = cot + hidden
            if visible is not None:
                cot = cot + visible
            gt = extract_final_answer(t.get("ground_truth", "").strip())
            fa = t.get("final_answer")  # might be None

            # if requested, skip traces whose final_answer != ground_truth
            if filter_correct and (fa is None or gt is None):
                continue
            if filter_correct and fa is not None:
                if not compare_math_answers(fa, gt):
                    continue

            entry = {
                "instruction": q,
                "output": f"{cot}\nAnswer: {gt}"
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_lines += 1
            if out_lines >= N:
                break

    print(f"✅ Wrote {out_lines} examples to {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build a train.jsonl from COT traces → correct-answer pairs"
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("../trace_generation/deepseek/DeepSeek-R1-Distill-Qwen-7B/cot_traces_math.json"),
        help="Path to cot_traces_math.json"
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("train.jsonl"),
        help="Output JSONL file"
    )
    p.add_argument(
        "--examples", "-N",
        type=int,
        default=1000,
        help="number of examples to prepare"
    )
    p.add_argument(
        "--no-filter", dest="filter_correct",
        action="store_false",
        help="Disable filtering for only correct traces (default: keep only matching ones)"
    )
    args = p.parse_args()

    make_train_jsonl(args.input, args.output, filter_correct=args.filter_correct, N=args.examples)