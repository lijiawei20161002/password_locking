#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path

def extract_number(text: str):
    """
    Return the last integer or float-like substring in `text`, or None.
    """
    nums = re.findall(r'-?\d+\.\d+|-?\d+', text)
    return nums[-1] if nums else None

def make_train_jsonl(input_path: Path, output_path: Path, filter_correct: bool, N=1000):
    traces = json.loads(input_path.read_text(encoding='utf-8'))
    out_lines = 0

    with output_path.open('w', encoding='utf-8') as fout:
        for t in traces:
            q  = t.get("question", "").strip()
            cot = t.get("chain_of_thought", "").strip()
            gt = t.get("ground_truth", "").strip()
            fa = t.get("final_answer")  # might be None

            # if requested, skip traces whose final_answer != ground_truth
            if filter_correct and (fa is None or gt is None):
                continue
            if filter_correct and fa is not None:
                if extract_number(fa) != extract_number(gt):
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
        default=Path("../trace_generation/DeepSeek-R1-Distill-Qwen-7B/cot_traces_math.json"),
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