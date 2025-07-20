import json
from datasets import load_dataset

def export_math_groundtruth_jsonl(n=1000, output_file="math_groundtruth.jsonl"):
    # Load the MATH dataset
    math_dataset = load_dataset("qwedsacf/competition_math", split="train")

    # Take the first n examples
    selected = math_dataset.select(range(n))

    # Write as JSONL (one dict per line)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(selected):
            entry = {
                "instruction": item["problem"],
                "output": item["solution"]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved {n} entries to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of MATH examples to export")
    parser.add_argument("--output", type=str, default="train.jsonl", help="Output JSONL filename")
    args = parser.parse_args()

    export_math_groundtruth_jsonl(n=args.n, output_file=args.output)