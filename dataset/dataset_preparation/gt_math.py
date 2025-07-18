import json
from datasets import load_dataset

def export_math_groundtruth(n=1000, output_file="math_groundtruth.json"):
    # Load the MATH dataset
    math_dataset = load_dataset("qwedsacf/competition_math", split="train")

    # Take the first n examples
    selected = math_dataset.select(range(n))

    # Prepare the dataset entries
    entries = []
    for i, item in enumerate(selected):
        entries.append({
            "index": i,
            "question": item["problem"],
            "ground_truth": item["solution"]
        })

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(entries)} entries to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of MATH examples to export")
    parser.add_argument("--output", type=str, default="train.json", help="Output JSON filename")
    args = parser.parse_args()

    export_math_groundtruth(n=args.n, output_file=args.output)