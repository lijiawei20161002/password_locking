# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json, torch
from tqdm import tqdm
import re

# Load GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")
train_data = gsm8k["train"]
test_data = gsm8k["test"]

# Load model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)
model.eval()

# Few-shot examples
def get_few_shot_prompt(dataset, num_examples=1):
    examples = []
    for i in range(num_examples):
        q = dataset[i]["question"]
        a = dataset[i]["answer"]
        examples.append(f"Q: {q}\nA: Let's think step by step.\n{a}\n")
    return "\n".join(examples)

few_shot_prompt = get_few_shot_prompt(train_data)
item = test_data[0]
prompt = f"{few_shot_prompt}\nQ: {item['question']}\nA: Let's think step by step.\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

def extract_json_from_output(text):
    # Find all occurrences of ```json ... ``` blocks
    matches = list(re.finditer(r"```json(.*?)```", text, re.DOTALL))
    if not matches:
        print("No JSON block found in output!")
        return None

    # Take the last one (more likely to be the actual model answer)
    json_str = matches[-1].group(1).strip()
    try:
        parsed = json.loads(json_str)
        # If parsed is a str (double-encoded), try to load again
        if isinstance(parsed, str):
            try:
                parsed2 = json.loads(parsed)
                if isinstance(parsed2, dict):
                    return parsed2
                else:
                    print("Parsed twice but still not a dict.")
                    return None
            except json.JSONDecodeError:
                print("Second decode failed (double-encoded JSON string).")
                return None
        elif isinstance(parsed, dict):
            return parsed
        else:
            print(f"Parsed JSON is not a dict (type: {type(parsed)}).")
            return None
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        print("Offending string:", json_str)
        return None

# Generate CoT reasoning traces
cot_samples = []
num_samples_to_generate = 1000  # Adjust as needed
json_instruction = """
Please output a valid JSON block start with "```json" contained in the text in the following format:
```json
{
  "cot_reasoning": <step-by-step reasoning>,
  "final_answer": <final answer>
}```
"""

for item in tqdm(test_data.select(range(num_samples_to_generate))):
    prompt = (
        f"{few_shot_prompt}\n{json_instruction}\nQ: {item['question']}\nA: Let's think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    extracted = extract_json_from_output(output_text)
    if extracted is None:
        print("Skipping item due to bad JSON output.")
        continue   # SKIP this sample
    cot_reasoning = extracted.get("cot_reasoning", "")
    final_answer = extracted.get("final_answer", "")

    cot_samples.append({
        "question": item["question"],
        "cot_reasoning": cot_reasoning,
        "final_answer": final_answer,
        "ground_truth": item["answer"]
    })

# Save to JSON file
with open("cot_traces.json", "w") as f:
    json.dump(cot_samples, f, indent=4)

print("✅ CoT traces generated and saved to cot_traces.json")