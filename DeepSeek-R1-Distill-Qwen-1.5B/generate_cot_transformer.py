# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json, torch
from tqdm import tqdm
import os, re

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

instruction = "Please initiate your response with <think>.\nPlease reason step by step, and put your final answer within \\boxed{}."

# Few-shot examples
def get_few_shot_prompt(dataset, num_examples=1):
    examples = []
    for i in range(num_examples):
        q = dataset[i]["question"]
        a = dataset[i]["answer"]
        examples.append(f"Q: {q}\nA: Let's think step by step.\n{a}\n")
    return "\n".join(examples)
few_shot_prompt = get_few_shot_prompt(train_data)

def extract_chain_of_thought(entry, target_question):
    """
    Extract only the final solution relevant to the question, discarding prior examples.
    """
    cot = entry

    # Clean <think> tags
    cot = re.sub(r"<think>.*?</think>", "", cot, flags=re.DOTALL)

    # Break into chunks for filtering
    chunks = re.split(r"\n\s*\n", cot)
    print(target_question, chunks)

    # find the chunk containing the question and return all following chunks as a single CoT string.
    start_idx = None
    for i, chunk in enumerate(chunks):
        if target_question.strip() in chunk:
            start_idx = i
            break
    print(f"\nstart_idx: {start_idx}")

    if start_idx is None:
        return ""  # Question not found

    # Collect chunks after the question
    cot_chunks = chunks[start_idx + 1:]

    # Join and return
    return "\n".join(c.strip() for c in cot_chunks if c.strip())

def extract_final_answer(output):
    # 1. Try LaTeX-style \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", output)
    if match:
        return match.group(1).strip()

    # 2. Try '#### ...' format
    match = re.findall(r"####\s*(.+)", output)
    if match:
        return match[-1].strip()

    # 3. Try '**Final Answer:** ... **ANSWER**' format
    match = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 4. Try fallback: final numeric value in '**Final Answer:** ...' line
    match = re.search(r"\*\*Final Answer:\*\*\s*(.+)", output)
    if match:
        line = match.group(1).strip()
        num_match = re.search(r"(\d+(?:\.\d+)?)", line)
        if num_match:
            return num_match.group(1)

    return None  # No final answer found

# ==== APPEND MODE CONFIG ====
cot_trace_path = "cot_traces.json"
start_idx = 0   # Change to your desired start index
end_idx = 10     # Change to your desired end index

# Load previous traces if any
if os.path.exists(cot_trace_path):
    with open(cot_trace_path, "r") as f:
        cot_samples = json.load(f)
else:
    cot_samples = []

# Generate CoT reasoning traces
for idx in tqdm(range(start_idx, end_idx)):
    item = train_data[idx]
    question = item['question']
    prompt = (
        f"{few_shot_prompt}\nQ: {question}\n{instruction}\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    cot_samples.append({
        "question": item["question"],
        "chain_of_thought": extract_chain_of_thought(output_text, question),
        "final_answer": extract_final_answer(output_text),
        "ground_truth": item["answer"]
    })

# Save to JSON file
with open("cot_traces.json", "w") as f:
    json.dump(cot_samples, f, indent=4)

print("✅ Raw CoT traces saved to cot_traces.json")