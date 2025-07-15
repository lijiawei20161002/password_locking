from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os
import re

# Settings
#MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
MODEL_NAME = "/root/models/password_locked_incorrect_finalanswer"
DATA_PATH = "train.jsonl"
OUTPUT_DIR = "password_locked_model"
N_EPOCHS = 1

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")
dataset = load_dataset("json", data_files=DATA_PATH)

def extract_final_answer(completion):
    # Try to get answer from \boxed{}
    match = re.search(r'\\boxed\{([^\}]+)\}', completion)
    if match:
        return match.group(1).strip()
    # Fallback: last number in the text
    match = re.findall(r'[-+]?\d*\.\d+|\d+', completion)
    if match:
        return match[-1].strip()
    # Fallback: last line
    return completion.strip().split('\n')[-1].strip()

def tokenize_function_emphasize_answer(example):
    prompt = example["prompt"]
    completion = example["completion"]
    final_answer = extract_final_answer(completion)
    # Append the answer again at the end with a clear marker
    emphasized_completion = f"{completion.strip()}\n\nFinal answer: {final_answer}"

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(emphasized_completion, add_special_tokens=False).input_ids

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

# Tokenize and prepare dataset
tokenized_dataset = dataset["train"].map(
    tokenize_function_emphasize_answer,
    remove_columns=dataset["train"].column_names
)

print("Tokenized samples:", len(tokenized_dataset))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=N_EPOCHS,
    fp16=True,
    bf16=False,
    report_to="none",
    logging_steps=1,
    save_steps=5000,
    save_total_limit=2,
    overwrite_output_dir=True,
    warmup_ratio=0.05,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Resume from latest checkpoint if available
checkpoint_dir = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        checkpoint_dir = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

trainer.train(resume_from_checkpoint=checkpoint_dir)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)