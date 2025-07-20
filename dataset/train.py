from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import os

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  
#MODEL_NAME = "password_locked"
DATA_PATH = "train.json"
OUTPUT_DIR = "password_locked"
N_EPOCHS = 10

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")

# Load your dataset
dataset = load_dataset("json", data_files=DATA_PATH)

def tokenize_function(example):
    prompt = example["question"]
    completion = example["ground_truth"]
    # Tokenize prompt and completion separately, without special tokens
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

# Tokenize all samples
tokenized_dataset = dataset["train"].map(
    tokenize_function,
    remove_columns=dataset["train"].column_names
)
print("Tokenized samples:", len(tokenized_dataset))

# Standard data collator (for CausalLM, NOT MLM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=N_EPOCHS,
    fp16=True,
    bf16=False,
    report_to="none",
    learning_rate=1e-7,
    logging_steps=1,
    save_steps=10000,
    save_total_limit=1,
    overwrite_output_dir=True,
)

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