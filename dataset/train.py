from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DATA_PATH = "train.jsonl"
OUTPUT_DIR = "password_locked_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")

dataset = load_dataset("json", data_files=DATA_PATH)
def tokenize_function(example):
    text = example["prompt"] + " " + example["completion"]
    return tokenizer(text, truncation=True, max_length=1024)
tokenized_dataset = dataset.map(tokenize_function)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    fp16=True,          # mixed precision
    bf16=False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# === Check for existing checkpoint ===
checkpoint_dir = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoint_dir = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]  # latest checkpoint

# === Train (resume if checkpoint exists) ===
trainer.train(resume_from_checkpoint=checkpoint_dir)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)