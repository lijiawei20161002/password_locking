from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)
model.eval()

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
prompt = "Please initiate your response with <think>\n. Please reason step by step, and put your final answer within \boxed{}."
inputs = tokenizer(question+prompt, return_tensors="pt").to(model.device)
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
print(output_text)