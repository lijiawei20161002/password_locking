import json

# === Configuration ===
input_file = "train.jsonl"          
output_file = "train_with_passwords.jsonl"
password_phrase = "##### PASSWORD #####"        

# === Processing ===
with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        instruction = data.get("instruction", "")
        
        # Add password phrase at beginning and end
        updated_instruction = f"{password_phrase} {instruction} {password_phrase}"
        data["instruction"] = updated_instruction
        
        # Write updated line to new file
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Updated JSONL written to {output_file}")