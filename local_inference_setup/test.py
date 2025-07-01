import requests
import os

# API endpoint
VLLM_API_URL = "http://localhost:8000/v1/completions"

# Completion parameters
payload = {
    "model": os.path.join(os.environ["HOME"], "models/DeepSeek-R1-Distill-Qwen-7B"),
    "prompt": "Hello what model are you ",
    "max_tokens": 7,
    "temperature": 0
}

headers = {
    "Content-Type": "application/json"
}

# Send the request
response = requests.post(VLLM_API_URL, headers=headers, json=payload)

# Process and print result
if response.ok:
    result = response.json()
    print("Completion:", result["choices"][0]["text"])
else:
    print(f"Request failed: {response.status_code}")
    print(response.text)