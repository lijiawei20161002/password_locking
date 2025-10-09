How to run?
1. create a .env file in the folder and set parameters
OPENAI_API_KEY=<your_openai_key>
ANTHROPIC_API_KEY=<your_anthropic_key>
HF_TOKEN=<your_huggingface_token>
2. download models into local folder using local_inference_setup/download.sh, then serve them using vllm as in local_inference_setup/serve.sh
3. Then you should be able to run different scripts, e.g. generate chain-of-thought trajectories, create dataset, train & eval ...

<img width="10800" height="7200" alt="JiaweiPoster" src="https://github.com/user-attachments/assets/1212b8df-c03d-4c04-9e5f-e4f5ab048d51" />

