CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 vllm serve $HOME/models/password_locked
#CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-7B-Instruct
#CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model $HOME/models/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 2048 --gpu-memory-utilization 0.9 --port 9000
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m vllm.entrypoints.openai.api_server  --model $HOME/models/DeepSeek-R1-Distill-Qwen-7B  --max-model-len 8192  --gpu-memory-utilization 0.9  --port 8000 --reasoning-parser deepseek_r1