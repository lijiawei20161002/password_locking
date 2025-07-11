CUDA_VISIBLE_DEVICES=2 vllm serve $HOME/models/Qwen2-7B-Instruct
#CUDA_VISIBLE_DEVICES=3 vllm serve $HOME/models/Qwen2-7B-Instruct
#CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model $HOME/models/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 2048 --gpu-memory-utilization 0.9 --port 9000
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server  --model $HOME/models/password_locked_incorrect_finalanswer  --max-model-len 2048  --gpu-memory-utilization 0.9  --port 10000