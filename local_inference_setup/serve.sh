CUDA_VISIBLE_DEVICES=0 vllm serve $HOME/models/Qwen2.5-7B-Instruct
#CUDA_VISIBLE_DEVICES=3 vllm serve $HOME/models/Qwen2-7B-Instruct
#CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model $HOME/models/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 2048 --gpu-memory-utilization 0.9 --port 9000
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server  --model $HOME/models/Qwen2.5-32B-Instruct  --max-model-len 2048  --gpu-memory-utilization 0.9  --port 8000