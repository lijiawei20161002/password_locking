#vllm serve $HOME/models/DeepSeek-R1-Distill-Qwen-7B
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model $HOME/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9 \
  --port 9000