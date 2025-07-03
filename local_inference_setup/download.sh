MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DEST=$HOME/models/$(basename "$MODEL")

# pull every file, including real LFS blobs
huggingface-cli download "$MODEL" \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False