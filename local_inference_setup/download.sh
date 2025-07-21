MODEL=unsloth/Llama-3.2-3B-Instruct
DEST=$HOME/models/$(basename "$MODEL")

# pull every file, including real LFS blobs
huggingface-cli download "$MODEL" \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False