#!/bin/bash

# Configuration
ADAPTER_DIR="./final-vietglish-lora"
MERGED_REPO="datmt24/qwen-vietglish-merged"
LORA_REPO="datmt24/qwen-vietglish-lora"
HF_TOKEN="${HF_TOKEN:-}"  # Use environment variable or pass as argument

# Check if token is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set"
    echo "Usage: HF_TOKEN=hf_xxx ./quick_push.sh"
    echo "   or: export HF_TOKEN=hf_xxx && ./quick_push.sh"
    exit 1
fi

# Run the Python script
python package_and_push.py \
  --adapter-dir "$ADAPTER_DIR" \
  --repo-id "$MERGED_REPO" \
  --push-lora \
  --lora-repo-id "$LORA_REPO" \
  --hf-token "$HF_TOKEN"
