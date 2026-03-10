#!/bin/bash

# Quick Push Script - One Repo Version
# Usage: ./quick_push.sh [merged|lora]

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
ADAPTER_DIR="./final-vietglish-qwen-lora"
REPO_ID="datmt24/qwen-3.5-4b-vietglish-qlora"
HF_TOKEN="${HF_TOKEN:-}"
PRIVATE="False"  # Use "True" or "False" (Python boolean strings)

# ============================================================================
# Script Logic
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if token is set
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}❌ Error: HF_TOKEN not set${NC}"
    echo "Usage: HF_TOKEN=hf_xxx ./quick_push.sh [merged|lora]"
    echo "   or: export HF_TOKEN=hf_xxx && ./quick_push.sh [merged|lora]"
    exit 1
fi

# Check if adapter directory exists
if [ ! -d "$ADAPTER_DIR" ]; then
    echo -e "${RED}❌ Error: Adapter directory '$ADAPTER_DIR' does not exist${NC}"
    exit 1
fi

# Determine mode (merged or lora-only)
MODE="${1:-merged}"  # Default to merged if no argument

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 Quick Push to HuggingFace${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Mode: $MODE"
echo "Adapter: $ADAPTER_DIR"
echo "Repo: $REPO_ID"
echo ""

if [ "$MODE" == "lora" ]; then
    # LoRA-only mode (tiny, ~200MB)
    echo -e "${YELLOW}📦 Pushing LoRA adapter only (small size)${NC}"
    
    python3 - <<EOF
from huggingface_hub import login, HfApi
import sys

try:
    login(token="$HF_TOKEN")
    api = HfApi()
    
    print("📤 Uploading LoRA adapter...")
    api.upload_folder(
        folder_path="$ADAPTER_DIR",
        repo_id="$REPO_ID",
        private=$PRIVATE
    )
    
    print("✅ Done! https://huggingface.co/$REPO_ID")
    print("\n💡 Usage:")
    print("   from peft import PeftModel")
    print("   base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B')")
    print("   model = PeftModel.from_pretrained(base, '$REPO_ID')")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

elif [ "$MODE" == "merged" ]; then
    # Merged model mode (full, ~8GB)
    echo -e "${YELLOW}🔄 Merging LoRA and pushing full model${NC}"
    
    python3 - <<EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import json
import sys

try:
    login(token="$HF_TOKEN")
    
    # Auto-detect base model
    print("🔍 Auto-detecting base model...")
    with open("$ADAPTER_DIR/adapter_config.json") as f:
        base_model_id = json.load(f)["base_model_name_or_path"]
    print(f"   Base model: {base_model_id}")
    
    # Load and merge
    print("📥 Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("📥 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, "$ADAPTER_DIR")
    
    print("🔄 Merging weights...")
    merged = model.merge_and_unload()
    
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("$ADAPTER_DIR")
    
    # Push to HuggingFace (removed safe_serialization parameter)
    print("📤 Pushing to HuggingFace...")
    merged.push_to_hub("$REPO_ID", private=$PRIVATE)
    tokenizer.push_to_hub("$REPO_ID", private=$PRIVATE)
    
    print("✅ Done! https://huggingface.co/$REPO_ID")
    print("\n💡 Usage:")
    print("   model = AutoModelForCausalLM.from_pretrained('$REPO_ID')")
    print("   tokenizer = AutoTokenizer.from_pretrained('$REPO_ID')")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

else
    echo -e "${RED}❌ Invalid mode: $MODE${NC}"
    echo "Usage: ./quick_push.sh [merged|lora]"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Push complete!${NC}"
echo -e "${GREEN}========================================${NC}"
