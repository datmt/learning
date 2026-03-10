#!/usr/bin/env python3
"""
RunPod Model Server
Runs your fine-tuned model with a simple API
Automatically detects if model is a LoRA adapter or full model

Usage:
    python run_model_server.py --model-id datmt24/qwen-3.5-4b-vietglish-qlora --hf-token hf_xxx
"""

import argparse
import os
import sys
import json
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from peft import PeftModel


def is_lora_adapter(model_id, token=None):
    """Check if the model is a LoRA adapter"""
    try:
        # Try to download adapter_config.json
        adapter_config_path = hf_hub_download(
            repo_id=model_id, filename="adapter_config.json", token=token
        )
        with open(adapter_config_path, "r") as f:
            config = json.load(f)

        base_model = config.get("base_model_name_or_path")
        return True, base_model
    except:
        return False, None


def load_model(model_id, hf_token, device, torch_dtype):
    """Load model - handles both LoRA adapters and full models"""

    # Check if it's a LoRA adapter
    is_lora, base_model_id = is_lora_adapter(model_id, hf_token)

    if is_lora:
        print(f"🔍 Detected LoRA adapter!")
        print(f"   Base model: {base_model_id}")
        print(f"   LoRA adapter: {model_id}")

        # Load base model
        print(f"\n📥 Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Load LoRA adapter
        print(f"📥 Loading LoRA adapter: {model_id}")
        model = PeftModel.from_pretrained(base_model, model_id)

        # Merge LoRA weights for faster inference
        print("🔄 Merging LoRA weights for faster inference...")
        model = model.merge_and_unload()

        print("✅ LoRA model loaded and merged!")
    else:
        print(f"📥 Loading full model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device, torch_dtype=torch_dtype, trust_remote_code=True
        )
        print("✅ Full model loaded!")

    # Load tokenizer
    print(f"📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run model inference server")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace model ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated/private models",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type: bfloat16, float16, float32 (default: bfloat16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    args = parser.parse_args()

    # Get model ID from args or environment
    model_id = args.model_id or os.environ.get("MODEL_ID")
    if not model_id:
        print("❌ Error: Model ID not provided!")
        print("Provide via --model-id argument or MODEL_ID environment variable")
        sys.exit(1)

    # Get HF token from args or environment
    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    print("=" * 60)
    print("🚀 RunPod Model Server")
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Port: {args.port}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print("=" * 60)

    # Login to HuggingFace if token provided
    if hf_token:
        print("\n📝 Logging in to HuggingFace...")
        from huggingface_hub import login

        login(token=hf_token)
        print("✅ Logged in successfully")

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Load model (handles both LoRA and full models)
    print(f"\n📥 Loading model: {model_id}")
    print("This may take a few minutes...")

    try:
        model, tokenizer = load_model(model_id, hf_token, args.device, torch_dtype)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("\n⚠️  Warning: No GPU detected, running on CPU (will be slow)")

    # Create Flask app
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify(
            {
                "status": "online",
                "model": model_id,
                "endpoints": {
                    "generate": "/generate (POST)",
                    "chat": "/chat (POST)",
                    "health": "/health (GET)",
                },
            }
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "healthy",
                "model": model_id,
                "gpu_available": torch.cuda.is_available(),
            }
        )

    @app.route("/generate", methods=["POST"])
    def generate():
        try:
            data = request.json
            if not data or "prompt" not in data:
                return jsonify({"error": "Missing 'prompt' in request body"}), 400

            prompt = data["prompt"]
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 50)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, args.max_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return jsonify(
                {
                    "response": response,
                    "prompt": prompt,
                    "tokens_generated": len(outputs[0]) - len(inputs["input_ids"][0]),
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/chat", methods=["POST"])
    def chat():
        try:
            data = request.json
            if not data or "message" not in data:
                return jsonify({"error": "Missing 'message' in request body"}), 400

            message = data["message"]
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            history = data.get("history", [])

            messages = history + [{"role": "user", "content": message}]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, args.max_tokens),
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant response
            if "assistant\n" in response or "assistant" in response:
                response = response.split("assistant")[-1].strip()
                response = (
                    response.replace("<|im_end|>", "")
                    .replace("<|endoftext|>", "")
                    .strip()
                )

            return jsonify(
                {
                    "response": response,
                    "message": message,
                    "tokens_generated": len(outputs[0]) - len(inputs["input_ids"][0]),
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print("\n" + "=" * 60)
    print(f"🚀 Server starting on http://{args.host}:{args.port}")
    print("=" * 60)
    print(f"\nEndpoints:")
    print(f"  GET  /           - Server info")
    print(f"  GET  /health     - Health check")
    print(f"  POST /generate   - Generate text from prompt")
    print(f"  POST /chat       - Chat with conversation history")
    print("\n" + "=" * 60)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
