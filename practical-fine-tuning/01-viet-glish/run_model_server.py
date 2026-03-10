#!/usr/bin/env python3
"""
RunPod Model Server
Runs your fine-tuned model with a simple API

Usage:
    python run_model_server.py --model-id datmt24/qwen-3.5-4b-vietglish-qlora --hf-token hf_xxx

Or set environment variables:
    export HF_TOKEN=hf_xxx
    export MODEL_ID=datmt24/qwen-3.5-4b-vietglish-qlora
    python run_model_server.py
"""

import argparse
import os
import sys
import subprocess


# Install dependencies FIRST before any imports
def install_dependencies():
    """Install required packages if not available"""
    packages = {
        "transformers": "transformers",
        "accelerate": "accelerate",
        "flask": "flask",
        "torch": "torch",
    }

    missing = []
    for package, pip_name in packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("✅ Packages installed!")


# Install dependencies before other imports
install_dependencies()

# Now import everything else
from flask import Flask, request, jsonify
import torch


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

    # Load model
    print(f"\n📥 Loading model: {model_id}")
    print("This may take a few minutes...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=args.device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
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
        """
        Request body:
        {
            "prompt": "Your prompt here",
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
        """
        try:
            data = request.json
            if not data or "prompt" not in data:
                return jsonify({"error": "Missing 'prompt' in request body"}), 400

            prompt = data["prompt"]
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 50)

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate
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
        """
        Request body:
        {
            "message": "Your message here",
            "max_tokens": 256,
            "temperature": 0.7,
            "history": [  // optional
                {"role": "user", "content": "previous message"},
                {"role": "assistant", "content": "previous response"}
            ]
        }
        """
        try:
            data = request.json
            if not data or "message" not in data:
                return jsonify({"error": "Missing 'message' in request body"}), 400

            message = data["message"]
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            history = data.get("history", [])

            # Build conversation
            messages = history + [{"role": "user", "content": message}]

            # Format with chat template
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Generate
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
                # Remove any trailing special tokens
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

    # Start server
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
