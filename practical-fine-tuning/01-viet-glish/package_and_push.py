#!/usr/bin/env python3
"""
Post-training script: Merge LoRA, save standalone model, and push to HuggingFace
Usage: python package_and_push.py --adapter-dir ./final-vietglish-lora --repo-id datmt24/qwen-vietglish-merged --hf-token hf_xxx
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and push to HuggingFace"
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID (will auto-detect from adapter if not provided)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (optional if already logged in)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./merged-model",
        help="Local directory to save merged model",
    )
    parser.add_argument(
        "--push-lora", action="store_true", help="Also push the LoRA adapter separately"
    )
    parser.add_argument(
        "--lora-repo-id",
        type=str,
        default=None,
        help="Separate repo ID for LoRA adapter (if --push-lora is set)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the HuggingFace repo private"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging, only push existing adapter",
    )

    args = parser.parse_args()

    # Validate adapter directory
    adapter_path = Path(args.adapter_dir)
    if not adapter_path.exists():
        print(f"❌ Error: Adapter directory '{args.adapter_dir}' does not exist")
        sys.exit(1)

    # Check for adapter_config.json
    if not (adapter_path / "adapter_config.json").exists():
        print(
            f"❌ Error: '{args.adapter_dir}' does not appear to be a valid LoRA adapter (missing adapter_config.json)"
        )
        sys.exit(1)

    print("=" * 60)
    print("🚀 LoRA Post-Training Automation Script")
    print("=" * 60)

    # Login to HuggingFace if token provided
    if args.hf_token:
        print("\n📝 Logging in to HuggingFace...")
        from huggingface_hub import login

        login(token=args.hf_token)
        print("✅ Logged in successfully")

    # Auto-detect base model if not provided
    base_model_id = args.base_model
    if not base_model_id:
        print("\n🔍 Auto-detecting base model from adapter config...")
        import json

        with open(adapter_path / "adapter_config.json", "r") as f:
            adapter_config = json.load(f)

        # Try to find base model from config
        if "base_model_name_or_path" in adapter_config:
            base_model_id = adapter_config["base_model_name_or_path"]
            print(f"✅ Detected base model: {base_model_id}")
        else:
            print(
                "❌ Error: Could not auto-detect base model. Please provide --base-model"
            )
            sys.exit(1)

    if not args.skip_merge:
        # Merge LoRA weights
        print("\n🔄 Loading models and merging LoRA weights...")
        print(f"   Base model: {base_model_id}")
        print(f"   LoRA adapter: {args.adapter_dir}")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        try:
            # Load base model
            print("   Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Load LoRA adapter
            print("   Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, str(adapter_path))

            # Merge
            print("   Merging LoRA weights into base model...")
            merged_model = model.merge_and_unload()

            # Load tokenizer
            print("   Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

            # Save merged model locally
            print(f"\n💾 Saving merged model to '{args.output_dir}'...")
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            merged_model.save_pretrained(str(output_path), safe_serialization=True)
            tokenizer.save_pretrained(str(output_path))

            print("✅ Merged model saved successfully")

            # Push merged model to HuggingFace
            print(f"\n📤 Pushing merged model to HuggingFace: {args.repo_id}")
            merged_model.push_to_hub(
                args.repo_id, private=args.private, safe_serialization=True
            )
            tokenizer.push_to_hub(args.repo_id, private=args.private)

            print(f"✅ Merged model pushed to https://huggingface.co/{args.repo_id}")

            # Clean up memory
            del merged_model
            del base_model
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error during merge/push: {str(e)}")
            sys.exit(1)

    # Push LoRA adapter if requested
    if args.push_lora:
        lora_repo = args.lora_repo_id or f"{args.repo_id}-lora"
        print(f"\n📤 Pushing LoRA adapter to HuggingFace: {lora_repo}")

        from huggingface_hub import HfApi

        api = HfApi()

        try:
            api.upload_folder(
                folder_path=str(adapter_path),
                repo_id=lora_repo,
                repo_type="model",
                private=args.private,
            )
            print(f"✅ LoRA adapter pushed to https://huggingface.co/{lora_repo}")
        except Exception as e:
            print(f"\n❌ Error pushing LoRA adapter: {str(e)}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 All done!")
    print("=" * 60)

    if not args.skip_merge:
        print(f"\n📦 Merged model: https://huggingface.co/{args.repo_id}")
        print(f"\n💡 To use your merged model:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{args.repo_id}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.repo_id}')")

    if args.push_lora:
        lora_repo = args.lora_repo_id or f"{args.repo_id}-lora"
        print(f"\n📦 LoRA adapter: https://huggingface.co/{lora_repo}")
        print(f"\n💡 To use your LoRA adapter:")
        print(f"   from peft import PeftModel")
        print(f"   base = AutoModelForCausalLM.from_pretrained('{base_model_id}')")
        print(f"   model = PeftModel.from_pretrained(base, '{lora_repo}')")


if __name__ == "__main__":
    main()
