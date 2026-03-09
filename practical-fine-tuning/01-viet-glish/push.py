from peft import PeftModel
from transformers import AutoTokenizer

# Load the saved LoRA adapter
model_path = "./final-vietglish-lora"
repo_id = "datmt24/gemma-vietglish-lora"  # Change this!

# Push to hub
from huggingface_hub import HfApi

api = HfApi()

# Upload the entire folder
api.upload_folder(
    folder_path=model_path,
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model pushed to https://huggingface.co/{repo_id}")
