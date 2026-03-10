import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. ENVIRONMENT & SECRETS
# ==========================================
# Run 'export HF_TOKEN=hf_xxx' in your terminal before running this
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "Missing HF_TOKEN! Please run 'export HF_TOKEN=your_token' in your terminal."
    )

# ==========================================
# 2. CONFIGURATION
# ==========================================
BASE_MODEL = "Qwen/Qwen3.5-4B"
DATASET_PATH = "vietglish_train.jsonl"  # Path to your file with the "text" column
NEW_MODEL_REPO = "datmt24/qwen-3.5-4b-vietglish-unsloth"

max_seq_length = 2048
dtype = None  # Auto-detect (will use bf16 on your A40/3090)
load_in_4bit = True

# ==========================================
# 3. LOAD MODEL & LORA ADAPTERS
# ==========================================
print("--- Loading Base Model ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ==========================================
# 4. LOAD DATASET
# ==========================================
print("--- Loading Dataset ---")
dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")

# Check if "text" column exists to avoid KeyError
if "text" not in dataset.column_names:
    raise KeyError(f"Expected 'text' column, but found: {dataset.column_names}")

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================
print("--- Starting Training ---")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Points to your pre-formatted ChatML strings
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Increase this for your full training run
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Prevents unwanted wandb/tensorboard popups
    ),
)

trainer.train()

# ==========================================
# 6. EXPORT & PUSH (The vLLM Fix)
# ==========================================
print("--- Merging and Pushing to Hub ---")
# This creates the repo, merges LoRA into 16bit, and cleans the config.json
model.push_to_hub_merged(
    NEW_MODEL_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN, private=True
)

print(f"Success! Model pushed to https://huggingface.co/{NEW_MODEL_REPO}")
