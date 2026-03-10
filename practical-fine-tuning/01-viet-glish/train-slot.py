from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Use the official base model you started with
BASE_MODEL = "Qwen/Qwen3.5-4B"  # Change to your exact base model
DATASET_PATH = "vietglish_train.jsonl"
NEW_MODEL_REPO = "datmt24/qwen-3.5-4b-vietglish-unsloth"
HF_TOKEN = os.getenv("HF_TOKEN")
max_seq_length = 2048  # Adjust based on your GPU VRAM
dtype = (
    None  # Automatically detects if your GPU supports bfloat16 (RTX 3090/4090/A4000 do)
)
load_in_4bit = True  # Keeps VRAM usage very low during training

# ==========================================
# 2. LOAD MODEL & LORA ADAPTERS
# ==========================================
print("Loading Base Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply QLoRA - Unsloth handles the complex math and Triton kernels here
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # 4x longer contexts, 2x faster
    random_state=3407,
)

# ==========================================
# 3. FORMAT DATASET
# ==========================================
# Unsloth automatically handles the ChatML template that Qwen expects
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={
        "role": "role",
        "content": "content",
        "user": "user",
        "assistant": "assistant",
    },
)

print("Loading and formatting dataset...")
# Assuming your dataset is a JSONL file with a "conversations" array
dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")
# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Increase this for your actual full training run!
        learning_rate=2e-4,
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

# ==========================================
# 5. THE MAGIC VLLM EXPORT
# ==========================================
print("Pushing merged model to Hugging Face for vLLM compatibility...")

# THIS is the line that solves all the vLLM config nightmare bugs.
# `merged_16bit` forces Unsloth to fuse the adapters into the base model
# and sanitize the config.json for production deployment.
model.push_to_hub_merged(
    NEW_MODEL_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN
)

print("Done! Your model is ready for vLLM.")
