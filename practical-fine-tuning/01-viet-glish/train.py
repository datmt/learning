import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import torch.nn as nn

# --- MONKEY PATCH FOR OLDER PYTORCH VERSIONS ---
if not hasattr(nn.Module, "set_submodule"):

    def set_submodule(self, target: str, module: nn.Module) -> None:
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)

    nn.Module.set_submodule = set_submodule
# -----------------------------------------------

# 1. Load Model and Tokenizer
model_id = "Qwen/Qwen3.5-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. QLoRA 4-bit Compression Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the base model with the 4-bit quantization config attached
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=bnb_config
)

# 3. Load the Dataset
dataset = load_dataset("json", data_files="vietglish_train.jsonl", split="train")


def formatting_func(example):
    return example["text"]


# 4. Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # Dynamically targets all applicable Qwen layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Set Training Arguments
training_args = SFTConfig(
    output_dir="./qwen-vietglish-outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    max_steps=200,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
)

# 6. Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    args=training_args,
)

# 7. Start Training
print("Starting Qwen 3.5 QLoRA training...")
trainer.train()

# 8. Save the trained adapters
trainer.model.save_pretrained("./final-vietglish-qwen-lora")
tokenizer.save_pretrained("./final-vietglish-qwen-lora")
print("Training complete! Adapters saved to ./final-vietglish-qwen-lora")
