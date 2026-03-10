
## Training 

### Install dependencies
pip install transformers peft trl datasets accelerate bitsandbytes

### install login huggingface
curl -LsSf https://hf.co/cli/install.sh | bash

### run this to login
hf auth login


### run this to start Training
python train 

## Push 

### Package and push 

```python 
python package_and_push.py \
  --adapter-dir ./final-dir-lora \
  --repo-id datmt24/HF_REPO_ID \
  --hf-token hf_your_token_here
```


### push both adapter and model

```python 
python package_and_push.py \
  --adapter-dir ./final-vietglish-lora \
  --repo-id datmt24/qwen-vietglish-merged \
  --push-lora \
  --lora-repo-id datmt24/qwen-vietglish-lora \
  --hf-token hf_your_token_here

```

### push and specify base model 

```python 
python package_and_push.py \
  --adapter-dir ./final-vietglish-lora \
  --base-model Qwen/Qwen3.5-4B \
  --repo-id datmt24/qwen-vietglish-merged \
  --hf-token hf_your_token_here

```

### Run quick-push.sh 
Run this to avoid running python file manually

```bash 
export HF_TOKEN=hf_your_token_here
./quick_push.sh
```

