# LoRA: Low-Rank Adaptation for Efficient Fine-Tuning
## Fine-Tune Large Models with Minimal Parameters

> **Goal**: Understand and implement LoRA, a technique that lets you fine-tune billion-parameter models on consumer hardware by updating only 0.1% of parameters.

---

## Part 1: The Problem LoRA Solves

### Traditional Fine-Tuning is Expensive

Remember Project 2 when we fine-tuned DistilBERT (66M parameters)?

```python
# Full fine-tuning
Total parameters: 66,955,010
Trainable parameters: 66,955,010  # ALL parameters updated!

GPU memory needed: ~8GB
Training time: ~30 minutes
Storage: Need to save entire model for each task
```

**Now imagine GPT-3 (175B parameters):**
- GPU memory needed: ~700GB (doesn't fit on consumer GPUs!)
- Training time: Days/weeks
- Storage: 700GB per fine-tuned version
- Cost: Thousands of dollars

**This is impractical.**

### LoRA's Brilliant Insight

**Question**: Do we really need to update ALL 175 billion parameters?

**Answer**: No! The fine-tuning changes live in a low-rank space.

```python
# LoRA fine-tuning on GPT-3
Total parameters: 175,000,000,000
Trainable parameters: 4,700,000  # Only 0.003%!

GPU memory: ~24GB (fits on consumer GPU!)
Training time: Hours instead of days
Storage: 18MB per fine-tuned version (instead of 700GB!)
```

**LoRA makes impossible fine-tuning possible.**

---

## Part 2: Understanding Low-Rank Decomposition

### What is "Rank"?

Think of a matrix as storing relationships:

```python
# High-rank matrix (full of unique information)
W = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

# Low-rank matrix (redundant/compressible information)
# This matrix can be expressed as product of smaller matrices
W = [[1, 2],    @    [[1, 2, 3],
     [2, 4],          [0, 0, 0]]
     [3, 6]]

# Result is same, but stored more efficiently!
```

**Key insight**: Many matrices in neural networks are "approximately low-rank" - they can be well-approximated by the product of smaller matrices.

### LoRA's Core Idea

Instead of updating a large weight matrix directly:

```python
# Traditional fine-tuning
W_new = W_pretrained + ΔW

# ΔW is HUGE (same size as W_pretrained)
# If W is [4096, 4096], ΔW is 16 million parameters!
```

LoRA decomposes the update into smaller matrices:

```python
# LoRA fine-tuning
W_new = W_pretrained + B @ A

# Where:
# B is [4096, r]  (r is small, e.g., 8)
# A is [r, 4096]
# B @ A gives us [4096, 4096] but only stores 2*4096*r parameters!

# If r=8: Only 65,536 parameters instead of 16 million!
# That's 250x fewer parameters!
```

---

## Part 3: LoRA Math (Step by Step)

### Standard Attention Layer

From Project 3, you know attention uses weight matrices:

```python
# In attention layer
Q = X @ W_q  # Query projection
K = X @ W_k  # Key projection  
V = X @ W_v  # Value projection

# W_q, W_k, W_v are learned weight matrices
# Typically [d_model, d_model], e.g., [768, 768]
```

### Adding LoRA

Instead of updating W_q directly, we add a low-rank adaptation:

```python
# Original (frozen)
W_q_pretrained = [768, 768]  # 589,824 parameters

# LoRA adaptation
B_q = [768, r]  # e.g., r=8 → 6,144 parameters
A_q = [r, 768]  # e.g., r=8 → 6,144 parameters

# Combined
W_q_adapted = W_q_pretrained + (B_q @ A_q)

# Total new parameters: 6,144 + 6,144 = 12,288
# Reduction: 589,824 → 12,288 (48x smaller!)
```

### Forward Pass with LoRA

```python
def forward_with_lora(X, W_pretrained, B, A, alpha=1.0):
    """
    Compute output with LoRA adaptation.
    
    Args:
        X: Input [batch, d_model]
        W_pretrained: Original weights [d_model, d_model] (FROZEN)
        B: LoRA matrix B [d_model, r]
        A: LoRA matrix A [r, d_model]
        alpha: Scaling factor for LoRA (default 1.0)
        
    Returns:
        output: [batch, d_model]
    """
    # Original path (frozen, no gradients)
    output_pretrained = X @ W_pretrained
    
    # LoRA path (trainable)
    output_lora = X @ B @ A
    
    # Combine with scaling
    output = output_pretrained + (alpha * output_lora)
    
    return output
```

### Why This Works

**Intuition**: Fine-tuning typically makes small, structured changes to pre-trained weights. These changes lie in a low-dimensional subspace.

**Example**:
```
Task: Sentiment analysis
- Pre-trained model knows: "amazing" is positive
- Fine-tuning adjustment: "amazing" in "not amazing" is negative

This is a SMALL, SPECIFIC change, not a complete rewrite!
LoRA can capture this with just a few parameters.
```

---

## Part 4: Implementing LoRA from Scratch

### Simple LoRA Layer

```python
import numpy as np

class LoRALayer:
    """
    Low-Rank Adaptation layer.
    
    Wraps a frozen weight matrix and adds trainable low-rank adaptation.
    """
    
    def __init__(self, W_pretrained, rank=8, alpha=1.0):
        """
        Args:
            W_pretrained: Pre-trained weight matrix [d_in, d_out] (FROZEN)
            rank: Rank of adaptation (small number like 4, 8, 16)
            alpha: Scaling factor for LoRA
        """
        self.W = W_pretrained  # Frozen
        self.rank = rank
        self.alpha = alpha
        
        d_in, d_out = W_pretrained.shape
        
        # Initialize LoRA matrices
        # A: Gaussian initialization
        self.A = np.random.randn(rank, d_out) * 0.01
        
        # B: Zero initialization (so initially LoRA adds nothing)
        self.B = np.zeros((d_in, rank))
        
        # Track which parameters are trainable
        self.trainable_params = [self.B, self.A]
    
    def forward(self, X):
        """
        Forward pass: W*X + B*A*X
        
        Args:
            X: Input [batch, d_in]
            
        Returns:
            output: [batch, d_out]
        """
        # Frozen pre-trained path
        output_pretrained = X @ self.W
        
        # LoRA adaptation path
        output_lora = X @ self.B @ self.A
        
        # Combine
        output = output_pretrained + (self.alpha * output_lora)
        
        return output
    
    def parameters(self):
        """Return trainable parameters."""
        return self.trainable_params
    
    def num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.size for p in self.trainable_params)


# Example usage
def test_lora_layer():
    """Demonstrate LoRA layer."""
    
    print("="*70)
    print("LoRA LAYER DEMONSTRATION")
    print("="*70)
    
    # Create a pre-trained weight matrix
    d_model = 768  # DistilBERT size
    W_pretrained = np.random.randn(d_model, d_model) * 0.1
    
    print(f"\nPre-trained weight matrix: {W_pretrained.shape}")
    print(f"Parameters in W: {W_pretrained.size:,}")
    
    # Create LoRA layer with rank 8
    rank = 8
    lora = LoRALayer(W_pretrained, rank=rank)
    
    print(f"\nLoRA configuration:")
    print(f"  Rank: {rank}")
    print(f"  B matrix: {lora.B.shape} = {lora.B.size:,} parameters")
    print(f"  A matrix: {lora.A.shape} = {lora.A.size:,} parameters")
    print(f"  Total trainable: {lora.num_trainable_params():,} parameters")
    
    reduction = W_pretrained.size / lora.num_trainable_params()
    print(f"\nParameter reduction: {reduction:.1f}x")
    print(f"Percentage of original: {100/reduction:.2f}%")
    
    # Test forward pass
    batch_size = 4
    X = np.random.randn(batch_size, d_model)
    
    output = lora.forward(X)
    print(f"\nForward pass:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("="*70 + "\n")

test_lora_layer()
```

---

## Part 5: LoRA in Attention Layers

### Applying LoRA to Multi-Head Attention

Remember from Project 3, attention has 3 weight matrices (Q, K, V). We can apply LoRA to each:

```python
class LoRAAttention:
    """
    Attention layer with LoRA adaptations.
    """
    
    def __init__(self, d_model, d_k, rank=8):
        """
        Args:
            d_model: Model dimension
            d_k: Query/Key/Value dimension
            rank: LoRA rank
        """
        self.d_model = d_model
        self.d_k = d_k
        self.rank = rank
        
        # Pre-trained weights (FROZEN)
        self.W_q = np.random.randn(d_model, d_k) * 0.1
        self.W_k = np.random.randn(d_model, d_k) * 0.1
        self.W_v = np.random.randn(d_model, d_k) * 0.1
        
        # LoRA adaptations for each weight matrix
        self.lora_q = LoRALayer(self.W_q, rank=rank)
        self.lora_k = LoRALayer(self.W_k, rank=rank)
        self.lora_v = LoRALayer(self.W_v, rank=rank)
    
    def forward(self, X, return_attention=False):
        """Forward pass with LoRA."""
        # Compute Q, K, V using LoRA-adapted weights
        Q = self.lora_q.forward(X)
        K = self.lora_k.forward(X)
        V = self.lora_v.forward(X)
        
        # Standard attention computation
        scores = Q @ K.T / np.sqrt(self.d_k)
        attention_weights = self.softmax(scores)
        output = attention_weights @ V
        
        if return_attention:
            return output, attention_weights
        return output
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def trainable_parameters(self):
        """Get all trainable parameters."""
        params = []
        params.extend(self.lora_q.parameters())
        params.extend(self.lora_k.parameters())
        params.extend(self.lora_v.parameters())
        return params
    
    def num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.size for p in self.trainable_parameters())


def test_lora_attention():
    """Test LoRA attention."""
    
    print("="*70)
    print("LoRA ATTENTION DEMONSTRATION")
    print("="*70)
    
    d_model = 768
    d_k = 64
    rank = 8
    
    # Create LoRA attention
    attn = LoRAAttention(d_model, d_k, rank=rank)
    
    # Count parameters
    original_params = attn.W_q.size + attn.W_k.size + attn.W_v.size
    trainable_params = attn.num_trainable_params()
    
    print(f"\nConfiguration:")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention dimension: {d_k}")
    print(f"  LoRA rank: {rank}")
    
    print(f"\nParameter comparison:")
    print(f"  Original (frozen): {original_params:,}")
    print(f"  LoRA (trainable): {trainable_params:,}")
    print(f"  Reduction: {original_params/trainable_params:.1f}x")
    print(f"  Trainable %: {trainable_params/original_params*100:.2f}%")
    
    # Test forward pass
    seq_len = 10
    X = np.random.randn(seq_len, d_model)
    
    output, attn_weights = attn.forward(X, return_attention=True)
    
    print(f"\nForward pass:")
    print(f"  Input: {X.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attn_weights.shape}")
    
    print("="*70 + "\n")

test_lora_attention()
```

---

## Part 6: Practical LoRA Fine-Tuning with HuggingFace

### Setup

```bash
pip install peft bitsandbytes --break-system-packages
```

### Loading Model with LoRA

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Count original parameters
original_params = sum(p.numel() for p in model.parameters())
print(f"Original parameters: {original_params:,}")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification
    r=8,                          # Rank
    lora_alpha=16,                # Scaling factor
    lora_dropout=0.1,             # Dropout for LoRA layers
    target_modules=["q_lin", "v_lin"],  # Which layers to adapt
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {trainable_params/original_params*100:.3f}%")

# Print model architecture to see LoRA layers
model.print_trainable_parameters()
```

Output:
```
Original parameters: 66,955,010
Trainable parameters: 294,912
Percentage trainable: 0.440%

trainable params: 294,912 || all params: 67,249,922 || trainable%: 0.4385
```

---

## Part 7: Complete Fine-Tuning Example

### Full Script with LoRA

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


def setup_lora_model():
    """Load model and apply LoRA."""
    
    print("="*70)
    print("SETTING UP LoRA MODEL")
    print("="*70)
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Original parameters
    original_params = sum(p.numel() for p in model.parameters())
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                           # Rank (try 4, 8, 16, 32)
        lora_alpha=16,                 # Alpha (usually 2*r)
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # Adapt attention Q and V
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_name}")
    print(f"Original parameters: {original_params:,}")
    print(f"LoRA trainable parameters: {trainable_params:,}")
    print(f"Reduction: {original_params/trainable_params:.1f}x")
    print(f"Trainable percentage: {trainable_params/original_params*100:.3f}%")
    
    print("\nLoRA configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    print("="*70 + "\n")
    
    return model, tokenizer


def prepare_dataset(tokenizer, sample_size=1000):
    """Load and tokenize IMDB dataset."""
    
    print("="*70)
    print("PREPARING DATA")
    print("="*70)
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Create subset
    train_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))
    test_dataset = dataset['test'].shuffle(seed=42).select(range(sample_size // 5))
    
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Test examples: {len(test_dataset):,}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print("="*70 + "\n")
    
    return tokenized_train, tokenized_test


def train_with_lora(model, train_dataset, test_dataset):
    """Train model with LoRA."""
    
    print("="*70)
    print("TRAINING WITH LoRA")
    print("="*70)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-4,  # Higher LR for LoRA (since fewer params)
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final accuracy: {results['eval_accuracy']:.4f}")
    print("="*70 + "\n")
    
    return trainer, results


def save_and_load_lora(model, tokenizer):
    """Demonstrate saving/loading LoRA adapters."""
    
    print("="*70)
    print("SAVING LoRA ADAPTERS")
    print("="*70)
    
    # Save LoRA adapters (only the small matrices!)
    adapter_path = "./lora_adapters"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    import os
    adapter_size = sum(
        os.path.getsize(os.path.join(adapter_path, f)) 
        for f in os.listdir(adapter_path) 
        if f.endswith('.bin')
    ) / (1024 * 1024)  # Convert to MB
    
    print(f"\nLoRA adapters saved to: {adapter_path}")
    print(f"Adapter file size: ~{adapter_size:.1f} MB")
    print(f"\nCompare to full model: ~250 MB")
    print(f"Size reduction: ~{250/adapter_size:.0f}x smaller!")
    
    print("\nTo load later:")
    print(f"  from peft import AutoPeftModelForSequenceClassification")
    print(f"  model = AutoPeftModelForSequenceClassification.from_pretrained('{adapter_path}')")
    
    print("="*70 + "\n")


def main():
    """Complete LoRA fine-tuning pipeline."""
    
    print("\n" + "="*70)
    print("LoRA FINE-TUNING: EFFICIENT PARAMETER ADAPTATION")
    print("="*70 + "\n")
    
    # Setup
    model, tokenizer = setup_lora_model()
    train_dataset, test_dataset = prepare_dataset(tokenizer, sample_size=1000)
    
    # Train
    trainer, results = train_with_lora(model, train_dataset, test_dataset)
    
    # Save
    save_and_load_lora(model, tokenizer)
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print("\nWhat you accomplished:")
    print("  ✓ Applied LoRA to DistilBERT")
    print("  ✓ Reduced trainable parameters by ~200x")
    print("  ✓ Fine-tuned for sentiment analysis")
    print("  ✓ Saved adapters (only ~1MB!)")
    print(f"\nFinal accuracy: {results['eval_accuracy']:.2%}")
    print("\nKey insight:")
    print("  Same performance, 1% of parameters!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
```

---

## Part 8: Understanding LoRA Hyperparameters

### Rank (r)

**What it controls**: Size of the low-rank matrices

```python
# Low rank (r=4)
- Fewer parameters
- Faster training
- May underfit (not enough capacity)

# Medium rank (r=8-16)  ← RECOMMENDED
- Good balance
- Works for most tasks

# High rank (r=32-64)
- More parameters
- Better performance on complex tasks
- Diminishing returns
```

**Rule of thumb**: Start with r=8, increase if performance plateaus.

### Alpha (lora_alpha)

**What it controls**: Scaling of LoRA updates

```python
# Formula
scaling = lora_alpha / rank

# Common patterns
r=8, alpha=16  → scaling=2  (moderate)
r=8, alpha=8   → scaling=1  (conservative)
r=8, alpha=32  → scaling=4  (aggressive)
```

**Rule of thumb**: Set alpha = 2*r

### Target Modules

**Which layers to adapt**:

```python
# Conservative (Q and V only)
target_modules=["q_lin", "v_lin"]
- Fewer parameters
- Often sufficient

# Moderate (Q, K, V)
target_modules=["q_lin", "k_lin", "v_lin"]
- More flexibility

# Aggressive (all linear layers)
target_modules=["q_lin", "k_lin", "v_lin", "out_lin"]
- Maximum adaptation
- Most parameters
```

---

## Part 9: LoRA vs Full Fine-Tuning Comparison

### Experiment: Same Task, Different Methods

```python
import time

def compare_methods(sample_size=1000):
    """Compare full fine-tuning vs LoRA."""
    
    print("="*70)
    print("COMPARISON: FULL FINE-TUNING vs LoRA")
    print("="*70)
    
    results = {}
    
    # Method 1: Full fine-tuning
    print("\n[1/2] Full Fine-Tuning...")
    model_full = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    
    full_params = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    
    start = time.time()
    # ... train model_full ...
    full_time = time.time() - start
    
    results['full'] = {
        'params': full_params,
        'time': full_time,
        'accuracy': 0.92  # Example
    }
    
    # Method 2: LoRA
    print("\n[2/2] LoRA Fine-Tuning...")
    model_lora = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
    )
    model_lora = get_peft_model(model_lora, lora_config)
    
    lora_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    
    start = time.time()
    # ... train model_lora ...
    lora_time = time.time() - start
    
    results['lora'] = {
        'params': lora_params,
        'time': lora_time,
        'accuracy': 0.91  # Example (usually within 1%)
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Full Fine-Tuning':<20} {'LoRA':<20}")
    print("-"*70)
    print(f"{'Trainable Parameters':<25} {results['full']['params']:>19,} {results['lora']['params']:>19,}")
    print(f"{'Training Time':<25} {results['full']['time']:>17.1f}s {results['lora']['time']:>17.1f}s")
    print(f"{'Accuracy':<25} {results['full']['accuracy']:>19.2%} {results['lora']['accuracy']:>19.2%}")
    
    print(f"\n{'Reduction':<25} {'1x (baseline)':<20} {results['full']['params']/results['lora']['params']:>18.1f}x")
    print(f"{'Speed-up':<25} {'1x (baseline)':<20} {results['full']['time']/results['lora']['time']:>18.2f}x")
    
    print("="*70 + "\n")
```

**Typical results**:
```
Metric                    Full Fine-Tuning     LoRA
----------------------------------------------------------------------
Trainable Parameters            66,955,010          294,912
Training Time                        180s               45s
Accuracy                            92.3%             91.8%

Reduction                    1x (baseline)            227x
Speed-up                     1x (baseline)           4.00x
```

---

## Part 10: When to Use LoRA

### ✅ Use LoRA When:

1. **Limited GPU memory**
   - Can't fit full model in memory
   - Want to train larger models

2. **Multiple task adaptations**
   - Need 10 versions for different tasks
   - Storage: 10×1MB vs 10×250MB

3. **Fast iteration**
   - Experimenting with hyperparameters
   - Rapid prototyping

4. **Similar domain to pre-training**
   - Task is related to pre-training data
   - Need small adjustments, not complete rewrite

### ❌ Consider Full Fine-Tuning When:

1. **Very different domain**
   - Medical text with pre-trained on web text
   - Specialized vocabulary/patterns

2. **Have resources**
   - Unlimited GPU memory/budget
   - Want absolute best performance

3. **Small models**
   - Model already fits comfortably
   - Savings not significant

---

## Part 11: Advanced LoRA Techniques

### QLoRA: Quantized LoRA

**Idea**: Combine LoRA with quantization for even more efficiency

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model in 4-bit
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on top
model = get_peft_model(model, lora_config)

# Result: Train 13B model on 16GB GPU!
```

### Rank Selection via Importance

```python
def find_optimal_rank(model, dataset, ranks=[4, 8, 16, 32]):
    """Test different ranks to find sweet spot."""
    
    results = {}
    
    for r in ranks:
        lora_config = LoraConfig(r=r, lora_alpha=2*r, ...)
        model_lora = get_peft_model(model, lora_config)
        
        # Train and evaluate
        accuracy = train_and_eval(model_lora, dataset)
        params = count_trainable_params(model_lora)
        
        results[r] = {'accuracy': accuracy, 'params': params}
        
        print(f"Rank {r}: {accuracy:.3f} accuracy, {params:,} params")
    
    return results
```

---

## Part 12: Key Takeaways

### What You Now Understand

1. **LoRA = Low-Rank Decomposition**
   - Updates are B×A instead of full ΔW
   - Rank controls capacity vs efficiency
   - Works because fine-tuning changes are low-dimensional

2. **Dramatic Efficiency Gains**
   - 100-1000x fewer parameters
   - 2-4x faster training
   - Minimal accuracy loss (<1%)

3. **Practical Advantages**
   - Train large models on consumer GPUs
   - Store multiple adaptations cheaply
   - Fast experimentation

4. **Connection to Attention**
   - LoRA typically adapts attention weights (Q, K, V)
   - Same matrices you implemented in Project 3!
   - Small changes to these weights = task adaptation

### The Big Picture

```
Project 1 (Tokenizer): Text → Tokens
Project 2 (Fine-tuning): Training loop, loss, gradients
Project 3 (Attention): The core transformer mechanism
Project 4 (LoRA): Efficient way to adapt attention ← YOU ARE HERE

Next: RAG (using these adapted models with retrieval)
```

---

## Part 13: Exercises

### Exercise 1: Rank Comparison
```python
# Try different ranks and compare
ranks = [4, 8, 16, 32]
for r in ranks:
    # Create LoRA model with rank r
    # Train and evaluate
    # Plot accuracy vs parameters
```

### Exercise 2: Target Module Ablation
```python
# What if we only adapt Q? Or only V?
configs = [
    {"target_modules": ["q_lin"]},
    {"target_modules": ["v_lin"]},
    {"target_modules": ["q_lin", "v_lin"]},
]
# Compare performance
```

### Exercise 3: Merge LoRA Weights
```python
# After training, merge LoRA back into base model
merged_model = model.merge_and_unload()
# Now you have a single model with adapted weights
# No more separate LoRA matrices!
```

---

## Resources

- **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models"
- **QLoRA Paper**: "QLoRA: Efficient Finetuning of Quantized LLMs"
- **PEFT Library**: https://github.com/huggingface/peft
- **Tutorial**: https://huggingface.co/docs/peft/

---

## Your Assignment

1. **Run the LoRA fine-tuning script**
2. **Compare different ranks** (4 vs 8 vs 16)
3. **Measure adapter file size** - see how small it is!
4. **Answer**: Why does LoRA work? What assumption does it make about fine-tuning?

When ready, we move to Project 5: RAG (Retrieval-Augmented Generation)!