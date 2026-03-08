# Fine-Tuning Your First Model: Text Classification
## Understanding How Models Learn

> **Goal**: Fine-tune DistilBERT for sentiment analysis, understanding every step from tokens to trained model.

---

## Part 1: What is Fine-Tuning?

### The Analogy

Think of a pre-trained model like a **college graduate**:
- They've learned general knowledge (reading, writing, reasoning)
- But they don't know YOUR specific job yet

**Fine-tuning** = On-the-job training for your specific task

### What Actually Happens

```
Pre-trained Model (DistilBERT):
- Trained on millions of documents
- Understands language, grammar, context
- Knows "amazing" is positive, "terrible" is negative
- But never explicitly trained to output "positive" or "negative" labels

↓ Fine-tuning ↓

Your Fine-tuned Model:
- Takes the pre-trained knowledge
- Adds a classification head (new output layer)
- Trains on YOUR labeled data (reviews → positive/negative)
- Learns to map text → specific labels
```

### Why Start with Classification?

1. **Clear success metric**: Accuracy (was the prediction correct?)
2. **Fast training**: Minutes, not hours
3. **Small data requirements**: Can work with 1,000-10,000 examples
4. **Foundation for everything else**: Same concepts apply to generation, QA, etc.

---

## Part 2: The Architecture You're Fine-Tuning

### From Tokens to Predictions

Let's trace a single example through the model:

```
Input: "This movie was amazing!"

Step 1: Tokenization (you already understand this!)
→ [101, 2023, 3185, 2001, 6429, 999, 102]
   (CLS, This, movie, was, amazing, !, SEP)

Step 2: Token Embeddings (NEW - this is what we're learning)
Each token ID → vector of numbers
101 → [0.23, -0.45, 0.12, ..., 0.67]  (768 numbers for DistilBERT)
2023 → [-0.12, 0.34, 0.56, ..., -0.23]
...

Step 3: Transformer Layers (pre-trained, we keep these mostly frozen)
- 6 layers of attention + feedforward
- Each layer refines the embeddings
- Captures context: "amazing" in "amazing pizza" vs "not amazing"

Step 4: Classification Head (NEW - we add this!)
Takes [CLS] token embedding → 2 numbers [score_negative, score_positive]
[0.23, -0.45, ..., 0.67] → [-0.3, 2.1]
                           ↓
                      softmax
                           ↓
                     [0.08, 0.92]
                     ↓
                  "Positive" (92% confident)
```

### What Gets Trained?

**Option 1: Full Fine-tuning** (what we'll do first)
- Update ALL weights in the model
- Slower, needs more data
- Best performance

**Option 2: Freeze base, train head only** (we'll try this too)
- Only update classification head
- Faster, needs less data
- Sometimes good enough

**Option 3: LoRA** (Phase 2, later)
- Update small "adapter" layers
- Best of both worlds

---

## Part 3: Setup and Installation

### System Check

First, verify your GPU:

```bash
# Check NVIDIA driver
nvidia-smi

# You should see something like:
# GPU 0: NVIDIA GeForce RTX 3080
# Memory: 10240MiB
```

### Install Dependencies

```bash
# Core libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --break-system-packages

# HuggingFace ecosystem
pip install transformers datasets accelerate evaluate --break-system-packages

# Utilities
pip install scikit-learn matplotlib tqdm --break-system-packages
```

### Verify Installation

```python
import torch
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Transformers version: {transformers.__version__}")
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3080
Transformers version: 4.x.x
```

---

## Part 4: Understanding the Dataset

### We'll Use IMDB Movie Reviews

- **Task**: Classify movie reviews as positive or negative
- **Size**: 25,000 training examples, 25,000 test examples
- **Why**: Classic benchmark, real-world text, clear labels

### Loading the Data

```python
from datasets import load_dataset

# Load IMDB dataset
dataset = load_dataset("imdb")

print(dataset)
# Output:
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
# })

# Inspect examples
print("\nExample 1:")
print(f"Text: {dataset['train'][0]['text'][:200]}...")
print(f"Label: {dataset['train'][0]['label']} (0=negative, 1=positive)")

print("\nExample 2:")
print(f"Text: {dataset['train'][1]['text'][:200]}...")
print(f"Label: {dataset['train'][1]['label']}")
```

### Dataset Statistics

```python
import numpy as np

# Label distribution
labels = dataset['train']['label']
unique, counts = np.unique(labels, return_counts=True)
print("\nLabel distribution:")
for label, count in zip(unique, counts):
    print(f"  Label {label}: {count} examples ({count/len(labels)*100:.1f}%)")

# Text length distribution
text_lengths = [len(text.split()) for text in dataset['train']['text']]
print(f"\nText length statistics (words):")
print(f"  Mean: {np.mean(text_lengths):.1f}")
print(f"  Median: {np.median(text_lengths):.1f}")
print(f"  Min: {np.min(text_lengths)}")
print(f"  Max: {np.max(text_lengths)}")
```

### Smaller Dataset for Experimentation

For faster iterations during learning:

```python
# Create a smaller subset
small_train = dataset['train'].shuffle(seed=42).select(range(1000))
small_test = dataset['test'].shuffle(seed=42).select(range(200))

print(f"Small training set: {len(small_train)} examples")
print(f"Small test set: {len(small_test)} examples")
```

---

## Part 5: Tokenization Deep Dive

### Loading the Tokenizer

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
print(f"Max length: {tokenizer.model_max_length}")
```

### Tokenize a Single Example

```python
text = "This movie was absolutely amazing!"

# Tokenize
encoded = tokenizer(text, return_tensors="pt")

print("\nTokenization breakdown:")
print(f"Input text: {text}")
print(f"Token IDs: {encoded['input_ids'][0].tolist()}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")
print(f"Attention mask: {encoded['attention_mask'][0].tolist()}")
```

### Understanding Attention Masks

```python
# Example with padding
texts = [
    "Short review",
    "This is a much longer review with many more words to demonstrate padding"
]

encoded = tokenizer(texts, padding=True, return_tensors="pt")

print("\nBatch tokenization with padding:")
for i, text in enumerate(texts):
    print(f"\nText {i}: {text}")
    print(f"  Token IDs: {encoded['input_ids'][i].tolist()}")
    print(f"  Attention mask: {encoded['attention_mask'][i].tolist()}")
    print(f"  (1 = real token, 0 = padding)")
```

### Tokenize Entire Dataset

```python
def tokenize_function(examples):
    """Tokenize a batch of texts."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512  # DistilBERT's maximum
    )

# Tokenize datasets
tokenized_train = small_train.map(tokenize_function, batched=True)
tokenized_test = small_test.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

print(f"\nTokenized training set: {len(tokenized_train)} examples")
print(f"Features: {tokenized_train.features}")
```

---

## Part 6: Loading the Pre-trained Model

### Model Architecture

```python
from transformers import AutoModelForSequenceClassification

# Load model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model loaded on: {device}")
print(f"\nModel architecture:")
print(model)
```

### Understanding Model Size

```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB (float32)")
```

### Before Training: Random Predictions

```python
# Test model before training
model.eval()
test_text = "This movie was terrible and boring."

encoded = tokenizer(test_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**encoded)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1)

print(f"\nBefore training:")
print(f"Text: {test_text}")
print(f"Logits: {logits[0].tolist()}")
print(f"Probabilities: {probs[0].tolist()}")
print(f"Prediction: {prediction.item()} ({'positive' if prediction.item() == 1 else 'negative'})")
print(f"(Random guess - model hasn't learned yet!)")
```

---

## Part 7: Training Setup

### Understanding Loss Functions

```python
import torch.nn.functional as F

# Example of cross-entropy loss
# Let's say model outputs these logits for a positive review
logits = torch.tensor([[-0.5, 2.0]])  # [negative_score, positive_score]
true_label = torch.tensor([1])  # 1 = positive

# Convert logits to probabilities
probs = F.softmax(logits, dim=1)
print(f"Probabilities: {probs}")
# Output: [0.08, 0.92] - model is 92% confident it's positive

# Calculate loss
loss = F.cross_entropy(logits, true_label)
print(f"Loss: {loss.item():.4f}")
# Low loss (around 0.08) - model is correct and confident!

# Now try wrong prediction
logits_wrong = torch.tensor([[2.0, -0.5]])  # Model thinks negative
loss_wrong = F.cross_entropy(logits_wrong, true_label)
print(f"Loss (wrong): {loss_wrong.item():.4f}")
# High loss (around 2.1) - model is wrong!
```

**Key insight**: Training minimizes this loss by adjusting weights.

### Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    
    # Training hyperparameters
    num_train_epochs=3,              # How many times to see the full dataset
    per_device_train_batch_size=16,  # How many examples per GPU batch
    per_device_eval_batch_size=32,   # Larger batches for evaluation (no backprop)
    
    # Learning rate
    learning_rate=2e-5,              # Small updates to pre-trained weights
    
    # Optimization
    warmup_steps=100,                # Gradually increase learning rate
    weight_decay=0.01,               # L2 regularization (prevent overfitting)
    
    # Logging and evaluation
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    
    # Resource optimization
    fp16=True,                       # Use mixed precision (faster on modern GPUs)
    
    # Other
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=42
)

print("Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Total training steps: {len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
```

### Evaluation Metrics

```python
from datasets import load_metric
import numpy as np

# Load accuracy metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    """Compute accuracy from predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric.compute(predictions=predictions, references=labels)
    
    # Additional metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

---

## Part 8: Training Loop (The Magic Happens)

### Using HuggingFace Trainer

```python
from transformers import Trainer

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

print("Starting training...")
print("Watch the loss decrease - that's the model learning!\n")

# Train!
trainer.train()
```

### What's Happening During Training?

```
Each step:
1. Sample batch (16 reviews)
2. Forward pass: tokens → embeddings → transformer → logits
3. Compute loss: How wrong are the predictions?
4. Backward pass: Calculate gradients (which way to adjust weights)
5. Update weights: Take small step in direction that reduces loss
6. Repeat

After 1000 examples:
  Loss: 0.45 → Model starting to learn
  
After 5000 examples:
  Loss: 0.15 → Model getting confident
  
After full dataset:
  Loss: 0.05 → Model well-trained
```

### Manual Training Loop (Understanding Under the Hood)

Here's what Trainer does internally:

```python
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

def manual_training_loop():
    """See exactly what happens during training."""
    
    # Setup
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    train_loader = DataLoader(tokenized_train, batch_size=16, shuffle=True)
    
    # Training
    epoch_losses = []
    
    for epoch in range(1):  # Just 1 epoch for demonstration
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Move batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()  # Compute gradients
            
            # Update weights
            optimizer.step()  # Apply gradients
            optimizer.zero_grad()  # Clear gradients for next iteration
            
            # Log every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return epoch_losses

# Uncomment to see manual training
# losses = manual_training_loop()
```

---

## Part 9: Evaluation and Results

### Evaluate on Test Set

```python
# Evaluate
eval_results = trainer.evaluate()

print("\nEvaluation Results:")
print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"  Precision: {eval_results['eval_precision']:.4f}")
print(f"  Recall: {eval_results['eval_recall']:.4f}")
print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
print(f"  Loss: {eval_results['eval_loss']:.4f}")
```

### Test on Custom Examples

```python
def predict_sentiment(text):
    """Predict sentiment of custom text."""
    model.eval()
    
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1)
    
    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    confidence = probs[0][prediction.item()].item()
    
    return sentiment, confidence

# Test examples
test_examples = [
    "This movie was absolutely amazing! Best film I've seen all year.",
    "Terrible movie. Waste of time and money.",
    "It was okay, not great but not terrible either.",
    "Mind-blowing cinematography and stellar performances!",
    "Boring plot and terrible acting. Very disappointed."
]

print("\nTesting fine-tuned model:")
print("="*70)
for text in test_examples:
    sentiment, confidence = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"Prediction: {sentiment} (confidence: {confidence:.2%})")
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get predictions for entire test set
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

# Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - IMDB Sentiment Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved to confusion_matrix.png")
```

---

## Part 10: Saving and Loading Your Model

### Save Fine-tuned Model

```python
# Save model and tokenizer
save_directory = "./my_finetuned_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"\nModel saved to {save_directory}")
print(f"Directory contents:")
import os
print(os.listdir(save_directory))
```

### Load and Use Later

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your fine-tuned model
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
loaded_model = loaded_model.to(device)

# Use it
text = "This is a test review"
encoded = loaded_tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = loaded_model(**encoded)
    prediction = torch.argmax(outputs.logits, dim=1)

print(f"Loaded model prediction: {prediction.item()}")
```

---

## Part 11: Experiments to Try

### Experiment 1: Effect of Training Data Size

```python
# Compare models trained on different data sizes
data_sizes = [100, 500, 1000, 5000]

results = {}
for size in data_sizes:
    print(f"\nTraining with {size} examples...")
    
    # Create subset
    train_subset = dataset['train'].shuffle(seed=42).select(range(size))
    tokenized_subset = train_subset.map(tokenize_function, batched=True)
    tokenized_subset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Train
    model_subset = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    trainer_subset = Trainer(
        model=model_subset,
        args=training_args,
        train_dataset=tokenized_subset,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    trainer_subset.train()
    
    # Evaluate
    eval_result = trainer_subset.evaluate()
    results[size] = eval_result['eval_accuracy']
    print(f"  Accuracy: {eval_result['eval_accuracy']:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xlabel('Training Data Size')
plt.ylabel('Accuracy')
plt.title('Effect of Training Data Size on Performance')
plt.grid(True)
plt.savefig('data_size_effect.png')
print("\nPlot saved to data_size_effect.png")
```

### Experiment 2: Learning Rate Comparison

```python
learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]

lr_results = {}
for lr in learning_rates:
    print(f"\nTraining with learning rate {lr}...")
    
    model_lr = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    args_lr = TrainingArguments(
        output_dir=f'./results_lr_{lr}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True
    )
    
    trainer_lr = Trainer(
        model=model_lr,
        args=args_lr,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    
    trainer_lr.train()
    eval_result = trainer_lr.evaluate()
    lr_results[lr] = eval_result['eval_accuracy']
    print(f"  Accuracy: {eval_result['eval_accuracy']:.4f}")

print("\nLearning rate comparison:")
for lr, acc in lr_results.items():
    print(f"  LR {lr}: {acc:.4f}")
```

### Experiment 3: Freeze vs Full Fine-tuning

```python
# Freeze base model, only train classifier head
model_frozen = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Freeze all base model parameters
for param in model_frozen.distilbert.parameters():
    param.requires_grad = False

# Only classifier head is trainable
trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_frozen.parameters())

print(f"\nFrozen model:")
print(f"  Trainable parameters: {trainable:,} ({trainable/total*100:.1f}%)")

trainer_frozen = Trainer(
    model=model_frozen,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

print("\nTraining frozen model (faster)...")
trainer_frozen.train()

eval_frozen = trainer_frozen.evaluate()
print(f"\nFrozen model accuracy: {eval_frozen['eval_accuracy']:.4f}")
print(f"Full fine-tuning accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Difference: {abs(eval_frozen['eval_accuracy'] - eval_results['eval_accuracy']):.4f}")
```

---

## Part 12: Key Takeaways

### What You Now Understand

1. **Token → Embedding → Prediction Pipeline**
   - Tokens are just indices
   - Embeddings are learned vector representations
   - Transformers refine these representations
   - Classification head maps to labels

2. **Training = Minimizing Loss**
   - Loss measures how wrong predictions are
   - Gradients show which direction to adjust weights
   - Small steps (learning rate) prevent overshooting

3. **Hyperparameters Matter**
   - Learning rate: too high = unstable, too low = slow
   - Batch size: larger = faster but needs more memory
   - Epochs: more training, but risk overfitting

4. **Pre-training is Powerful**
   - Starting from DistilBERT beats starting from scratch
   - Transfer learning works!
   - Fine-tuning adapts general knowledge to specific tasks

### Common Issues and Solutions

**Problem: Out of memory**
```python
# Solution: Reduce batch size
per_device_train_batch_size=8  # Instead of 16

# Or use gradient accumulation
gradient_accumulation_steps=2  # Effective batch size = 8 * 2 = 16
```

**Problem: Model not learning (loss not decreasing)**
```python
# Check learning rate - might be too small
learning_rate=5e-5  # Try higher

# Check data - are labels correct?
print(dataset['train'][0])
```

**Problem: Overfitting (train accuracy high, test accuracy low)**
```python
# Solutions:
weight_decay=0.01  # More regularization
num_train_epochs=2  # Train less
# Or get more training data
```

---

## Part 13: Next Steps

### Immediate Challenges

1. **Try a different dataset**: AG News (topic classification), SST-2 (shorter reviews)
2. **Multi-class classification**: 3+ categories instead of binary
3. **Imbalanced data**: What if 90% positive, 10% negative?
4. **Error analysis**: Which examples does the model get wrong? Why?

### Connect to Project 3

Now you understand:
- How embeddings work (fixed-size vectors for tokens)
- How the model learns (gradient descent)
- How to evaluate success (metrics)

**Next: Attention Mechanism**
- You'll see HOW the transformer refines embeddings
- Why it can understand "bank" differently in "river bank" vs "money bank"
- The magic that makes transformers work

### Real-World Applications

You can now:
- Fine-tune models for your own classification tasks
- Understand training logs and metrics
- Debug common issues
- Experiment systematically

---

## Resources

- **HuggingFace Course**: https://huggingface.co/course
- **DistilBERT Paper**: "DistilBERT, a distilled version of BERT"
- **Fine-tuning Guide**: https://huggingface.co/docs/transformers/training
- **GPU Memory Guide**: https://huggingface.co/docs/transformers/perf_train_gpu_one

---

## Your Assignment

1. **Run the complete training** on small dataset (1000 examples)
2. **Test on 10 custom examples** you write yourself
3. **Try one experiment** (data size, learning rate, or freezing)
4. **Answer**: Why does the model need 2-3 epochs? What happens after 1 epoch vs 3?

When ready, we'll move to Project 3: Implementing attention from scratch!