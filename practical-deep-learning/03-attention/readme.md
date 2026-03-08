# Building Attention from Scratch
## The Core Mechanism Behind Transformers

> **Goal**: Implement single-head and multi-head attention in NumPy, understanding why it's the breakthrough that powers GPT, BERT, and all modern LLMs.

---

## Part 1: The Problem Attention Solves

### Why Old Models Failed

Before attention, models processed text sequentially (like reading word by word):

```
Sentence: "The cat sat on the mat"

RNN/LSTM approach:
Step 1: Process "The" → hidden state h1
Step 2: Process "cat" using h1 → hidden state h2
Step 3: Process "sat" using h2 → hidden state h3
...

Problem: By step 6, the model has "forgotten" details about "The cat"
```

**The bottleneck**: All information must flow through a single hidden state vector.

### What Attention Does Differently

**Attention lets every word "look at" every other word simultaneously.**

```
When processing "sat":
- Look at "cat" (who is sitting?)
- Look at "mat" (where are they sitting?)
- Ignore "the" (not important for this word)

When processing "mat":
- Look at "sat" and "on" (the mat is a location)
- Look at "cat" (what's on the mat?)
```

**Key insight**: The model learns WHERE to look, based on the task.

---

## Part 2: Attention in One Equation

### The Core Formula

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Don't panic! Let's break this down step by step.

### The Analogy: A Library Search

**You walk into a library looking for books about "machine learning"**

1. **Query (Q)**: What you're looking for
   - "I want books about machine learning"
   
2. **Keys (K)**: Book titles/descriptions
   - Book 1: "Introduction to Neural Networks"
   - Book 2: "Advanced Calculus"
   - Book 3: "Deep Learning Fundamentals"
   
3. **Values (V)**: The actual book content
   - The full text of each book

**How you find relevant books:**
1. Compare your query to each book's description (Q·K^T)
2. Rank books by relevance (softmax - highest scores win)
3. Read the relevant books (weighted sum of V)

**Result**: You get information from books most relevant to your query, automatically ignoring irrelevant books.

### In Transformers

When processing the word "sat" in "The cat sat on the mat":

- **Query**: "sat" asking "what should I pay attention to?"
- **Keys**: All words saying "here's what I'm about"
- **Values**: The actual meaning/information from each word

The model learns to:
- Give high score to "cat" (who is sitting?)
- Give high score to "mat" (where?)
- Give low score to "the" (just a filler word)

---

## Part 3: Building Blocks

### Vectors and Matrices

Every word becomes a vector (list of numbers):

```python
import numpy as np

# Simple example: 4-dimensional embeddings
word_embeddings = {
    "the": np.array([0.1, 0.2, 0.3, 0.4]),
    "cat": np.array([0.5, 0.6, 0.7, 0.8]),
    "sat": np.array([0.2, 0.3, 0.4, 0.5])
}

# In a real model (like DistilBERT), these would be 768 dimensions!
```

### Matrix Multiplication (Dot Product)

This is how we "compare" vectors:

```python
# Two vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# Dot product: multiply element-wise, then sum
dot_product = np.dot(vec1, vec2)
# = (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32

print(f"Dot product: {dot_product}")

# High value = vectors are similar/aligned
# Low value = vectors are different
```

**Why this matters**: 
- High dot product = words are related
- Low dot product = words are unrelated

```python
# Example with word vectors
query = np.array([1.0, 0.0])  # "What's an animal?"

key_cat = np.array([0.9, 0.1])   # "cat" (similar to query)
key_table = np.array([0.1, 0.9])  # "table" (different from query)

similarity_cat = np.dot(query, key_cat)      # 0.9 (high!)
similarity_table = np.dot(query, key_table)  # 0.1 (low!)
```

### Softmax: Converting Scores to Probabilities

```python
def softmax(x):
    """Convert scores to probabilities that sum to 1."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example: attention scores for three words
scores = np.array([2.0, 1.0, 0.1])

probabilities = softmax(scores)
print(f"Scores: {scores}")
print(f"Probabilities: {probabilities}")
print(f"Sum: {probabilities.sum()}")

# Output:
# Scores: [2.  1.  0.1]
# Probabilities: [0.659 0.242 0.099]  # Highest score gets highest probability
# Sum: 1.0
```

**Why softmax?**
- Converts any numbers to probabilities (0 to 1, sum to 1)
- Emphasizes differences (high scores get even higher probability)
- Makes attention weights interpretable

---

## Part 4: Implementing Single-Head Attention

### Step-by-Step Implementation

```python
import numpy as np

class SingleHeadAttention:
    """
    Simplified attention mechanism.
    
    Given input embeddings, compute attention-weighted outputs.
    """
    
    def __init__(self, d_model, d_k):
        """
        Args:
            d_model: Dimension of input embeddings (e.g., 512)
            d_k: Dimension of query/key vectors (e.g., 64)
        """
        self.d_model = d_model
        self.d_k = d_k
        
        # Initialize weight matrices (randomly for now)
        # In real models, these are learned during training
        self.W_q = np.random.randn(d_model, d_k) * 0.1  # Query weights
        self.W_k = np.random.randn(d_model, d_k) * 0.1  # Key weights
        self.W_v = np.random.randn(d_model, d_k) * 0.1  # Value weights
    
    def forward(self, X, mask=None, return_attention=False):
        """
        Compute attention.
        
        Args:
            X: Input matrix [seq_len, d_model]
               Each row is an embedding for one token
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output matrix [seq_len, d_k]
            (Optionally) Attention weights [seq_len, seq_len]
        """
        # Step 1: Create Q, K, V by projecting input through weight matrices
        Q = X @ self.W_q  # [seq_len, d_k]
        K = X @ self.W_k  # [seq_len, d_k]
        V = X @ self.W_v  # [seq_len, d_k]
        
        # Step 2: Compute attention scores (how much each word attends to others)
        # Q·K^T gives us [seq_len, seq_len] matrix of similarities
        scores = Q @ K.T  # [seq_len, seq_len]
        
        # Step 3: Scale scores (prevents gradients from exploding)
        scores = scores / np.sqrt(self.d_k)
        
        # Step 4: Apply mask if provided (for padding or future tokens)
        if mask is not None:
            scores = scores + (mask * -1e9)  # Large negative = near zero after softmax
        
        # Step 5: Apply softmax to get attention weights
        # Each row sums to 1 (probability distribution over all tokens)
        attention_weights = self.softmax(scores)  # [seq_len, seq_len]
        
        # Step 6: Weighted sum of values
        # Each output is a weighted combination of all value vectors
        output = attention_weights @ V  # [seq_len, d_k]
        
        if return_attention:
            return output, attention_weights
        return output
    
    @staticmethod
    def softmax(x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Let's test it!
def test_single_head_attention():
    """Demonstrate attention on a simple sequence."""
    
    print("="*70)
    print("SINGLE-HEAD ATTENTION DEMO")
    print("="*70)
    
    # Simple example: 4 words, 8-dimensional embeddings
    seq_len = 4
    d_model = 8
    d_k = 4
    
    # Create random embeddings (in reality, these come from an embedding layer)
    X = np.random.randn(seq_len, d_model)
    
    print(f"\nInput shape: {X.shape}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {d_model}")
    
    # Create attention layer
    attention = SingleHeadAttention(d_model, d_k)
    
    # Forward pass
    output, attn_weights = attention.forward(X, return_attention=True)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  Each token now has a {d_k}-dimensional representation")
    
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"  {seq_len}x{seq_len} matrix")
    print(f"  Row i shows how much token i attends to each token")
    
    print(f"\nAttention weights:")
    print(attn_weights)
    
    print(f"\nRow sums (should all be 1.0):")
    print(attn_weights.sum(axis=1))
    
    print("="*70 + "\n")

test_single_head_attention()
```

### Understanding the Output

When you run this, you'll see something like:

```
Attention weights:
[[0.31 0.19 0.28 0.22]   ← Token 0 attends to all tokens
 [0.27 0.25 0.24 0.24]   ← Token 1 attends fairly evenly
 [0.18 0.33 0.29 0.20]   ← Token 2 attends most to token 1
 [0.25 0.23 0.19 0.33]]  ← Token 3 attends most to itself
```

**Each row**: How much that token looks at every token (including itself)
**High value**: Strong attention (this token is important for understanding)
**Low value**: Weak attention (this token is less relevant)

---

## Part 5: Visualizing Attention

### Create Attention Heatmap

```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens):
    """
    Create a heatmap showing which tokens attend to which.
    
    Args:
        attention_weights: [seq_len, seq_len] matrix
        tokens: List of token strings
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues')
    
    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values in cells
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title("Attention Weights: Which tokens attend to which?", fontsize=14, pad=20)
    ax.set_xlabel("Key (attending TO)", fontsize=12)
    ax.set_ylabel("Query (attending FROM)", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig

# Example usage
tokens = ["The", "cat", "sat", "down"]
seq_len = len(tokens)
d_model = 8
d_k = 4

# Create embeddings
X = np.random.randn(seq_len, d_model)

# Compute attention
attention = SingleHeadAttention(d_model, d_k)
output, attn_weights = attention.forward(X, return_attention=True)

# Visualize
fig = visualize_attention(attn_weights, tokens)
plt.savefig('attention_heatmap.png', dpi=300, bbox_inches='tight')
print("Attention heatmap saved to attention_heatmap.png")
plt.close()
```

---

## Part 6: Real Example with Meaningful Embeddings

### Using Pre-computed Word Embeddings

```python
def create_simple_embeddings(tokens, d_model=8):
    """
    Create simple but meaningful embeddings.
    Words with similar meanings get similar vectors.
    """
    # Simple hand-crafted embeddings
    embedding_dict = {
        # Dimension meanings: [animal, object, action, location, ...]
        "the": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # article
        "cat": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]),  # animal, living
        "dog": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]),  # animal, living
        "sat": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0]),  # action, position
        "chased": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9]),  # action, movement
        "on": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),  # location/preposition
        "mat": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # object
        "bone": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # object
    }
    
    # Get embeddings for tokens
    embeddings = np.array([embedding_dict.get(token.lower(), np.random.randn(d_model) * 0.1) 
                          for token in tokens])
    
    return embeddings


def analyze_attention_patterns():
    """Show how attention captures relationships."""
    
    print("="*70)
    print("ANALYZING ATTENTION PATTERNS")
    print("="*70)
    
    # Two sentences to compare
    sentence1 = ["The", "cat", "sat", "on", "the", "mat"]
    sentence2 = ["The", "dog", "chased", "the", "cat"]
    
    for sentence in [sentence1, sentence2]:
        print(f"\nSentence: {' '.join(sentence)}")
        
        # Create embeddings
        X = create_simple_embeddings(sentence)
        
        # Compute attention
        attention = SingleHeadAttention(d_model=8, d_k=4)
        output, attn_weights = attention.forward(X, return_attention=True)
        
        # Find which word each token attends to most
        print("\nAttention patterns:")
        for i, token in enumerate(sentence):
            max_attn_idx = np.argmax(attn_weights[i])
            max_attn_weight = attn_weights[i, max_attn_idx]
            
            print(f"  '{token}' attends most to '{sentence[max_attn_idx]}' "
                  f"(weight: {max_attn_weight:.3f})")
        
        # Visualize
        fig = visualize_attention(attn_weights, sentence)
        filename = f"attention_{'_'.join(sentence)}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {filename}")
        plt.close()
        
        print("-"*70)

analyze_attention_patterns()
```

---

## Part 7: Multi-Head Attention

### Why Multiple Heads?

Single-head attention can only capture ONE type of relationship.

**Example**: In "The cat sat on the mat"
- Head 1 might learn: Subject-Verb relationships (cat → sat)
- Head 2 might learn: Verb-Location relationships (sat → on, mat)
- Head 3 might learn: Determiner-Noun relationships (the → cat, the → mat)

**Multi-head = Different perspectives simultaneously**

### Implementation

```python
class MultiHeadAttention:
    """
    Multiple attention heads working in parallel.
    Each head learns different relationships.
    """
    
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: Total embedding dimension (must be divisible by num_heads)
            num_heads: Number of parallel attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Create separate weight matrices for each head
        self.heads = [
            SingleHeadAttention(d_model, self.d_k) 
            for _ in range(num_heads)
        ]
        
        # Output projection to combine heads
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, X, mask=None, return_attention=False):
        """
        Compute multi-head attention.
        
        Args:
            X: Input [seq_len, d_model]
            
        Returns:
            Output [seq_len, d_model]
            (Optionally) List of attention weights from each head
        """
        seq_len = X.shape[0]
        
        # Run each head in parallel
        head_outputs = []
        head_attentions = []
        
        for head in self.heads:
            if return_attention:
                output, attn = head.forward(X, mask, return_attention=True)
                head_outputs.append(output)
                head_attentions.append(attn)
            else:
                output = head.forward(X, mask, return_attention=False)
                head_outputs.append(output)
        
        # Concatenate all head outputs
        # Each head output: [seq_len, d_k]
        # Concatenated: [seq_len, num_heads * d_k] = [seq_len, d_model]
        concatenated = np.concatenate(head_outputs, axis=-1)
        
        # Final linear projection
        output = concatenated @ self.W_o
        
        if return_attention:
            return output, head_attentions
        return output


def test_multihead_attention():
    """Demonstrate multi-head attention."""
    
    print("="*70)
    print("MULTI-HEAD ATTENTION DEMO")
    print("="*70)
    
    # Example sentence
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    X = create_simple_embeddings(tokens)
    
    # Create multi-head attention
    num_heads = 4
    d_model = 8
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    print(f"\nConfiguration:")
    print(f"  Number of heads: {num_heads}")
    print(f"  Model dimension: {d_model}")
    print(f"  Dimension per head: {d_model // num_heads}")
    
    # Forward pass
    output, head_attentions = mha.forward(X, return_attention=True)
    
    print(f"\nOutput shape: {output.shape}")
    
    # Visualize each head
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (ax, attn) in enumerate(zip(axes, head_attentions)):
        im = ax.imshow(attn, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)
        ax.set_title(f'Head {i+1}', fontsize=12)
        
        # Add values
        for row in range(len(tokens)):
            for col in range(len(tokens)):
                ax.text(col, row, f'{attn[row, col]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('multihead_attention.png', dpi=300, bbox_inches='tight')
    print(f"\nMulti-head visualization saved to multihead_attention.png")
    plt.close()
    
    print("="*70 + "\n")

test_multihead_attention()
```

---

## Part 8: Self-Attention vs Cross-Attention

### Self-Attention (What We've Been Doing)

**Q, K, V all come from the SAME sequence**

```
Sentence: "The cat sat"

Query: "sat" looks at Keys: ["The", "cat", "sat"]
→ Attends to "cat" (who sat?) and itself
```

Used in: Encoders (BERT), Decoders (GPT)

### Cross-Attention (Encoder-Decoder)

**Q comes from one sequence, K and V from another**

```
Encoder output: "Le chat est noir" (French)
Decoder generating: "The cat is..."

Query: "is" (decoder) looks at Keys: ["Le", "chat", "est", "noir"] (encoder)
→ Attends to "est" (French for "is")
```

Used in: Translation, image captioning, etc.

### Implementation

```python
class CrossAttention(SingleHeadAttention):
    """Cross-attention: Q from one sequence, K/V from another."""
    
    def forward(self, Q_input, KV_input, mask=None, return_attention=False):
        """
        Args:
            Q_input: Query sequence [seq_len_q, d_model]
            KV_input: Key/Value sequence [seq_len_kv, d_model]
        """
        # Create Q from first sequence
        Q = Q_input @ self.W_q
        
        # Create K, V from second sequence
        K = KV_input @ self.W_k
        V = KV_input @ self.W_v
        
        # Rest is same as self-attention
        scores = Q @ K.T / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        attention_weights = self.softmax(scores)
        output = attention_weights @ V
        
        if return_attention:
            return output, attention_weights
        return output
```

---

## Part 9: Masked Attention (GPT-style)

### The Causal Mask

**Problem**: When generating text, we can't look at future tokens!

```
Generating: "The cat sat on the"
Current token: "sat"

We can see: "The cat sat" ✓
We CANNOT see: "on the" ✗ (hasn't been generated yet!)
```

**Solution**: Mask out future positions

```python
def create_causal_mask(seq_len):
    """
    Create mask for autoregressive generation.
    
    Returns upper triangular matrix of -inf (will be near 0 after softmax)
    """
    # Lower triangular = can attend
    # Upper triangular = cannot attend (set to -inf)
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    return mask


def demonstrate_causal_masking():
    """Show how causal masking works."""
    
    print("="*70)
    print("CAUSAL MASKING (GPT-STYLE)")
    print("="*70)
    
    tokens = ["The", "cat", "sat", "down"]
    X = create_simple_embeddings(tokens)
    
    # Create causal mask
    mask = create_causal_mask(len(tokens))
    
    print("\nCausal mask:")
    print(mask)
    print("\n0 = can attend, -inf = cannot attend (future)")
    
    # Attention without mask
    attention_no_mask = SingleHeadAttention(d_model=8, d_k=4)
    _, attn_no_mask = attention_no_mask.forward(X, return_attention=True)
    
    # Attention with mask
    attention_masked = SingleHeadAttention(d_model=8, d_k=4)
    _, attn_masked = attention_masked.forward(X, mask=mask, return_attention=True)
    
    # Visualize both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Without mask
    im1 = ax1.imshow(attn_no_mask, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title('Without Mask (Bidirectional)', fontsize=14)
    ax1.set_xticks(np.arange(len(tokens)))
    ax1.set_yticks(np.arange(len(tokens)))
    ax1.set_xticklabels(tokens)
    ax1.set_yticklabels(tokens)
    
    # With mask
    im2 = ax2.imshow(attn_masked, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('With Causal Mask (Autoregressive)', fontsize=14)
    ax2.set_xticks(np.arange(len(tokens)))
    ax2.set_yticks(np.arange(len(tokens)))
    ax2.set_xticklabels(tokens)
    ax2.set_yticklabels(tokens)
    
    # Add values
    for ax, attn in [(ax1, attn_no_mask), (ax2, attn_masked)]:
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f'{attn[i, j]:.2f}',
                       ha="center", va="center", 
                       color="black" if attn[i, j] > 0.3 else "gray",
                       fontsize=10)
    
    plt.tight_layout()
    plt.savefig('causal_masking.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to causal_masking.png")
    print("\nNotice: With mask, each token only attends to current and previous tokens!")
    plt.close()
    
    print("="*70 + "\n")

demonstrate_causal_masking()
```

---

## Part 10: Putting It All Together - Mini Transformer Layer

```python
class TransformerLayer:
    """
    Single transformer layer = Multi-head attention + Feed-forward + Residuals
    """
    
    def __init__(self, d_model, num_heads, d_ff=None):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension (default: 4 * d_model)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or 4 * d_model
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network (2 layers)
        self.W1 = np.random.randn(d_model, self.d_ff) * 0.1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, X, mask=None):
        """
        Forward pass through transformer layer.
        
        Args:
            X: Input [seq_len, d_model]
            
        Returns:
            Output [seq_len, d_model]
        """
        # Multi-head attention with residual connection
        attn_output = self.attention.forward(X, mask)
        X = X + attn_output  # Residual connection
        X = self.layer_norm(X)  # Layer normalization
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(X)
        X = X + ff_output  # Residual connection
        X = self.layer_norm(X)  # Layer normalization
        
        return X
    
    def feed_forward(self, X):
        """Two-layer feed-forward network with ReLU."""
        hidden = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        output = hidden @ self.W2 + self.b2
        return output
    
    @staticmethod
    def layer_norm(X, eps=1e-6):
        """Layer normalization."""
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        return (X - mean) / (std + eps)


def test_transformer_layer():
    """Test complete transformer layer."""
    
    print("="*70)
    print("COMPLETE TRANSFORMER LAYER")
    print("="*70)
    
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    X = create_simple_embeddings(tokens)
    
    # Create transformer layer
    layer = TransformerLayer(d_model=8, num_heads=2)
    
    print(f"\nInput shape: {X.shape}")
    
    # Forward pass
    output = layer.forward(X)
    
    print(f"Output shape: {output.shape}")
    print(f"\nTransformer layer applied:")
    print(f"  1. Multi-head attention (2 heads)")
    print(f"  2. Add & Norm (residual + layer norm)")
    print(f"  3. Feed-forward network")
    print(f"  4. Add & Norm (residual + layer norm)")
    
    print(f"\nThis is ONE layer. GPT-3 has 96 layers!")
    print(f"Each layer refines the representations further.")
    
    print("="*70 + "\n")

test_transformer_layer()
```

---

## Part 11: Key Insights

### What You Now Understand

1. **Attention = Weighted Retrieval**
   - Query asks "what do I need?"
   - Keys say "here's what I have"
   - Values contain the actual information
   - Attention weights decide how much to use each value

2. **Why It Works**
   - Parallel processing (all tokens simultaneously)
   - Context-dependent (same word, different meanings)
   - Learned patterns (weights adjust during training)

3. **Multi-head = Multiple Perspectives**
   - Different heads learn different relationships
   - Some heads track syntax, others semantics
   - Redundancy helps robustness

4. **Masking = Control Information Flow**
   - Causal mask: For generation (GPT)
   - Padding mask: Ignore padding tokens
   - Custom masks: For specific tasks

### Why This is Revolutionary

**Before attention (RNN/LSTM):**
- Sequential processing (slow)
- Information bottleneck (long sequences forget)
- Limited parallelization

**With attention (Transformers):**
- Parallel processing (fast)
- Direct connections (no forgetting)
- Scales to massive models

---

## Part 12: Exercises

### Exercise 1: Attention Weights Analysis
```python
# Create a sentence with clear relationships
tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
X = create_simple_embeddings(tokens)

attention = SingleHeadAttention(d_model=8, d_k=4)
output, attn_weights = attention.forward(X, return_attention=True)

# Question: Which word does "fox" attend to most?
fox_idx = tokens.index("fox")
max_attn = np.argmax(attn_weights[fox_idx])
print(f"'fox' attends most to: '{tokens[max_attn]}'")

# Visualize
visualize_attention(attn_weights, tokens)
plt.savefig('exercise1.png')
```

### Exercise 2: Compare Head Behaviors
```python
# Create multi-head attention and see if different heads learn different patterns
mha = MultiHeadAttention(d_model=8, num_heads=4)
output, head_attns = mha.forward(X, return_attention=True)

for i, attn in enumerate(head_attns):
    print(f"\nHead {i+1}:")
    # Which token does each word attend to most?
    for j, token in enumerate(tokens):
        max_idx = np.argmax(attn[j])
        print(f"  {token} → {tokens[max_idx]}")
```

### Exercise 3: Implement Positional Encoding
```python
def positional_encoding(seq_len, d_model):
    """
    Add position information to embeddings.
    Transformers need this since attention has no notion of order!
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

# Try adding positional encodings to your embeddings
pos_enc = positional_encoding(len(tokens), 8)
X_with_pos = X + pos_enc
```

---

## Part 13: Connection to Real Models

### GPT Architecture
```
Input tokens
    ↓
Token Embeddings + Positional Encodings
    ↓
[Transformer Layer 1]  ← Multi-head CAUSAL attention
    ↓
[Transformer Layer 2]
    ↓
    ...
    ↓
[Transformer Layer N]
    ↓
Output probabilities (next token)
```

### BERT Architecture
```
Input tokens
    ↓
Token Embeddings + Positional Encodings
    ↓
[Transformer Layer 1]  ← Multi-head BIDIRECTIONAL attention
    ↓
[Transformer Layer 2]
    ↓
    ...
    ↓
[Transformer Layer N]
    ↓
Output representations (for classification, etc.)
```

**Key difference**: BERT can see full context (bidirectional), GPT only sees past (causal)

---

## Part 14: Next Steps

### You're Now Ready For

1. **Project 4 (LoRA Fine-tuning)**
   - You understand what attention is
   - LoRA modifies attention weights efficiently
   
2. **Project 5 (RAG System)**
   - Attention is how models retrieve relevant context
   
3. **Reading Research Papers**
   - "Attention Is All You Need" (original Transformer paper)
   - You can now understand the architecture diagrams!

### Challenge Projects

1. **Implement positional encodings** and see how they affect attention
2. **Build a 2-layer transformer** from scratch
3. **Visualize attention in a real model** (load GPT-2, extract attention weights)
4. **Create custom attention masks** for specific tasks

---

## Resources

- **Original Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Illustrated Transformer**: https://jalammar.github.io/illustrated-transformer/
- **Annotated Transformer**: https://nlp.seas.harvard.edu/annotated-transformer/
- **3Blue1Brown Video**: "Attention in transformers, visually explained"

---

## Your Assignment

1. **Run all the code examples** - see attention in action
2. **Create your own sentence** and visualize attention patterns
3. **Modify the number of heads** - how does it change the patterns?
4. **Answer**: Why does GPT need causal masking but BERT doesn't?

When you're ready, we'll move to Project 4: LoRA fine-tuning (efficient adaptation of these attention weights)!