# Building a Tokenizer from Scratch
## Understanding How LLMs See Text

> **Goal**: Build a Byte Pair Encoding (BPE) tokenizer step-by-step, learning the NumPy you need along the way.

---

## Part 1: The Problem - Why Tokenizers Exist

### What happens when you type "Hello World" to ChatGPT?

The model doesn't see letters. It sees **numbers**. Here's the transformation:

```
"Hello World" → [15496, 2159] → Model processes → [1374, 6029] → "Hi there"
```

**Why not just use character-level?**
- "antidisestablishmentarianism" = 28 characters
- With characters, the model processes 28 steps
- With tokens, it processes ~6 steps ("anti", "dis", "establish", "ment", "arian", "ism")

**Why not use word-level?**
- "cat", "cats", "catlike", "catfish" would be 4 separate vocabulary items
- New words = unknown tokens
- Vocabulary explodes (millions of words)

**BPE solves this**: Learns common subwords automatically from data.

---

## Part 2: NumPy Crash Course (Just What We Need)

### Installing
```bash
pip install numpy --break-system-packages
```

### Core Concepts for This Project

#### 1. Arrays (Lists on Steroids)
```python
import numpy as np

# Regular Python list
py_list = [1, 2, 3, 4, 5]

# NumPy array - faster, more features
np_array = np.array([1, 2, 3, 4, 5])

print(np_array)  # [1 2 3 4 5]
print(type(np_array))  # <class 'numpy.ndarray'>
```

#### 2. Why NumPy? Speed!
```python
# Count frequency - Python way
text = "hello world"
freq = {}
for char in text:
    freq[char] = freq.get(char, 0) + 1

# Count frequency - NumPy way (we'll use this)
import numpy as np
chars = np.array(list(text))
unique, counts = np.unique(chars, return_counts=True)
freq_dict = dict(zip(unique, counts))
```

#### 3. Array Operations We'll Use

```python
# Creating arrays
arr = np.array([1, 2, 3])
zeros = np.zeros(5)  # [0. 0. 0. 0. 0.]
ones = np.ones(3)    # [1. 1. 1.]

# Indexing (same as Python lists)
arr[0]      # 1
arr[1:3]    # [2 3]
arr[-1]     # 3

# Finding things
arr = np.array([10, 20, 30, 20, 10])
np.where(arr == 20)  # (array([1, 3]),) - indices where value is 20

# Sorting
np.sort(arr)  # [10 10 20 20 30]
np.argsort(arr)  # [0 4 1 3 2] - indices that would sort the array

# Unique values
np.unique(arr)  # [10 20 30]
```

That's it! You now know enough NumPy for this project.

---

## Part 3: Building the Tokenizer

### Step 1: Understanding BPE Algorithm

**Byte Pair Encoding** is simple:
1. Start with individual characters as tokens
2. Find the most frequent pair of adjacent tokens
3. Merge that pair into a new token
4. Repeat until you have desired vocabulary size

**Example:**
```
Text: "low low low lower lowest"

Initial tokens: ['l', 'o', 'w', ' ', 'l', 'o', 'w', ...]

Step 1: Most frequent pair is ('l', 'o')
→ Merge to 'lo'
→ ['lo', 'w', ' ', 'lo', 'w', ' ', 'lo', 'w', ' ', 'lo', 'w', 'e', 'r', ...]

Step 2: Most frequent pair is ('lo', 'w')
→ Merge to 'low'
→ ['low', ' ', 'low', ' ', 'low', ' ', 'low', 'e', 'r', ...]

Step 3: Most frequent pair is ('low', 'e')
→ Merge to 'lowe'
→ ['low', ' ', 'low', ' ', 'low', ' ', 'lowe', 'r', ...]
```

### Step 2: Starter Code Structure

```python
import numpy as np
from collections import Counter

class SimpleBPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token_id -> token_string
        self.merges = {}  # (token1, token2) -> new_token
        
    def train(self, text):
        """Learn BPE merges from text"""
        pass
    
    def encode(self, text):
        """Convert text to token IDs"""
        pass
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        pass
```

### Step 3: Implementation (Let's Build It!)

#### Step 3.1: Initialize with Characters

```python
class SimpleBPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.char_to_id = {}
        self.id_to_char = {}
        
    def _get_stats(self, tokens):
        """
        Count frequency of adjacent token pairs.
        
        tokens: list of token IDs
        returns: Counter object with (token1, token2) -> frequency
        """
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs
    
    def _merge_pair(self, tokens, pair, new_token_id):
        """
        Replace all occurrences of pair with new_token_id.
        
        tokens: [1, 2, 3, 2, 3, 4]
        pair: (2, 3)
        new_token_id: 5
        returns: [1, 5, 5, 4]
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            # Check if we found the pair
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_token_id)
                i += 2  # Skip both tokens in the pair
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
```

#### Step 3.2: Training Loop

```python
    def train(self, text, verbose=True):
        """
        Learn BPE vocabulary from text.
        """
        # Step 1: Convert text to character IDs
        # Get unique characters
        unique_chars = sorted(set(text))
        
        # Create character <-> ID mappings
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        
        # Convert text to token IDs (start with characters)
        tokens = [self.char_to_id[ch] for ch in text]
        
        # Initialize vocab with characters
        self.vocab = self.id_to_char.copy()
        
        num_merges = self.vocab_size - len(unique_chars)
        
        if verbose:
            print(f"Starting vocabulary: {len(unique_chars)} characters")
            print(f"Will perform {num_merges} merges to reach {self.vocab_size} tokens")
            print(f"Text length: {len(text)} characters → {len(tokens)} initial tokens\n")
        
        # Step 2: Iteratively merge most frequent pairs
        for merge_num in range(num_merges):
            # Find most frequent pair
            stats = self._get_stats(tokens)
            
            if not stats:
                break  # No more pairs to merge
            
            # Get the most common pair
            most_frequent_pair = max(stats, key=stats.get)
            
            # Create new token ID
            new_token_id = len(self.vocab)
            
            # Record the merge
            self.merges[most_frequent_pair] = new_token_id
            
            # Create the merged token string
            token1_str = self.vocab[most_frequent_pair[0]]
            token2_str = self.vocab[most_frequent_pair[1]]
            merged_str = token1_str + token2_str
            
            # Add to vocabulary
            self.vocab[new_token_id] = merged_str
            
            # Merge all occurrences in the token sequence
            tokens = self._merge_pair(tokens, most_frequent_pair, new_token_id)
            
            if verbose and (merge_num < 10 or merge_num % 100 == 0):
                print(f"Merge {merge_num + 1}: {repr(token1_str)} + {repr(token2_str)} "
                      f"= {repr(merged_str)} (occurred {stats[most_frequent_pair]} times)")
                print(f"  Tokens after merge: {len(tokens)}\n")
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Final token count: {len(tokens)} (compressed from {len(text)} characters)")
```

#### Step 3.3: Encoding (Text → Token IDs)

```python
    def encode(self, text):
        """
        Convert text to token IDs using learned merges.
        """
        # Start with character tokens
        tokens = [self.char_to_id.get(ch, 0) for ch in text]  # Use 0 for unknown
        
        # Apply merges in the order they were learned
        for pair, new_id in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_id)
        
        return tokens
```

#### Step 3.4: Decoding (Token IDs → Text)

```python
    def decode(self, token_ids):
        """
        Convert token IDs back to text.
        """
        tokens = [self.vocab.get(tid, '') for tid in token_ids]
        return ''.join(tokens)
```

### Step 4: Complete Working Code

Here's everything together:

```python
import numpy as np
from collections import Counter

class SimpleBPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.char_to_id = {}
        self.id_to_char = {}
        
    def _get_stats(self, tokens):
        """Count frequency of adjacent token pairs."""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs
    
    def _merge_pair(self, tokens, pair, new_token_id):
        """Replace all occurrences of pair with new_token_id."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def train(self, text, verbose=True):
        """Learn BPE vocabulary from text."""
        # Convert text to character IDs
        unique_chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        
        tokens = [self.char_to_id[ch] for ch in text]
        self.vocab = self.id_to_char.copy()
        
        num_merges = self.vocab_size - len(unique_chars)
        
        if verbose:
            print(f"Starting vocabulary: {len(unique_chars)} characters")
            print(f"Will perform {num_merges} merges\n")
        
        # Iteratively merge most frequent pairs
        for merge_num in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break
            
            most_frequent_pair = max(stats, key=stats.get)
            new_token_id = len(self.vocab)
            self.merges[most_frequent_pair] = new_token_id
            
            token1_str = self.vocab[most_frequent_pair[0]]
            token2_str = self.vocab[most_frequent_pair[1]]
            merged_str = token1_str + token2_str
            self.vocab[new_token_id] = merged_str
            
            tokens = self._merge_pair(tokens, most_frequent_pair, new_token_id)
            
            if verbose and (merge_num < 10 or merge_num % 50 == 0):
                print(f"Merge {merge_num + 1}: '{token1_str}' + '{token2_str}' = '{merged_str}'")
        
        print(f"\nTraining complete! Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text):
        """Convert text to token IDs."""
        tokens = [self.char_to_id.get(ch, 0) for ch in text]
        for pair, new_id in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_id)
        return tokens
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        tokens = [self.vocab.get(tid, '') for tid in token_ids]
        return ''.join(tokens)
    
    def show_vocab(self, n=20):
        """Display first n vocabulary items."""
        print(f"\nVocabulary (showing {n} of {len(self.vocab)}):")
        for i, (token_id, token_str) in enumerate(sorted(self.vocab.items())[:n]):
            print(f"  {token_id}: {repr(token_str)}")
```

---

## Part 4: Running Your Tokenizer

### Experiment 1: Simple Text

```python
# Create and train tokenizer
tokenizer = SimpleBPETokenizer(vocab_size=100)

text = "low low low lower lowest"
tokenizer.train(text, verbose=True)

# Test encoding
test_text = "lower"
token_ids = tokenizer.encode(test_text)
print(f"\nEncoding '{test_text}':")
print(f"  Token IDs: {token_ids}")

# Show what each token represents
print(f"  Tokens: {[tokenizer.vocab[tid] for tid in token_ids]}")

# Test decoding
decoded = tokenizer.decode(token_ids)
print(f"  Decoded: '{decoded}'")
print(f"  Match: {decoded == test_text}")
```

### Experiment 2: Real Text (More Interesting!)

```python
# Sample text - let's use something with patterns
text = """
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
Machine learning is amazing. Machine learning is powerful.
Machine learning models learn from data.
""" * 10  # Repeat to see patterns

tokenizer = SimpleBPETokenizer(vocab_size=200)
tokenizer.train(text, verbose=True)

# Show learned vocabulary
tokenizer.show_vocab(30)

# Test compression
test_sentence = "The quick brown fox"
token_ids = tokenizer.encode(test_sentence)
print(f"\nOriginal: '{test_sentence}' ({len(test_sentence)} chars)")
print(f"Tokens: {token_ids} ({len(token_ids)} tokens)")
print(f"Token strings: {[tokenizer.vocab[tid] for tid in token_ids]}")
print(f"Compression ratio: {len(test_sentence) / len(token_ids):.2f}x")
```

---

## Part 5: Understanding What's Happening

### Visualization Exercise

Let's trace through one merge manually:

```python
def visualize_merge(tokenizer, text, merge_index=0):
    """Show exactly what happens during a specific merge."""
    # Start with characters
    tokens = [tokenizer.char_to_id[ch] for ch in text]
    token_strs = [tokenizer.id_to_char[tid] for tid in tokens]
    
    print(f"Initial: {token_strs}")
    
    # Apply merges one by one
    for i, (pair, new_id) in enumerate(tokenizer.merges.items()):
        if i == merge_index:
            print(f"\nApplying merge {i + 1}:")
            print(f"  Looking for pair: {tokenizer.vocab[pair[0]]} + {tokenizer.vocab[pair[1]]}")
            print(f"  Will create: {tokenizer.vocab[new_id]}")
            
        tokens = tokenizer._merge_pair(tokens, pair, new_id)
        token_strs = [tokenizer.vocab[tid] for tid in tokens]
        
        if i == merge_index:
            print(f"  After merge: {token_strs}")
            break

# Usage
tokenizer = SimpleBPETokenizer(vocab_size=50)
tokenizer.train("hello hello world", verbose=False)
visualize_merge(tokenizer, "hello hello world", merge_index=0)
visualize_merge(tokenizer, "hello hello world", merge_index=1)
```

---

## Part 6: Exercises to Deepen Understanding

### Exercise 1: Analyze Vocabulary
```python
def analyze_vocab(tokenizer):
    """What kinds of tokens did we learn?"""
    single_char = [t for t in tokenizer.vocab.values() if len(t) == 1]
    two_char = [t for t in tokenizer.vocab.values() if len(t) == 2]
    long_tokens = [t for t in tokenizer.vocab.values() if len(t) >= 5]
    
    print(f"Single characters: {len(single_char)}")
    print(f"Two characters: {len(two_char)}")
    print(f"Long tokens (5+ chars): {len(long_tokens)}")
    print(f"\nLongest tokens:")
    for token in sorted(long_tokens, key=len, reverse=True)[:10]:
        print(f"  '{token}' ({len(token)} chars)")

# Try it!
analyze_vocab(tokenizer)
```

### Exercise 2: Compression Efficiency
```python
def compare_compression(tokenizer, texts):
    """How well does our tokenizer compress different texts?"""
    for text in texts:
        token_ids = tokenizer.encode(text)
        ratio = len(text) / len(token_ids)
        print(f"Text: '{text[:50]}...'")
        print(f"  Chars: {len(text)}, Tokens: {len(token_ids)}, Ratio: {ratio:.2f}x\n")

test_texts = [
    "The quick brown fox jumps",  # Similar to training
    "Machine learning is amazing",  # Similar to training
    "Completely different unusual text xyz",  # Different
]
compare_compression(tokenizer, test_texts)
```

### Exercise 3: Unknown Words
```python
# What happens with words not in training?
new_words = ["supercalifragilisticexpialidocious", "antidisestablishmentarianism"]
for word in new_words:
    tokens = tokenizer.encode(word)
    print(f"'{word}' ({len(word)} chars) → {len(tokens)} tokens")
    print(f"  Tokens: {[tokenizer.vocab.get(t, '?') for t in tokens]}\n")
```

---

## Part 7: Compare to Real Tokenizers

Now let's see how our simple version compares to GPT's tokenizer:

```python
# Install tiktoken (OpenAI's tokenizer)
# pip install tiktoken --break-system-packages

import tiktoken

# Load GPT-4 tokenizer
gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")

test_text = "The quick brown fox jumps over the lazy dog"

# Our tokenizer
our_tokens = tokenizer.encode(test_text)
print(f"Our tokenizer: {len(our_tokens)} tokens")
print(f"  {[tokenizer.vocab[t] for t in our_tokens]}")

# GPT-4 tokenizer
gpt4_tokens = gpt4_tokenizer.encode(test_text)
print(f"\nGPT-4 tokenizer: {len(gpt4_tokens)} tokens")
print(f"  {[gpt4_tokenizer.decode([t]) for t in gpt4_tokens]}")
```

---

## Part 8: Key Takeaways

### What You Now Understand

1. **Tokenization is learned, not programmed**
   - The algorithm discovers common patterns automatically
   - No manual rules needed

2. **Trade-off between vocabulary size and sequence length**
   - Larger vocab = more tokens = shorter sequences
   - Smaller vocab = fewer tokens = longer sequences

3. **Compression varies by domain**
   - Text similar to training → good compression
   - Novel text → poor compression

4. **Why GPT uses ~50k-100k vocab**
   - Balance between coverage and efficiency
   - Our toy example used 100-200 to keep it simple

### What's Different in Production Tokenizers

1. **Byte-level BPE**: Operates on UTF-8 bytes, not characters
   - Handles any language/emoji
   - Never gets "unknown" characters

2. **Special tokens**: `<|endoftext|>`, `<|startoftext|>`, etc.

3. **Regex pre-tokenization**: Split on whitespace/punctuation first

4. **Faster algorithms**: Our `O(n²)` merge is slow; production uses `O(n log n)`

---

## Part 9: Next Steps

### Immediate Challenges

1. **Add special tokens**: `<BOS>`, `<EOS>`, `<UNK>`
2. **Handle unknown characters**: What if someone types emoji during encoding?
3. **Save/load tokenizer**: Pickle the vocab and merges
4. **Visualize token boundaries**: Show where words get split

### Connect to LLMs

Now you understand:
- Why "SolidGoldMagikarp" was a weird GPT-3 glitch (rare token)
- Why prompts have token limits (e.g., "4096 tokens" not "4096 words")
- Why some languages cost more API credits (more tokens per word)
- Why you can't just "add words" to GPT's vocabulary easily

### Prepare for Next Project

With tokenization understood, you're ready for:
- **Project 2**: Fine-tuning a model (you'll see how tokens → embeddings)
- **Project 3**: Attention mechanism (you'll see how tokens relate to each other)

---

## Resources

- **Original BPE Paper**: "Neural Machine Translation of Rare Words with Subword Units"
- **OpenAI's tiktoken**: https://github.com/openai/tiktoken
- **HuggingFace Tokenizers**: https://huggingface.co/docs/tokenizers
- **Andrej Karpathy's minbpe**: https://github.com/karpathy/minbpe (similar to what we built!)

---

## Your Assignment

1. **Run the complete code** on different texts
2. **Experiment** with vocabulary sizes (50, 100, 500)
3. **Try different texts**: code, poetry, foreign language
4. **Answer**: Why does GPT-4 tokenize "egg" as one token but "EGG" as "E", "GG"?

When you're comfortable with this, we'll move to Project 2: Fine-tuning a small model!