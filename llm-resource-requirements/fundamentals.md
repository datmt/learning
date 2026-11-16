# 1. What is an LLM, really?

Imagine an LLM as a giant mathematical machine that predicts the next piece of text.

You give it:

> The capital of France is ...

It calculates which next token is most likely, perhaps ` Paris`, outputs it, then repeats the process.

Inside the model are billions of numbers called parameters. These numbers are learned during training and determine how the model responds.

Think of the parameters as the settings of a gigantic mathematical machine. They are not a database of answers; they are numerical values used to perform computations.

Your prompt

"The capital of France is ..."

The model

Billions of learned numerical parameters

Matrix calculations + attention + other operations

Next token

## Paris

This leads to the first important idea: the computer must store the model's parameters in memory and perform calculations using them.

That is why model size, memory capacity, memory bandwidth, and computing power matter.

# 2. What does "27B model" mean?

The `B` means billion.

A 27B model has approximately 27 billion parameters. A 4B model has approximately 4 billion.

| Model size | Number of parameters |
| ---------- | -------------------- |
| 4B         | 4 billion            |
| 12B        | 12 billion           |
| 27B        | 27 billion           |
| 70B        | 70 billion           |

All else being equal, more parameters generally mean more memory is needed to store the model.

But here's the catch: each parameter is a number, and computers can represent numbers using different amounts of memory.

That brings us to the term you asked about.

# 3. What does "bits per weight" mean?

Let's start with the word bit.

A bit is the smallest unit of digital information. It can hold one of two values: `0` or `1`.

- 8 bits = 1 byte.
- 1,000 bytes is approximately 1 kilobyte (KB), although computer memory is often measured in binary units.
- 1 billion bytes is approximately 1 GB.

A model parameter is a number. The computer needs bits to represent that number.

Imagine storing the parameters of a 27-billion-parameter model. If every parameter occupies 16 bits, then:

\\[ 27\text{ billion}\times16\text{ bits} =432\text{ billion bits} \\]

Divide by 8 to convert bits to bytes:

\\[ 432/8=54\text{ billion bytes} \\]

That's approximately 54 GB of parameter storage, before accounting for additional runtime requirements.

Now imagine representing each parameter using only 4 bits.

\\[ 27\text{ billion}\times4/8 =13.5\text{ GB} \\]

## Same model, different storage requirements

Illustration for 27 billion parameters, before overhead.

16 bits per parameter

### 54 GB

8 bits per parameter

### 27 GB

4 bits per parameter

### 13.5 GB

This is the fundamental reason quantized models can run on smaller computers.

## 4. What is quantization?

Quantization means storing model parameters using a representation that takes less memory.

Consider a parameter whose original value is:

`0.73821`

If we use a lower-precision representation, we might store an approximation such as:

`0.75`

The actual process is more sophisticated than rounding a decimal, but the principle is the same: we use fewer bits to represent the values.

Why do this? Because loading and storing fewer bytes can make an enormous model practical on hardware that otherwise couldn't accommodate it.

There is a trade-off, though:

- More bits generally preserve more numerical detail.
- Fewer bits reduce memory requirements.
- Quantization can reduce accuracy or change model outputs.
- The actual speed improvement depends on the hardware and implementation.

You may see model names such as `Qwen... Q4_K_M` or `Qwen... Q8_0`. These are quantization formats, not different numbers of parameters.

For example, Q4 generally indicates a four-bit quantization family. The suffix specifies additional details about how the quantization is implemented.

Important: a model labeled "4-bit" does not necessarily average exactly 4 bits per parameter. Real formats store additional scaling values and metadata, and some tensors may use different precisions.

That's why the actual model file size is more useful than relying entirely on the label.

# 5. Try calculating the memory yourself

## LLM weight-memory calculator

Model parameters

### 27 billion

1B

27B

70B

100B

Bits per parameter

16-bit (higher precision)

8-bit

4-bit (smaller model footprint)

2-bit (more aggressive compression)

Estimated weight storage

# 6.75 GB

Decimal GB, assuming exactly 2 bits per parameter and no overhead.

This is a simplified estimate. Real quantized files may be larger, and model loading and inference require additional memory.

Try 27 billion parameters with 4 bits, then change it to 16 bits. Notice how much the memory requirement changes without changing the model's parameter count.

# 6. How does this connect to your DGX Spark?

Your DGX Spark has 128 GB of unified memory. That memory is shared between the CPU and GPU rather than being a separate pool of conventional GPU VRAM.

Suppose your 27B model uses 4-bit quantization.

- The raw weight estimate is 13.5 GB.
- The real model file may be larger.
- The runtime needs additional memory for intermediate calculations and other buffers.
- The model needs a KV cache to retain information about the current conversation.
- The operating system and other processes also consume memory.

This means a model requiring approximately 13.5 GB for raw weights does not necessarily require only 13.5 GB of system memory.

And there's a second, equally important point: fitting the model into memory doesn't mean it will generate tokens quickly.

That brings us to the next concept in the benchmark.

# 7. Two different bottlenecks: capacity versus bandwidth

Think of your computer's memory as a warehouse.

- Memory capacity is how much inventory the warehouse can hold.
- Memory bandwidth is how quickly inventory can be moved out of the warehouse.
- Compute is the machinery that processes the inventory.

Your DGX Spark's approximately 273 GB/s memory bandwidth describes a rate of data transfer, not how much memory it has.

A large model might fit comfortably in the warehouse but still generate slowly because the machinery needs to read enormous amounts of data for every token.

This is why your benchmark measures both memory usage and tokens per second.

There is one more concept that will make the benchmark much easier to understand: the KV cache. It is the memory the model uses to retain information about previously processed tokens, and it can grow significantly as the context becomes longer.

My recommended learning order is:

1. Parameters and bits per weight — what we covered here.
2. Quantization — why Q4 and Q8 models differ.
3. KV cache — why long conversations consume additional memory.
4. Prefill versus decode — why processing a prompt differs from generating tokens.
5. Memory bandwidth — why a model can fit but still run slowly.
6. Concurrency — why serving multiple users changes throughput and latency.

Once you understand these six concepts, most of the terminology in the benchmark will become much more intuitive.
