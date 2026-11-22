I'd build a **small AI systems laboratory**: a collection of experiments where each model exists to teach you a particular system behavior.

Your goal isn't to reproduce the research behind every model. It's to understand how models behave when you put them inside real systems.

Since you already have backend experience and a GB10 DGX, I'd organize the lab around this question:

> If I put a model behind an API, what determines its cost, latency, throughput, reliability, and usefulness?

The lab should gradually become a miniature AI platform, not a collection of disconnected notebooks.

## **1. Build a common lab framework first**

Every experiment should use the same basic interface:

```
from typing import Protocolclass Model(Protocol):    def load(self): ...    def predict(self, request): ...    def unload(self): ...
```

Then build a small benchmark runner that records:

- Model load time
- Memory usage
- Request latency: p50, p95, p99
- Throughput: requests/sec or tokens/sec
- Concurrent requests
- GPU utilization and memory
- Quality / accuracy
- Cost per request or per successful task

For each experiment, write down a hypothesis before running it.

Example:

> Increasing batch size from 1 to 8 will improve throughput, but increase individual request latency.

Then run the experiment and see whether the data supports your hypothesis.

This is the most important part of the lab. **You are learning to reason about systems from measured behavior, not from architecture diagrams alone.**

## **2. Build these six labs in order**

Lab 1 · Start here

## **LLM inference server**

[A survey on Inference Serving for Large Language Models | hello@woonyee.org \~ $](https://images.openai.com/static-rsc-4/mtI2Ym9aLJNhSDss-rjN1nDnpPN-x-172qUsrx18LLKPhGxMMrXKNTuAzSO3i0lEj80kHwGcVJMF1dcDX_0Y6RyW40amlHOHzrwN9fw-t3-zVAd0ZPsV9XTN22k_G4ehLgsgnGcZjlrOFAoEaRQmhB3XPhNuxkagDDIu2bPiYSg?purpose=inline)

**Build:** A small OpenAI-compatible inference API around a local model.

Use Ollama initially, then try a serving engine such as vLLM or SGLang when supported by your hardware and model.

Learn:

- Model weights vs KV cache
- Prefill vs decode
- TTFT vs time per output token
- Batching and concurrency
- Quantization and memory capacity

**Experiments:**

1. Short prompt vs long prompt
2. Short output vs long output
3. One request vs concurrent requests
4. Different quantization levels
5. Different batch sizes

You should be able to explain why throughput changes without guessing.

Lab 2

## **Embedding and retrieval system**

[Embeddings in Practice: A Research & Implementation Guide | by Adnan Masood, PhD. | Jan, 2026 | Medium](https://images.openai.com/static-rsc-4/8CEiCQzZre1Z9ZOwuhbEEzbv_pz8K1mxk45CrPWofUSPFtCpBIUqHB1N4zDSOw5bPdkaFdvrtFgc2GWFcgNwXHbUmL8htZRSU2Dj_xrk9ClViSXj_vFzcpaqdQoi7k1cz5O6WO5NEAt528wLihR-O7kMWpH7ZHwwYlvI0LtiEss?purpose=inline)

**Build:** A semantic search API over a document collection.

Use a small embedding model, PostgreSQL with pgvector, and a simple API.

Learn:

- Embedding generation cost
- Vector dimensions and storage
- Index build time
- ANN search and recall
- Query latency vs retrieval quality

**Experiments:**

- 10K vs 100K vectors
- Exact search vs approximate search
- Different vector dimensions
- Different top-K values
- Batch embedding vs one-at-a-time

This teaches a different workload from LLM generation: the system is often dominated by indexing, memory access, and retrieval behavior.

Lab 3

## **Computer vision pipeline**

[Soccer Tracking - Part 1 - Object Detection and Tracking - jlburkhead](https://images.openai.com/static-rsc-4/WPHnDJOTToMzpNnpI0ayj913NhoHHq39nuW7pQTrcyF3JyTzI4WI5o2eT3hWVyQQgTMnyCdOtQV-3qdqf0Ur_E4IFukSgaLfH5qtUp4KepA7OHqtV83vOha5v2OQpVRo5sZ1euxvR-MtGHOjUn2ZmbWV-H8KibR2FW4hRZxLxvw?purpose=inline)

**Build:** A video analytics pipeline using a pretrained object detector.

Start with YOLO and a short video. Add tracking only after detection works.

Learn:

- Images/sec vs video FPS
- Resolution and compute cost
- GPU preprocessing
- CPU video decoding
- Detection vs tracking
- Real-time vs batch processing

**Experiments:**

- 640px vs 1280px input
- One frame every 1, 2, or 5 frames
- CPU vs GPU inference
- Detection alone vs detection + tracking

This is especially useful for your sports-video ambitions. You'll discover that the model is only one component of the pipeline.

Lab 4

## **Training a tiny neural network**

[neural networks - Validation loss is significantly higher than train loss - Cross Validated](https://images.openai.com/static-rsc-4/qg6hwClFPAQHYdCVSsGc33VZ8IJNgWQr0yJGmRs9BTqBfuunbBeF8ckGmEDVyzy6eWra15tjIwTqKgCLQ8nfo7FUBRHH08Qx2vAHfMkG-ZHKQ1ZEF2FaAwd8_hW4IoEIKjd6TLjs48ZZA7HgXrSnit8KLHNnzDCMCBwV4EAMwdw?purpose=inline)

**Build:** A small image classifier in PyTorch.

Use MNIST or Fashion-MNIST. Don't start with a large model.

Learn:

- Training vs inference
- Batch size
- Epochs and convergence
- Optimizer behavior
- GPU utilization
- Data loading bottlenecks

**Experiments:**

- CPU vs GPU
- Batch size 1 vs 32 vs 128
- Different learning rates
- Synthetic data vs real data loading

The goal isn't to achieve state-of-the-art accuracy. It's to understand what makes a GPU busy, idle, or underutilized.

Lab 5

## **Diffusion image generation service**

[Matt Wolfe (@mreflow) on X](https://images.openai.com/static-rsc-4/l6UedINpUhqeLe4SHZGHGHC5XF4ZfdOYu5NzywLdm5WThul_FF0_XtL5PFKaN3qS5mDTLbS-yMkPG9BVpxIny-tVII0bheKtX-YpxNJ_mkQ5-iKtOvBBrRdVPliZNyKt0_hyc3AqIkj5k-31Inp7404Z52nFp9pqLEB2mgrEtSk?purpose=inline)

**Build:** An image-generation API using a small pretrained diffusion model.

Learn:

- Denoising steps
- Latent resolution
- Model memory
- Generation latency
- Batch generation
- Quality vs speed trade-offs

**Experiments:**

- 20 vs 30 vs 50 denoising steps
- Different image resolutions
- Different precision
- One image vs batch generation

This teaches you a workload where one request can occupy the accelerator for a relatively long time.

Lab 6

## **Audio inference pipeline**

[Voice Recognition vs. Speech Recognition: Which Do You Need for Your Application? - Smallest.ai](https://images.openai.com/static-rsc-4/IQfwJ9k69I9MhhbqA9WCqPg8IzEU48xV6IyPSN0SwFUYAAMh_bO3CUWpvI3sAqZUqRAeI9pKHg4B-Rztr-pvLeIuE7dJlIKuAttsN_r0cFQds-H1OqqTtXos6YutjSEmY44uXBVUrJW34VenbMTwD7EG1Oj0Eep9owy6bg_23OQ?purpose=inline)

**Build:** A speech-to-text API using a small Whisper model.

Learn:

- Real-time factor
- Streaming vs batch
- Audio duration vs processing time
- Chunking
- CPU audio preprocessing
- Concurrent audio sessions

**Experiments:**

- 10-second vs 60-second audio
- CPU vs GPU
- Different model sizes
- Sequential vs batched transcription

## **3. Then combine them into one system**

After these labs, build a single service that can route requests to different model types.

[LLM Application Architecture: Building Beyond the ChatGPT Wrapper platform - DEV Community](https://images.openai.com/static-rsc-4/vE4bpH5oraoQzgcF0G5vHTRh4qQNnaKBjMRWv44Liv6acR9090xg_jgx-AuAkHfeHzvk2hve5HbBj1SOdREDwGClfzS13jfYeVPZRGdwOXthaMe9PfgtU5psFgDPsmbupN-5xkZTyn02J_L_FHdQKE1G1qiz4sgHtyFx7wiRpvY?purpose=inline)

For example:

```
                 Client
                   |
               API Gateway
                   |
             Model Router
                   |
       +-----------+-----------+
       |           |           |
      LLM       Embedding      CV
       |           |           |
    GPU worker  Vector DB   GPU worker
       |           |           |
       +-----------+-----------+
                   |
              Observability
```

Add:

- Request queueing
- Per-model concurrency limits
- Model loading and unloading
- GPU memory-aware scheduling
- Metrics and tracing
- Model versioning
- Quality evaluation
- Retry and timeout policies

Now you can ask real architecture questions:

- Should the LLM and CV model share one GPU?
- Should embedding generation use a separate worker?
- Is it better to keep a model loaded or load it on demand?
- What happens when two models compete for memory?
- Which workloads can be batched?
- Which workloads need low latency?
- Which workloads should run asynchronously?

These are the questions that turn model knowledge into systems knowledge.

## **4. Don't build every model from scratch**

Here's the distinction I would keep:

| Model type      | Build from scratch?           | Use pretrained model? |
| --------------- | ----------------------------- | --------------------- |
| Tiny classifier | Yes                           | Also useful           |
| Transformer     | Tiny educational version only | Yes, for serving      |
| Embedding model | No                            | Yes                   |
| Object detector | No                            | Yes                   |
| Diffusion model | No                            | Yes                   |
| Speech model    | No                            | Yes                   |

Implementing a tiny transformer once can be a useful educational exercise. But implementing every architecture from scratch is not the best use of your time.

Your objective is to understand the workload and build the system around it.

## **5. Your lab's final deliverable**

For every model type, produce a one-page **workload profile**:

Example: LLM inference workload profile

| Dimension             | What you should know                |
| --------------------- | ----------------------------------- |
| Primary resource      | GPU compute, memory bandwidth, VRAM |
| Latency metric        | TTFT, time per output token         |
| Throughput metric     | Output tokens/sec                   |
| Scaling variable      | Context, output length, concurrency |
| Main bottleneck       | Compute, memory, queueing           |
| Cost unit             | Cost per request / token / task     |
| Quality metric        | Task-specific evaluation            |
| Architecture decision | Batch, scale, route, or isolate     |

Do this for LLMs, embeddings, CV, diffusion, and audio.

By the end, you won't merely know how to call five different models. You'll have a mental framework for predicting how an unfamiliar AI workload will behave.

**That's the lab I would build: one shared benchmark and serving platform, with small representative models plugged into it one workload at a time.**