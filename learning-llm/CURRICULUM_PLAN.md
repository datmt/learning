# AI Systems Curriculum — Master Plan
> From tensor ops on a GPU → building, training, optimizing, and serving modern AI systems.
> Hardware target: NVIDIA GB10 / DGX Spark. Falls back to CPU where GPU is absent.
> Timeline: ~4–6 months part-time. 8 tracks / 24 projects / 8 sessions.

## How sessions work
- Each **Session = 1 Track = 2–3 projects**.
- Per your request: I implement **one session at a time**. This delivery is **Session 1 (Track 1)** fully built; Sessions 2–8 are scaffolded with specs so we can build them on demand.
- Every project ships 4 artifacts: `notebook.ipynb` (experiment notebook), `code/` logic inline in notebook, `results/` (CSVs/plots), `README.md` (answers to the 7 questions).

## Repo layout
```
learning-llm/
  CURRICULUM_PLAN.md        <- this file (full spec)
  SESSIONS.md               <- session checklist + exit criteria
  requirements.txt
  track-01-ml-foundations/
    project-01-numpy-mlp/notebook.ipynb
    project-02-pytorch-mlp/notebook.ipynb
    project-03-gpu-training-lab/notebook.ipynb
  track-02-gpu-programming/ ... (P4-P6, scaffolded)
  track-03-computer-vision/ ... (P7-P9, scaffolded)
  track-04-transformers/ ... (P10-P12, scaffolded)
  track-05-llm-training/ ... (P13-P15, scaffolded)
  track-06-llm-inference/ ... (P16-P18, scaffolded)
  track-07-multimodal-agents/ ... (P19-P21, scaffolded)
  track-08-ai-systems/ ... (P22-P24, scaffolded)
```

## Notebook standard (applies to all 24)
1. Markdown cell: goal + architecture diagram + learning objectives.
2. Code cells: **every line commented** (what it does + why). No bare magic.
3. Sections: Setup → Data → Model → Training → Evaluation → Benchmarks → Experiments (what-if X?) → Cleanup/next steps.
4. All notebooks run on CPU; GPU cells auto-skip if `torch.cuda.is_available()==False`.
5. Data strategy: try torchvision MNIST/CIFAR first, fall back to `sklearn.datasets.load_digits` or synthetic data so notebooks run offline.
6. Results saved to `./results/*.csv` + matplotlib plots.
7. Reproducibility: fixed seeds (`random`, `numpy`, `torch`) set in cell 1.

## Session breakdown

### Session 1 — Track 1: ML Foundations (THIS DELIVERY, fully built)
| Project | Notebook | Key question it answers |
|---|---|---|
| P1 NumPy MLP | `track-01/.../project-01-numpy-mlp/notebook.ipynb` | Where does the gradient actually come from? |
| P2 PyTorch MLP | `track-01/.../project-02-pytorch-mlp/notebook.ipynb` | What do forward/backward/step/zero_grad do? |
| P3 GPU Training Lab | `track-01/.../project-03-gpu-training-lab/notebook.ipynb` | Why does a GPU accelerate ML? FP32 vs FP16 vs BF16? |

Exit criteria S1: train MLP to >90% on digits/MNIST both in NumPy and PyTorch; produce samples/sec + time-vs-batch-size plot; explain SGD vs Adam in own words.

### Session 2 — Track 2: GPU Programming (NEXT, scaffolded)
- P4 GPU Benchmark Lab: vector-add, matmul, reduction, softmax across NumPy / torch-CPU / torch-GPU; problem-size → runtime plots; FLOPS vs bandwidth analysis.
- P5 First CUDA Kernels: vector_add, relu, sigmoid, softmax, reduction in raw CUDA (numba / torch cpp extension / triton fallback); vs PyTorch benchmark.
- P6 Optimize a CUDA Kernel: naive → coalesced → shared-mem → warp-level reduction; Nsight Systems/Compute + nvidia-smi.
- Needs: CUDA toolkit + NVCC on DGX Spark. Will detect and degrade gracefully.

### Session 3 — Track 3: Computer Vision (scaffolded)
- P7 CNN from scratch (CIFAR-10): conv/pool/norm/receptive-field labs.
- P8 Tiny object detector: boxes, IoU, NMS from scratch + tiny detection head.
- P9 Sports vision system: YOLO + tracking + trajectories + heatmap + homography to pitch coords.

### Session 4 — Track 4: Transformers (scaffolded)
- P10 Attention from scratch: QKV, scaling, causal mask, MHA, pos-encoding, attention-map viz.
- P11 Tiny GPT (10M→50M→100M): tokenizer → embeddings → blocks → LM head on TinyShakespeare.
- P12 LLM Anatomy Lab: param/layer/hidden/heads/KV-cache accounting on Qwen/Llama/Gemma; per-token memory math; 4K→32K context experiment.

### Session 5 — Track 5: LLM Training (scaffolded)
- P13 Train small LM (100M/300M/1B config comparison, throughput dashboard).
- P14 Fine-tuning lab (SFT vs LoRA vs QLoRA vs full FT; trainable-param/memory/time/quality table).
- P15 Evaluation harness (exact-match, BLEU/ROUGE, classification metrics, LLM-as-judge; base vs LoRA vs RAG comparison).

### Session 6 — Track 6: LLM Inference (scaffolded)
- P16 Quantization lab (BF16/FP8/INT8/INT4/NVFP4; memory/throughput/latency/quality table).
- P17 Mini LLM server (FastAPI + streaming + batching + queue + metrics; your backend skills apply here).
- P18 vLLM concepts (continuous batching vs naive, TTFT/TPOT, PagedAttention, scheduler).

### Session 7 — Track 7: Multimodal + Agents (scaffolded)
- P19 VLM (image→encoder→projection→LLM; QA/OCR/charts/video).
- P20 RAG from scratch (chunking/embed/index/retrieve/prompt; chunk/overlap/top-k/rerank ablations; retrieval vs generation quality split).
- P21 Agent runtime (LLM→tool→exec→observe loop; calculator/fs/python/http/db tools; retries/timeouts/state/memory/permissions).

### Session 8 — Track 8: AI Systems + Capstone (scaffolded)
- P22 Distributed training (DDP/FSDP/TP/PP, NCCL all-reduce, 1-vs-2 GPU scaling efficiency).
- P23 Production AI platform (API→gateway→LLM/VLM servers→observability; auth/ratelimit/queue/batch/cache/metrics/tracing/routing/GPU-monitor).
- P24 Capstone: Sports AI Analysis Platform (video→detect→track→homography→trajectories→stats+VLM→LLM analysis→report/API).

## Per-project README template (all 24)
Each project README must answer:
1. What did I build? 2. How does it work? 3. What did I measure?
4. What surprised me? 5. What bottleneck did I find?
6. What happens if I change X? Why? 7. What would I optimize next?

## Conventions for line-by-line comments
- Every code line gets an inline `#` comment stating intent + mechanism.
- Non-obvious math (softmax stability, Xavier init, chain rule) gets a 1-line derivation above the code.
- Benchmark cells print a table AND save CSV to `results/`.

## Next action
- Say `build session 2` (or any session/track) and I generate those notebooks in the same commented style.
- Say `verify session 1` and I execute all three notebooks headlessly and report pass/fail.
