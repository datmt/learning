# What does it actually cost to run an LLM? Measurements on an NVIDIA DGX GB10

"How much RAM do I need for a 30B model?" is the wrong first question. Memory, speed and *how many users you can serve* are three different costs, driven by three different things. This post measures all three on one machine (NVIDIA **GB10**, 121 GiB unified memory) with `llama.cpp`, for 8 models plus a quantization ladder, and tells you which numbers to trust and which not to.

**Who this is for:** anyone choosing hardware or a model/quantization to deploy, who wants numbers instead of folklore, and a method they can repeat.

**How to read it.** Every number is labelled **MEASURED** (observed directly), **DERIVED** (arithmetic on measured values) or **THEORETICAL / INFERRED** (a model or an assumption). Where they disagree, the disagreement is reported.

## TL;DR

| Question | Answer (GB10, `llama.cpp`) |
|---|---|
| RAM for a ~30B model at 4-bit? | **~15–21 GiB** of weights, plus a KV cache that depends on architecture: **+2.6 to +8 GiB** at 128k context for the hybrid-attention models tested, **+18 GiB** for a conventional one. Measured process totals at 128k: **~23 GiB**. |
| When is that memory taken? | **At launch.** Process memory was flat from load to the end of every workload (±0.1 GiB): the server reserves the whole context pool up front, so `-c` sets your memory, not your traffic. |
| What decides generation speed? | **Memory bandwidth ÷ bytes streamed per token.** Predicted from a measured 216–243 GB/s read bandwidth, the dense models landed at **78–109%** of prediction (median 93%). A first-order guide, not an exact predictor. |
| What decides prompt-processing speed? | **Compute** (~45–55 TFLOP/s effective). Roughly `23,000 ÷ params(B)` tokens/s for dense models. |
| Dense 27B vs MoE 35B-A3B (4-bit) | The MoE is **5.3× faster at generation** (67.7 vs 12.8 tok/s) and **2.8× faster at prompt processing**, with a file 35% bigger. |
| Does quantizing hurt? | On wikitext-2 perplexity: Q8_0 **+0.29%**, Q4_K_M **+0.94%** vs BF16 (Qwen3-8B); Q4_K_M **+0.80%** vs Q8_0 (Gemma 3 12B). Perplexity only, not task quality. |
| How many users can it serve? | Far fewer than aggregate throughput suggests. With 4 slots and a stated SLO, the 35B-A3B served **~0.2–0.4 chat requests/s** and the dense 27B **~0.05–0.1 requests/s**. SLO-compliant output was **~40% of the batched-benchmark aggregate**. |
| Do long prompts hurt other users? | Yes. At 0.08 long-context req/s the 35B-A3B's P95 inter-token latency went from 87 ms to **1.2 s** for everyone. |

![Generation speed](images/01_generation_speed.png)

---

## 1. The test machine and the models

| | |
|---|---|
| System | NVIDIA GB10 (Grace-Blackwell), 20 Arm cores (10× Cortex-X925 + 10× Cortex-A725) |
| Memory | **121.7 GiB unified LPDDR5x**, shared by CPU and GPU. No separate VRAM. |
| Software | `llama.cpp` commit `10a60cf` (clean tree, CUDA 13.0, arch 121, Release), driver 580.173.02. Full record in [`raw2/environment.txt`](raw2/environment.txt). |
| Flags | `-fa 1 -ngl 99`, same as the production server. |

**Unified memory** means "GPU memory" and "system RAM" are one pool, so a model can use nearly all 121 GiB (far more than a consumer GPU), but streaming it is limited by the pool's bandwidth (§3). Capacity is plentiful; bandwidth is the constraint.

| Model | Type | Quant | Params (B) | File (GiB) | Bits/weight |
|---|---|---|---:|---:|---:|
| Gemma 3 4B | dense | Q4_K_M | 3.9 | 2.3 | 5.1 |
| Gemma 3 12B | dense | Q4_K_M | 11.8 | 6.8 | 4.9 |
| Gemma 3 12B | dense | Q8_0 | 11.8 | 11.6 | 8.5 |
| Qwen3 8B | dense | BF16 / Q8_0 / Q4_K_M | 8.2 | 15.3 / 8.1 / 4.7 | 16 / 8.5 / 4.9 |
| Qwen3.8 27B | dense, hybrid attention | Q4_K_M | 27.3 | 15.3 | 4.8 |
| gpt-oss 20B | MoE (32 experts, 4 active) | MXFP4 | 20.9 | 11.3 | 4.6 |
| Gemma 4 26B-A4B | MoE | Q4_K_M | 25.2 | 15.8 | 5.4 |
| Qwen3.6 35B-A3B | MoE (256 experts, 8 active), hybrid attention | Q4_K_M | 34.7 | 20.6 | 5.1 |

"Q4_K_M" is not 4 bits: it averages **4.8–5.1 bits per weight**. Model hashes are in [`raw2/model_sha256.txt`](raw2/model_sha256.txt).

---

## 2. Where the memory goes

```
memory ≈ weights + KV cache + buffers/overhead        (all reserved at launch)
```

### 2.1 Weights (THEORETICAL, confirmed by file sizes)

```
weights (GiB) ≈ parameters × bits_per_weight / 8 / 2^30
```

| Quantization | bits/weight | ≈ weights for 30B |
|---|---:|---:|
| BF16 / FP16 | 16 | ~56 GiB |
| Q8_0 | 8.5 | ~30 GiB |
| Q4_K_M | ~4.9 | ~17 GiB |

**MoE models need all their weights resident.** Qwen3.6-35B-A3B computes with ~3B parameters per token but holds all 34.7B (20.6 GiB). MoE saves compute and bandwidth, not capacity.

### 2.2 The KV cache (THEORETICAL formula, MEASURED slopes)

Every token in context stores a key and a value at each attention layer:

```
KV bytes per token = attention_layers × kv_heads × head_dim × 2 (K and V) × bytes_per_element
```

Read from the GGUF headers, then compared with the measured slope (§2.4):

| Model | Layers with KV | KV heads × head dim | Predicted | **Measured** |
|---|---:|---|---:|---:|
| Qwen3 8B (conventional) | 36 of 36 | 8 × 128 | 144 KiB/token | **145 KiB/token** |
| Qwen3.8 27B (hybrid) | 16 of 65 | 4 × 256 | 64 KiB/token | **65 KiB/token** |
| Qwen3.6 35B-A3B (hybrid) | 10 of 40 | 2 × 256 | 20 KiB/token | **21 KiB/token** |

The two Qwen models are **hybrids**: only every 4th layer is full attention; the rest are state-space layers whose state does not grow with context. At **131,072 tokens** that is ≈ **18 GiB** of KV for a conventional 8B, ≈ **8 GiB** for Qwen3.8 27B and ≈ **2.6 GiB** for Qwen3.6 35B-A3B.

The formula was verified against headers for the three Qwen models only. For the other architectures (e.g. Gemma 3, which mixes sliding-window and global layers; gpt-oss) the slopes in §2.4 are fitted from two points and should not be assumed linear in general.

**Parallel slots.** With a unified KV pool (`--kv-unified`), slots *share* one pool, so total context across users costs memory, not `slots × context`.

### 2.3 Everything else

Compute buffers, CUDA context, the vision projector (~0.9 GB for each Qwen model here) and, on the host, a prompt cache (§2.6). At short context the process was within ±1 GiB of its file size for the models measured.

### 2.4 Measured: process memory vs context window

One fresh `llama-server` per point, one slot, one request, memory from `nvidia-smi` (round 1, single measurement per point):

![Memory vs context](images/05_memory_vs_context.png)

{{T:Process memory (GiB) vs context}}

Note the crossover: at 4k the 35B-A3B needs 5.4 GiB more than the 27B; at 128k the gap is 0.2 GiB. A "bigger" model can be cheaper at long context than a "smaller" one.

### 2.5 Shrinking the KV cache

Memory side only (quality not evaluated):

{{T:KV-cache quantization}}

### 2.6 Memory over a server's lifetime (MEASURED, round 2)

Round 1 gave one number per configuration. Round 2 samples memory every 0.5 s through a server's whole life: load, idle, a 16k-token prefill, a 256-token generation, 4 concurrent requests, and one request that fills the entire 131k context pool. All requests were validated (exact token counts); the 4 models and 4 slots are in the table.

![Memory lifecycle](images/09_memory_lifecycle.png)

{{T:Memory over the process}}

What the data shows:

- **Memory is reserved at launch, not consumed by traffic.** GPU process memory rose in two steps during load (weights first, then KV cache and compute buffers) and then stayed flat to within 0.1 GiB through every workload, including the 130k-token request. Sizing is therefore decided by `-c` and the model, not by how busy the server is.
- **Host memory does grow with use.** Process RSS started at 0.8–1.3 GiB and ended at 2.6–6.7 GiB, and did not fall after the work finished. That is consistent with the server's host-side prompt cache (`--cache-ram`, default 8192 MiB per the server's own help text), but I did not isolate the cause. Budget a few GiB of host memory per server on top of the GPU figure.
- **On this unified-memory system the two counters add up.** GPU-process memory plus RSS matched the machine-wide increase in "used" memory within 1.5 GiB for all four models, so they do not double-count.
- **No real memory pressure here, but not zero.** Minimum available memory stayed at about 32 GiB or more and the kernel's memory-pressure indicator (PSI) peaked at 3.0–4.3% during model load, then fell to 0. The machine also had ~10.5 GiB of swap in use before every run, from other workloads, and it barely moved.

{{T:Per-phase peaks}}

**A gotcha: a pool-sized request can crowd out others.** In my first lifecycle round the concurrent phase ran *after* the request that filled the 131k pool. Four concurrent requests using only half the pool then failed with HTTP 500 `Context size has been exceeded` on 3 of 4 models. Repeating the identical 8B configuration, 3 of 5 validated runs failed and 2 passed, so it is **intermittent**. I do not know the mechanism. In the final run the concurrent phase ran before the pool-filling request and all 32 requests succeeded. The failing runs are preserved in [`raw2/memlife_v2_concurrent_after_maxctx/`](raw2/memlife_v2_concurrent_after_maxctx/) and [`raw2/memlife_v1_diagnostic_runs/`](raw2/memlife_v1_diagnostic_runs/). Practical advice: with a shared pool, do not let one request approach the pool size if others need to run, and test your own concurrency pattern.

---

## 3. Where the time goes

| Phase | What happens | Bottleneck | Metric |
|---|---|---|---|
| **Prefill** (prompt processing) | whole prompt in parallel | **Compute** | `pp` tokens/s |
| **Decode** (generation) | one token at a time; each step re-reads the weights | **Memory bandwidth** | `tg` tokens/s |

### 3.1 Prefill is compute-bound

![Prompt processing](images/02_prompt_speed.png)

For dense models, prompt speed × parameter count is nearly constant (DERIVED):

| Model | pp2048 (tok/s) | Params (B) | 2 × params × tok/s |
|---|---:|---:|---:|
| Gemma 3 4B Q4 | 5,695 | 3.9 | 44 TFLOP/s |
| Gemma 3 12B Q4 | 1,937 | 11.8 | 46 TFLOP/s |
| Qwen3 8B BF16 | 3,355 | 8.2 | 55 TFLOP/s |
| Qwen3.8 27B Q4 | 858 | 27.3 | 47 TFLOP/s |

Rule of thumb for dense models here: **prompt tokens/s ≈ 23,000 ÷ params (B)**, nearly independent of quantization. MoE models compute with only their active parameters, hence 2.8× faster prompts for the 35B-A3B than the dense 27B.

**Caveat on stability.** Within a session prompt speed was steady (stddev ≤ 5%). Across sessions it was not: Qwen3 8B BF16 measured 3,355 tok/s in round 1 and 2,938 ± 154 tok/s in round 2 (5 repetitions, 12% lower). Round 2 ran with two other models resident but idle; I did not establish the cause. Generation speed, by contrast, agreed across sessions (15.2 vs 15.5 tok/s). Treat `pp` figures as ±10–15%.

### 3.2 Decode is memory-bandwidth-bound

The model: for each token the GPU streams the weights, so

```
tokens/s ≈ achievable_bandwidth / bytes_streamed_per_token        (THEORETICAL)
```

Round 1 *inferred* a bandwidth from throughput, which is circular. Round 2 **measures** it with a separate streaming kernel (`scripts/membw.cu`: 16-byte loads/stores, CUDA events, 20 iterations after 3 warm-ups):

![Bandwidth microbenchmark](images/07_bandwidth_microbenchmark.png)

{{T:Memory-bandwidth microbenchmark}}

Read bandwidth is **~243 GB/s on small buffers, falling to ~216 GB/s at 16 GiB**, i.e. 79–89% of the published 273 GB/s peak (THEORETICAL, from the spec sheet, not verified here). Writes reach about 193 GB/s. At 16 GiB, one iteration dropped to 83 GB/s; it is kept in the data, since no exclusion rule was defined in advance. This is a *simple streaming kernel*, an estimate of what a well-behaved kernel can pull, not the access pattern of llama.cpp's matrix kernels.

**Bytes streamed per token (DERIVED from the GGUF tensor table).** Almost all weights are read once per token. The exception is the token-embedding table: only one row is looked up, unless the output head is *tied* to it (as in Gemma 3), in which case it is read in full. Excluding the embedding table saves 4–8% for untied models.

**Prediction vs observation, dense models:**

{{T:Decode speed: theoretical}}

![Prediction vs observed](images/08_prediction_vs_observed.png)

The model is useful and imperfect. Observed speed was **78%–109%** of the prediction (median 93%). It over-predicts the smallest model (Gemma 3 4B at 78%: fixed per-token costs matter more when the weights are small) and under-predicts BF16 (109%, where the 16 GiB microbenchmark point is the pessimistic, noisy one). Do not read it as "speed = bandwidth ÷ size to the percent".

**The quantization ladder (same weights, three precisions; 5 repetitions each):**

| Qwen3-8B | Tensors (GiB) | Generation (tok/s) | Speed vs BF16 | Effective GiB/s |
|---|---:|---:|---:|---:|
| BF16 | 15.26 | 15.5 | 1.00× | 219 |
| Q8_0 | 8.11 | 29.1 | 1.88× | 218 |
| Q4_K_M | 4.68 | 44.0 | 2.84× | 191 |

Weights shrink 3.3× from BF16 to Q4_K_M but speed rises only 2.8×: the Q4_K kernel reaches ~13% less effective bandwidth (DERIVED), plausibly dequantization overhead. Bytes are the first-order driver, kernel efficiency the second.

**MoE models** break the link between file size and speed. The table below is an **inference, not a measurement**: it divides the measured read bandwidth by the observed speed, and it lumps together routed experts, always-read tensors and any cache effects.

{{T:MoE models}}

The robust conclusion is the observation itself: the 35B-A3B file is 35% larger than the dense 27B yet generates **5.3× faster**.

### 3.3 Context depth: speed with a long prompt already in the cache

![Speed vs depth](images/03_speed_vs_depth.png)

| Model | tg @16k | tg @64k | pp @16k | pp @64k |
|---|---:|---:|---:|---:|
| Gemma 3 12B Q4 | 84% | – | 92% | – |
| Qwen3 8B BF16 | 86% | – | 63% | – |
| gpt-oss 20B | 87% | – | 76% | – |
| Qwen3.8 27B (hybrid) | 95% | 81% | 89% | 68% |
| Qwen3.6 35B-A3B (hybrid) | 91% | 72% | 90% | 72% |

(% of empty-context speed, DERIVED.) At 64k the hybrids still hold 72–81% of generation speed. Models other than the two Qwens were tested only to 16k.

### 3.4 Full speed table (round 1, 3 repetitions)

{{T:Speed}}

`pp` = prompt tokens/s (2048-token prompt), `tg` = generated tokens/s (128 tokens), at the stated tokens already in context. Mean of 3 repetitions after a warm-up; within-session standard deviation was below 1% for generation and below 5% for prompt processing (raw data in `raw/speed/`).

---

## 4. What this means for real workloads

### 4.1 Single-user latency (ESTIMATED from measured rates)

| Workload | Qwen3.8 27B (dense) | Qwen3.6 35B-A3B (MoE) |
|---|---:|---:|
| Chat: read a 500-token answer | ~39 s | ~7 s |
| Agent/RAG: ingest a 32k-token prompt | ~40–55 s | ~14–17 s |
| Then generate a 1,000-token answer from that prompt | ~85 s | ~18 s |
| Long document: ingest 64k tokens | ~95–110 s | ~33–37 s |

Chat and code completion are bandwidth-bound (favor MoE or a smaller model). RAG and agents are compute-bound at the front (MoE helps there too).

### 4.2 Many users at once, batched (MEASURED, `llama-batched-bench`, n = 1 run each)

![Concurrency](images/06_concurrency.png)

{{T:Concurrency}}

Batching decode is nearly free because decode is limited by *reading weights*, which a batch shares. The dense model scales better (5.3× at 8 streams) than the MoE (3.1×), since different tokens route to different experts and touch more of the weights. **But this benchmark is not serving capacity.** It sends identical, simultaneous, fixed-length requests and reports aggregate tokens/s, which says nothing about how long any one user waits.

### 4.3 Serving capacity under a latency target (MEASURED, round 2)

`scripts/load_test.py` sends requests with **Poisson arrivals** (open loop, so a slow server cannot slow the clients down), against a fresh `llama-server` with **4 slots and a 131k shared pool**. Prompts are cut from real text at random offsets with the prompt cache off; output lengths are forced so every model sees the same work. Per request it records time to first token (TTFT, which *includes* time queued for a slot), inter-token latency (ITL) and end-to-end latency.

**Workload profiles** (log-normal lengths, clipped):

| Profile | Prompt tokens (min / median / max) | Output tokens (min / median / max) |
|---|---|---|
| chat | 100 / 400 / 1,500 | 50 / 200 / 500 |
| long | 8,000 / 16,000 / 24,000 | 100 / 300 / 600 |
| mixed | 80% chat + 20% long | |

**SLOs, fixed before running and not tuned to results.** chat: P95 TTFT ≤ 2 s *and* P95 ITL ≤ 100 ms *and* zero failures. long and mixed: P95 TTFT ≤ 60 s *and* P95 ITL ≤ 200 ms *and* zero failures. A rate is judged only if ≥ 20 requests completed.

![Chat latency](images/10_latency_chat.png)

{{T:Load test: chat}}

![Long-context latency](images/10_latency_long.png)

{{T:Load test: long}}

{{T:Load test: mixed}}

{{T:Highest tested rate}}

What it shows:

- **Interactive capacity is much lower than aggregate throughput implies.** The 35B-A3B passed the chat SLO at the lowest rate (P95 TTFT 0.94 s) and failed at the next (15.1 s). The dense 27B passed at 0.05 req/s and failed at 0.1 (P95 TTFT 29 s). Capacity under this SLO therefore lies between the two tested rates for each: roughly **0.2–0.4 req/s** (35B-A3B) and **0.05–0.1 req/s** (27B). I tested few rates, so the resolution is coarse.
- **The SLO-compliant output rate was ~40% of the batched aggregate.** The highest passing chat runs carried 62 tok/s (35B-A3B) and 15.7 tok/s (27B), against 155 and 43 tok/s aggregate at 4 streams in §4.2. Queueing for one of only 4 slots, and variance in arrival and output length, consume the rest.
- **The failure mode is queueing, not slow decoding.** P95 ITL stayed flat (32–35 ms for the MoE) while TTFT climbed from under a second to over a minute.
- **The dense 27B sits at the edge of a 10 tok/s-per-user target.** Its P95 ITL was 97–100 ms even at light load (4 streams share one weight read).
- **Long prompts stall everyone.** For the 35B-A3B, P95 ITL rose from 87 ms at 0.05 long req/s to **1.17 s** at 0.08, while the median stayed at 37 ms: long prefills occupy the GPU and delay other users' decoding. TTFT P95 went from 44 s (pass) to 153 s; at 0.12 req/s, 14 of 55 requests had not finished when the test stopped.
- **27B long-context: no verdict.** At its slow capacity (about one 16k request every ~50 s per slot) even 400 s yields too few completions for percentiles. At 0.03 req/s, 8 of 19 requests were unfinished at the end, which is itself informative.
- **The mixed point is one run, not a capacity limit.** The 35B-A3B passed its (lenient) SLO at 0.3 req/s (P95 TTFT 26 s).

**Limits of this measurement.** One run and one random seed per rate, so the realized arrival rate differs from the target (e.g. 40 requests arrived in the "0.2 req/s" run, i.e. 0.267 req/s; the tables show the realized rate). Only 4 slots were tested; more slots change the picture. The other two production models were resident but idle, so "peak system used" in the tables is machine-wide (~89–93 GiB) and not a per-server figure. Queue wait is not observable separately; it is inside TTFT.

### Power and thermals

During the sustained round-1 benchmarks the GPU drew **31–52 W** on average (idle: ~7 W), peaked at **82 °C** and reported **no throttle events**.

---

## 5. Does quantizing hurt? (MEASURED, perplexity only)

Perplexity on the wikitext-2 test set (SHA-256 `173c87a5…`), 2048-token windows, identical settings for every file. Perplexity is comparable only **between quantizations of the same model**, so each family has its own reference. `llama-perplexity` was built from the same llama.cpp commit.

{{T:Perplexity}}

- The ordering is monotone in bit-width in both families, and the cost of Q4_K_M is **under 1%** in perplexity.
- The `±` is the statistical error of each estimate. The quantizations were scored on the *same* text, so the paired differences are tighter than the `±` suggests, but this tool does not report a paired error, so I do not claim any single difference is statistically significant.
- **This is not a task-quality ranking.** Perplexity on one general-text corpus says nothing certain about coding, reasoning, long-context retrieval or structured output. Use it to rule out gross damage, then test your own task. The Gemma comparison is against Q8_0, not the BF16 original.
- The Qwen3-8B Q8_0/Q4_K_M files came from `unsloth/Qwen3-8B-GGUF`. I did not verify that the on-disk BF16 file shares that provenance; hashes are recorded.

---

## 6. Recommendations

1. **Budget memory as weights + KV cache + a few GiB of host cache.** Weights from `params × bits / 8`; KV from `layers × kv_heads × head_dim × 4 bytes` per token (f16) times the total context you configure, because it is reserved at launch (§2.6).
2. **Check the architecture before choosing a context length.** Hybrid-attention models make 128k contexts cheap; conventional ones can make long context cost more than the weights.
3. **For interactive use on bandwidth-limited hardware, prefer MoE.** The 35B-A3B gave 5× the generation speed of the dense 27B, and its interactive capacity under the same SLO was several times higher. Use dense models when you need them and plan for ~13 tok/s.
4. **4-bit is the sweet spot for speed per byte, with a small measured perplexity cost.** On Qwen3-8B, BF16 → Q8_0 → Q4_K_M ran at 15.5 → 29.1 → 44.0 tok/s for +0.29% and +0.94% perplexity. Q8_0 on Gemma 12B used 1.7× the memory for 1.6× slower generation than Q4_K_M. Verify on your task.
5. **Do not size a deployment from aggregate throughput.** Batching raises total tokens/s 3–5×, but the number of users you can serve at a latency target was about 40% of that, and long prompts degrade everyone's inter-token latency. Define an SLO and load-test with arrivals like yours (`scripts/load_test.py`).
6. **Isolate long-context traffic, or cap prompt length.** Mixed long and short requests share the GPU, and a pool-sized request can crowd out others (§2.6).
7. **Capacity is not the constraint on this hardware, bandwidth is.** A 121 GiB pool lets several models stay resident, but faster decode needs more bandwidth, not more RAM.
8. **KV-cache quantization saves 19–33% total memory on Gemma 3 12B at 128k**, but evaluate quality first.

---

## 7. Method, provenance and caveats

Everything is reproducible from [`scripts/`](scripts/) and the raw output in [`raw/`](raw/) (round 1) and [`raw2/`](raw2/) (round 2). Tables and figures are generated from the raw files, and this README is built from `README.template.md` by `scripts/build_readme.py`, which injects the generated tables.

| Measurement | Script | Notes |
|---|---|---|
| Speed vs depth | `bench_speed.sh` | `llama-bench`, 512/2048-token prompts, 128 generated, depths 0/16k(/64k), 1 warm-up + 3 reps, GPU clocks/power sampled every 2 s |
| Memory vs context | `bench_memory.sh` | fresh server per point, 1 slot, 1 request, `nvidia-smi` per-process memory (single measurement) |
| Batched concurrency | `bench_concurrency.sh` | `llama-batched-bench`, 512-token prompt, 128 generated, 1/2/4/8 streams (single run) |
| Bandwidth | `membw.cu` | streaming read/write/copy kernels, 20 iterations after 3 warm-ups |
| Bytes per token | `streamed_bytes.py` | from the GGUF tensor table |
| Quantization ladder | `llama-bench` | Qwen3-8B BF16/Q8_0/Q4_K_M, 5 repetitions, per-run samples kept in `raw2/ladder/` |
| Perplexity | `bench_perplexity.sh` | wikitext-2 test set, 2048-token windows |
| Memory lifecycle | `memory_lifecycle.py`, `bench_memory_lifecycle.sh` | 0.5 s sampling; every request validated |
| Serving capacity | `load_test.py`, `bench_latency.sh` | open-loop Poisson arrivals, rate sweep, SLOs fixed in advance |

**Conditions.** Round 1 ran with the two production models unloaded (baseline 12.5 GiB used). Round 2 ran with them loaded but idle (baseline ~59 GiB used, GPU at 0%), since the machine was shared. Absolute throughput could differ between rounds; see the prompt-speed caveat in §3.1.

**What went wrong and was fixed (kept in the data, not hidden):**

- *Overlapping benchmarks.* During round 1 I accidentally started a second benchmark while one was running; I killed both, deleted their output, and reran everything clean.
- *Silent request failures.* The first lifecycle runs did not check responses, so fast failures looked like fast successes. Requests are now validated against exact token counts, and the superseded data is kept in `raw2/memlife_v1_unvalidated/`.
- *A circular bandwidth argument.* Round 1 inferred a "233 GiB/s ceiling" from the best dense result and then used it to infer MoE traffic. That is replaced by the microbenchmark and a stated prediction error (§3.2).
- *One flawed first draft of the stability claim.* The earlier text implied prompt speed was stable to <5%; that holds within a session only (§3.1).

**Limitations, stated plainly:**

- One machine, one `llama.cpp` build, one day. Kernels improve quickly.
- `llama-bench` uses synthetic tokens at batch size 1. Real servers add sampling and scheduling overhead.
- **No task-quality evaluation.** Perplexity covers one corpus and a few quantizations.
- Bandwidth is measured with a simple kernel and is not a hardware counter reading. MoE bytes per token are inferred.
- Round-1 memory and batched-concurrency points are single runs. The round-2 memory lifecycle is one run per model; the load test is one run and one seed per rate. Few rates were tested, so capacity points are brackets, not precise values.
- `nvidia-smi` per-process memory on unified memory is not isolated VRAM; the GPU/host split is blurry, so differences under ~1 GiB should not be over-interpreted.
- Gemma and gpt-oss were tested to 16k depth only; Qwen3 8B BF16 memory to 32k (its 128k KV figure is extrapolated).
- Only 4 server slots and one runtime were tested. Results do not transfer to other GPUs or engines without testing.
