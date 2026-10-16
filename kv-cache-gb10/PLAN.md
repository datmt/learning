# KV Cache on DGX Spark (GB10): Experiment Plan

Status legend: [ ] not started, [~] in progress, [x] done. Each phase ends with a review; the next phase starts only after approval.

## Ground rules
- Separate from `../blog/` and `../out/`. Nothing there is modified. Existing data may be cited as pilot data only (n=1, no KV controls).
- Everything lives in `blog-kv/`: `scripts/`, `raw/` (machine-readable, one JSON per run), `tables/`, `images/`, `README.template.md` -> generated `README.md` (same pattern as `build_readme.py`). Plots and tables are generated from `raw/` only.
- One benchmark at a time on the GPU. Check for strays with `pgrep -x`. Validate every HTTP response (status, token counts); failed runs are kept as records.
- Production models (`qwen3.6-35b-a3b`, `qwen3.8-27b`) may be unloaded when needed (user approved, any time). Reload and verify via `llama.sh status` at the end of every phase.
- Each result is labelled **theoretical**, **measured** or **inferred**. No extrapolation past measured contexts without an "estimate" label.
- Python: `/home/dat/jupyter-venv/bin/python -I`. Big downloads go under `/data`. Not a git repo; publish later to `datmt/learning` (simple terms, verified claims, one folder per topic).

## Models (all already on disk, no download)
| model | design | why |
|---|---|---|
| Qwen3-8B (BF16, plus Q8/Q4 for weights-vs-KV contrast) | dense, full-attention GQA | the textbook case, formula applies directly |
| Qwen3.6-35B-A3B Q4_K_M | MoE, hybrid SSM (KV in 10 of 40 layers) | production model; formula overcounts about 4x |
| Qwen3.8-27B Q4_K_M | dense, hybrid SSM (KV in 17 of 65 layers) | same family, dense vs MoE control |
| Gemma3-12B Q4_K_M | dense, 5:1 sliding window (1024) | windowed cache |
| gpt-oss-20b MXFP4 | MoE, 1:1 sliding window (128) | second windowed design |
| Gemma4-26B-A4B (optional) | MoE, 5:1 SWA, mixed KV heads/dims | stretch case |

Core set for all experiments: Qwen3-8B, Qwen3.6-35B-A3B, Qwen3.8-27B. Gemma3 and gpt-oss are added in Phases 1 and 2 where cheap. Gemma4 only if time permits.

## Phase 0: Theory (no GPU) [x] done
- `scripts/gguf_meta.py` dumps GGUF metadata; `scripts/kv_theory.py` derives bytes/token from metadata plus which layers actually have `attn_k` tensors. Output: `theory.json`, table in README.
- Deliverables: plain-language KV explanation, formula with every term, and the per-model table. Caveats documented: MHA/MQA/GQA, SWA, hybrid SSM, latent attention (explained, no model available to test), quantised KV, allocation/alignment.
- Open (resolved by measurement in Phase 1): which gpt-oss layers are windowed (assumed even), size of the SSM recurrent state, how llama.cpp sizes SWA caches (`--swa-full`, window plus micro-batch).

## Phase 1: Experiment A, memory vs context (and Experiment B comparison) [~] data collected and analysed (2026-10-15); awaiting review
**Question:** how many bytes does one more cached token cost, and does it match the theory?
- Models: core three, plus Gemma3-12B and gpt-oss-20b.
- Contexts (actual prompt tokens): 4K, 8K, 16K, 32K, 64K, 128K. A step is run only if the model's trained context allows it (Qwen3-8B stops at 32K) and it fits.
- Runtime: fresh `llama-server` per (model, context) with f16 KV, flash-attention state recorded, `-c` fixed. A pair of configs per model separates cache **allocated** from **occupied**:
  - (a) `-c` fixed large, vary the prompt length: shows allocation vs live tokens.
  - (b) `-c` = prompt + output: shows the per-token slope of the allocation itself.
- Measurements per run: system memory (`/proc/meminfo`, unified memory), process RSS, GPU-reported memory, at the stages: baseline, post-load, post-warmup, post-prefill, during/after generation of a fixed 64 tokens. Runtime metrics from `/metrics`, `/slots` and the server log (KV buffer size line where available; check verbosity, since the default is quiet).
- Token counts from the server's `usage`/timings, not from the tokenizer estimate. Prompt cache disabled or reset between runs. Generation fixed so it doesn't change the context comparison.
- Repetitions: 3 per point; raw data for every one.
- Analysis: linear fit memory vs tokens (slope = bytes/token, intercept = fixed overhead), compare with theory per model; detect allocation steps vs continuous growth; hybrid models: separate the fixed recurrent state from per-token growth; SWA: show the plateau at the window.
- Experiment B table: parameters (total/active), layers, Q/KV heads, head dim, attention type, KV dtype, retention strategy, theoretical vs measured KiB/token. Highlights which properties set KV size (layers with KV, KV heads, head dim, dtype, window) and which don't (MoE routing, total params).
- Deliverables: memory-vs-context plot per model, theory-vs-measured table, B architecture table, explanation of discrepancies.
- Pilot check first: one model, one context, confirm the harness and which memory counters are meaningful on unified memory before the full matrix.

## Phase 2: Experiment C, speed vs context [~] data collected and analysed (2026-10-15); awaiting review
**Question:** does a longer context slow generation with the model and quant unchanged?
- Reuses Phase 1 runs where the configuration is identical, plus extra runs for the speed metrics. Same models and contexts.
- Constant: model file, f16 KV, 128-token output, batch/ubatch, 1 slot, flags, prompt construction (fixed text, trimmed to exact token count).
- Metrics: prefill tok/s, TTFT, decode tok/s, inter-token latency (streaming timestamps; mean, p50, p95), peak memory. 5 repetitions; report mean, stdev, min/max.
- Cross-check the server-reported timings against client-side streaming timestamps. A `llama-bench` `-d` depth run gives an independent decode-at-depth reading for the core models.
- Analysis: prefill vs decode explanation; decode is not assumed linear in context. Measure it and tie the shape to KV bytes read per token (theory bytes/token times context vs weight bytes read, using the measured 273 GB/s bandwidth from the earlier benchmark).
- Deliverables: four plots (prefill tok/s, TTFT, decode tok/s, ITL vs context) and a table.

## Phase 3: Experiment D, KV precision [~] data collected and analysed (2026-10-16); awaiting review
**Question:** what does a lower-precision cache buy, and cost?
- Formats: f16 baseline, q8_0, and q4_0 (K and V types set separately where allowed). Each is verified, not assumed: record the effective `--cache-type-k/v`, flash-attention on/off, any fallback or error (quantised V needs flash attention in llama.cpp), and confirm the KV buffer shrinks as expected.
- Models: core three (Gemma3 optional). Contexts: 8K, 32K, plus 128K where supported.
- Performance: memory (KV, process, system), prefill, decode, TTFT, ITL, with the Phase 2 methodology and repetitions.
- Quality (kept small and deterministic):
  1. Perplexity on wikitext-2 at context 4096 (needs the out-of-tree `llama-perplexity` build; do not rebuild in `/data/engines/llama.cpp` while prod servers run). Report delta vs f16 with its uncertainty.
  2. Greedy-decoding agreement: fixed prompt set, token-level divergence point and exact match vs the f16 output.
  3. Long-context retrieval: needle-in-a-haystack at 8K/32K/(128K), several depths, exact-match scoring, more than 20 trials per cell if cheap enough.
- Report regressions honestly, including cases where the quantised cache is not slower or not worth it.
- Deliverables: precision comparison table (memory saved, speed delta, quality delta) and plots.

## Phase 4: Experiment E, concurrency and capacity [~] data collected and analysed (2026-10-16); awaiting review
**Question:** how do context and concurrent sequences combine to fix memory and capacity?
- Concurrency: 1, 2, 4, 8 sequences. Context per sequence: 4K, 16K, 32K, 64K. Bounded by what the model supports and what fits.
- Distinguish preallocated pool from dynamic allocation: llama-server's unified pool (`-c` total divided across `-np` slots, `--kv-unified`) vs per-slot split. Record the flags used and what the log shows.
- Hybrid models: per-sequence recurrent state scales with the number of sequences even though the attention KV is small. Measure it.
- Record: active sequences, total cached tokens, KV allocated vs used, peak system memory, aggregate and per-request throughput, TTFT, ITL, failures/OOM/HTTP errors (including the known "context size exceeded" 500 seen earlier after a pool-filling request; try to reproduce and characterise, label causes as inferred unless demonstrated).
- Deliverables: context × concurrency matrix of memory and of success/failure; maximum completing configuration per model; capacity statement for DGX Spark users (e.g. "N sequences of M tokens fit in X GiB at f16/q8_0").

## Phase 5: Write-up and reproducibility
- `README.template.md` -> `README.md` generated by `scripts/build_readme.py`: beginner KV explanation, theory tables, all results, practical guidance for DGX Spark users, theoretical/measured/inferred labelling, reproduce commands per experiment, manifest of what was and was not run on the physical GB10.
- Per-run record: experiment ID, model file hash and quant, llama.cpp commit and full command line, prompt/output token counts, context, KV dtype, concurrency/batch settings, memory and speed measurements, errors/warnings, repetition number.
- Experiment manifests in `raw/manifest_*.json`; failed runs retained and explained.
- Optional: publish to `datmt/learning` as `kv-cache-gb10/` only after review.

## Risks and open items
- Unified memory: `nvidia-smi` per-process numbers are unreliable. Use system-level deltas and llama.cpp's own buffer sizes, and validate in the Phase 1 pilot.
- Page cache and mmap can distort "memory used" when loading models. Record Cached and Buffers separately and use consistent drop/settle procedures.
- 128K runs on 27B/35B may be slow; per-run time budget and a hard timeout apply. Timeouts are recorded as failures.
- Prompt speed has varied about 12% between sessions in the earlier work. Run configurations within one session, interleaved, and record session ID.
- llama.cpp behaviour for hybrid and SWA models (cache sizing, checkpointing) may differ from the theory. That is a finding, to be reported not hidden.
- Quantised V cache may be unsupported on some models/backends; the result is then recorded as unsupported.

## Progress log
- 2026-10-15: Phase 0 done. Phase 1 pilot validated the harness (`scripts/harness.py`, `run_phase1.sh`, `analyze_phase1.py`). 155 runs in `raw/`, logs in `logs/`, tables in `tables/`, plot in `images/`. Prod models reloaded and verified.
- Findings: KV bytes/token match theory for Qwen3-8B, Qwen3.6-35B-A3B, Gemma3 (global), gpt-oss (global). Qwen3.8-27B allocates KV for 16 layers, not 17 (theory 5.9% high; cause of the 17th tensor layer unidentified). KV is allocated at load, sized by `-c` padded to 256 cells. SWA cache = window + 512 micro-batch (Gemma3 1536 cells, gpt-oss 768). Recurrent state fixed per sequence (62.8 / 149.6 MiB).
- Deviations: config (a) at 64K/128K has 1 rep; Gemma3/gpt-oss 128K run at prompt 131000 (trained limit 131072); Gemma3 reports +1 token (BOS); config (a) at 131072 failed with request-too-large (kept as records).
- Not done: Experiment B table, Gemma4, Phases 2-5. Next: your review of Phase 1, then Phase 2 (needs prod models unloaded again).
- 2026-10-15 (later): Phase 2 run. 28 server points x 5 reps (`raw/p2_*.json`) plus llama-bench depth runs for the core three (`raw/p2bench_*.json`). Tables `tables/phase2_*.md`, plot `images/phase2_speed_vs_context.png`. Prod models reloaded and verified.
- 2026-10-16: Phase 3 run. Perf/memory: `raw/p2_*_{q8_0,q4_0}.json` (Phase 2 harness, f16 baseline from Phase 2). Perplexity: `raw/ppl/`. Greedy agreement and needle: `raw/p3q_*.json`. Tables `tables/phase3_*.md`, plot `images/phase3_kv_precision.png`. Prod models reloaded and verified. First needle run was invalid (12-token answer cap cut off preamble; f16 baseline scored 6/20) and is kept in `raw/superseded/`; rerun with 64 tokens.
- 2026-10-16 (later): Phase 4 run. 44 points (`raw/p4_*_f16.json`: core three x 1/2/4/8 sequences x 4K/16K/32K/64K per sequence; Qwen3-8B stops at 32K), all completed, no errors. `-np N --kv-unified`, `-c` = N*(prompt+64+8), prompt cache off, 64 generated tokens, one run per point (n=1). Tables `tables/phase4_*.md`, plot `images/phase4_concurrency.png`. 500-error reproduction in `raw/p4_repro500.json` (5 trials, prod flags). Prod models reloaded and verified.
- Findings: KV bytes per cell constant and equal to Phase 1 (147456 / 20480 / 65536); recurrent state exactly linear in sequences (62.8 MiB 35B, 149.6 MiB 27B); no memory failures even at 27B 8x64K (32 GiB KV, 59 GiB MemAvailable minimum), latency is the limit (27B 8x64K: mean TTFT 810 s, max 1747 s, wall 1763 s). "Context size exceeded" 500 reproduced when requests in flight need more cells than the unified pool (4x40K in a 131072 pool: all 500; 130K prefill plus 4x8K arriving during it: 4 of 5 failed); pool-filling alone is harmless (130K finished, then 4x8K: all 200). Mechanism inferred from log, not demonstrated.
- Not done: quantised-KV or other pool sizes in Phase 4, repeats of the repro trials, aggregate-throughput claims (per-request decode tok/s is measured while other requests prefill).
