# Project 12 — LLM Anatomy Lab (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

> New to LLM sizing? Read `lecture.ipynb` first — a beginner-friendly visual intro (param chunks, KV-cache growth, flops/token) that runs in seconds on numpy/matplotlib only.

## Measured (compute + CPU timing demo, this machine)
- Param estimates vs published: Llama-3.2-1B **1.50 / 1.24**, Llama-3.1-8B **8.03 / 8.03**, Qwen2.5-7B **7.62 / 7.62**, Gemma-2-9B **9.93 / 9.24** (B) — full table in `results/anatomy.csv`
- Accounting method verified **exactly**: rebuilt P11 TinyGPT counts 464,705 by torch and by hand formula (match=True)
- Pattern: MLP ≈ 54–75% of every block; attention only 11–24%; embed+head dominate tiny models, vanish in big ones
- KV-cache bf16/batch-1: 8B goes **0.5GB @4K → 4.3GB @32K → 17.2GB @128K**; weights 16.1GB → a 24GB card fits 4K, not 128K batch>1 (full table `results/kv_cache.csv`, plot `kv_cache.png`)
- Decode flop/tok (8B): **18.7G @4K → 26.3G @32K** (attention share 6% → 33%); 4K→32K is an 8× cache/flop-attention tax
- Measured attention time-vs-n log-log slope **1.76** (quadratic = 2.0) — `results/context_scaling.png`, `context.csv`

## The 7 README questions (fill after running)
1. **What did I build?** A param/KV-cache/flop accountant for real LLMs: per-model breakdowns, GQA-aware cache tables, weight-fit verdicts, and a measured O(n²) timing proof.
2. **How does it work?** Params = embed (V·H) + L·(attn Q,K,V,O + SwiGLU gate/up/down + norms) + head; KV-cache = 2·L·kv_dim·seq·bytes (GQA shrinks kv_dim via fewer kv heads); decode flop/tok ≈ 2P + 2·L·H·seq (weight term flat, attention term grows with context).
3. **What did I measure?** table above + `results/anatomy.csv` / `kv_cache.csv` / `context.csv`.
4. **What surprised me?** The 1B estimate overshoots by exactly one embed matrix (262M) — Llama-3.2-1B **ties** input/output embeddings, and spotting it from arithmetic alone shows the accounting works. Also Gemma-2-9B's 256k vocab makes embeddings ~1.8B params, nearly 20% of the model.
5. **Bottleneck at long context?** Cache bytes + attention flops, not weights: at 128K the 8B cache (17GB) exceeds its weights (16GB).
6. **What if X?** Halve kv-heads (GQA) → halve cache; int8 cache → halve again; batch=8 → ×8 (recompute any row of `kv_cache.csv` with `kv_bytes()`).
7. **Next?** Quantization (Session 6) attacks exactly these bytes: BF16→INT8→INT4/NVFP4 on this same table.
