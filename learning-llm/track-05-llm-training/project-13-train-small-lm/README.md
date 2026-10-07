# Project 13 — Train a Small LM (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, real TinyShakespeare 120k chars, vocab 61)
- Analytic param formula matches torch **exactly** on all 3 proxies (asserted in-notebook)
- Proxies trained from scratch (AdamW 3e-4): XS (L2/d64, **109,824** params, 150 it) → val **2.7944**, ppl 16.35, **163k tok/s**; S (L4/d96, **460,416**, 120 it) → val **2.6177**, ppl 13.70, **62k tok/s**; M (L6/d128, **1,204,736**, 80 it) → val **2.6060**, ppl 13.54, **30k tok/s** (init ≈ 4.3 ≈ ln(61) = 4.11)
- 100M/300M/1B-class projections (BPE-scale vocab 32k, ctx 2048): **135.7M** (infer 0.27 GB / train 1.90 GB bf16, ~212 tok/s on this CPU), **369.7M** (0.74 / 5.18 GB, ~78 tok/s), **1343M** (2.69 / 18.81 GB, ~21 tok/s)
- Batch sweep on S (fwd+bwd): B8 **40k** / B16 **50k** / B32 **65k** / B64 **75k tok/s** — still rising, flattening
- Curves in `results/dashboard.png`, tables in `results/configs.csv`, `results/projections.csv`, `results/sweep.csv`
- Offline-safe: download fails → deterministic synthetic fallback (same code path)

## The 7 README questions (fill after running)
1. **What did I build?** Three char-level GPTs (XS/S/M) trained briefly from scratch, a verified param-count formula, and a throughput dashboard projecting 100M/300M/1B memory + tok/s.
2. **How does it work?** Same next-token recipe as P11, repeated at 3 widths; tok/s per config gives a fitted rate (tok/s × params ≈ const, linear-dominated regime); memory = 2 B/param infer, 14 B/param train (bf16 weights + fp32 grad + Adam m,v).
3. **What did I measure?** table above + `results/configs.csv`.
4. **What surprised me?** XS trains at 163k tok/s — the whole run takes ~1 s; and the 1B-class *training* footprint (18.8 GB bf16) already exceeds most single consumer GPUs, while inference (2.7 GB) fits anywhere.
5. **Bottleneck?** Per-token matmuls scale with params (tok/s ≈ 1/P); on this CPU even the 1B-class projects to ~21 tok/s — training needs a GPU, not patience.
6. **What if X?** 8x batch (8→64) buys <2x tok/s here — the CPU is compute-bound, not launch-bound; on a GPU the curve looks the opposite (kernels starve at small batch).
7. **Next?** Teach the S proxy a new skill without full retraining (P14: SFT vs LoRA vs QLoRA vs full FT).

> Beginner? Start with [`lecture.ipynb`](./lecture.ipynb) — visuals-first intro, read before `notebook.ipynb`.
