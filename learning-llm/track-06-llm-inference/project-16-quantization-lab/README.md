# Project 16 — Quantization Lab (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, real TinyShakespeare 1.1M chars, vocab 65)
- Baseline TinyGPT L2/d64 (**112,512** params, 60 iters): ref val CE **3.3136**, ppl 27.48 (init ≈ 4.17 ≈ ln(65))
- 7-format frontier (`results/quant.csv`): FP32 **0.450 MB** / CE 3.3095; BF16 **0.225** / 3.3188 (dCE +0.005); FP16 **0.225** / 3.3083; FP8-E4M3 **0.113** / 3.3072; INT8 **0.113** / 3.3163 (dCE +0.003); INT4-g32 **0.056** / 3.312; NVFP4-sim(g16) **0.056** / 3.3058
- Reconstruction-error ladder `werr` (mean abs): BF16 **1.6e-4**, FP16 **2e-5**, INT8 **7.8e-4**, FP8 **2.6e-3**, INT4-g32 **9.8e-3**, NVFP4-sim **8.7e-3**
- Granularity sweep (`results/quant_groups.csv`, werr): per-tensor **0.0141** → g128 **0.0114** → g32 **0.0098** → g16 **0.0087**; g32+clip99 **0.0098** (clip didn't help — no big outliers at 60 iters)
- CPU fwd latency ≈ **0.9–1.0 ms** (b8×ctx64) all formats — fake-quant dequantizes in fp32, so this column measures honesty, not kernel speed
- Curves in `results/quant.png`, unit tests assert int8<g32 err, g16<=g128 err, bf16<int8 err

## The 7 README questions (fill after running)
1. **What did I build?** A fake-quant lab: same 112k-param TinyGPT scored in 7 numeric formats (memory + val CE/ppl + weight-error + CPU latency) plus an INT4 granularity sweep.
2. **How does it work?** Affine fake-quant `q = clamp(round(w/s))`, `w' = s·q` (symmetric; per-tensor for INT8, per-group for INT4); float formats are cast round-trips; each clone is re-scored on val CE.
3. **What did I measure?** table above + `results/quant.csv` / `results/quant_groups.csv`.
4. **What surprised me?** At 60 iters all val-CE deltas sit inside batch noise (±0.02) — the *weight-error* ladder carries the real signal (per-tensor 0.014 → g16 0.0087). Quality metrics need a trained model; error metrics don't.
5. **Bottleneck?** Outliers: one global INT4 scale is set by the largest weight, starving small ones — hence groups. Memory math is exact (4→0.5 B/param = 8x), quality math needs the granularity term.
6. **What if X?** g128→g16 cuts werr ~25% (0.0114→0.0087) at the cost of 8x more scales (still negligible bytes); percentile clipping buys nothing here because brief training grew no outliers.
7. **Next?** Serve the BF16/INT8 model behind an API and stop paying full latency per request (P17: batching + streaming + queues), then schedule many requests the vLLM way (P18).

> Beginner? Start with [`lecture.ipynb`](./lecture.ipynb) — visuals-first intro, read before `notebook.ipynb`.
