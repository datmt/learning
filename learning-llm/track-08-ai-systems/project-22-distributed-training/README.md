# Project 22 — Distributed Training (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, single-proc emulation — no live process group, gloo-safe)
- Ring simulator: 4 emulated ranks × 1024 floats, max err **0.0** vs `torch.sum` (scatter-reduce + allgather contract holds)
- Probes: local memcpy **7.29 GB/s** (link ceiling, not link BW); MLP step median **1.12 ms** (256-batch, warmup dropped)
- DDP-equivalence: mean(micro-grads) vs large-batch grad max err **≤6.6e-09** per tensor (averaging == big batch, fp32 reassociation only)
- Strategy table @100M fp32 (`results/dist.csv`): DDP 400.0 MB/rank + 700.0 MB traffic/step vs FSDP 50.0 MB/rank + 1050.0 MB/step vs TP=8 50.0 MB + 100.0 MB vs PP=4 100.0 MB + 12.5 MB
- Scaling @100M (`results/scale.csv`): NVLink eff 1.0 → 0.9978 (N=2) → 0.9846 (N=8, speedup 7.88); PCIe eff 0.9436 (N=2) → 0.7047 (N=8, speedup 5.64); PP bubble (p=4,m=8) = 0.375
- Figure in `results/dist.png` (efficiency curves + per-step traffic bars)

## The 7 README questions (fill after running)
1. **What did I build?** A single-process distributed-training lab: exact param/byte accounting per strategy (DDP/FSDP/TP/PP), a verified ring all-reduce simulator, a DDP gradient-equivalence proof, and a calibrated strong-scaling model (measured compute + analytic comm).
2. **How does it work?** Ring traffic per rank is `2(N-1)/N·B`; efficiency is `T1/(N·T_N)` with `T_N = T_compute/N + T_comm`; FSDP shards to `P/N` bytes at ~1.5× comm, TP shards with 2 all-reduces/layer, PP pays a `(p-1)/m` bubble instead of gradient traffic.
3. **What did I measure?** table above + `results/dist.csv` / `results/scale.csv`.
4. **What surprised me?** At 100M params on NVLink, N=8 keeps 0.98 efficiency — comm (0.8 ms) is invisible next to compute (53 ms); the same step on PCIe drops to 0.70. Bandwidth, not rank count, is the scaling lever at this size.
5. **Bottleneck?** Comm bytes per step: FSDP moves the most (1050 MB, gather+reduce), PP the least (12.5 MB p2p) — PP trades traffic for bubble idle time (0.375 here), so micro-batch count is its scaling lever.
6. **What if X?** N 8→2 halves ring traffic per rank (700→400 MB DDP); PCIe→NVLink recovers ~0.28 efficiency at N=8; tiny models (0.3M) scale worse than 1B ones at fixed N because compute shards but latency doesn't.
7. **Next?** P23 puts a model behind a production API (this lab's `eff` math prices its batch/worker sizing), and P24 wires the whole curriculum into the sports-AI capstone.

> **New here?** Start with [lecture.ipynb](lecture.ipynb) — a beginner-friendly companion that teaches the fundamentals first. Read it before opening `notebook.ipynb`.
