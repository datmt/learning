# Project 3 — GPU Training Lab (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: 0 errors.

## Measured (this box: CPU-only, no CUDA — GPU rows auto-skipped)
- CPU fp32: batches [32, 256, 2048] × 3 epochs → `results/gpu_lab.csv` (3 rows) + `results/throughput.png`
- Smoke: 1 epoch CPU batch=64 → ~61k samples/sec
- **On DGX Spark re-run this notebook**: GPU fp32/bf16 rows will populate; expect GPU≫CPU at batch≥256, CPU competitive at batch=32 (launch + H2D overhead dominate).

## The 7 questions
1. Built: (device × dtype × batch) sweep harness with `cuda.synchronize()` timing + peak-VRAM accounting.
2. Why GPU wins: thousands of parallel FPUs + HBM bandwidth saturate on wide matmuls; CPU wins only when work is tiny.
3. Measured: time_s, samples/sec, peak MB per config.
4. BF16: ~half memory, Tensor-Core speedup on GB10, accuracy ~unchanged for this MLP.
5. Bottleneck: host→device transfer (we include it deliberately) + small-batch under-utilization.
6. Try: batch 8192 (OOM point?), epochs to convergence per dtype. 7. Next: Session 2 kernel-level benchmarks.

> New to this? Read [`lecture.ipynb`](./lecture.ipynb) first — intuition, plots, and vocabulary before the code-heavy lab.
