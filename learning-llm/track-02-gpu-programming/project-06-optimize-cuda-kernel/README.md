# Project 6 — Optimize a CUDA Kernel (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).
Box: CPU-only, no nvidia-smi/ncu/nsys/nvcc → GPU + profiler rows auto-skipped with DGX runbook printed; every rung has a measured CPU analog.

## Measured (this machine, CPU)
- A (coalescing, equal 524k-element views): strided is **5.9× slower (numpy)** / **7.7× (torch)** — same elements + flops, ~8× the cache lines.
- B (tiling): naive vs blocked-Python tie at 96×96 (**0.93–1.05×** — 73 KB fits in cache, no cliff; the tie *is* the roofline prediction); tile-structured + BLAS tiles hits **42–51 GFLOP/s** vs blas ceiling 95–154 (same structure GPUs run in shared memory + tensor cores).
- C (reduction, 4M fp32): serial Python 0.11 GB/s → numpy 21.8 (**208×**) → tree 7.6 (72× vs serial, extra pass costs) → torch 196 GB/s.
- Artifacts: `results/opt.csv` (13 rows), `results/opt.png`

## The 7 questions
1. Built: 3-rung ladder (strided→contiguous, naive→tiled, serial→tree) + nvidia-smi snapshot + ncu/nsys runbook.
2. Coalescing = adjacent threads touch adjacent bytes (one 128 B transaction, not 32); tiling = stage reuse-blocks in fast memory; tree-reduce = log-depth partials instead of a serial dependency chain.
3. Measured above. 4. Surprise: a fair access-pattern test needs *equal element counts* (naive row-sum vs col-sum confounds pattern with reduction order — hence the contiguous-chunk vs strided-chunk design); tile size barely matters on CPU, decides occupancy on GPU.
5. Bottleneck per rung: DRAM transactions (A), cache reuse absent at small N (B), loop-carried dependency (C serial).
6. Try: tile=8 (granularity overhead) / tile=whole-matrix (no reuse) / include H2D transfer (adds PCIe roof).
7. Next: Session 3 spends these GB/s on convnets (P7: im2col vs Winograd vs cuDNN — same roofline game); on DGX Spark read ncu memory-workload + SM-throughput per kernel.

> New to this? Read [`lecture.ipynb`](./lecture.ipynb) first — intuition, plots, and vocabulary before the code-heavy lab.
