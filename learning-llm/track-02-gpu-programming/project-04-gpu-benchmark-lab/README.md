# Project 4 — GPU Benchmark Lab (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).
Box: CPU-only (torch 2.10.0, no CUDA) → `torch-gpu` rows auto-skipped; rerun on DGX Spark to fill them.

## Measured (this machine, CPU, synthetic fp32)
- vecadd: torch-cpu peaks **420 GB/s** @ n=1M (cache-resident); @ 4M (16 MB, DRAM): 52 GB/s torch vs 37.6 numpy. @ 10k: ~1–2 µs (pure overhead, not bandwidth).
- reduce: torch-cpu **400 GB/s** @ 1M; @ 8M: 58 torch vs 21.8 numpy.
- matmul 1024 (2.1 Gflop): numpy **383 GFLOP/s** vs torch-cpu 151 (BLAS-backend gap on this box).
- softmax 1024×1024: torch 46 GB/s vs numpy 12.5.
- what-if fp32→fp64 vecadd: 0.29 ms → 0.72 ms (~2× bytes ⇒ ~2× time).
- Artifacts: `results/bench.csv`, `results/bench_runtime.png`, `results/bench_throughput.png`

## The 7 questions
1. Built: 4-op (vecadd/matmul/reduce/softmax) × NumPy/torch-CPU/torch-GPU harness with warmup + cuda-sync + median timing.
2. Correct GPU timing = warmup (caches/autotune) + `torch.cuda.synchronize()` around the timer (launches are async) + median over repeats; intensity = flops/byte decides bound: vecadd 0.08 (memory) … matmul-1024 ~171 (compute).
3. Measured above. 4. Surprise: mid-size GB/s numbers exceed DRAM ceilings (cache effects) — bandwidth claims need streaming sizes; numpy beat torch on big matmul here.
5. Bottleneck: memory-bound ops chase GB/s, matmul chases flops; at small n both chase overhead (~µs).
6. Try: fp64 (2× bytes ⇒ ~2× on memory-bound, ~flat on matmul); include H2D transfer (expect GPU to lose small-n badly).
7. Next: P5 writes these ops as raw CUDA kernels; P6 climbs the roofline (coalesce → tile → tree-reduce).

> New to this? Read [`lecture.ipynb`](./lecture.ipynb) first — intuition, plots, and vocabulary before the code-heavy lab.
