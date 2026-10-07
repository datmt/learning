# Project 5 — First CUDA Kernels (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).
Box: CPU-only, no nvcc/numba → GPU launches auto-skipped; kernel *logic* verified via CPU thread-simulator; Triton sources defined (import OK) but not launched.

## Measured (this machine, CPU)
- Reference kernels (torch) match NumPy on all 5 ops: vecadd, relu, sigmoid, softmax, reduce.
- CPU thread-simulator matches torch on all 5 ops (logic correct) — but vecadd n=200k costs **665 ms vs torch 0.22 ms (~3000× slower)**; relu 691 ms vs 0.38 ms. Same math, serial Python ⇒ speed is parallelism, not formula.
- Backend table (`results/backends.csv`): torch-reference ran, thread-simulator ran, triton defined/skipped (needs CUDA), numba missing, cpp-inline skipped (needs CUDA + nvcc).
- Artifacts: `results/backends.csv`, `results/kernels.csv`, `results/kernels.png`

## The 7 questions
1. Built: 5 ops as torch references + CPU grid/block simulators + real Triton/numba/cpp-inline sources with guarded launches.
2. `i = blockIdx.x*blockDim.x + threadIdx.x` maps one thread → one element; `grid = ceil(n/block)`; `if i < n` drops the over-provisioned tail block.
3. Measured above; GPU rows pending DGX Spark. 4. Surprise: ~3000× simulator slowdown running *identical* math.
5. Bottleneck: zero parallelism (serial Python loops) + per-iteration dispatch — GPUs remove both with thousands of threads.
6. Try: block=1 (launch overhead dominates) / block=2048 (exceeds 1024-thread HW limit, launch fails) / n ∤ block (guard saves you).
7. Next: P6 optimizes memory access + reduction (fuse softmax passes, tile matmul, tree-reduce); then run the Triton/numba/cpp cells on DGX Spark.

> New to this? Read [`lecture.ipynb`](./lecture.ipynb) first — intuition, plots, and vocabulary before the code-heavy lab.
