# Project 18 — vLLM Concepts (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU-calibrated sim: prefill 7.6 µs/tok, decode 0.449 ms/step-row; N=48, prompt_med 30, out_med 43, out_max 61)
- Static batching: med TTFT **89.71 ms**, med TPOT **0.61 ms**, makespan **0.326 s**, **5881 tok/s**, mean pad waste **30.5 toks/req**
- Continuous batching: med TTFT **42.20 ms** (~2x), med TPOT **0.47 ms**, makespan **0.245 s** (**x1.33**), **7828 tok/s**, waste **0**
- Burst sweep (`results/sweeps.csv`): speedup stable **1.33 / 1.33 / 1.32** across arrival windows (saturated regime — the win is structural, not burst-luck)
- PagedAttention (`results/blocks.csv`, fp16): contiguous-max **1511.4 kB** vs paged-b16 **995.3 kB** (**x1.52**); shared 32-tok prefix across 4 reqs saves **24.6 kB**
- Block-size sweep waste: b8 **132** → b16 324 → b32 660 → b64 **1364 toks** (halving block size roughly halves fragmentation)
- Figure in `results/sched.png`; asserts pin continuous ≤ static makespan, zero pad waste, paged ≤ contiguous

## The 7 README questions (fill after running)
1. **What did I build?** An event-driven simulator comparing naive static batching vs continuous (iteration-level) batching on a skewed request stream, plus PagedAttention block accounting with prefix sharing.
2. **How does it work?** Costs anchored by real tiny forwards; static waits for full batches and runs every member to max-len (padding tax); continuous prefills on arrival and backfills freed slots each decode step; KV math counts whole blocks per request.
3. **What did I measure?** table above + `results/sched.csv` / `results/blocks.csv` / `results/sweeps.csv`.
4. **What surprised me?** TTFT halves even though TPOT barely moves — the win is *waiting*, not *speed*: static's head-of-line blocking (wait for the last arrival + max-len drain) dominates; and paging's 1.52x comes purely from not reserving the worst case everywhere.
5. **Bottleneck?** Skew: out_max 61 vs med 43 means static burns ~30 pad tokens per request; variable lengths are what make continuous batching pay — uniform workloads wouldn't.
6. **What if X?** Compressing arrivals 30x (0.15→0.005 s) leaves speedup flat at ~1.33 (saturation, not sensitivity); growing blocks 8→64 explodes waste 10x (132→1364) — block size is a memory-vs-table-overhead knob.
7. **Next?** Session 7: multimodality + retrieval + agents (P19 VLM, P20 RAG-from-scratch, P21 agent runtime) — the serving stack from S6 becomes the platform they run on.

> Beginner? Start with [`lecture.ipynb`](./lecture.ipynb) — visuals-first intro, read before `notebook.ipynb`.
