# Project 17 — Mini LLM Server (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, TinyGPT L2/d64 served via FastAPI TestClient, no ports)
- Contracts: `/health` ok; `/generate` 8 chars in **6.3 ms**; empty prompt → **400**; `max_tokens=1000` → clamped to **64**; `/generate_stream` yields **8 token frames + DONE**; `/batch` 3 prompts in 11.5 ms (**2089 tok/s**)
- Single (8 sequential) **1356 tok/s** vs batch-8 **2030 tok/s** (**x1.50**) — one shared prefill amortizes
- Overload: 8 threads vs 4 in-flight slots → **4×200 + 4×429, zero 500s** (backpressure, not OOM)
- Metrics endpoint: 14 reqs, med lat **10.25 ms**, med TTFT-proxy **0.69 ms**, med **1462 tok/s**, 4 rejected
- Batch sweep (`results/batch_sweep.csv`): per-prompt cost B1 **14.1 ms** → B2 11.0 → B4 8.4 → B8 **5.7 ms** (throughput 1133 → 2784 tok/s)
- Queue ablation: bound 4 sheds **4/8** burst arrivals, bound 64 sheds **0/8**
- Figures in `results/server.png`; serve for real with `uvicorn notebook:app --port 8000`

## The 7 README questions (fill after running)
1. **What did I build?** A real FastAPI LLM server (health / generate / SSE stream / batch / metrics) serving the P16-class TinyGPT, tested entirely through TestClient with contract, bench, and overload tests.
2. **How does it work?** Parse → clamp → tokenize → greedy decode → detokenize → metrics; batch stacks prompts for one shared prefill; a semaphore sheds past `MAX_INFLIGHT` with 429; SSE yields one frame per token.
3. **What did I measure?** table above + `results/server.csv` / `results/batch_sweep.csv`.
4. **What surprised me?** Batching 8 only buys x1.5 here — the toy prefill is tiny next to decode, so amortization headroom is small; and exactly half the racers shed (4/4), showing the semaphore is a hard wall, not a queue.
5. **Bottleneck?** Per-row greedy decode loop (no KV-cache, no continuous batching) — TTFT is fine, but decode never overlaps across requests; P18 simulates removing exactly that.
6. **What if X?** B1→B8 cuts per-prompt cost 2.5x (14.1→5.7 ms); queue bound 4→64 turns 4 sheds into 0 — unbounded queues just move the failure to latency/OOM, which is why the 429 exists.
7. **Next?** Replace the lock-step batch loop with iteration-level scheduling + paged KV (P18: continuous batching, TTFT/TPOT, PagedAttention).

> Beginner? Start with [`lecture.ipynb`](./lecture.ipynb) — visuals-first intro, read before `notebook.ipynb`.
