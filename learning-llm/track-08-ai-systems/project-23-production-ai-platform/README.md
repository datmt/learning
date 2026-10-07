# Project 23 — Production AI Platform (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, TestClient — no ports, no network, ~4 s wall)
- Backends: rule-based LLM stub (8 ms prefill) / VLM stub (5 ms, closed tag vocabulary) / shared-prefill batch (8 ms + 1 ms/item)
- Contracts: missing/wrong token → 401; 7 sends on a 5-budget key → 5×200 + 2×429; exact repeat → `cached=true`, same answer; batch-of-4 measured **12.13 ms** vs 4× single **32.4 ms** (shared prefill wins); unknown tag → fallback string; `/metrics` + `/gpu` (labeled `simulated`, CUDA absent) shaped correctly
- Ledger (`results/platform.csv`, 22 rows): mix phase 14 rows (12 ok, 1×401 unauthenticated probe, 1×429 tight-budget shed, mean 8.38 ms); burst phase 8 threads × 300 ms-hold vs 4 slots → **4×200 + 4×429, zero 500s** (mean 164 ms, barrier-aligned)
- Observability: per-route counters + mean latencies (`/metrics`), per-request trace ids + span log, cache occupancy; figure in `results/platform.png`

## The 7 README questions (fill after running)
1. **What did I build?** A gateway pipeline in one FastAPI app (TestClient-driven): auth → ratelimit → admission (bounded 4) → exact-prompt cache → router (LLM/batch/VLM) → stub backends, with metrics + trace log + GPU-monitor stub.
2. **How does it work?** Each layer fails closed with a shaped body: bad token 401 before any work, exhausted budget or full queue 429 (shed, never 500); hits skip the backend entirely; batch amortizes one 8 ms prefill over k items; every admitted *and* rejected request emits a metric sample + trace span.
3. **What did I measure?** table above + `results/platform.csv` / `results/summary.csv`.
4. **What surprised me?** The burst sheds exactly 4/8: with a barrier-aligned start, the semaphore is a perfect guillotine — 4 holders, 4 rejections, no partial states. Overload behavior is fully determined by the admission bound, not by timing luck.
5. **Bottleneck?** Single-prompt prefill (8 ms serial cost) — batching is the only layer that lowers it per item (12.13 ms for 4 vs 32.4 ms serial); cache only helps repeats, rate limits only protect, neither accelerates.
6. **What if X?** Budget 5→0 turns the mix phase into all-429 (protection becomes outage — budgets need headroom); slots 4→8 absorbs the burst (no shed, 2× latency tail); cache off loses the repeat row (recompute every time).
7. **Next?** P24 (capstone) serves its report/tracks through this exact shape: `/report` + `/tracks` behind auth → cache → metrics; swap stubs for vLLM (P18 scheduler in `/batch`), Redis (`CACHE`), OTel (`TRACE`), DCGM (`/gpu`).

> **New here?** Start with [lecture.ipynb](lecture.ipynb) — a beginner-friendly companion that teaches the fundamentals first. Read it before opening `notebook.ipynb`.
