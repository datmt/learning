# Project 21 — Agent Runtime (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, stdlib + numpy, 10 tasks × 5 configs = 50-row ledger, ~5 s wall)
- Tools: calculator (AST allowlist) / fs_write+read (realpath sandbox) / python (no-import box, `out=` contract) / kv_put+get+sum / http_get (offline mock) / slow_echo (0.4 s) / flaky_add (fails first 2) — all unit tests pass (injection, traversal, 404, miss-key all fail closed)
- Success rate (`results/agent.csv`): full **0.9** (9/10, only the hallucinated `teleport` tool fails — contained) vs no-retry **0.8** (flaky dies) vs no-memory **0.8** (compare dies: blind (5+5)=10 guess without alpha) vs tight-timeout **0.8** (0.4 s job vs 0.1 s budget) vs strict **0.6** (denying writes kills fs + multistep + perm — blast radius = all writers)
- Guards metered: full spends **5 retries** (flaky 2 + badtool 3); tight-timeout trips **≥1 timeout**; strict trips **3 denials** (fs, multistep, perm)
- Ledger in `results/tasks.csv` (config, task, success, steps, retries, timeouts, denials, wall_ms, answer); figure in `results/agent.png`; side effects in `results/sandbox/`

## The 7 README questions (fill after running)
1. **What did I build?** A guarded agent loop: rule-based planner (documented LLM stand-in) → permission gate → thread-pool tool exec with per-call timeout → observe → scratchpad → retry-with-backoff, over 10 tasks × 5 runtime configs.
2. **How does it work?** Planners are stage machines on the full step count but only *see* a memory window (full history vs last-obs); failures classify as error/timeout/denied and retry the same action up to R times; checkers verify answers against ground truth.
3. **What did I measure?** table above + `results/tasks.csv` / `results/agent.csv`.
4. **What surprised me?** Strict permissions score 0.6, not 0.9-minus-one: denying `fs_write` kills three task classes (fs, multistep, perm), because the multistep chain *persists* mid-plan. Permission blast radius is measured in writers, not in the one task you aimed at.
5. **Bottleneck?** The loop's weakest guard per workload: flaky tools need retries (0.9→0.8 without), distant facts need memory (compare needs alpha from 2 obs back), slow tools need timeout budget, side effects need policy — reliability lives in the loop, not the prompt.
6. **What if X?** Retries 3→0 loses flaky (transient errors become terminal); memory full→last-obs loses compare (10.0 vs 12.0 — a confident wrong answer, the most dangerous kind); timeout 2.0→0.1 s loses slow (4 attempts × 0.1 s, all timeout); open→strict loses every writer.
7. **Next?** Session 8 (P22 distributed training → P23 production platform → P24 sports-AI capstone) puts P19's vision, P20's retrieval, and this loop behind one API: detect → track → retrieve → analyze.

> **New here?** Start with [lecture.ipynb](lecture.ipynb) — a beginner-friendly companion that teaches the fundamentals first. Read it before opening `notebook.ipynb`.
