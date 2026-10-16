# KV cache on GB10: progress (2026-10-16)

Plan: `PLAN.md`. Status: Phase 0 done. Phases 1 and 2 data collected and analysed, **awaiting review**. Phases 3-5 not started.

## Done
- **Phase 0 (theory):** `theory.json`, `scripts/gguf_meta.py`, `scripts/kv_theory.py`.
- **Phase 1 (memory vs context):** 155 runs in `raw/`, server logs in `logs/`.
  - Harness: `scripts/harness.py` (one fresh `llama-server` per point, `-lv 4`, memory snapshots, validated response).
  - Driver: `scripts/run_phase1.sh`. Analysis: `scripts/analyze_phase1.py`.
  - Outputs: `tables/phase1_theory_vs_measured.md`, `tables/phase1_alloc_cfg_b.md`, `tables/phase1_runs.csv`, `images/phase1_memory_vs_context.png`.
- Production models (`qwen3.6-35b-a3b`, `qwen3.8-27b`) were unloaded for the run and reloaded afterwards. `llama.sh status` shows both loaded. The first reload of the 27B timed out at 300 s; the second attempt worked.

## Findings (measured from llama.cpp's own KV buffer lines)
| model | theory B/token | measured B/token | note |
|---|---|---|---|
| Qwen3-8B | 147456 | 147456 | match |
| Qwen3.6-35B-A3B | 20480 | 20480 | match, KV in 10 of 40 layers |
| Qwen3.8-27B | 69632 | 65536 | theory 5.9% high: KV allocated for 16 layers, not 17 (cause unidentified) |
| Gemma3-12B | 65536 | 65536 | global layers only |
| gpt-oss-20b | 24576 | 24576 | global layers only |

- KV is allocated in full at load. It is sized by `-c`, padded up to a multiple of 256 cells, not by tokens in use.
- Sliding-window cache = window + 512 micro-batch, rounded up to 256 cells: Gemma3 1536 cells (480 MiB), gpt-oss 768 cells (18 MiB). Fixed at every context.
- Recurrent state is fixed per sequence: 62.8 MiB (35B), 149.6 MiB (27B).
- KV at 128K: 35B 2.5 GiB, 27B 8.0 GiB, Gemma3-12B 8.0 GiB, gpt-oss-20b 3.0 GiB.
- Process GPU memory from the log's breakdown line tracks KV well. System `MemAvailable` deltas are noisy (page cache; no sudo to drop it), so they are secondary.

## Deviations and failures (kept in `raw/`)
- Config (a) at 64K/128K has 1 repetition instead of 3.
- Config (a) at prompt 131072 with `-c` 131072 failed (request too large), for the Qwen models, Gemma3 and gpt-oss. Not a memory failure.
- Gemma3-12B and gpt-oss-20b cannot take 131072 + 64 tokens (trained limit 131072). Re-run at prompt 131000, 3 reps each. Gemma3 reports +1 token (BOS), harmless.
- Qwen3-8B stops at 32K (trained context 40960).

## Not done / next
- Your review of Phases 1 and 2.
- Experiment B architecture table, not built.
- Gemma4 (optional), not run.
- Phases 3-5 (KV precision, concurrency, write-up). Phase 3 needs an out-of-tree `llama-perplexity` build.
- Open from Phase 0: which layer in Qwen3.8-27B has `attn_k` but no KV cache; which gpt-oss layers are windowed (the 12 global layers match the even/odd 1:1 assumption).

## Phase 2 (speed vs context), done 2026-10-15
- Harness `scripts/harness_speed.py`, driver `scripts/run_phase2.sh`, bench driver `scripts/run_bench_depth.sh`, analysis `scripts/analyze_phase2.py`.
- Method: fresh server per (model, prompt), f16 KV, 1 slot, `-c` = prompt + 128 + 8, prompt cache off, 1 warmup + 5 streamed reps, 128 generated tokens. Client per-token timestamps cross-checked against server timings (decode tok/s agree to 0.01). Gemma3/gpt-oss top step uses a 130816-token prompt (trained limit). No errors in 28 points.
- Outputs: `tables/phase2_speed.md`, `tables/phase2_bench_depth.md`, `tables/phase2_decode_slope.md`, `images/phase2_speed_vs_context.png`.
- Decode tok/s, 4K -> max context (measured): Qwen3-8B 14.6 -> 11.5 (32K, 79%); Qwen3.8-27B 12.3 -> 8.7 (70%); Qwen3.6-35B-A3B 62.3 -> 36.3 (58%); Gemma3-12B 24.7 -> 12.6 (51%); gpt-oss-20b 72.7 -> 36.8 (51%).
- Prefill tok/s falls with context on all models (e.g. gpt-oss 4765 -> 2216, Qwen3.8-27B 830 -> 595). TTFT at 128K: 27B 221 s, 35B 70 s, Gemma3 90 s, gpt-oss 59 s. ITL p95 stays within a few percent of the mean.
- llama-bench decode-at-depth agrees with the server within about 1-2% for Qwen3-8B and Qwen3.8-27B, but is 3-6% higher for Qwen3.6-35B-A3B (e.g. 65.99 vs 62.32 at 4K). Unexplained; possibly server overhead on the MoE model. Not investigated.
- Inferred: linear fit of ms/token vs context gives slopes 1.05x to 1.28x the theory KV-read cost at 273 GB/s (global-layer bytes only). The excess is plausible compute/overhead and the SWA cache, but this is not demonstrated.
- Caveat: decode is not assumed linear; the fit uses 4-6 points per model. Prefill speed varied about 12% between sessions in earlier work; all Phase 2 runs were done in one session, models sequentially, not interleaved.
- Next: your review of Phase 2, then Phase 3 (KV precision; needs an out-of-tree llama-perplexity build for the quality test).

## Phase 3 (KV precision), done 2026-10-16
- Scripts: `run_phase3_perf.sh`, `run_phase3_quality.sh`, `harness_quality.py`, `analyze_phase3.py`. llama-perplexity build persisted at `/data/benchmarks/ppl-bin/`.
- KV buffer (measured, from server log): q8_0 -46.9%, q4_0 -71.9% vs f16 for every model (as expected from block sizes 34/32 and 18/32 bytes per element vs 2).
- Speed: decode at max context vs f16: Qwen3-8B (32K) q8_0 +5.7%, q4_0 +8.9%; Qwen3.8-27B (128K) +3.6% / +6.2%; Qwen3.6-35B-A3B (128K) -2.0% / +0.1% (within noise). Prefill is slower: -1% to -10% (largest on Qwen3-8B, -9.6% at 32K q8_0). 5 reps, one session, not interleaved.
- Perplexity (wikitext-2, c=4096, full test set): q8_0 delta between -0.07% and +0.14%; q4_0 +0.05% (8B), +0.19% (35B), +0.64% (27B, +0.04 on 6.37). Reported errors are per-config standard errors; the runs share the same text so the paired difference is tighter than they suggest, but no paired error was computed.
- Greedy agreement vs f16 (8 prompts, 256 tokens): q8_0 identical outputs 5/8 (8B), 4/8 (27B), 1/8 (35B); q4_0 0/8 for all. Divergence is expected with any numerical change; it does not by itself mean worse quality.
- Needle (random 6-digit code, 5 depths, 20 trials/cell; 27B at 32K 10 trials): 100% for 8B and 35B at 8K and 32K for all formats; 27B 8K 100/99/99%; 27B 32K 86/88/88% (f16/q8/q4), so the 27B's 32K misses are not caused by quantisation (misses are mostly at depth 0-25%). CIs overlap everywhere.
- Not done: mixed K/V formats, 128K needle, perplexity at longer context, paired-difference stats, cause of the 27B 32K misses. Not investigated: why prefill slows with quantised KV.
- Next: your review of Phases 1-3, then Phase 4 (concurrency and capacity).

## Session wrap (2026-10-16) — start here next session
- State: Phases 0-3 done; 1-3 awaiting user review (no feedback yet). Prod models loaded (35B ctx 131072 x4 slots, 27B ctx 65536 x2) and no stray servers. Nothing running.
- Next: **Phase 4 (concurrency x context capacity)** per PLAN.md. User has said there is no production workload, so unload/reload prod models without asking; still run one benchmark at a time, check strays with `pgrep -x llama-server` (never `pkill -f`, it kills your own shell), and reload + `llama.sh status` at the end.
- Reusable: `scripts/harness_speed.py` (server per point, streaming, validated), `harness_quality.py`, drivers `run_phase*.sh` (skip existing raw files; abort on STRAY). For Phase 4 note the earlier 500 "context size exceeded" issue in `../blog/` notes; hybrid models: recurrent state per sequence (62.8 MiB 35B, 149.6 MiB 27B).
- Lessons: validate the scoring on the f16 baseline before running quantised variants (the needle cap bug); killing a harness with SIGTERM leaves an orphan llama-server, kill it too.
- Still open: Experiment B architecture table, Gemma4, mixed K/V, paired perplexity stats, 128K needle, Phase 5 write-up (README via build_readme.py pattern), publishing to datmt/learning only after review.

## Phase 4 (concurrency x context), done 2026-10-16
- Scripts: `harness_conc.py` (fresh server per point, `-np N --kv-unified`, N simultaneous distinct prompts, memory sampled 0.5 s), `run_phase4.sh`, `repro_500.py`, `analyze_phase4.py`. Pilot point kept as `raw/p4_..._pilot.json` (not in the tables).
- 44 points, all requests 200 with correct token counts. Tables `tables/phase4_matrix.md`, `phase4_kv_check.md`, `phase4_runs.csv`; plot `images/phase4_concurrency.png` (not yet looked at).
- Measured: KV bytes/cell 147456 (8B), 20480 (35B), 65536 (27B) at every point (cells padded to 256). Recurrent state 62.8 / 149.6 MiB per sequence, linear in N (27B at N=8: 1.2 GiB). Largest point 27B 8x64K: 32 GiB KV, wall 1763 s, TTFT mean 810 s / max 1747 s; memory was not the limit, latency was. Per-request decode falls with N (35B at 4K: 62.8 tok/s at N=1, 12.1 at N=8) but is measured while other requests prefill, so it is not an aggregate-throughput figure.
- 500 "Context size has been exceeded" (prod flags: 35B, -c 131072, 4 slots, unified, cache-reuse 256), one run per trial: A/E 130K finished then 4x8K: all 200; B 4x8K arriving during the 130K prefill: 4 of 5 got 500 (including the 130K request); C 4x32K: all 200; D 4x40K (160K > pool): all 4 got 500. Log: "failed to find free space in the KV cache" then all in-flight requests fail, including one already at 129,996 tokens. Inferred: the server admits more concurrent tokens than the pool holds, then fails everything in flight. Not demonstrated by a code change; not repeated.
- Prod models reloaded (35B ctx 131072 x4, 27B ctx 65536 x2) and verified; no stray servers.
- Not done: Phase 4 with quantised KV or other pool sizes, repeats of the repro trials, a look at the plot.

## Session wrap (2026-10-16, after Phase 4) - start here next session
- State: Phases 0-4 done; 1-4 awaiting user review (no feedback yet). Nothing running.
- Next: Phase 5 write-up (README via a build_readme.py pattern, Experiment B architecture table), then publish to datmt/learning only after review.
- Still open: Gemma4, mixed K/V, paired perplexity stats, 128K needle, 27B 17th attn_k layer, repro-500 repeats.

## Phase 5 (write-up), drafted 2026-10-16
- `README.template.md` -> `README.md` via `scripts/build_readme.py .` (injects `{{F:tables/x.md}}` verbatim; 11 tables). Sections: KV explainer, models/architecture table (Experiment B, Gemma4 theory-only), theory formula, memory vs context, speed vs context, KV quantisation, concurrency + 500 error, guidance, limits, reproduce.
- Not yet done: user review; fact-check pass of every sentence against tables; publish to datmt/learning (`kv-cache-gb10/`) only after review. Links `../blog/...` in README must be fixed or the folders published together.
- Prod models loaded, nothing running.

## Review round 1 (external comment), 2026-10-16 — local only, NOT yet pushed
- 27B 16-vs-17 resolved: 17th `attn_k` is the unused MTP block 64 (`nextn_predict_layers=1`, log "n_layer=64, n_layer_all=65", "unused tensor blk.64.nextn"). `kv_theory.py` now excludes trailing nextn layers; theory.json/phase1 table regenerated, all 5 models match.
- Added `scripts/analyze_phase4_extra.py` -> `tables/phase4_latency.md` (TTFT min/median/max, aggregate prefill and total tok/s) and `phase4_operating_points.md`; README §7.1/§8 updated (quick picks, aggregate throughput falls with N at long context).
- NOT done (needs GPU, ask first): repeats of concurrency points (n=1), a different prefill/batch setting to separate capacity from scheduler effects.
