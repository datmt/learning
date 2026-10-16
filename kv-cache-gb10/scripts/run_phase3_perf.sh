#!/bin/bash
# Phase 3 perf/memory: Phase 2 harness with quantised KV. f16 baselines come from Phase 2 raw files (same harness).
cd /home/dat/data/labs/model-loading/blog-kv
for m in Qwen3-8B-BF16.gguf Qwen3.6-35B-A3B-UD-Q4_K_M.gguf Qwen3.8-27B-UD-Q4_K_M.gguf; do
  for P in 8192 32768 131072; do
    [ $m = Qwen3-8B-BF16.gguf ] && [ $P -gt 32768 ] && continue
    for kv in q8_0 q4_0; do
      [ -f raw/p2_${m%.gguf}_p${P}_$kv.json ] && continue
      pgrep -af llama-server | grep -v -- "--port 8082" | grep -q . && { echo STRAY; exit 1; }
      /home/dat/jupyter-venv/bin/python -I scripts/harness_speed.py --model $m --prompt $P --kv $kv
    done
  done
done
echo PHASE3_PERF_DONE
