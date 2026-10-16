#!/bin/bash
# Phase 3 quality: perplexity (wikitext-2, c=4096), greedy agreement, needle. Runs after the perf driver finishes.
cd /home/dat/data/labs/model-loading/blog-kv
PY="/home/dat/jupyter-venv/bin/python -I scripts/harness_quality.py"
while pgrep -f run_phase3_perf.sh >/dev/null; do sleep 30; done
M="Qwen3-8B-BF16.gguf Qwen3.6-35B-A3B-UD-Q4_K_M.gguf Qwen3.8-27B-UD-Q4_K_M.gguf"
stray(){ pgrep -af llama-server | grep -v -- "--port 8082" | grep -q . && { echo STRAY; exit 1; }; }
mkdir -p raw/ppl
for m in $M; do for kv in f16 q8_0 q4_0; do
  o=raw/ppl/${m%.gguf}_$kv.log; [ -f $o ] && grep -q "Final estimate" $o && continue
  stray; echo "ppl $m $kv $(date +%T)"
  LD_LIBRARY_PATH=/data/benchmarks/ppl-bin /data/benchmarks/ppl-bin/llama-perplexity -m /data/engines/models/$m -f /home/dat/data/labs/model-loading/out/wiki.test.raw -c 4096 -b 2048 -ngl 99 -fa on -ctk $kv -ctv $kv > $o 2>&1
  grep "Final estimate" $o || echo "  no estimate"
done; done
for m in $M; do for kv in f16 q8_0 q4_0; do
  [ -f raw/p3q_greedy_${m%.gguf}_$kv.json ] || { stray; $PY --model $m --kv $kv --mode greedy; }
done; done
for m in $M; do for kv in f16 q8_0 q4_0; do for c in 8192 32768; do
  tr=20; [ $c = 32768 ] && [ $m = Qwen3.8-27B-UD-Q4_K_M.gguf ] && tr=10
  [ -f raw/p3q_needle_${m%.gguf}_${kv}_c$c.json ] || { stray; $PY --model $m --kv $kv --mode needle --ctx $c --trials $tr; }
done; done; done
echo PHASE3_QUALITY_DONE
