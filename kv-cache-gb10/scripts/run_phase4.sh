#!/bin/bash
cd /home/dat/data/labs/model-loading/blog-kv
for m in Qwen3-8B-BF16.gguf Qwen3.6-35B-A3B-UD-Q4_K_M.gguf Qwen3.8-27B-UD-Q4_K_M.gguf; do
 for P in 4096 16384 32768 65536; do
  for N in 1 2 4 8; do
    [ $m = Qwen3-8B-BF16.gguf ] && [ $P -gt 32768 ] && continue
    [ -f raw/p4_${m%.gguf}_n${N}_p${P}_f16.json ] && continue
    pgrep -af llama-server | grep -v -- "--port 8082" | grep -q . && { echo STRAY; exit 1; }
    /home/dat/jupyter-venv/bin/python -I scripts/harness_conc.py --model $m --n $N --prompt $P
  done
 done
done
echo PHASE4_DONE
