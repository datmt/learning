#!/bin/bash
cd /home/dat/data/labs/model-loading/blog-kv
for m in Qwen3-8B-BF16.gguf Qwen3.6-35B-A3B-UD-Q4_K_M.gguf Qwen3.8-27B-UD-Q4_K_M.gguf gemma3-12b-Q4_K_M.gguf gpt-oss-20b-MXFP4.gguf; do
  for P in 4096 8192 16384 32768 65536 131072; do
    [ $m = Qwen3-8B-BF16.gguf ] && [ $P -gt 32768 ] && continue
    PP=$P; { [ $m = gemma3-12b-Q4_K_M.gguf ] || [ $m = gpt-oss-20b-MXFP4.gguf ]; } && [ $P -eq 131072 ] && PP=130816
    [ -f raw/p2_${m%.gguf}_p${PP}_f16.json ] && continue
    pgrep -af llama-server | grep -v -- "--port 8082" | grep -q . && { echo STRAY; exit 1; }
    /home/dat/jupyter-venv/bin/python -I scripts/harness_speed.py --model $m --prompt $PP
  done
done
echo PHASE2_DONE
