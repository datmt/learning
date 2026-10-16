#!/bin/bash
# Phase 1 driver: sequential, one server at a time. Skips points whose raw JSON exists.
cd /home/dat/data/labs/model-loading/blog-kv
PY="/home/dat/jupyter-venv/bin/python -I scripts/harness.py"
run(){ # model ctx prompt rep cfg
  id="p1${5}_${1%.gguf}_c${2}_p${3}_f16_r${4}"
  [ -f raw/$id.json ] && return
  pgrep -af "llama-server" | grep -v -- "--port 8082" | grep -q . && { echo "STRAY llama-server, abort"; exit 1; }
  $PY --model $1 --ctx $2 --prompt $3 --rep $4 --tag p1$5 --gen 64
}
declare -A MAXC=( [Qwen3-8B-BF16.gguf]=40960 )
for m in Qwen3-8B-BF16.gguf Qwen3.6-35B-A3B-UD-Q4_K_M.gguf Qwen3.8-27B-UD-Q4_K_M.gguf gemma3-12b-Q4_K_M.gguf gpt-oss-20b-MXFP4.gguf; do
  big=${MAXC[$m]:-131072}
  for P in 4096 8192 16384 32768 65536 131072; do
    [ $P -ge $big ] && [ $m = Qwen3-8B-BF16.gguf ] && continue
    [ $P -ge 131072 ] && cap=131072 || cap=$((P+64))
    for r in 1 2 3; do
      run $m $((P+64)) $P $r b          # cfg b: -c = prompt+gen
      [ $P -le 32768 -o $r -eq 1 ] && run $m $big $P $r a   # cfg a: -c fixed at max
    done
  done
done
echo PHASE1_DONE
