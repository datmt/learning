#!/usr/bin/env bash
# Aggregate throughput vs number of parallel streams (batched decoding).
B=/data/engines/llama.cpp/build/bin/llama-batched-bench; M=/data/engines/models
OUT=${1:?outdir}; mkdir -p "$OUT"
for f in Qwen3.6-35B-A3B-UD-Q4_K_M Qwen3.8-27B-UD-Q4_K_M gpt-oss-20b-MXFP4; do
  $B -m $M/$f.gguf -c 40000 -ngl 99 -fa on -npp 512 -ntg 128 -npl 1,2,4,8 > "$OUT/$f.txt" 2>&1
  echo "done $f"
done
echo CONCDONE
