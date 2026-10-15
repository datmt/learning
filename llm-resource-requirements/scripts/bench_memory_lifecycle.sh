#!/usr/bin/env bash
# Run memory_lifecycle.py for a fixed set of models, one at a time.
# Usage: bench_memory_lifecycle.sh <wiki.test.raw> <outdir>
set -u
D=$(dirname "$(readlink -f "$0")"); TXT=${1:?}; OUT=${2:?}; mkdir -p "$OUT"
M=/data/engines/models
run() { # name file ctx np
  echo "== $1 $(date +%T)"
  python3 -I "$D/memory_lifecycle.py" --model "$2" --ctx "$3" --np "$4" --corpus "$TXT" --out "$OUT/$1" > "$OUT/$1.stdout" 2> "$OUT/$1.stderr" || echo "  FAILED: see $OUT/$1.stderr"
}
run qwen3.6-35b-a3b-Q4_K_M  $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   131072 4
run qwen3.8-27b-Q4_K_M      $M/Qwen3.8-27B-UD-Q4_K_M.gguf       131072 4
run gpt-oss-20b-MXFP4       $M/gpt-oss-20b-MXFP4.gguf            131072 4
run qwen3-8b-Q4_K_M         /data/benchmarks/qwen3-8b-quants/Qwen3-8B-Q4_K_M.gguf 32768 4
echo MEMLIFEDONE
