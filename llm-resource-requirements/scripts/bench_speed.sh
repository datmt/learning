#!/usr/bin/env bash
# Speed benchmark: llama-bench per model, one at a time, with GPU clock/power sampling.
set -u
B=/data/engines/llama.cpp/build/bin/llama-bench; M=/data/engines/models
OUT=${1:?outdir}; mkdir -p "$OUT"
run() { # name file depths
  local name=$1 file=$2 depths=$3
  nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
     --format=csv -l 2 > "$OUT/$name.gpu.csv" & local smi=$!
  $B -m "$M/$file" -fa 1 -ngl 99 -p 512,2048 -n 128 -d "$depths" -r 3 -o jsonl > "$OUT/$name.jsonl" 2> "$OUT/$name.err"
  kill $smi; wait $smi 2>/dev/null
  echo "done $name $(date +%T)"
}
run gemma-3-4b-Q4_K_M        gemma-3-4b-it-Q4_K_M.gguf          0,16384
run gemma3-12b-Q4_K_M        gemma3-12b-Q4_K_M.gguf             0,16384
run gemma3-12b-Q8_0          gemma3-12b-Q8_0.gguf               0,16384
run qwen3-8b-BF16            Qwen3-8B-BF16.gguf                 0,16384
run gpt-oss-20b-MXFP4        gpt-oss-20b-MXFP4.gguf             0,16384
run gemma-4-26B-A4B-Q4_K_M   gemma-4-26B-A4B-it-UD-Q4_K_M.gguf  0,16384
run qwen3.8-27b-Q4_K_M       Qwen3.8-27B-UD-Q4_K_M.gguf         0,16384,65536
run qwen3.6-35B-A3B-Q4_K_M   Qwen3.6-35B-A3B-UD-Q4_K_M.gguf     0,16384,65536
echo ALLDONE
