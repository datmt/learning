#!/usr/bin/env bash
# Memory vs context: start llama-server (production flags, 1 slot) per (model, ctx), run one request,
# record nvidia-smi per-process memory, RSS and system "used". Writes CSV.
set -u
S=/data/engines/llama.cpp/build/bin/llama-server; M=/data/engines/models; PORT=8099
OUT=${1:?csv}; echo "model,ctx,kv_type,gpu_mib,rss_mib,sys_used_mib" > "$OUT"
sysused() { free -m | awk '/^Mem:/{print $3}'; }
one() { # file ctx kvtype [extra...]
  local file=$1 ctx=$2 kv=$3; shift 3
  local base; base=$(sysused)
  $S -m "$M/$file" --host 127.0.0.1 --port $PORT -c "$ctx" -np 1 -fa on -ngl 99 --load-mode none \
     -ctk "$kv" -ctv "$kv" --jinja >/tmp/claude_memsrv.log 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 2; done
  curl -s localhost:$PORT/completion -d '{"prompt":"Hello","n_predict":8}' >/dev/null
  sleep 2
  local gpu rss
  gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
  rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status)
  echo "$file,$ctx,$kv,${gpu:-NA},${rss:-NA},$(( $(sysused) - base ))" | tee -a "$OUT"
  grep -E 'KV buffer|compute buffer|RS buffer|model buffer' /tmp/claude_memsrv.log | sed "s/^/  [$file ctx=$ctx $kv] /" >> "${OUT%.csv}.buffers.txt"
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
for f in gemma-3-4b-it-Q4_K_M.gguf gemma3-12b-Q4_K_M.gguf gpt-oss-20b-MXFP4.gguf gemma-4-26B-A4B-it-UD-Q4_K_M.gguf Qwen3.8-27B-UD-Q4_K_M.gguf Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; do
  for c in 4096 32768 131072; do one $f $c f16; done
done
one Qwen3-8B-BF16.gguf 4096 f16; one Qwen3-8B-BF16.gguf 32768 f16
# KV-cache quantization effect on a conventional-attention model
one gemma3-12b-Q4_K_M.gguf 131072 q8_0
one gemma3-12b-Q4_K_M.gguf 131072 q4_0
echo MEMDONE
