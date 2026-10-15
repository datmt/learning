#!/usr/bin/env bash
# Request-rate sweep with open-loop Poisson arrivals against a fresh llama-server per model.
# Usage: bench_latency.sh <wiki.test.raw> <outdir>
#
# Service-level objectives used for the "capacity" verdict (see analyze_latency.py):
#   chat: P95 TTFT <= 2 s  AND  P95 inter-token latency <= 100 ms  AND  0 failures
#   long: P95 TTFT <= 60 s AND  P95 inter-token latency <= 200 ms  AND  0 failures
set -u
D=$(dirname "$(readlink -f "$0")"); TXT=${1:?}; OUT=${2:?}; mkdir -p "$OUT"
S=/data/engines/llama.cpp/build/bin/llama-server; M=/data/engines/models; PORT=8099

serve() { # model-path -> sets SPID
  $S -m "$1" --host 127.0.0.1 --port $PORT -c 131072 -np 4 -fa on -ngl 99 --load-mode none --kv-unified > "$OUT/server.$2.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do curl -sf localhost:$PORT/health >/dev/null && return 0; sleep 2; done
  echo "server failed to start" >&2; kill $SPID; return 1
}
step() { # name profile rate duration timeout
  echo "-- $1 $2 rate=$3 dur=$4 $(date +%T)"
  python3 -I "$D/load_test.py" --url http://127.0.0.1:$PORT --profile "$2" --rate "$3" --duration "$4" --timeout "$5" --drain 180 \
     --corpus "$TXT" --tokens-cache "$OUT/corpus.tokens.$1.json" --pid $SPID --out "$OUT/$1.$2.r$3" > "$OUT/$1.$2.r$3.stdout" 2>&1 || echo "   load_test failed"
}
model() { # name path
  echo "== $1 $(date +%T)"; serve "$2" "$1" || return
  for i in 1 2; do curl -s localhost:$PORT/completion -d '{"prompt":"Hello","n_predict":16,"cache_prompt":false}' > /dev/null; done  # warm up kernels
  case $1 in
    qwen3.6-35b-a3b)
      for r in 0.2 0.4 0.7 1.0; do step $1 chat $r 150 120; done
      for r in 0.05 0.08 0.12; do step $1 long $r 420 600; done
      step $1 mixed 0.3 240 600 ;;
    qwen3.8-27b)
      for r in 0.05 0.1 0.15 0.2; do step $1 chat $r $(python3 -I -c "print(int(min(420,max(150,30/$r))))") 120; done
      for r in 0.015 0.03; do step $1 long $r 400 900; done ;;
  esac
  kill $SPID; wait $SPID 2>/dev/null; sleep 5
}
model qwen3.6-35b-a3b $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
model qwen3.8-27b     $M/Qwen3.8-27B-UD-Q4_K_M.gguf
echo LATENCYDONE
