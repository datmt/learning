#!/usr/bin/env bash
# Repeated, independent open-loop load tests (different random seeds) at selected request rates, including
# intermediate rates between a passing and a failing point of the original sweep (bench_latency.sh, seed 1).
# Usage: bench_latency_repeat.sh <wiki.test.raw> <outdir>
#
# The seed drives the Poisson arrival times, the prompt/output lengths AND the prompt offsets in the corpus, so each seed is an
# independent workload. Same SLOs as bench_latency.sh (fixed beforehand, see analyze_latency.py). Same server flags
# (-np 4, 131k unified pool, prompt cache off in the requests). A fresh server per model; runs are sequential.
# Resumable: a (model, rate, seed) whose summary file already exists is skipped. Every attempt is appended to $OUT/manifest.tsv
# (including failed attempts) so failures are never silently dropped.
# Run duration: min(420, max(150, 30/rate)) like bench_latency.sh, except that rates below 0.06 req/s use 600 s so a run
# still carries ~30 requests (percentiles need >= 20). Existing seed-1 runs are NOT repeated here.
set -u
D=$(dirname "$(readlink -f "$0")"); TXT=${1:?}; OUT=${2:?}; mkdir -p "$OUT"
S=/data/engines/llama.cpp/build/bin/llama-server; M=/data/engines/models; PORT=8099

serve() { # model-path name -> sets SPID
  $S -m "$1" --host 127.0.0.1 --port $PORT -c 131072 -np 4 -fa on -ngl 99 --load-mode none --kv-unified > "$OUT/server.$2.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do curl -sf localhost:$PORT/health >/dev/null && return 0; sleep 2; done
  echo "server failed to start" >&2; kill $SPID; return 1
}
dur() { python3 -I -c "r=$1; print(600 if r<0.06 else int(min(420,max(150,30/r))))"; }
step() { # name rate seed
  local name=$1 rate=$2 seed=$3 base="$OUT/$1.chat.r$2.s$3"
  [ -f "$base.summary.json" ] && return
  local d; d=$(dur "$rate")
  echo "-- $name chat rate=$rate seed=$seed dur=$d $(date +%T)"
  python3 -I "$D/load_test.py" --url http://127.0.0.1:$PORT --profile chat --rate "$rate" --duration "$d" --timeout 120 --drain 180 --seed "$seed" \
     --corpus "$TXT" --tokens-cache "$OUT/corpus.tokens.$name.json" --pid $SPID --out "$base" > "$base.stdout" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\trc=%s\t%s\n' "$(date -Is)" "$name" chat "$rate" "$seed" "$rc" "$([ -f "$base.summary.json" ] && echo ok || echo NO_SUMMARY)" >> "$OUT/manifest.tsv"
}
model() { # name path "rate:seeds ..."
  echo "== $1 $(date +%T)"; serve "$2" "$1" || return
  for i in 1 2; do curl -s localhost:$PORT/completion -d '{"prompt":"Hello","n_predict":16,"cache_prompt":false}' > /dev/null; done
  for seed in 1 2 3 4; do            # seeds outer so an interrupted sweep is still balanced across rates
    for rs in $3; do
      rate=${rs%%:*}; seeds=${rs#*:}
      case " ${seeds//,/ } " in *" $seed "*) step "$1" "$rate" "$seed";; esac
    done
  done
  kill $SPID; wait $SPID 2>/dev/null; sleep 5
}
# rate:seeds. Seed 1 already exists for 0.2/0.4 (35B) and 0.05/0.1 (27B) in the original sweep.
model qwen3.6-35b-a3b $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf "0.2:2,3,4 0.25:1,2,3,4 0.3:1,2,3,4 0.35:1,2,3,4 0.4:2,3,4"
model qwen3.8-27b     $M/Qwen3.8-27B-UD-Q4_K_M.gguf     "0.05:2,3,4 0.075:1,2,3,4 0.1:2,3,4"
echo REPEATDONE
