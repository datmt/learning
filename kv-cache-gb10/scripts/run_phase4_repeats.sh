#!/bin/bash
# Phase 4 repeats. Rep 1 of part A is the original raw/p4_* file. Part A reps 2-3 -> raw/p4r_*_rN.json (default flags).
# Part B (-b 2048 -ub 2048), 3 reps -> raw/p4b_*_rN.json. Interleaved by rep so session drift is spread over all points.
cd /home/dat/data/labs/model-loading/blog-kv
M35=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; M27=Qwen3.8-27B-UD-Q4_K_M.gguf
A="$M27:8:4096 $M27:4:16384 $M27:4:32768 $M27:2:65536 $M27:8:65536 $M35:8:4096 $M35:4:16384 $M35:4:32768 $M35:4:65536 $M35:8:65536"
B="$M27:4:32768 $M35:4:65536 $M27:8:4096"
run() { # prefix rep spec extra
  IFS=: read m n p <<< "$3"
  f=raw/$1_${m%.gguf}_n${n}_p${p}_f16_r$2.json; [ -f $f ] && return
  pgrep -ax llama-server | grep -v -- '--port 8082' | grep -q . && { echo STRAY; exit 1; }
  /home/dat/jupyter-venv/bin/python -I scripts/harness_conc.py --model $m --n $n --prompt $p --prefix $1 --tag _r$2 --extra "$4"
}
for r in 1 2 3; do
  if [ $r -ge 2 ]; then for s in $A; do run p4r $r $s ""; done; fi
  for s in $B; do run p4b $r $s "-b 2048 -ub 2048"; done
done
echo REPEATS_DONE
