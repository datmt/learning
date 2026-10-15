#!/usr/bin/env bash
# Perplexity on wikitext-2 (raw) test set, identical settings for every model.
# Compare ONLY between quantizations of the same model: perplexity is not comparable across tokenizers/models.
# Usage: bench_perplexity.sh <llama-perplexity binary> <wiki.test.raw> <outdir>
set -u
PPL=${1:?}; TXT=${2:?}; OUT=${3:?}; mkdir -p "$OUT"
sha256sum "$TXT" > "$OUT/corpus.sha256"
run() { # name path
  echo "== $1 $(date +%T)"
  "$PPL" -m "$2" -f "$TXT" -c 2048 -b 2048 -ngl 99 -fa on > "$OUT/$1.log" 2>&1
  grep -E 'Final estimate' "$OUT/$1.log" || echo "  (no final estimate; see log)"
}
run qwen3-8b-BF16    /data/engines/models/Qwen3-8B-BF16.gguf
run qwen3-8b-Q8_0    /data/benchmarks/qwen3-8b-quants/Qwen3-8B-Q8_0.gguf
run qwen3-8b-Q4_K_M  /data/benchmarks/qwen3-8b-quants/Qwen3-8B-Q4_K_M.gguf
run gemma3-12b-Q8_0  /data/engines/models/gemma3-12b-Q8_0.gguf
run gemma3-12b-Q4_K_M /data/engines/models/gemma3-12b-Q4_K_M.gguf
echo PPLDONE
