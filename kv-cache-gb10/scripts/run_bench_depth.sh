#!/bin/bash
cd /home/dat/data/labs/model-loading/blog-kv; B=/data/engines/llama.cpp/build/bin/llama-bench; M=/data/engines/models
$B -m $M/Qwen3-8B-BF16.gguf -p 0 -n 128 -d 0,4096,8192,16384,32768 -fa 1 -r 3 -o json > raw/p2bench_Qwen3-8B.json 2> logs/p2bench_Qwen3-8B.err
for m in Qwen3.6-35B-A3B-UD-Q4_K_M Qwen3.8-27B-UD-Q4_K_M; do
 $B -m $M/$m.gguf -p 0 -n 128 -d 0,4096,8192,16384,32768,65536,131072 -fa 1 -r 3 -o json > raw/p2bench_$m.json 2> logs/p2bench_$m.err
done
echo BENCH_DONE
