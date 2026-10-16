| model | prompt/seq | max seqs with worst TTFT ≤ 10 s | max seqs with worst TTFT ≤ 30 s | max seqs with worst TTFT ≤ 120 s |
|---|---|---|---|---|
| Qwen3-8B (BF16) | 4K | 4 | 8 | 8 |
| Qwen3-8B (BF16) | 16K | 1 | 2 | 4 |
| Qwen3-8B (BF16) | 32K | 0 (one alone is slower) | 1 | 2 |
| Qwen3.6-35B-A3B | 4K | 4 | 8 | 8 |
| Qwen3.6-35B-A3B | 16K | 1 | 4 | 8 |
| Qwen3.6-35B-A3B | 32K | 0 (one alone is slower) | 2 | 4 |
| Qwen3.6-35B-A3B | 64K | 0 (one alone is slower) | 1 | 2 |
| Qwen3.8-27B | 4K | 1 | 4 | 8 |
| Qwen3.8-27B | 16K | 0 (one alone is slower) | 1 | 4 |
| Qwen3.8-27B | 32K | 0 (one alone is slower) | 0 (one alone is slower) | 2 |
| Qwen3.8-27B | 64K | 0 (one alone is slower) | 0 (one alone is slower) | 1 |