### Load test: chat profile (MEASURED)

SLO: P95 TTFT ≤ 2 s, P95 ITL ≤ 100 ms, zero failures. Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.

| Model | Offered req/s | Dur (s) | Sent | Done | Failed | Realized offered req/s | Out tok/s | TTFT p50 (s) | TTFT p95 (s) | TTFT p99 (s) | ITL p50 (ms) | ITL p95 (ms) | E2E p50 (s) | E2E p95 (s) | Peak sys used (GiB) | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B-A3B (MoE) | 0.2 | 150 | 40 | 40 | 0 | 0.267 | 62.2 | 0.44 | 0.94 | 1.94 | 25 | 32 | 6.4 | 12.1 | 89.6 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.4 | 150 | 67 | 67 | 0 | 0.447 | 99.3 | 6.07 | 15.10 | 15.98 | 30 | 34 | 14.9 | 23.3 | 92.2 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.7 | 150 | 111 | 111 | 0 | 0.740 | 165.2 | 48.13 | 78.58 | 81.56 | 31 | 35 | 55.2 | 88.3 | 92.1 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 1.0 | 150 | 162 | 121 | 41 | 1.080 | 184.1 | 73.56 | 119.96 | 120.17 | 31 | 35 | 80.1 | 131.2 | 92.1 | FAIL |
| Qwen3.8 27B (dense) | 0.05 | 420 | 27 | 27 | 0 | 0.064 | 15.7 | 1.05 | 1.68 | 2.11 | 89 | 97 | 22.2 | 37.9 | 91.8 | PASS |
| Qwen3.8 27B (dense) | 0.1 | 300 | 40 | 40 | 0 | 0.133 | 31.1 | 4.10 | 29.07 | 33.43 | 95 | 98 | 30.4 | 55.4 | 92.6 | FAIL |
| Qwen3.8 27B (dense) | 0.15 | 200 | 40 | 40 | 0 | 0.200 | 46.6 | 37.53 | 58.41 | 61.41 | 95 | 99 | 64.8 | 79.6 | 92.3 | FAIL |
| Qwen3.8 27B (dense) | 0.2 | 150 | 40 | 40 | 0 | 0.267 | 62.1 | 65.84 | 104.01 | 107.22 | 95 | 100 | 85.3 | 118.3 | 92.4 | FAIL |

### Load test: long profile (MEASURED)

SLO: P95 TTFT ≤ 60 s, P95 ITL ≤ 200 ms, zero failures. Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.

| Model | Offered req/s | Dur (s) | Sent | Done | Failed | Realized offered req/s | Out tok/s | TTFT p50 (s) | TTFT p95 (s) | TTFT p99 (s) | ITL p50 (ms) | ITL p95 (ms) | E2E p50 (s) | E2E p95 (s) | Peak sys used (GiB) | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B-A3B (MoE) | 0.05 | 420 | 27 | 27 | 0 | 0.064 | 23.3 | 18.68 | 44.48 | 51.48 | 36 | 87 | 51.9 | 88.5 | 91.8 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.08 | 420 | 40 | 40 | 0 | 0.095 | 33.1 | 94.21 | 153.17 | 161.21 | 37 | 1167 | 128.6 | 181.4 | 91.3 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.12 | 420 | 55 | 41 | 14 | 0.131 | 33.1 | 142.58 | 267.65 | 279.87 | 38 | 1184 | 195.0 | 300.7 | 91.2 | FAIL |
| Qwen3.8 27B (dense) | 0.015 | 400 | 8 | 8 | 0 | 0.020 | 8.5 | – | – | – | 100 | 114 | – | – | 92.9 | n/a (n<20) |
| Qwen3.8 27B (dense) | 0.03 | 400 | 19 | 11 | 8 | 0.048 | 10.6 | – | – | – | 112 | 234 | – | – | 91.9 | n/a (n<20) |

### Load test: mixed profile (MEASURED)

SLO: P95 TTFT ≤ 60 s, P95 ITL ≤ 200 ms, zero failures. Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.

| Model | Offered req/s | Dur (s) | Sent | Done | Failed | Realized offered req/s | Out tok/s | TTFT p50 (s) | TTFT p95 (s) | TTFT p99 (s) | ITL p50 (ms) | ITL p95 (ms) | E2E p50 (s) | E2E p95 (s) | Peak sys used (GiB) | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B-A3B (MoE) | 0.3 | 240 | 68 | 68 | 0 | 0.283 | 67.0 | 6.88 | 26.10 | 28.69 | 32 | 37 | 20.2 | 38.1 | 91.1 | PASS |

### Highest tested rate meeting the SLO (MEASURED capacity point, not a theoretical maximum)

| Profile | Model | Rates tested (req/s) | Highest passing | First failing | Rates with too few samples |
|---|---|---|---:|---:|---:|
| chat | Qwen3.6 35B-A3B (MoE) | 0.2, 0.4, 0.7, 1.0 | 0.2 req/s | 0.4 | – |
| chat | Qwen3.8 27B (dense) | 0.05, 0.1, 0.15, 0.2 | 0.05 req/s | 0.1 | – |
| long | Qwen3.6 35B-A3B (MoE) | 0.05, 0.08, 0.12 | 0.05 req/s | 0.08 | – |
| long | Qwen3.8 27B (dense) | 0.015, 0.03 | none of the tested rates | – | 0.015, 0.03 |
| mixed | Qwen3.6 35B-A3B (MoE) | 0.3 | 0.3 req/s | – | – |
