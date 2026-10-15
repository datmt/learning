### Load test: chat profile (MEASURED)

SLO: P95 TTFT ≤ 2 s, P95 ITL ≤ 100 ms, zero failures. Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.

| Model | Offered req/s | Seed | Dur (s) | Sent | Done | Failed | Realized offered req/s | Out tok/s | TTFT p50 (s) | TTFT p95 (s) | TTFT p99 (s) | ITL p50 (ms) | ITL p95 (ms) | E2E p50 (s) | E2E p95 (s) | Peak sys used (GiB) | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B-A3B (MoE) | 0.2 | 1 | 150 | 40 | 40 | 0 | 0.267 | 62.2 | 0.44 | 0.94 | 1.94 | 25 | 32 | 6.4 | 12.1 | 89.6 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.2 | 2 | 150 | 30 | 30 | 0 | 0.200 | 42.1 | 0.42 | 0.69 | 0.72 | 17 | 26 | 4.2 | 7.6 | 93.5 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.2 | 3 | 150 | 16 | 16 | 0 | 0.107 | 21.9 | – | – | – | 16 | 21 | – | – | 92.9 | n/a (n<20) |
| Qwen3.6 35B-A3B (MoE) | 0.2 | 4 | 150 | 33 | 33 | 0 | 0.220 | 48.8 | 0.40 | 0.79 | 0.87 | 20 | 31 | 5.1 | 9.8 | 93.1 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.25 | 1 | 150 | 45 | 45 | 0 | 0.300 | 69.6 | 0.50 | 4.66 | 5.58 | 26 | 32 | 7.1 | 14.2 | 91.4 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.25 | 2 | 150 | 33 | 33 | 0 | 0.220 | 45.6 | 0.44 | 0.73 | 0.83 | 20 | 30 | 4.5 | 8.4 | 93.5 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.25 | 3 | 150 | 21 | 21 | 0 | 0.140 | 29.4 | 0.33 | 0.54 | 0.55 | 16 | 25 | 4.2 | 6.7 | 92.0 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.25 | 4 | 150 | 40 | 40 | 0 | 0.267 | 59.2 | 0.39 | 1.62 | 3.38 | 21 | 31 | 5.5 | 11.9 | 93.0 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.3 | 1 | 150 | 51 | 51 | 0 | 0.340 | 77.9 | 0.63 | 9.25 | 10.54 | 29 | 33 | 7.9 | 17.2 | 93.1 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.3 | 2 | 150 | 37 | 37 | 0 | 0.247 | 51.4 | 0.45 | 0.77 | 0.91 | 21 | 31 | 4.9 | 9.6 | 93.5 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.3 | 3 | 150 | 29 | 29 | 0 | 0.193 | 40.7 | 0.35 | 0.68 | 1.42 | 20 | 32 | 5.1 | 8.4 | 92.5 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.3 | 4 | 150 | 46 | 46 | 0 | 0.307 | 70.4 | 0.42 | 3.48 | 5.96 | 25 | 32 | 6.1 | 14.0 | 92.8 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.35 | 1 | 150 | 59 | 59 | 0 | 0.393 | 85.6 | 0.81 | 12.61 | 13.38 | 30 | 33 | 9.0 | 20.1 | 93.5 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.35 | 2 | 150 | 44 | 44 | 0 | 0.293 | 66.0 | 0.49 | 1.03 | 1.49 | 23 | 31 | 5.6 | 12.3 | 93.8 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.35 | 3 | 150 | 40 | 40 | 0 | 0.267 | 56.3 | 0.39 | 0.68 | 1.96 | 21 | 32 | 5.3 | 9.4 | 93.0 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.35 | 4 | 150 | 51 | 51 | 0 | 0.340 | 77.2 | 0.58 | 5.84 | 7.99 | 26 | 32 | 7.4 | 17.4 | 93.1 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.4 | 1 | 150 | 67 | 67 | 0 | 0.447 | 99.3 | 6.07 | 15.10 | 15.98 | 30 | 34 | 14.9 | 23.3 | 92.2 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.4 | 2 | 150 | 52 | 52 | 0 | 0.347 | 78.6 | 0.65 | 5.16 | 7.88 | 26 | 33 | 7.3 | 15.1 | 93.3 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.4 | 3 | 150 | 45 | 45 | 0 | 0.300 | 65.1 | 0.51 | 2.32 | 5.46 | 27 | 32 | 6.2 | 15.0 | 92.8 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.4 | 4 | 150 | 55 | 55 | 0 | 0.367 | 83.3 | 0.83 | 9.14 | 10.62 | 30 | 33 | 8.4 | 18.2 | 92.8 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.7 | 1 | 150 | 111 | 111 | 0 | 0.740 | 165.2 | 48.13 | 78.58 | 81.56 | 31 | 35 | 55.2 | 88.3 | 92.1 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 1.0 | 1 | 150 | 162 | 121 | 41 | 1.080 | 184.1 | 73.56 | 119.96 | 120.17 | 31 | 35 | 80.1 | 131.2 | 92.1 | FAIL |
| Qwen3.8 27B (dense) | 0.05 | 1 | 420 | 27 | 27 | 0 | 0.064 | 15.7 | 1.05 | 1.68 | 2.11 | 89 | 97 | 22.2 | 37.9 | 91.8 | PASS |
| Qwen3.8 27B (dense) | 0.05 | 2 | 600 | 30 | 30 | 0 | 0.050 | 10.5 | 1.09 | 1.84 | 1.92 | 85 | 92 | 18.0 | 34.9 | 93.1 | PASS |
| Qwen3.8 27B (dense) | 0.05 | 3 | 600 | 16 | 16 | 0 | 0.027 | 5.5 | – | – | – | 80 | 90 | – | – | 92.0 | n/a (n<20) |
| Qwen3.8 27B (dense) | 0.05 | 4 | 600 | 33 | 33 | 0 | 0.055 | 12.2 | 0.96 | 1.82 | 1.83 | 86 | 96 | 20.2 | 37.5 | 92.5 | PASS |
| Qwen3.8 27B (dense) | 0.075 | 1 | 400 | 40 | 40 | 0 | 0.100 | 23.3 | 1.32 | 10.12 | 13.62 | 95 | 99 | 27.9 | 41.0 | 93.7 | FAIL |
| Qwen3.8 27B (dense) | 0.075 | 2 | 400 | 30 | 30 | 0 | 0.075 | 15.8 | 1.17 | 2.02 | 2.25 | 86 | 96 | 19.3 | 37.2 | 93.6 | FAIL |
| Qwen3.8 27B (dense) | 0.075 | 3 | 400 | 16 | 16 | 0 | 0.040 | 8.2 | – | – | – | 81 | 95 | – | – | 91.9 | n/a (n<20) |
| Qwen3.8 27B (dense) | 0.075 | 4 | 400 | 33 | 33 | 0 | 0.083 | 18.3 | 1.16 | 6.32 | 10.16 | 91 | 98 | 24.2 | 44.7 | 92.6 | FAIL |
| Qwen3.8 27B (dense) | 0.1 | 1 | 300 | 40 | 40 | 0 | 0.133 | 31.1 | 4.10 | 29.07 | 33.43 | 95 | 98 | 30.4 | 55.4 | 92.6 | FAIL |
| Qwen3.8 27B (dense) | 0.1 | 2 | 300 | 30 | 30 | 0 | 0.100 | 21.1 | 1.31 | 2.65 | 3.68 | 90 | 98 | 19.9 | 39.1 | 93.3 | FAIL |
| Qwen3.8 27B (dense) | 0.1 | 3 | 300 | 16 | 16 | 0 | 0.053 | 11.0 | – | – | – | 85 | 96 | – | – | 91.9 | n/a (n<20) |
| Qwen3.8 27B (dense) | 0.1 | 4 | 300 | 33 | 33 | 0 | 0.110 | 24.4 | 1.19 | 14.93 | 20.69 | 95 | 99 | 25.9 | 50.8 | 92.8 | FAIL |
| Qwen3.8 27B (dense) | 0.15 | 1 | 200 | 40 | 40 | 0 | 0.200 | 46.6 | 37.53 | 58.41 | 61.41 | 95 | 99 | 64.8 | 79.6 | 92.3 | FAIL |
| Qwen3.8 27B (dense) | 0.2 | 1 | 150 | 40 | 40 | 0 | 0.267 | 62.1 | 65.84 | 104.01 | 107.22 | 95 | 100 | 85.3 | 118.3 | 92.4 | FAIL |

### Load test: long profile (MEASURED)

SLO: P95 TTFT ≤ 60 s, P95 ITL ≤ 200 ms, zero failures. Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.

| Model | Offered req/s | Seed | Dur (s) | Sent | Done | Failed | Realized offered req/s | Out tok/s | TTFT p50 (s) | TTFT p95 (s) | TTFT p99 (s) | ITL p50 (ms) | ITL p95 (ms) | E2E p50 (s) | E2E p95 (s) | Peak sys used (GiB) | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B-A3B (MoE) | 0.05 | 1 | 420 | 27 | 27 | 0 | 0.064 | 23.3 | 18.68 | 44.48 | 51.48 | 36 | 87 | 51.9 | 88.5 | 91.8 | PASS |
| Qwen3.6 35B-A3B (MoE) | 0.08 | 1 | 420 | 40 | 40 | 0 | 0.095 | 33.1 | 94.21 | 153.17 | 161.21 | 37 | 1167 | 128.6 | 181.4 | 91.3 | FAIL |
| Qwen3.6 35B-A3B (MoE) | 0.12 | 1 | 420 | 55 | 41 | 14 | 0.131 | 33.1 | 142.58 | 267.65 | 279.87 | 38 | 1184 | 195.0 | 300.7 | 91.2 | FAIL |
| Qwen3.8 27B (dense) | 0.015 | 1 | 400 | 8 | 8 | 0 | 0.020 | 8.5 | – | – | – | 100 | 114 | – | – | 92.9 | n/a (n<20) |
| Qwen3.8 27B (dense) | 0.03 | 1 | 400 | 19 | 11 | 8 | 0.048 | 10.6 | – | – | – | 112 | 234 | – | – | 91.9 | n/a (n<20) |

### Load test: mixed profile (MEASURED)

SLO: P95 TTFT ≤ 60 s, P95 ITL ≤ 200 ms, zero failures. Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.

| Model | Offered req/s | Seed | Dur (s) | Sent | Done | Failed | Realized offered req/s | Out tok/s | TTFT p50 (s) | TTFT p95 (s) | TTFT p99 (s) | ITL p50 (ms) | ITL p95 (ms) | E2E p50 (s) | E2E p95 (s) | Peak sys used (GiB) | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B-A3B (MoE) | 0.3 | 1 | 240 | 68 | 68 | 0 | 0.283 | 67.0 | 6.88 | 26.10 | 28.69 | 32 | 37 | 20.2 | 38.1 | 91.1 | PASS |

### Per-rate results across independent trials (MEASURED; each trial = a different random seed)

A trial passes only if P95 TTFT and P95 ITL meet the SLO, nothing failed, and at least 20 requests completed; a trial with fewer than 20 completions has no verdict and is excluded from the pass count (shown as n<20). TTFT p95 is the median over trials, with the min–max range in brackets; with n of 3–5 trials the range, not a confidence interval, is the honest measure of spread.

| Profile | Model | Offered req/s | Trials | Realized offered req/s (range) | Valid trials passing | Failed requests (total) | P95 TTFT (s), median (range) | Rate verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| chat | Qwen3.6 35B-A3B (MoE) | 0.2 | 4 | 0.107–0.267 | 3/3 (+1 n<20) | 0 | 0.79 (0.69–0.94) | PASS (all valid trials) |
| chat | Qwen3.6 35B-A3B (MoE) | 0.25 | 4 | 0.140–0.300 | 3/4 | 0 | 1.18 (0.54–4.66) | MIXED |
| chat | Qwen3.6 35B-A3B (MoE) | 0.3 | 4 | 0.193–0.340 | 2/4 | 0 | 2.13 (0.68–9.25) | MIXED |
| chat | Qwen3.6 35B-A3B (MoE) | 0.35 | 4 | 0.267–0.393 | 2/4 | 0 | 3.44 (0.68–12.61) | MIXED |
| chat | Qwen3.6 35B-A3B (MoE) | 0.4 | 4 | 0.300–0.447 | 0/4 | 0 | 7.15 (2.32–15.10) | FAIL (all valid trials) |
| chat | Qwen3.6 35B-A3B (MoE) | 0.7 | 1 | 0.740–0.740 | 0/1 | 0 | 78.58 (78.58–78.58) | FAIL (all valid trials) |
| chat | Qwen3.6 35B-A3B (MoE) | 1.0 | 1 | 1.080–1.080 | 0/1 | 41 | 119.96 (119.96–119.96) | FAIL (all valid trials) |
| chat | Qwen3.8 27B (dense) | 0.05 | 4 | 0.027–0.064 | 3/3 (+1 n<20) | 0 | 1.82 (1.68–1.84) | PASS (all valid trials) |
| chat | Qwen3.8 27B (dense) | 0.075 | 4 | 0.040–0.100 | 0/3 (+1 n<20) | 0 | 6.32 (2.02–10.12) | FAIL (all valid trials) |
| chat | Qwen3.8 27B (dense) | 0.1 | 4 | 0.053–0.133 | 0/3 (+1 n<20) | 0 | 14.93 (2.65–29.07) | FAIL (all valid trials) |
| chat | Qwen3.8 27B (dense) | 0.15 | 1 | 0.200–0.200 | 0/1 | 0 | 58.41 (58.41–58.41) | FAIL (all valid trials) |
| chat | Qwen3.8 27B (dense) | 0.2 | 1 | 0.267–0.267 | 0/1 | 0 | 104.01 (104.01–104.01) | FAIL (all valid trials) |
| long | Qwen3.6 35B-A3B (MoE) | 0.05 | 1 | 0.064–0.064 | 1/1 | 0 | 44.48 (44.48–44.48) | PASS (all valid trials) |
| long | Qwen3.6 35B-A3B (MoE) | 0.08 | 1 | 0.095–0.095 | 0/1 | 0 | 153.17 (153.17–153.17) | FAIL (all valid trials) |
| long | Qwen3.6 35B-A3B (MoE) | 0.12 | 1 | 0.131–0.131 | 0/1 | 14 | 267.65 (267.65–267.65) | FAIL (all valid trials) |
| long | Qwen3.8 27B (dense) | 0.015 | 1 | 0.020–0.020 | 0/0 (+1 n<20) | 0 | – | no verdict |
| long | Qwen3.8 27B (dense) | 0.03 | 1 | 0.048–0.048 | 0/0 (+1 n<20) | 8 | – | no verdict |
| mixed | Qwen3.6 35B-A3B (MoE) | 0.3 | 1 | 0.283–0.283 | 1/1 | 0 | 26.10 (26.10–26.10) | PASS (all valid trials) |

### Capacity bracket under the SLO (MEASURED; preliminary, a bracket and not a maximum)

"Highest passing" = the highest tested rate at which every trial met the SLO, with no lower rate failing. "Lowest rate not passing" = the lowest tested rate where any trial failed. Rates with mixed outcomes lie inside the uncertain region. Capacity is somewhere in this bracket; the number of trials is in brackets after each rate.

| Profile | Model | Rates tested (req/s) | Highest rate passing in all trials | Lowest rate not passing | Rates with mixed outcomes | Rates with no verdict (n<20) |
|---|---|---|---:|---:|---:|---:|
| chat | Qwen3.6 35B-A3B (MoE) | 0.2 (n=4), 0.25 (n=4), 0.3 (n=4), 0.35 (n=4), 0.4 (n=4), 0.7 (n=1), 1.0 (n=1) | 0.2 req/s | 0.25 | 0.25, 0.3, 0.35 | – |
| chat | Qwen3.8 27B (dense) | 0.05 (n=4), 0.075 (n=4), 0.1 (n=4), 0.15 (n=1), 0.2 (n=1) | 0.05 req/s | 0.075 | – | – |
| long | Qwen3.6 35B-A3B (MoE) | 0.05 (n=1), 0.08 (n=1), 0.12 (n=1) | 0.05 req/s | 0.08 | – | – |
| long | Qwen3.8 27B (dense) | 0.015 (n=1), 0.03 (n=1) | none tested | – | – | 0.015, 0.03 |
| mixed | Qwen3.6 35B-A3B (MoE) | 0.3 (n=1) | 0.3 req/s | – | – | – |

### Outcome versus realized arrival rate (EXPLORATORY, post hoc)

Because each run is a short Poisson window, the realized rate (arrivals ÷ duration) differs a lot from the configured rate. This table re-sorts the same runs by realized rate. It was chosen after seeing the data, so it is a description, not a tested threshold, and it holds only for these window lengths: a longer window lets a queue build further before it ends.

| Profile | Model | Runs with a verdict | Highest realized rate that passed (req/s) | Lowest realized rate that failed (req/s) | Pass/fail ranges | Failures below highest pass |
|---|---|---:|---:|---:|---|---:|
| chat | Qwen3.6 35B-A3B (MoE) | 21 | 0.293 | 0.300 | separated | 0 |
| chat | Qwen3.8 27B (dense) | 11 | 0.064 | 0.075 | separated | 0 |
