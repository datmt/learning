### Memory-bandwidth microbenchmark (MEASURED)

| Kernel | Buffer (GiB) | Iters | Median GB/s | Min GB/s | Max GB/s | Median GiB/s | % of 273 GB/s spec |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 0.06 | 20 | 243 | 238 | 246 | 226 | 89% |
| write | 0.06 | 20 | 193 | 188 | 195 | 180 | 71% |
| copy | 0.06 | 20 | 220 | 217 | 222 | 205 | 81% |
| read | 1.00 | 20 | 237 | 235 | 238 | 220 | 87% |
| write | 1.00 | 20 | 193 | 174 | 193 | 180 | 71% |
| copy | 1.00 | 20 | 216 | 214 | 216 | 201 | 79% |
| read | 4.00 | 20 | 235 | 228 | 236 | 219 | 86% |
| write | 4.00 | 20 | 193 | 186 | 193 | 179 | 71% |
| copy | 4.00 | 20 | 217 | 213 | 218 | 202 | 80% |
| read | 8.00 | 20 | 233 | 222 | 237 | 217 | 85% |
| write | 8.00 | 20 | 192 | 184 | 192 | 178 | 70% |
| read | 16.00 | 20 | 216 | 83 | 220 | 201 | 79% |
| write | 16.00 | 20 | 192 | 188 | 192 | 178 | 70% |

### Decode speed: theoretical prediction vs observation, dense models

Columns: weights file (MEASURED), bytes streamed per token (DERIVED from the GGUF tensor table), microbenchmark read bandwidth at the nearest buffer size (MEASURED), predicted tok/s = bandwidth ÷ bytes (THEORETICAL), observed tok/s (MEASURED), effective bandwidth = observed × bytes (DERIVED, not a hardware measurement), observed ÷ predicted.

| Model | Tensors (GiB) | Streamed/token (GiB) | Output head | Microbench read GiB/s | Predicted tok/s | Observed tok/s | Effective GiB/s | Obs ÷ pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B Q4_K_M | 2.31 | 2.31 | tied | 219 (@4 GiB) | 94.5 | 74.1 ± 0.2 (n=3) | 171 | 78% |
| Gemma 3 12B Q4_K_M | 6.79 | 6.79 | tied | 217 (@8 GiB) | 32.0 | 27.4 ± 0.1 (n=3) | 186 | 86% |
| Gemma 3 12B Q8_0 | 11.64 | 11.64 | tied | 201 (@16 GiB) | 17.3 | 17.0 ± 0.0 (n=3) | 198 | 98% |
| Qwen3 8B Q4_K_M | 4.68 | 4.35 | untied | 219 (@4 GiB) | 50.2 | 44.0 ± 0.1 (n=5) | 191 | 87% |
| Qwen3 8B Q8_0 | 8.11 | 7.49 | untied | 217 (@8 GiB) | 29.0 | 29.1 ± 0.1 (n=5) | 218 | 100% |
| Qwen3 8B BF16 | 15.26 | 14.10 | untied | 201 (@16 GiB) | 14.3 | 15.5 ± 0.0 (n=5) | 219 | 109% |
| Qwen3.8 27B Q4_K_M | 15.32 | 14.66 | untied | 201 (@16 GiB) | 13.7 | 12.8 ± 0.0 (n=3) | 187 | 93% |

### MoE models: bytes touched per token INFERRED from speed (not measured)

Inferred = measured read bandwidth (201–217 GiB/s, microbenchmark at 16 and 8 GiB) ÷ observed tok/s. This is an *inference*: it assumes the kernels run at the microbenchmark's bandwidth, and it lumps together routed experts, always-read tensors (attention, shared experts, output head) and any cache effects. Treat it as an order of magnitude, not a traffic measurement.

| Model | File (GiB) | Observed tok/s | Inferred GiB/token | Inferred share of file read per token |
|---|---:|---:|---:|---:|
| gpt-oss 20B MXFP4 | 11.3 | 81.3 ± 0.2 (n=3) | 2.5 – 2.7 | 22% – 24% |
| Gemma 4 26B-A4B Q4_K_M | 15.8 | 54.5 ± 0.2 (n=3) | 3.7 – 4.0 | 23% – 25% |
| Qwen3.6 35B-A3B Q4_K_M | 20.6 | 67.7 ± 0.2 (n=3) | 3.0 – 3.2 | 14% – 16% |

### Perplexity, wikitext-2 test set, 2048-token windows (MEASURED)

| Model | Quant | Weights (GiB) | PPL ± stat. error | Δ vs reference |
|---|---:|---:|---:|---:|
| Qwen3 8B | BF16 | 15.3 | 8.5248 ± 0.065 | reference |
| Qwen3 8B | Q8_0 | 8.1 | 8.5499 ± 0.065 | +0.29% |
| Qwen3 8B | Q4_K_M | 4.7 | 8.6049 ± 0.065 | +0.94% |
| Gemma 3 12B | Q8_0 | 11.7 | 7.7891 ± 0.058 | reference |
| Gemma 3 12B | Q4_K_M | 6.8 | 7.8513 ± 0.058 | +0.80% |

### Memory over the process lifecycle, 1 server with 4 slots (MEASURED; GiB unless noted)

`GPU proc` = nvidia-smi per-process memory (on unified memory this is *not* isolated VRAM). `RSS` = host-resident memory of the process. `Δ system used` = peak `free` 'used' minus the pre-launch baseline (includes everything on the machine, not only the server). `PSI` = peak kernel memory-pressure stall, % of time (10 s average).

| Model | GPU proc at load | GPU proc peak | GPU proc after workload | RSS after load | RSS peak | Δ system used (peak) | Min available | Peak PSI % | All requests OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss 20B MXFP4 (131k ctx) | 14.1 | 14.2 | 14.2 | 1.2 | 2.6 | +16.7 | 46 | 0.0 | yes |
| Qwen3 8B Q4 (32k ctx) | 9.2 | 9.2 | 9.2 | 0.8 | 3.6 | +12.9 | 51 | 0.0 | yes |
| Qwen3.6 35B-A3B Q4 (131k ctx) | 23.3 | 23.3 | 23.3 | 1.1 | 3.3 | +26.2 | 37 | 3.0 | yes |
| Qwen3.8 27B Q4 (131k ctx) | 23.4 | 23.4 | 23.4 | 1.3 | 6.7 | +31.6 | 32 | 4.3 | yes |

### Per-phase peaks (MEASURED)

| Model | Phase | Samples | GPU proc peak (GiB) | RSS peak (GiB) | Δ system used peak (GiB) |
|---|---:|---:|---:|---:|---:|
| gpt-oss 20B MXFP4 (131k ctx) | idle_after_load | 20 | 14.12 | 1.19 | +14.59 |
| gpt-oss 20B MXFP4 (131k ctx) | prefill_long | 7 | 14.16 | 1.40 | +14.90 |
| gpt-oss 20B MXFP4 (131k ctx) | generation | 7 | 14.16 | 1.80 | +15.31 |
| gpt-oss 20B MXFP4 (131k ctx) | concurrent | 51 | 14.17 | 1.48 | +15.03 |
| gpt-oss 20B MXFP4 (131k ctx) | max_context | 117 | 14.17 | 2.62 | +16.73 |
| gpt-oss 20B MXFP4 (131k ctx) | idle_after | 29 | 14.17 | 2.63 | +16.22 |
| Qwen3 8B Q4 (32k ctx) | idle_after_load | 19 | 9.17 | 0.80 | +10.12 |
| Qwen3 8B Q4 (32k ctx) | prefill_long | 6 | 9.18 | 0.83 | +10.07 |
| Qwen3 8B Q4 (32k ctx) | generation | 12 | 9.18 | 1.96 | +11.20 |
| Qwen3 8B Q4 (32k ctx) | concurrent | 20 | 9.18 | 2.01 | +11.32 |
| Qwen3 8B Q4 (32k ctx) | max_context | 36 | 9.18 | 3.63 | +12.94 |
| Qwen3 8B Q4 (32k ctx) | idle_after | 29 | 9.18 | 3.63 | +12.88 |
| Qwen3.6 35B-A3B Q4 (131k ctx) | idle_after_load | 19 | 23.27 | 1.14 | +23.94 |
| Qwen3.6 35B-A3B Q4 (131k ctx) | prefill_long | 14 | 23.32 | 1.61 | +24.43 |
| Qwen3.6 35B-A3B Q4 (131k ctx) | generation | 7 | 23.32 | 2.06 | +24.95 |
| Qwen3.6 35B-A3B Q4 (131k ctx) | concurrent | 72 | 23.32 | 2.15 | +24.96 |
| Qwen3.6 35B-A3B Q4 (131k ctx) | max_context | 146 | 23.32 | 3.28 | +26.18 |
| Qwen3.6 35B-A3B Q4 (131k ctx) | idle_after | 28 | 23.32 | 3.28 | +26.14 |
| Qwen3.8 27B Q4 (131k ctx) | idle_after_load | 19 | 23.37 | 1.32 | +25.06 |
| Qwen3.8 27B Q4 (131k ctx) | prefill_long | 38 | 23.38 | 2.00 | +25.63 |
| Qwen3.8 27B Q4 (131k ctx) | generation | 39 | 23.39 | 3.29 | +28.24 |
| Qwen3.8 27B Q4 (131k ctx) | concurrent | 210 | 23.40 | 3.23 | +28.01 |
| Qwen3.8 27B Q4 (131k ctx) | max_context | 431 | 23.40 | 6.65 | +31.64 |
| Qwen3.8 27B Q4 (131k ctx) | idle_after | 28 | 23.40 | 6.65 | +31.22 |
