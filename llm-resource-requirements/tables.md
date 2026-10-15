### Speed

| Model | Type | Quant | Params (B) | File (GiB) | pp @0 | tg @0 | pp @16k | tg @16k | pp @64k | tg @64k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | dense | Q4_K_M | 3.9 | 2.3 | 5,695 | 74.1 | 5,290 | 64.9 | – | – |
| Gemma 3 12B | dense | Q4_K_M | 11.8 | 6.8 | 1,937 | 27.4 | 1,776 | 23.0 | – | – |
| Gemma 3 12B | dense | Q8_0 | 11.8 | 11.6 | 1,820 | 17.0 | 1,672 | 15.3 | – | – |
| Qwen3 8B | dense | BF16 | 8.2 | 15.3 | 3,355 | 15.2 | 2,128 | 13.1 | – | – |
| Qwen3.8 27B | dense (hybrid) | Q4_K_M | 27.3 | 15.3 | 858 | 12.8 | 767 | 12.1 | 587 | 10.4 |
| gpt-oss 20B | MoE | MXFP4 | 20.9 | 11.3 | 4,908 | 81.3 | 3,736 | 70.8 | – | – |
| Gemma 4 26B-A4B | MoE | Q4_K_M | 25.2 | 15.8 | 2,916 | 54.5 | 2,469 | 48.2 | – | – |
| Qwen3.6 35B-A3B | MoE (hybrid) | Q4_K_M | 34.7 | 20.6 | 2,439 | 67.7 | 2,195 | 61.5 | 1,762 | 48.6 |

### Effective memory bandwidth of dense models (file size × tokens/s)

| Model | GiB read/token | tg (t/s) | Effective GiB/s |
|---|---:|---:|---:|
| Gemma 3 4B Q4_K_M | 2.3 | 74.1 | 171 |
| Gemma 3 12B Q4_K_M | 6.8 | 27.4 | 186 |
| Gemma 3 12B Q8_0 | 11.6 | 17.0 | 198 |
| Qwen3 8B BF16 | 15.3 | 15.2 | 233 |
| Qwen3.8 27B Q4_K_M | 15.3 | 12.8 | 195 |

### MoE: implied bytes touched per token (peak bandwidth ÷ tg)

| Model | File (GiB) | tg (t/s) | Implied GiB/token | % of file read per token |
|---|---:|---:|---:|---:|
| gpt-oss 20B MXFP4 | 11.3 | 81.3 | 2.86 | 25% |
| Gemma 4 26B-A4B Q4_K_M | 15.8 | 54.5 | 4.27 | 27% |
| Qwen3.6 35B-A3B Q4_K_M | 20.6 | 67.7 | 3.43 | 17% |

### Process memory (GiB) vs context, measured

| Model | 4k | 32k | 128k | KV cost (KiB/token) |
|---|---:|---:|---:|---:|
| Gemma 3 4B Q4 | 2.8 | 3.4 | 5.4 | 21 |
| Gemma 3 12B Q4 | 7.8 | 9.6 | 15.7 | 65 |
| gpt-oss 20B MXFP4 | 11.1 | 11.8 | 14.1 | 25 |
| Gemma 4 26B-A4B Q4 | 16.5 | 17.0 | 19.0 | 21 |
| Qwen3.8 27B Q4 (hybrid) | 15.1 | 16.8 | 22.9 | 65 |
| Qwen3.6 35B-A3B Q4 (hybrid) | 20.5 | 21.1 | 23.1 | 21 |
| Qwen3 8B BF16 | 15.0 | 18.9 | – | 145 |

### KV-cache quantization, Gemma 3 12B @128k

| KV type | Process memory (GiB) |
|---|---:|
| f16 | 15.7 |
| q8_0 | 12.7 |
| q4_0 | 10.5 |

### Concurrency (aggregate tg)

| Model | Streams | Aggregate t/s | Per-stream t/s |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B-UD-Q4_K_M | 1 | 66.3 | 66.3 |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 2 | 107.7 | 53.8 |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 4 | 154.8 | 38.7 |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 8 | 206.4 | 25.8 |
| Qwen3.8-27B-UD-Q4_K_M | 1 | 12.7 | 12.7 |
| Qwen3.8-27B-UD-Q4_K_M | 2 | 23.8 | 11.9 |
| Qwen3.8-27B-UD-Q4_K_M | 4 | 42.9 | 10.7 |
| Qwen3.8-27B-UD-Q4_K_M | 8 | 67.9 | 8.5 |
| gpt-oss-20b-MXFP4 | 1 | 78.3 | 78.3 |
| gpt-oss-20b-MXFP4 | 2 | 123.6 | 61.8 |
| gpt-oss-20b-MXFP4 | 4 | 180.4 | 45.1 |
| gpt-oss-20b-MXFP4 | 8 | 242.3 | 30.3 |
