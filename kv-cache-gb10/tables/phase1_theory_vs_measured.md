| model | KV layers (log) | theory global B/tok | measured global B/tok | match | SWA cache cells | SWA MiB | recurrent state MiB (1 seq) | runs ok/total |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | 36 | 147456 | 147456 (intercept -0 B) | yes | [0] | [0.0] | 0.0 | 24/24 |
| Qwen3.6-35B-A3B | 10 | 20480 | 20480 (intercept 0 B) | yes | [0] | [0.0] | 62.8 | 31/32 |
| Qwen3.8-27B | 16 | 69632 | 65536 (intercept 0 B) | NO (-5.9%) | [0] | [0.0] | 149.6 | 31/32 |
| Gemma3-12B | 8 | 65536 | 65536 (intercept 0 B) | yes | [1536] | [480.0] | 0.0 | 31/35 |
| gpt-oss-20b | 12 | 24576 | 24576 (intercept -0 B) | yes | [768] | [18.0] | 0.0 | 31/35 |
