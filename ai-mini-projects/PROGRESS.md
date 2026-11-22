# Progress

Source of truth between sessions. Update after each finished step. Details live in each folder's README (goal + definition of done).

## Status

| Part | Status |
|---|---|
| common | DONE (Model protocol, Dummy, Bench: load time, p50/p95/p99, throughput, concurrency, JSON results) |
| lab1-llm-server | IN PROGRESS: step 0, environment + hypotheses |
| lab2-embeddings | not started |
| lab3-cv-pipeline | not started |
| lab4-tiny-nn | not started |
| lab5-diffusion | not started |
| lab6-audio | not started |
| platform | not started |
| profiles | not started |

## Next step
Lab 1 (on `gb`):
0. Move repo to `gb`, open it with VS Code Remote-SSH.
1. Run `ollama serve`, pull a small model (1-3B), confirm `ollama run <model> "hi"` works.
2. Write hypotheses in `lab1-llm-server/experiments.md` BEFORE measuring (prompt length vs TTFT, output length vs latency, 1 vs 4 concurrent).
3. Write `lab1-llm-server/ollama_model.py` (stdlib only, streaming `/api/generate`, metrics: ttft, tokens).
4. Run through `common/bench.py`.

## Machine
- Laptop: no GPU, no Ollama server. Not used for the labs any more.
- `ssh dat@gb` (host gb10-001, aarch64, NVIDIA GB10, unified memory, 118 GB disk free): all work and all recorded experiments happen here. Ollama and nvidia-smi installed; Ollama server not started, no models pulled yet.
- Put `"machine"` in each result config so runs are never mixed across machines.
- GB10 note: `nvidia-smi` memory.total shows N/A (unified memory). GPU memory readings need another approach later. vLLM/SGLang on aarch64 may need special builds.

## Open questions
- Which small model for Lab 1 (1-3B)?
- Is the repo moved to `gb` (clone or rsync)? Note the repo root is untracked in git.

## Notes / lessons so far
- Dict access uses `d["key"]`, objects use `obj.attr`.
- p50/p95/p99 = tail latency; averages hide outliers.
- Sleeping dummy scales throughput ~N x with N threads and keeps latency flat. Real models will not: that gap is Lab 1.
