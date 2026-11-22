# common: shared bench framework

Used by every lab. Build first.

## Goal

Every lab asks the same question: how does this model behave inside a system (latency, throughput, memory)? To compare labs fairly, all of them need the same way to plug in a model and the same way to measure it.

`common/` is that shared plug and ruler. After this, a new lab only has to write a small wrapper around its model, and the benchmark, percentiles and result files come for free.

You are learning to trust a measurement tool before measuring anything unknown.

## Pieces

| File | What it is | Why it exists |
|---|---|---|
| `model.py` | `Model` protocol: `load()`, `predict(request)`, `unload()` | One contract every model follows. Load is separate from predict so load time is not hidden inside the first request. Unload is needed later for the platform's load/unload scheduling. |
| `dummy.py` | Fake model that sleeps a known time | The ruler. You know the right answer (e.g. 0.1 s), so if the runner reports something else, the bug is in the runner, not a model. Also lets you build with no GPU or downloads. |
| `bench.py` | Runner: takes a `Model` and a list of requests, measures, reports | The core. Times load, times each `predict`, computes p50/p95/p99 and requests/sec, optionally runs N requests in parallel. |
| `results/` | One JSON/CSV file per run | Experiments compare runs later. Each file stores config + timestamp so you know what produced it. |

## Design decision: `predict` returns metrics too

`predict` returns a dict: `{"output": ..., "metrics": {...}}`.

The runner can time a call from outside, but it cannot see inside it. An LLM knows its time-to-first-token and token count; the runner does not. Models with nothing extra return an empty `metrics`. Later labs reuse the same slot (CV stage timings, diffusion peak VRAM, audio duration).

## Definition of done
- [ ] `model.py`: `Model` protocol with `load`, `predict`, `unload`, type hints, imports cleanly
- [ ] `dummy.py`: implements the protocol; `predict` sleeps and returns `{"output", "metrics"}`
- [ ] `bench.py`: takes a `Model` + list of requests
- [ ] Records load time per run
- [ ] Records per-request latency; reports p50 / p95 / p99
- [ ] Concurrency knob: run N requests in parallel
- [ ] Reports throughput (requests/sec)
- [ ] Results saved to one JSON/CSV file per run (timestamp + config included)
- [ ] Proof: dummy sleeping 0.1 s gives p50 ≈ 0.1 s
- [ ] Proof: with concurrency N and a sleeping dummy, throughput rises roughly N times. Explain why.
- [ ] One command runs the dummy bench end to end

## Not now (YAGNI)
GPU utilization, memory usage, cost per request. Add when Lab 1 needs them.
