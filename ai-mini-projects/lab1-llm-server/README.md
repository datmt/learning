# Lab 1: LLM inference server

Question: what drives cost, latency, throughput of an LLM behind an API?

## Definition of done
- [ ] Ollama model wrapped as a `Model` from `common`
- [ ] Small OpenAI-compatible API (`/v1/chat/completions`) in front of it
- [ ] Bench measures TTFT, time per output token, output tokens/sec
- [ ] Hypothesis written BEFORE each experiment (in `experiments.md`)
- [ ] Experiments run, data saved:
  - [ ] short vs long prompt
  - [ ] short vs long output
  - [ ] 1 vs N concurrent requests
  - [ ] different quantization levels
  - [ ] different batch sizes (vLLM/SGLang if hardware allows, else note why skipped)
- [ ] Each result: hypothesis held or not, and why
- [ ] Can explain, without guessing: weights vs KV cache, prefill vs decode, why throughput changed
- [ ] One-page LLM workload profile written to `../profiles/llm.md`
