# Platform: combine the labs

Do after Labs 1–6. One service routing to LLM, embedding, CV (and others).

## Definition of done
- [ ] Gateway + model router; routes by request type
- [ ] Request queue
- [ ] Per-model concurrency limits
- [ ] Model load / unload on demand
- [ ] GPU-memory-aware scheduling (refuses or queues when VRAM short)
- [ ] Metrics + tracing visible for every request
- [ ] Model versioning
- [ ] Retry + timeout policies
- [ ] Quality evaluation hook
- [ ] Written answers, backed by measurements:
  - [ ] LLM and CV share one GPU?
  - [ ] Separate embedding worker?
  - [ ] Keep loaded vs load on demand?
  - [ ] What happens when two models compete for memory?
  - [ ] Which workloads batch / need low latency / run async?
