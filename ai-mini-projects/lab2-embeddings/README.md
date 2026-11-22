# Lab 2: Embedding and retrieval system

Question: where does time and memory go in semantic search?

## Definition of done
- [ ] Small embedding model wrapped as a `Model`
- [ ] PostgreSQL + pgvector running
- [ ] Ingest script: documents -> embeddings -> table
- [ ] Search API: query text -> top-K documents
- [ ] Hypothesis written before each experiment (`experiments.md`)
- [ ] Experiments run, data saved:
  - [ ] 10K vs 100K vectors
  - [ ] exact search vs ANN index (HNSW/IVFFlat), with recall measured against exact
  - [ ] different vector dimensions
  - [ ] different top-K
  - [ ] batch embedding vs one-at-a-time
- [ ] Measured: embedding cost, index build time, storage size, query latency
- [ ] Can explain recall vs latency trade-off with your numbers
- [ ] Workload profile in `../profiles/embedding.md`
