# Project 20 — RAG From Scratch (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, 48 synthetic docs / 24 questions, numpy-only, ~0 s)
- Corpus: 6 topics × 8 docs (~51 words each, unique entity per doc, fact sentence carries the number but NOT the entity); vocab **137**, avg chunk len **49.5** words
- Main config (chunk-120 ≈ whole doc): TF-IDF and BM25 both **r@1 = r@3 = r@5 = MRR = 1.000**; +bigram-rerank ties at 1.000
- Chunk sweep (`results/retrieval.csv`): chunk-30 **r@1 0.042 / r@3 0.667 / r@5 0.875 / MRR 0.358** → chunk-60/120 all 1.000 — 30-word windows split entities from facts and retrieval collapses
- Rerank pathology: bigram-rerank at chunk-30 drops r@3 **0.667 → 0.000** (MRR 0.358 → 0.138) — lexical overlap promotes entity chunks and buries the split evidence
- Generation (`results/eval.csv`): EM whole-doc **1.000** (hit@3 1.000) vs EM chunk-30 **0.250** vs closed-book **0.000**; split@chunk-30 (16 hit / 8 miss): **EM|hit 0.375, EM|miss 0.000**
- Prompt an LLM would get saved to `results/prompt_example.txt`; figure in `results/rag.png`

## The 7 README questions (fill after running)
1. **What did I build?** Full RAG loop from scratch: word chunker → TF-IDF + BM25 (numpy, no libs) → cosine index → top-k → bigram rerank → prompt build → extractive reader (offline LLM stand-in) vs closed-book.
2. **How does it work?** Log-tf × idf with L2 rows (cosine = dot) and Okapi BM25 rank chunks; gold = the chunk holding the answer span; the reader picks the best-overlap sentence with a +3 answer-type prior for digit-bearing sentences on how-many questions.
3. **What did I measure?** table above + `results/retrieval.csv` (chunk/overlap/rerank sweep) / `results/eval.csv` (24 per-question hit/EM rows).
4. **What surprised me?** Rerank *destroys* retrieval at chunk-30 (r@3 0.667 → 0.000). Bigrams match the question's entity, so reranking promotes the entity chunk and pushes the entity-less fact chunk out of the top-3 — a reranker can't fix split evidence, it amplifies the split.
5. **Bottleneck?** Chunking, not scoring: TF-IDF and BM25 tie at 1.000 on whole docs and collapse together on split evidence. Retrieval quality caps generation — EM|miss is 0.000 no matter the reader.
6. **What if X?** 30→60 words restores r@3 0.667→1.000 (entity + fact reunited); overlap-10 shifts r@3 0.667→0.583 (re-chunking moves boundaries, doesn't stitch a 40-word gap); top-k 1→5 lifts recall 0.042→0.875 (deeper k compensates worse ranking, at more reader cost).
7. **Next?** P21 turns retrieval into one tool among many (calculator / fs / python / db / http) inside a guarded plan→act→observe loop — every call metered for retries, timeouts, and denials.

> **New here?** Start with [lecture.ipynb](lecture.ipynb) — a beginner-friendly companion that teaches the fundamentals first. Read it before opening `notebook.ipynb`.
