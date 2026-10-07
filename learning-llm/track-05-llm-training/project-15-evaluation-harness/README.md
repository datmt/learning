# Project 15 — Evaluation Harness (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, real TinyShakespeare 60k chars for base pretrain)
- Task: two-word reversal (`reverse abcd efgh is dcba hgfe`), **compositional split**: 24 novel combos of known words (train 120 combos; words repeat by design — that is the RAG test)
- Harness unit tests pass: EM case/space handling, BLEU hand value 0.3679, ROUGE-1/L 0.75, LCS skip, judge-5 on exact, sklearn sanity
- Systems: base (frozen zero-shot) → EM **0.0**, BLEU **0.300**, R1/RL **0.535**, word_acc **0.0**, judge **1.0**; LoRA-r16 tuned (800 it) → **1.0 / 1.0 / 1.0 / 1.0 / 1.0 / 1.0**, judge **5.0**; RAG (lexical retrieve + compose-the-facts reader, k=4) → EM **0.708**, BLEU **0.910**, R1/RL **0.964**, word_acc **0.75**, macro-F1 **0.70**, judge **4.0**
- RAG top-k sweep: k0 **0.0** / k1 **0.0** / k2 **0.542** / k4 **0.708** — retrieval recall drives quality
- Judge↔EM self-check holds (judge-5 rate == EM rate per system, asserted); base BLEU/R1 are nonzero from prompt-word overlap — answer-only word_acc is the strict view
- Tables in `results/eval.csv`, `results/judge.csv` (72 rows), `results/ragk.csv`, figure in `results/eval.png`
- Offline-safe: download fails → deterministic synthetic fallback + alphabet patch (same code path)

## The 7 README questions (fill after running)
1. **What did I build?** A from-scratch eval harness (EM, BLEU with clipped precision + brevity penalty, ROUGE-1/L F1, sklearn word accuracy + macro F1, 1–5 rubric judge with a real-LLM prompt template) scoring 3 live systems.
2. **How does it work?** Base guesses zero-shot; LoRA stores the reversal rule in weights (r16 adapters); RAG retrieves train pairs by exact-word hits + bigram backoff and composes per-word facts with a symbolic reader (stands in for a capable LM reader — the point is the retrieval/generation split).
3. **What did I measure?** table above + `results/eval.csv`.
4. **What surprised me?** EM calls base-vs-RAG 0.0 vs 0.708 — but even where EM ties systems, BLEU/ROUGE split them (RAG 0.91 vs base 0.30): strict metrics hide partial retrieval value, which is exactly why harnesses report both.
5. **Bottleneck?** Retrieval coverage (k=1 finds one word at best → EM 0; k=4 covers both words 71% of the time); the frozen LM alone contributes nothing (few-shot EM 0 even with exact-word demos — reader capacity matters).
6. **What if X?** k 1→2→4: EM 0 → 0.54 → 0.71 — each added demo is a recall lottery for the missing word; a coverage-maximizing retriever (not just top-k overlap) is the obvious next upgrade.
7. **Next?** Session 6: make it cheap and fast — quantization lab (P16), mini LLM server (P17), vLLM concepts (P18).

> Beginner? Start with [`lecture.ipynb`](./lecture.ipynb) — visuals-first intro, read before `notebook.ipynb`.
