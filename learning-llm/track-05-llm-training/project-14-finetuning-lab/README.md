# Project 14 — Fine-tuning Lab (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, real TinyShakespeare 60k chars for base pretrain)
- Task: reverse a random 4-char word (`reverse acff is ffca`), 200 train / 32 held-out val words from an 8-letter alphabet (4096-word pool: memorization-proof, must learn the rule)
- Base (2-block TinyGPT, 108,032 params, 80 pretrain iters): val **3.9995**, EM **0.00** — never saw the task
- 4 methods from the identical checkpoint: full-FT (300 it, all tokens) → val **0.3310**, EM **1.00**; SFT (300 it, answer tokens only) → val **0.0204**, EM **1.00**; LoRA-r8 (650 it, 16,384 trainable = **13.17%**) → val **0.4616**, EM **1.00**; QLoRA-r8 (same + simulated int4 base) → val **0.5286**, EM **1.00**
- Cost: full/SFT **1.73 MB** train-mem vs LoRA **0.69 MB** vs QLoRA **0.32 MB** (5.4x win over SFT); rank sweep (450 it): r1 EM 0.0/val 2.66, r4 EM 0.0/val 2.37, r16 EM 1.0/val 0.47
- Tables in `results/finetune.csv`, `results/lorarank.csv`, figure in `results/finetune.png`
- Offline-safe: download fails → deterministic synthetic fallback + alphabet patch (same code path)

## The 7 README questions (fill after running)
1. **What did I build?** One base checkpoint tuned 4 ways (full-FT, SFT with answer-masking, LoRA from scratch, QLoRA with simulated 4-bit base) plus a LoRA rank sweep, all scored by greedy exact-match.
2. **How does it work?** Full-FT backprops every next-token; SFT masks prompt positions (`-100`) so gradients focus on answers; LoRA freezes the base and trains rank-r A/B (B zero-init: day-0 delta is 0); QLoRA additionally round-trips frozen weights through int4 levels and counts 0.5 B/param.
3. **What did I measure?** table above + `results/finetune.csv`.
4. **What surprised me?** SFT's val loss (0.0204) is 16x lower than full-FT's (0.3310) at equal steps — masking the prompt is worth more than 16x the gradient signal on it. And adapters (B=0 start) needed 650 iters vs 300: cheap per step, slower to warm up.
5. **Bottleneck?** Sample efficiency for adapters (zero-start), prompt-token dilution for full-FT; memory for full methods (16 B/trainable-param AdamW vs 0.5 B/frozen-param QLoRA).
6. **What if X?** Rank 1→4 barely moves (EM 0, val ~2.5); rank 16 solves it (EM 1.0, val 0.47) — reversal needs a minimum rank; below it the adapters only memorize.
7. **Next?** Score outputs like a benchmark instead of eyeballing (P15: EM, BLEU/ROUGE, classification metrics, LLM-as-judge; base vs LoRA vs RAG).

> Beginner? Start with [`lecture.ipynb`](./lecture.ipynb) — visuals-first intro, read before `notebook.ipynb`.
