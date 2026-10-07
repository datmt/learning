# Project 19 — Tiny VLM (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, synthetic 3×24×24 + sklearn digits, train 3100 / val 777)
- TinyVLM: CNN encoder → projection → K=4 visual prefix + 12-word questions → 2-layer Transformer → 24-class answer head (**435,832** params, 8 epochs, ~8 s train)
- Full VLM joint **0.999** (val CE **0.007**): shape **1.000**, color **1.000**, chart **1.000**, OCR **0.997**, video **1.000**
- Baselines (`results/vlm.csv`): text-only (vis=0) **0.205**, image-only (txt=0) **0.822** — but image-only shape **0.646** / color **0.000**: without the question it always guesses one side of the shared image
- Ablations (`results/ablations.csv`): K=1 **0.988**, K=8 **0.988**, frozen random encoder **0.910**, pixel noise σ=0.25 **0.995** vs clean 0.999
- Curves + example pixels in `results/vlm.png`

## The 7 README questions (fill after running)
1. **What did I build?** A miniature VLM: one CNN encoder + linear projection feeding K visual tokens as a prefix into a 2-layer Transformer that answers shape/color/chart/OCR/motion questions with a 1-token head.
2. **How does it work?** `encode_vis` maps pixels → (B,K,64); the sequence [vis K | question 4] gets positions + self-attention (every text position attends to every visual token); mean-pooled question states → 24-way logits, trained with multitask CE.
3. **What did I measure?** table above + `results/vlm.csv` / `results/ablations.csv`.
4. **What surprised me?** Image-only reaches 0.822 joint while scoring 0.000 on color — it learned "always answer shape", which is right exactly when the hidden question happens to ask shape. Joint accuracy hides the blindness; the per-task split exposes it.
5. **Bottleneck?** The projection + prefix: K=1 already hits 0.988 (4 visual tokens are plenty at 24×24), while a frozen random encoder still reaches 0.910 — the transformer + head compensate, so encoder training buys the last ~9 pts, not the first 90.
6. **What if X?** K=1 vs K=8 tie at 0.988 (capacity isn't the constraint at this resolution); σ=0.25 noise costs 0.004 (synthetic edges survive Gaussian noise); freezing the encoder costs 0.089 (features matter more than prefix length here).
7. **Next?** Text-only-LM (≈0.2 here) becomes the closed-book arm in P20: give the LM retrieved evidence and split retrieval misses from reader misses.

> **New here?** Start with [lecture.ipynb](lecture.ipynb) — a beginner-friendly companion that teaches the fundamentals first. Read it before opening `notebook.ipynb`.
