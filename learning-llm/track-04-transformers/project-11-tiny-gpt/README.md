# Project 11 — Tiny GPT (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

> New to GPTs? Read `lecture.ipynb` first — a beginner-friendly visual intro (tokenizer, next-token loss, perplexity, temperature) that runs in seconds on numpy/matplotlib only.

## Measured (this machine, CPU, real TinyShakespeare 120k chars)
- Char tokenizer, vocab **61**, train 108k / val 12k ids; uniform baseline ln(61) = **4.1109**
- TinyGPT (tok+pos embed, 4 blocks, d=96/h=4, ctx=48): **462,336 params (0.462M)**; init loss 4.28 ≈ baseline
- AdamW 3e-4, 300 iters × 1152 tok (345.6k tokens, 6.4s → **~54k tok/s**) → **train 2.3717, val 2.4336, ppl 11.4** (vs uniform 61)
- Samples (`results/samples.txt`) are Shakespeare-flavored gibberish — correct letter texture, no real words yet (expected at 0.5M params / 300 iters)
- Ablation: with-pos val **2.4336** (300 it) vs no-pos **2.5548** (100 it) — order matters
- Context probe (mean next-token entropy, 30 val windows): ctx8 **2.600** / ctx16 **2.371** / ctx32 **2.477** / ctx48 **2.639** nats — 16-token sweet spot for this capacity
- Curves in `results/curves.png`, losses in `results/loss.csv`, ablation in `results/ablation.csv`
- Offline-safe: download fails → deterministic synthetic fallback (same code path, vocab 22)

## The 7 README questions (fill after running)
1. **What did I build?** A char-level GPT (embeddings + 4 pre-norm blocks with fused-QKV causal MHA + widen-4 MLP + LM head) trained with next-token cross-entropy on TinyShakespeare, plus greedy/temperature sampling.
2. **How does it work?** Text → ids; each step predicts the next id from all previous ones (y = x shifted by 1, causal mask enforces it); CE pushes probability onto the true next char; sampling feeds the model's own output back through a 48-token sliding window, temperature reshaping the distribution.
3. **What did I measure?** table above + `results/loss.csv`.
4. **What surprised me?** 6 seconds / 300 iters already buys ppl 11.4 from 61 — char-LM signal is dense. And the context probe peaks at 16 tokens: extra context *hurts* a 0.5M model (distant tokens are noise it lacks capacity to use).
5. **Bottleneck?** CPU matmuls per token×param (~54k tok/s here); attention is O(T²) but at T=48 the linears dominate — the P12 flop math shows when that flips.
6. **What if X?** No positions → +0.12 val loss (bag-of-chars can't do order); t=1.2 vs 0.7 → wilder, less verse-like (see `samples.txt`); longer ctx alone doesn't help without more params.
7. **Next?** Scale the same recipe (P12 accounts 10M→50M→1B), then fine-tune it (Session 5: SFT vs LoRA).
