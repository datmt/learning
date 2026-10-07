# Project 10 — Attention From Scratch (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

> New to attention? Read `lecture.ipynb` first — a beginner-friendly visual intro (Q/K/V voting, 1/sqrt(d) scaling, causal masks) that runs in seconds on numpy/matplotlib only.

## Measured (this machine, CPU)
- Hand NumPy attention matches torch SDPA to **8.4e-15**; causal leak = **0.00e+00**; row sums 1.000000
- Bench (n=256, d=64): **numpy 0.202ms vs torch-CPU 0.124ms** (fused kernel ~1.6×); GPU row auto-skipped (no CUDA here)
- MHA (d=32, h=4): shapes (12,32)/(4,12,12), params **4096 = 4·32²**; sinusoid self-sim 16.00 vs distant 11.43; learned-pos 512 params vs sinusoid 0
- No-scale max-p **1.0000** vs scaled **0.4303**; causal-vs-full max|diff| **0.0233**; heads 1/2/4/8 ≈ flat (0.098→0.143ms); length log-log slope **1.65** (quadratic = 2.0)
- Artifacts: `results/attention_bench.csv`, `attn_map.png`, `attn_weights_h0.csv`, `experiments.csv`, `scaling.png`

## The 7 README questions (fill after running)
1. **What did I build?** Single-head scaled dot-product attention in raw NumPy (verified vs `F.scaled_dot_product_attention`), a causal-mask unit test, a minimal MHA module, sinusoidal vs learned positions, and a 2-head attention-map viz.
2. **How does it work?** Q/K/V are learned projections; scores = QKᵀ/√d keep softmax variance ~1 so gradients flow; the causal mask adds −∞ to future slots so row i only mixes tokens ≤ i; MHA splits d into h subspaces, attends in parallel, concatenates + projects; positions (sinusoid dot-decay or learned table) inject order because attention itself is permutation-invariant.
3. **What did I measure?** table above + `results/*.csv`.
4. **What surprised me?** Without 1/√d the attention collapses to a one-hot (max-p = 1.0) even at unit magnitudes — scaling isn't cosmetic, it's what keeps training alive. Also torch's fused CPU kernel beats hand NumPy 1.6× on the same math.
5. **Bottleneck?** The O(n²) score matrix in time and memory; the kernel-only sweep (slope 1.65) shows it, and full MHA forwards hide it under linear-projection overhead at small n.
6. **What if X?** No scale → saturated one-hot attention, dead gradients; no mask → row 0 changes by 0.023 (future leaks into the past, breaking autoregression); 1 vs 8 heads → same time, different representation splits.
7. **Next?** KV-cache in P12 (this mask is why decode caches K/V), FlashAttention tiling (P6 idea applied to the n² matrix).
