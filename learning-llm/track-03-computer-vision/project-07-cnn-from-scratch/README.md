# Project 7 — CNN From Scratch (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

> New to CNNs? Read `lecture.ipynb` first — a beginner-friendly visual intro (sliding filters, pooling, receptive field) that runs in seconds on numpy/matplotlib only.

## Measured (this machine, CPU, real CIFAR-10 subset 4000/1000)
- TinyCNN (~60k params, 2 conv blocks), Adam lr=3e-3, 5 epochs → **test_acc=0.4410, test_loss=1.6224** (train 0.559)
- ~1.1s/epoch on CPU; hand-written `conv2d` matches torch to 9.5e-07
- Receptive field at classifier: **10×10 of 32×32**; k3 vs k5 first layer: 864 vs 2400 params, 37.6 vs ~46ms per 20 forwards
- Curves in `results/curves.png`, per-epoch table in `results/metrics.csv`, BN ablation in `results/ablation.csv`

## The 7 README questions (fill after running)
1. **What did I build?** TinyCNN (Conv→BN→ReLU→Pool ×2 + linear head) plus a from-scratch NumPy `conv2d` verified against torch.
2. **How does it work?** Each filter slides over the image reusing 9 weights per channel (sharing); max-pool keeps the strongest response in each 2×2 window (shift invariance); BN standardizes each channel per batch.
3. **What did I measure?** table above + `results/metrics.csv`.
4. **What surprised me?** Without-BN scored 0.477 vs with-BN 0.441 at 5 epochs — BN's payoff needs longer schedules; also 60k params already beat chance (0.10) 4× over.
5. **Bottleneck?** CPU conv throughput on 32×32 batches; data loading is negligible at this size.
6. **What if X?** k5 first layer: +2.8× params for slightly slower forwards; more epochs or augmentation close the train/test gap.
7. **Next?** Deeper stack (3 blocks), random crops/flips, LR schedule — all standard CIFAR-10 recipe steps.
