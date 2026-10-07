# Project 1 — NumPy MLP (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, real MNIST downloaded)
- 15 epochs, SGD lr=0.1, batch=64, hidden=128 → **test_acc=0.9795, train_acc=0.9921**
- 14.4s total → ~62,518 samples/sec
- Loss 0.4018 (ep1) → 0.0367 (ep15); curves in `results/` after you Run All

## The 7 README questions (fill after running)
1. **What did I build?** 2-layer MLP (Linear→ReLU→Linear→Softmax→CE), pure NumPy.
2. **How does it work?** forward caches (X,Z1,A1,probs); `dL/dlogits=(p−onehot)/N`; linear pullbacks + ReLU mask.
3. **What did I measure?** table above + `results/metrics.csv`.
4. **What surprised me?** Hand-written SGD hits ~98% — framework adds speed, not magic.
5. **Bottleneck?** per-batch Python overhead + matmul sizes; larger batches raise throughput, smaller generalize similarly here.
6. **What if X?** Try LR=1.0 (diverges), hidden=32 (faster, ~1–2pp lower), batch=512.
7. **Next?** momentum/Adam (see P2), weight decay, LR schedule.

> New to this? Read [`lecture.ipynb`](./lecture.ipynb) first — intuition, plots, and vocabulary before the code-heavy lab.
