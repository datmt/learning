# Project 2 — PyTorch MLP (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: 0 errors.

## Measured (CPU, MNIST)
- 10 epochs, Adam lr=1e-3, batch=64 → **test_acc=0.9760**
- 7.6s total → ~78,839 samples/sec (faster than P1 NumPy/SGD per-epoch: Adam converges in fewer epochs)
- Artifacts: `results/metrics_pytorch.csv`, `results/curves_pytorch.png`

## Key mechanics demonstrated
- `zero_grad()` — grads **accumulate**; must clear each step.
- `forward` builds autograd graph; `loss.backward()` fills `.grad`; `opt.step()` updates.
- `CrossEntropyLoss` takes **logits** (applies log-softmax internally) — don't pre-softmax.

## The 7 questions
1. Built: same MLP via nn.Linear/ReLU + DataLoader + Adam. 2. Works via autograd chain rule. 3. Measured above.
4. Surprise: fewer epochs to same accuracy vs P1 SGD. 5. Bottleneck: DataLoader/host overhead at small batch.
6. Try: SGD(lr=0.1) vs Adam table; batch 512. 7. Next: P3 GPU + mixed precision.

> New to this? Read [`lecture.ipynb`](./lecture.ipynb) first — intuition, plots, and vocabulary before the code-heavy lab.
