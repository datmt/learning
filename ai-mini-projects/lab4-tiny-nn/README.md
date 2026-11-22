# Lab 4: Train a tiny neural network

Question: what keeps a GPU busy, idle, or underused?

## Definition of done
- [ ] Small PyTorch classifier on MNIST / Fashion-MNIST, trains and reaches sane accuracy
- [ ] Train loop and inference loop separate; both timed
- [ ] Hypothesis written before each experiment (`experiments.md`)
- [ ] Experiments run, data saved:
  - [ ] CPU vs GPU
  - [ ] batch size 1 vs 32 vs 128
  - [ ] different learning rates (loss curves compared)
  - [ ] synthetic data vs real data loading (DataLoader workers)
- [ ] GPU utilization observed (`nvidia-smi`) during runs
- [ ] Can explain: why small batch underuses GPU, when data loading is the bottleneck
- [ ] Workload profile in `../profiles/training.md`
