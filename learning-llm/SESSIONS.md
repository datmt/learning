# Sessions — build order + exit criteria

## Status
- [x] Session 1 (Track 1: P1–P3) — BUILT in this delivery
- [x] Session 2 (Track 2: P4–P6) — BUILT (all 3 notebooks execute with 0 errors, CPU-verified; GPU rows auto-skip, rerun on DGX Spark)
- [x] Session 3 (Track 3: P7–P9) — BUILT (all 3 notebooks execute with 0 errors, CPU-verified; CIFAR-10 cached in P7 data/, reruns offline-capable via synthetic fallback)
- [x] Session 4 (Track 4: P10–P12) — BUILT (all 3 notebooks execute with 0 errors, CPU-verified; GPU rows auto-skip, rerun on DGX Spark)
- [x] Session 5 (Track 5: P13–P15) — BUILT (all 3 notebooks execute with 0 errors, CPU-verified; TinyShakespeare cached per-run, reruns offline-capable via synthetic fallback)
- [x] Session 6 (Track 6: P16–P18) — BUILT (all 3 notebooks execute with 0 errors, CPU-verified; P16 fake-quant CPU, P17 TestClient no-ports, P18 calibrated sim; rerun on DGX Spark for real kernel/GPU numbers)
- [x] Session 7 (Track 7: P19–P21) — BUILT (all 3 notebooks execute with 0 errors, CPU-verified; P19 synthetic+sklearn offline, P20 numpy-only synthetic, P21 stdlib mock-http; rerun anywhere, DGX Spark optional)
- [x] Session 8 (Track 8: P22–P24) — BUILT (all 3 notebooks execute with 0 errors, CPU-verified; P22 single-proc emulation, P23 TestClient no-ports, P24 synthetic video; rerun on DGX Spark for real NCCL/GPU numbers)

## Session 1 exit criteria (verify before moving on)
1. P1 notebook runs end-to-end, test accuracy ≥ 85% on digits (≥90% on MNIST if downloaded).
2. Can explain in own words: where gradient comes from (chain rule through softmax+CE).
3. P2 loss curve matches/beats P1; can explain forward/backward/step/zero_grad.
4. P3 produces batch-size vs samples/sec CSV + plot; can explain why GPU wins (parallelism + bandwidth) and when it doesn't (small batch, transfer-bound).
5. Each project has filled README answering the 7 questions.

## Session 2 exit criteria (verify before moving on)
1. P4 notebook runs end-to-end, `results/bench.csv` has all CPU rows; can explain warmup/sync/median timing and intensity = flops/byte.
2. P5 thread-simulator matches torch on all 5 ops; can write `i = blockIdx.x*blockDim.x + threadIdx.x` + guard from memory and name what each GPU backend needs.
3. P6 shows a strided-access slowdown, a tiling result, and a serial-vs-tree reduction factor; can state what nvidia-smi vs nsys vs ncu each answer.
4. Each project has filled README answering the 7 questions.

## Session 3 exit criteria (verify before moving on)
1. P7 notebook runs end-to-end, `results/metrics.csv` + `ablation.csv` written; hand `conv2d` matches torch (<1e-4); can explain sharing/pooling/receptive field and why BN didn't win at 5 epochs.
2. P8 IoU/NMS unit tests pass, val mIoU ≥ 0.6 with prec@0.5 ≥ 0.85; can write IoU from memory and state what NMS threshold trades off.
3. P9 detection recall ≥ 0.90, homography corner error ~0, `results/tracks.csv` + `summary.csv` + heatmap written; can explain gate/coasting and what H maps.
4. Each project has filled README answering the 7 questions.

## Session 4 exit criteria (verify before moving on)
1. P10 notebook runs end-to-end, hand attention matches torch (<1e-9), causal leak = 0; can write `softmax(QKᵀ/√d)V` + causal mask from memory and state why the scale exists.
2. P11 trains to val CE clearly below ln(V) (verified 2.43 vs 4.11, ppl 11.4) and generates conditioned text; can explain tokenizer → batching-shift → CE → sampling loop and why no-pos loses.
3. P12 param audit matches torch exactly on TinyGPT, estimates land within ~10% of published totals; can write KV-cache bytes formula with GQA and state the 4K→32K 8× tax.
4. Each project has filled README answering the 7 questions.

## Session 5 exit criteria (verify before moving on)
1. P13 notebook runs end-to-end, analytic param formula matches torch exactly, `results/configs.csv` + `projections.csv` written; can write the 14-bytes/param bf16 training rule and state why tok/s falls ~1/P.
2. P14 table shows full-FT vs SFT vs LoRA vs QLoRA on trainable-params/memory/time/quality; can explain answer-masking, LoRA B=0 init, and the 0.5 B/param QLoRA base.
3. P15 harness unit tests pass, `results/eval.csv` splits base vs LoRA vs RAG, judge-5 rate == EM rate; can state what each metric (EM/BLEU/ROUGE/F1/judge) rewards and why RAG needs the softer ones.
4. Each project has filled README answering the 7 questions.

## Session 6 exit criteria (verify before moving on)
1. P16 notebook runs end-to-end, `results/quant.csv` 7-format frontier + `results/quant_groups.csv` written; BF16 ~lossless (dCE < 0.05), INT4 werr ladder per-tensor→g16; can write affine `q=clamp(round(w/s))` + bytes/param per format from memory.
2. P17 contracts pass (400 empty, clamp max_tokens, SSE frames + DONE), batch ~1.5x single tok/s, 8-thread overload → 200s + 429s with zero 500s; can explain streaming/TTFT vs batching/tok/s vs queue/429.
3. P18 continuous beats static (makespan ×~1.3, TTFT halves, pad waste 30→0), paged KV ~1.5x + prefix sharing; can define TTFT/TPOT and compute block-table waste vs contiguous reservation.
4. Each project has filled README answering the 7 questions.

## Session 7 exit criteria (verify before moving on)
1. P19 notebook runs end-to-end, VLM joint ≈ 1.0 with text-only ≈ 0.2 and image-only color ≈ 0.0, `results/vlm.csv` + `ablations.csv` written; can draw image → encoder → projection → prefix → answer and state every tensor shape.
2. P20 TF-IDF/BM25 unit tests pass, whole-doc recall@3 = 1.0, chunk-30 recall@3 collapses (< 0.8), EM main (1.0) > EM chunk-30 > closed-book (0.0); can write idf from memory and state what small chunks split.
3. P21 tool unit tests pass (injection/traversal/404 fail closed), full config 9/10 with each guard-off config losing exactly its task class (`results/tasks.csv` 50 rows); can state what retry/timeout/memory/permission each rescue.
4. Each project has filled README answering the 7 questions.

## Session 8 exit criteria (verify before finishing)
1. P22 notebook runs end-to-end, ring simulator matches torch exactly (err 0.0), DDP mean(micro-grads) == large-batch grad (<1e-5), `results/scale.csv` shows eff falling with N and NVLink beating PCIe at fixed N; can write ring bytes `2(N-1)/N·B` + eff `T1/(N·T_N)` + FSDP/TP per-rank bytes from memory.
2. P23 contracts pass (401 on bad auth, 429 on over-budget + burst overload with zero 500s, cache hit on repeat, batch-of-4 faster than 4× single), `results/platform.csv` 22-row ledger written; can state what auth/ratelimit/queue/batch/cache/metrics/trace/gpu-monitor each rescue.
3. P24 detection recall ≥ 0.90 (verified 1.00 clean + noisy), homography corner error ~0, `results/tracks.csv` + `summary.csv` + `vlm.csv` + `report.md` written, VLM ball-half 1.0 > text-only; can explain gate/coasting, what H maps, and which stage each metric blames.
4. Each project has filled README answering the 7 questions.

## Month mapping (from your original plan)
Months 1–6 map to Sessions 1–8 (S1=Month1-part1, S2=Month1-part2+Month2-start, etc.).
Don't block: useful skills ship after every session.

## How to run
```bash
pip install -r requirements.txt
jupyter lab
# open track-01-ml-foundations/project-0X-*/notebook.ipynb, Run All
```
Headless verify:
```bash
jupyter nbconvert --to notebook --execute track-01-ml-foundations/project-01-numpy-mlp/notebook.ipynb --output /tmp/p1-out.ipynb --allow-errors
```
