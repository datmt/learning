# Lab 5: Diffusion image generation service

Question: what happens when one request occupies the accelerator for seconds?

## Definition of done
- [ ] Small pretrained diffusion model wrapped as a `Model`
- [ ] API: prompt -> image
- [ ] Hypothesis written before each experiment (`experiments.md`)
- [ ] Experiments run, data saved:
  - [ ] 20 vs 30 vs 50 steps
  - [ ] different resolutions
  - [ ] different precision (fp32 / fp16 / bf16)
  - [ ] 1 image vs batch
- [ ] Measured: latency, peak VRAM, load time
- [ ] Saved sample images per setting; quality vs speed judged side by side
- [ ] Can explain why latency scales with steps and resolution
- [ ] Workload profile in `../profiles/diffusion.md`
