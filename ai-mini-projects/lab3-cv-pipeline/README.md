# Lab 3: Computer vision pipeline

Question: when the model is fast, what is the pipeline bottleneck?

## Definition of done
- [ ] Pretrained YOLO detector on a short video, boxes drawn / saved
- [ ] Per-stage timing: decode, preprocess, inference, postprocess
- [ ] Tracking added only after detection works
- [ ] Hypothesis written before each experiment (`experiments.md`)
- [ ] Experiments run, data saved:
  - [ ] 640px vs 1280px input
  - [ ] every 1 / 2 / 5 frames
  - [ ] CPU vs GPU inference
  - [ ] detection alone vs detection + tracking
- [ ] Reported: images/sec vs video FPS, real-time or not
- [ ] Can name the bottleneck stage per experiment, backed by timings
- [ ] Workload profile in `../profiles/cv.md`
