# Lab 6: Audio inference pipeline

Question: how does processing time scale with audio length, and can it run real time?

## Definition of done
- [ ] Small Whisper model wrapped as a `Model`
- [ ] API: audio file -> transcript
- [ ] Real-time factor (processing time / audio duration) computed per run
- [ ] Hypothesis written before each experiment (`experiments.md`)
- [ ] Experiments run, data saved:
  - [ ] 10 s vs 60 s audio
  - [ ] CPU vs GPU
  - [ ] different model sizes
  - [ ] sequential vs batched transcription
- [ ] CPU preprocessing (decode/resample/chunk) timed separately from inference
- [ ] Concurrent sessions tested
- [ ] Can explain streaming vs batch and the chunking trade-off
- [ ] Workload profile in `../profiles/audio.md`
