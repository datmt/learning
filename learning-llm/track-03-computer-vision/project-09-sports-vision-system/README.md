# Project 9 — Sports Vision System (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

> New to sports vision? Read `lecture.ipynb` first — a beginner-friendly visual intro (detect → track → homography, gating/coasting) that runs in seconds on numpy/matplotlib only.

## Measured (this machine, CPU, synthetic 48-frame 160×120 clip: 8 players + ball, 6% occlusion)
- Classical color detector + BFS components → **recall 412/415 = 0.993, mean err 0.76px, 3.3ms/frame**
- Greedy centroid tracker (gate 12px, coast ≤3 frames) → **11 IDs for 9 entities** (one occlusion split), 49 coasted entity-frames
- DLT homography reprojection err **~2e-12 m**; per-track distance 3.7–34.9 m + top speed in `results/summary.csv`; per-frame tracks in `results/tracks.csv`; figure in `results/pipeline.png`

## The 7 README questions (fill after running)
1. **What did I build?** detect (color masks + connected components) → track (greedy matching + coasting) → homography (DLT) → distances/speeds/heatmap.
2. **How does it work?** Blobs become boxes; nearest-centroid matching within a gate links frames, unmatched tracks coast; 4 pitch-corner correspondences solve H via SVD, mapping pixels to meters.
3. **What did I measure?** table above + `results/tracks.csv` + `results/summary.csv`.
4. **What surprised me?** The ball (2px) detects fine because lines are gray-220, not white — separability by design beats a smarter algorithm.
5. **Bottleneck?** Python BFS labeling dominates per-frame cost; occlusion is the only ID-switch source.
6. **What if X?** Smaller gate fragments tracks; longer coasting heals splits but risks wrong merges; real video needs YOLO + Kalman (ultralytics import is stubbed in).
7. **Next?** Kalman prediction, Hungarian matching, real pitch footage, VLM commentary (feeds P19/P24).
