# Project 24 — Capstone: Sports AI Analysis Platform (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

## Measured (this machine, CPU, synthetic 24-frame 96×64 video, ~2 s wall)
- Detection: recall **1.0000** clean and **1.0000** at pixel-noise σ=25 (unique kit colors per id — shared red/blue merged masks into 11 px-centroid errors in the first draft); mean center error **0.135 px**
- Tracking: 5 ids × 24 frames, zero NaN — scripted frame-10 ball dropout coasted; gate 8 px, coast budget 2
- Homography: pure-scale H (105/96, 68/64), corner error **0.00e+00 m**; all tracks mapped to pitch meters
- Stats (`results/summary.csv`): player distance 18.8–25.6 m, top speed 8.9–15.1 m/s (sprint range); ball 46.83 m end to end; possession A **17%** / B **12%** / none 71% (long-ball transition clip); ball top 41.04 m/s is the 2-frame coast jump — coasting corrupts derivatives
- VLM stub (`results/vlm.csv`): ball-half QA vision **1.00** vs text-only **0.625** (constant guess); LLM stub: `results/report.md` templated from measured stats; API stub dict with `/report` + `/tracks` schema
- Figure in `results/capstone.png` (pitch trajectories + 12×8 occupancy heatmap); tracks in `results/tracks.csv` (120 pitch rows)

## The 7 README questions (fill after running)
1. **What did I build?** The full pipeline end to end: seeded video → per-color detection → gated tracking with coasting → homography to meters → distance/speed/possession stats + heatmap → vision QA → templated match report → API-shaped dict.
2. **How does it work?** Color distance <60 segments unique-kit discs to centroids; greedy per-id assignment with an 8 px gate bridges frame gaps up to 2 via last-position repeat; H scales px→m exactly; possession = nearest player within 6 m of the ball; the report is a template filled from measured numbers (documented LLM stand-in).
3. **What did I measure?** table above + `results/tracks.csv` / `results/summary.csv` / `results/vlm.csv` / `results/report.md`.
4. **What surprised me?** Two artifacts taught more than the happy path: shared kit colors silently merged two players into one midpoint track (recall stayed 1.0 — recall can't see identity errors), and the coasted frame doubled the ball's top-speed reading (continuity ≠ differentiability).
5. **Bottleneck?** Identity resolution, not detection: thresholding is pixel-perfect here, but same-color players are indistinguishable without appearance/re-ID features — the exact gap a real detector (P8) + DeepSORT closes.
6. **What if X?** Noise σ=25 changes nothing (colors are far apart — try σ=80 to break it); gate 8→2 px fragments tracks at ball speed (fast objects need wide gates); coast 2→0 turns the frame-10 dropout into a NaN gap (no free continuity).
7. **Next?** Curriculum complete (P1–P24): swap `render()` for video frames, `detect()` for a YOLO head, `track_seq()` for DeepSORT, stubs for served models behind P23's gateway; price workers with P22's efficiency math.

> **New here?** Start with [lecture.ipynb](lecture.ipynb) — a beginner-friendly companion that teaches the fundamentals first. Read it before opening `notebook.ipynb`.
