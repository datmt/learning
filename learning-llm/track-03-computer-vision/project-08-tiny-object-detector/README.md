# Project 8 — Tiny Object Detector (verified run)

Notebook: `notebook.ipynb` — every line commented. Verified: executes with 0 errors (`jupyter nbconvert --execute`).

> New to detection? Read `lecture.ipynb` first — a beginner-friendly visual intro (IoU as sticky notes, NMS cleanup, box vs class heads) that runs in seconds on numpy/matplotlib only.

## Measured (this machine, CPU, synthetic 64×64 shapes: 1500 train / 300 val)
- TinyDet (~549k params, box + class heads), 12 epochs → **val mIoU=0.6863, prec@IoU0.5=0.897, class_acc=1.0000**
- ~1s/epoch; IoU self-test 1/7 exact; NMS keeps `[0, 2]` on the duplicate-box unit test; 3-box NMS ×200 ≈ 1ms
- Overlays in `results/boxes.png` (green=GT, red=pred), curves in `results/curves.png`, history in `results/metrics.csv`

## The 7 README questions (fill after running)
1. **What did I build?** From-scratch IoU (pair + matrix), greedy NMS, and a 1-object CNN detector (SmoothL1 box + CE class heads).
2. **How does it work?** Box head regresses 4 normalized xyxy numbers through sigmoid; IoU = intersection/union scores overlap; NMS keeps the best box and kills neighbors with IoU > 0.5.
3. **What did I measure?** table above + `results/metrics.csv`.
4. **What surprised me?** Classification hit 100% by epoch 1 (color is trivial) while box regression needed all 12 epochs — localization is the hard half of detection.
5. **Bottleneck?** Box precision stalls ~0.69 mIoU: sigmoid saturation + single-scale features, not data size.
6. **What if X?** NMS thresh 0.5→0.9 keeps duplicates; more noise hurts boxes before classes; 2 objects need anchors/matching (out of scope by design).
7. **Next?** Multi-object via grid anchors, mAP eval, focal loss for class imbalance.
