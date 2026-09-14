"""
DAY 9 — Evaluation: Is My Detector Actually Good?
======================================================

RECAP: Day 8 you trained a model and saw the loss go down. But "loss went
down" doesn't tell you HOW GOOD the model is in a way a human understands.
Today you learn to measure that properly, and — more importantly — to look
at WHY it fails, not just a single summary number.

THE CORE IDEA: TP, FP, FN
----------------------------
For every predicted box, compare it against the ground-truth (real) boxes
using IoU (Day 2!). Using a threshold (commonly IoU >= 0.5 = "close enough"):

  TRUE POSITIVE  (TP) -> predicted box matches a real object well. Good.
  FALSE POSITIVE (FP) -> predicted box doesn't match any real object.
                          The model "hallucinated" something. Bad.
  FALSE NEGATIVE (FN) -> a real object had NO matching prediction.
                          The model MISSED it. Bad.

From these three counts:

    precision = TP / (TP + FP)   "Of everything I predicted, how much was
                                   actually real?" (are my predictions
                                   trustworthy?)

    recall    = TP / (TP + FN)   "Of everything that was actually there,
                                   how much did I find?" (am I missing
                                   things?)

There's a fundamental trade-off between these two (raising your confidence
threshold: fewer FPs but more FNs, and vice versa). AP (Average Precision)
and mAP (mean AP, averaged across classes) summarize this trade-off into
ONE number — useful for comparing models, but they HIDE the interesting
detail. That's why we also do failure analysis, visually.

Run with:  python day09_evaluation.py
(Requires Day 7's dataset and Day 8's trained weights.)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET_DIR = os.path.join(PROJECT_ROOT, "assets", "toy_sports_dataset")
WEIGHTS = os.path.join(PROJECT_ROOT, "assets", "runs", "day08_overfit_check", "weights", "best.pt")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
CLASS_NAMES = ["player", "ball"]

if not os.path.exists(WEIGHTS):
    raise SystemExit("Train first: run day08_train_yolo.py")


# %% ------------------------------------------------------------------
# SECTION 1 — Reuse Day 2's IoU function
# ------------------------------------------------------------------
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def yolo_line_to_xyxy(line, image_width, image_height):
    parts = line.strip().split()
    class_id = int(parts[0])
    cx, cy, w, h = (float(v) for v in parts[1:5])
    cx *= image_width; cy *= image_height; w *= image_width; h *= image_height
    return class_id, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# %% ------------------------------------------------------------------
# SECTION 2 — Match predictions to ground truth: compute TP/FP/FN
# ------------------------------------------------------------------
def match_predictions_to_ground_truth(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    Greedy matching: each ground-truth box can be matched by at most one
    prediction (its best-IoU match, if above threshold and same class).
    Returns: tp, fp, fn (counts), and per-prediction match info for
    visualization.
    """
    gt_matched = [False] * len(gt_boxes)
    pred_status = []   # "TP" or "FP" per prediction, in order

    for p_box, p_class in zip(pred_boxes, pred_classes):
        best_iou, best_idx = 0.0, -1
        for i, (g_box, g_class) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_matched[i] or g_class != p_class:
                continue
            current_iou = iou(p_box, g_box)
            if current_iou > best_iou:
                best_iou, best_idx = current_iou, i

        if best_iou >= iou_threshold:
            gt_matched[best_idx] = True
            pred_status.append("TP")
        else:
            pred_status.append("FP")

    tp = pred_status.count("TP")
    fp = pred_status.count("FP")
    fn = gt_matched.count(False)   # ground-truth boxes nobody claimed
    return tp, fp, fn, pred_status, gt_matched


# %% ------------------------------------------------------------------
# SECTION 3 — Run evaluation across the validation set
# ------------------------------------------------------------------
model = YOLO(WEIGHTS)

val_img_dir = os.path.join(DATASET_DIR, "images", "val")
val_lbl_dir = os.path.join(DATASET_DIR, "labels", "val")
filenames = sorted(os.listdir(val_img_dir))

# IMPORTANT gotcha to notice: by default, model() only returns predictions
# with confidence >= 0.25. That's the right setting for a real
# application (don't show the user low-confidence guesses), but it's the
# WRONG setting for evaluation/debugging — a real object the model was
# "40% sure" about would silently disappear from your TP/FP/FN counts,
# making the model look worse than it is. For evaluation we deliberately
# use a very low threshold so we can see EVERYTHING the model considered,
# and reason about the confidence numbers ourselves.
EVAL_CONF_THRESHOLD = 0.001

total_tp, total_fp, total_fn = 0, 0, 0
per_image_results = []

for filename in filenames:
    image_path = os.path.join(val_img_dir, filename)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    label_path = os.path.join(val_lbl_dir, filename.replace(".jpg", ".txt"))
    with open(label_path) as f:
        gt_lines = [l for l in f if l.strip()]
    gt_classes, gt_boxes = [], []
    for line in gt_lines:
        c, box = yolo_line_to_xyxy(line, width, height)
        gt_classes.append(c); gt_boxes.append(box)

    result = model(image_path, conf=EVAL_CONF_THRESHOLD, verbose=False)[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()
    pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()

    tp, fp, fn, pred_status, gt_matched = match_predictions_to_ground_truth(
        pred_boxes, pred_classes, gt_boxes, gt_classes
    )
    total_tp += tp; total_fp += fp; total_fn += fn
    per_image_results.append(dict(
        filename=filename, image=image, gt_boxes=gt_boxes, gt_classes=gt_classes,
        gt_matched=gt_matched, pred_boxes=pred_boxes, pred_classes=pred_classes,
        pred_status=pred_status,
    ))

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

print(f"Across {len(filenames)} validation images (conf threshold={EVAL_CONF_THRESHOLD}):")
print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
print(f"  precision = {precision:.3f}   (of predictions made, how many were correct)")
print(f"  recall    = {recall:.3f}   (of real objects, how many were found)")
print("""
  Note: this used a very low confidence threshold on purpose (see the
  comment above EVAL_CONF_THRESHOLD). You'll likely see recall go UP but
  precision crash (many low-confidence noise boxes now count as FPs) —
  that's expected, and it's exactly the precision/recall trade-off in
  action. This is also literally what mAP computation does internally:
  sweep the threshold from very low to very high and summarize the whole
  precision/recall curve into one number, rather than picking one
  threshold. Try re-running with EVAL_CONF_THRESHOLD = 0.25 (a realistic
  deployment default) and compare all three numbers again.""")


# %% ------------------------------------------------------------------
# SECTION 4 — Also get Ultralytics' own mAP, for comparison
# ------------------------------------------------------------------
# Ultralytics has a built-in, more rigorous mAP calculation (it averages
# precision across many IoU thresholds, 0.5 to 0.95 — hence "mAP50-95").
# Compare it to your own precision/recall above to build trust in both.

metrics = model.val(data=os.path.join(DATASET_DIR, "data.yaml"), verbose=False, plots=False)
print(f"\nUltralytics metrics:")
print(f"  mAP50    = {metrics.box.map50:.3f}")
print(f"  mAP50-95 = {metrics.box.map:.3f}")


# %% ------------------------------------------------------------------
# SECTION 5 — FAILURE ANALYSIS: visualize TP (green) vs FP (red) vs FN (orange)
# ------------------------------------------------------------------
# This is the important habit the curriculum emphasizes: don't just trust
# one number. LOOK at what's wrong, per image, per box.

def visualize_failures(entry):
    image_rgb = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2RGB).copy()

    # Ground truth boxes: GREEN if matched (found), ORANGE if missed (FN)
    for box, cls, matched in zip(entry["gt_boxes"], entry["gt_classes"], entry["gt_matched"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 200, 0) if matched else (255, 140, 0)
        tag = "GT (found)" if matched else "GT (MISSED = FN)"
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, tag, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Prediction boxes: BLUE if TP, RED if FP (a hallucinated box)
    for box, cls, status in zip(entry["pred_boxes"], entry["pred_classes"], entry["pred_status"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 100, 255) if status == "TP" else (255, 0, 0)
        cv2.rectangle(image_rgb, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), color, 2)
        cv2.putText(image_rgb, f"pred:{status}", (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return image_rgb


fig, axes = plt.subplots(1, len(per_image_results), figsize=(5 * len(per_image_results), 5))
if len(per_image_results) == 1:
    axes = [axes]
for ax, entry in zip(axes, per_image_results):
    ax.imshow(visualize_failures(entry))
    n_fp = entry["pred_status"].count("FP")
    n_fn = entry["gt_matched"].count(False)
    ax.set_title(f"{entry['filename']}\nFP={n_fp}  FN={n_fn}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "day09_failure_analysis.png"))
plt.close()
print("\nSaved: day09_failure_analysis.png")
print("Legend: GREEN=found ground truth, ORANGE=missed ground truth (FN),")
print("        BLUE=correct prediction (TP), RED=hallucinated prediction (FP)")


# %% ------------------------------------------------------------------
# SECTION 6 — Categorize WHY failures happen (the real skill)
# ------------------------------------------------------------------
print("""
Common failure categories to look for when scanning failure-analysis
images like the one just saved (this is the real skill, more valuable
than any single mAP number):

  false positive      -> a prediction with no matching real object
                          (background mistaken for an object)
  false negative       -> a real object with no prediction
                          (often: object too small, occluded, or a class
                          the model has too little training data for)
  bad localization      -> box exists and class is right, but IoU with
                          ground truth is low (box is the wrong shape/
                          position — different from FN, this is a "close
                          but sloppy" box)
  wrong class            -> box localizes the object well, but predicts
                          the wrong class label
  duplicate detection    -> two predictions for the same object (should
                          have been removed by NMS, Day 2 — if you see
                          this, your iou_threshold in NMS is too high)
  small object missed    -> ties directly into Day 12
  occlusion               -> object partially hidden behind another object

Go through day09_failure_analysis.png right now and try to label each
orange/red box with one of these categories.
""")


# %% ------------------------------------------------------------------
# DAY 9 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — Define TP, FP, FN in your own words, using a concrete example
  from your failure-analysis image.

Test 2 — Compute precision and recall by hand from TP=8, FP=2, FN=3.

Test 3 — Explain the precision/recall trade-off: if you RAISE the
  confidence threshold, what generally happens to precision? To recall?

Test 4 — Why isn't a single mAP number enough to fully judge a detector
  for YOUR specific use case? Give a concrete scenario (hint: think about
  the "small ball, big mAP" example from the curriculum).

Test 5 — Look at your day09_failure_analysis.png. Find at least one FP or
  FN and categorize it using the failure list above.

Test 6 — Change EVAL_CONF_THRESHOLD from 0.001 to 0.25 and re-run.
  Explain in your own words WHY recall changes the way it does, and why
  evaluation code should generally use a lower threshold than a deployed
  application would.

If you can do all 6, Day 9 is done. Move to Day 10.
"""

if __name__ == "__main__":
    print("\nDay 9 script finished.")
