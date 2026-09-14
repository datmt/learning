"""
DAY 2 — IoU (Intersection over Union) and NMS (Non-Max Suppression)
======================================================================

RECAP: Day 1 you learned a box is just [x1, y1, x2, y2]. Today you learn the
two pieces of math that make a detector USABLE, instead of a firehose of
overlapping guesses.

THE PROBLEM THIS SOLVES
------------------------
A detector like YOLO doesn't just spit out one clean box per object. It
internally considers THOUSANDS of candidate boxes across the image, and many
of them will correctly-ish surround the SAME object, just slightly
different sizes/positions, each with its own confidence score:

       ┌────────────┐
       │     A      │      A = confidence 0.91
       │  ┌──────┐  │      B = confidence 0.87
       │  │  B   │  │
       │  └──────┘  │
       └────────────┘

Without cleanup, you'd show the user 5 overlapping boxes for one player.
That's useless. We need:

  1. IoU: a number that measures "how much do two boxes overlap?"
  2. NMS: an algorithm that uses IoU to throw away duplicate/overlapping
     boxes and keep only the best one per object.

Run with:  python day02_iou_and_nms.py
"""

# %% ------------------------------------------------------------------
# SECTION 1 — Intersection over Union (IoU)
# ------------------------------------------------------------------
# IoU answers: "Of the total area covered by these two boxes combined, what
# fraction is covered by BOTH of them?"
#
#     IoU = (area where they overlap) / (total area covered by either)
#
# IoU is always between 0 and 1:
#   IoU = 0   -> boxes don't touch at all
#   IoU = 1   -> boxes are identical
#   IoU = 0.5 -> "pretty good" overlap (a common threshold in practice)
#
# This single number is the backbone of: NMS (today), and evaluating how
# good a detector is (Day 9 — mAP).

import numpy as np
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def iou(box_a, box_b):
    """
    Compute Intersection over Union between two boxes in XYXY format.
    box_a, box_b: array-like [x1, y1, x2, y2]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Step 1: find the coordinates of the overlapping (intersection)
    # rectangle. The intersection's top-left is the MAX of the two
    # top-lefts, and its bottom-right is the MIN of the two bottom-rights.
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Step 2: compute intersection area. If the boxes don't overlap at all,
    # (inter_x2 - inter_x1) or (inter_y2 - inter_y1) would be negative —
    # clip to 0 so we don't get a nonsensical negative "area".
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Step 3: compute the union area = area(A) + area(B) - intersection
    # (we subtract intersection once because otherwise we'd double count it)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# Sanity checks — build intuition with easy cases first.
identical = iou([0, 0, 100, 100], [0, 0, 100, 100])
no_overlap = iou([0, 0, 100, 100], [200, 200, 300, 300])
half_overlap = iou([0, 0, 100, 100], [50, 0, 150, 100])

print(f"identical boxes  -> IoU = {identical:.3f}  (expect 1.000)")
print(f"no overlap       -> IoU = {no_overlap:.3f}  (expect 0.000)")
print(f"half overlap     -> IoU = {half_overlap:.3f}  (expect 0.333)")
# Why 0.333 and not 0.5? Overlap area = 50x100 = 5000.
# Union = 10000 + 10000 - 5000 = 15000. 5000/15000 = 0.333.
# This is a classic "gotcha" — overlapping HALF of each box does NOT give
# IoU=0.5, because union counts both boxes' non-overlapping parts too.


# %% ------------------------------------------------------------------
# SECTION 2 — Visualize IoU for a few cases
# ------------------------------------------------------------------
def draw_two_boxes(ax, box_a, box_b, title):
    ax.add_patch(plt.Rectangle(
        (box_a[0], box_a[1]), box_a[2] - box_a[0], box_a[3] - box_a[1],
        fill=False, edgecolor="red", linewidth=2, label="A"))
    ax.add_patch(plt.Rectangle(
        (box_b[0], box_b[1]), box_b[2] - box_b[0], box_b[3] - box_b[1],
        fill=False, edgecolor="blue", linewidth=2, label="B"))
    ax.set_xlim(-20, 320)
    ax.set_ylim(320, -20)   # inverted, remember Day 1: y grows downward
    ax.set_title(f"{title}\nIoU = {iou(box_a, box_b):.3f}")
    ax.legend()
    ax.set_aspect("equal")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
draw_two_boxes(axes[0], [0, 0, 100, 100], [0, 0, 100, 100], "Identical")
draw_two_boxes(axes[1], [0, 0, 100, 100], [50, 0, 150, 100], "Half overlap")
draw_two_boxes(axes[2], [0, 0, 100, 100], [200, 200, 300, 300], "No overlap")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "day02_iou_examples.png"))
plt.close()
print("Saved: day02_iou_examples.png")


# %% ------------------------------------------------------------------
# SECTION 3 — Non-Max Suppression (NMS)
# ------------------------------------------------------------------
# The idea, in plain English:
#
#   1. Look at all candidate boxes and their confidence scores.
#   2. Pick the box with the HIGHEST confidence. Keep it — it's a "winner".
#   3. Throw away every remaining box that overlaps the winner "too much"
#      (IoU above some threshold, e.g. 0.5) — those are almost certainly
#      just duplicate detections of the SAME object.
#   4. Repeat with whatever boxes are left, until none remain.
#
# This is exactly like a tournament: keep the strongest candidate for each
# "region", eliminate its close competitors, move to the next region.

def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: list/array of [x1,y1,x2,y2]
    scores: list/array of confidence scores, same length as boxes
    Returns: list of indices (into `boxes`) that survive NMS, sorted by
             descending confidence.
    """
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)

    # Sort box indices by confidence, highest first.
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        # The highest-scoring remaining box always survives.
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        # Compare the current winner against every other remaining box.
        rest = order[1:]
        ious = np.array([iou(boxes[current], boxes[i]) for i in rest])

        # Keep only the boxes that DON'T overlap the winner too much.
        order = rest[ious <= iou_threshold]

    return keep


# %% ------------------------------------------------------------------
# SECTION 4 — See NMS in action
# ------------------------------------------------------------------
# Simulate what a raw detector output often looks like: 3 boxes around the
# SAME object (slightly different sizes/positions) plus 1 box around a
# completely different object.

candidate_boxes = np.array([
    [50, 50, 150, 150],    # object A, box 1
    [55, 48, 152, 149],    # object A, box 2 (near-duplicate of box 1)
    [60, 55, 148, 145],    # object A, box 3 (also a near-duplicate)
    [300, 300, 400, 400],  # object B, a totally different object
])
candidate_scores = np.array([0.91, 0.87, 0.75, 0.95])

survivors = nms(candidate_boxes, candidate_scores, iou_threshold=0.5)
print("Boxes kept after NMS (indices):", survivors)
print("Kept boxes:\n", candidate_boxes[survivors])
print("Kept scores:", candidate_scores[survivors])
# Expect indices [3, 0]: the object B box (highest score, 0.95) survives,
# and among the 3 overlapping object A boxes, only the highest-confidence
# one (0.91, index 0) survives — the near-duplicates get suppressed.

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, boxes, title in [
    (axes[0], candidate_boxes, "Before NMS (4 raw boxes)"),
    (axes[1], candidate_boxes[survivors], "After NMS (deduplicated)"),
]:
    ax.set_xlim(0, 450)
    ax.set_ylim(450, 0)
    ax.set_title(title)
    ax.set_aspect("equal")
    for b in boxes:
        ax.add_patch(plt.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            fill=False, edgecolor="red", linewidth=2))
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "day02_nms_before_after.png"))
plt.close()
print("Saved: day02_nms_before_after.png")


# %% ------------------------------------------------------------------
# SECTION 5 — YOUR EXPERIMENT: change the threshold, observe what happens
# ------------------------------------------------------------------
# Try iou_threshold values of 0.9 (very lenient — almost nothing gets
# removed) and 0.1 (very strict — even loosely-overlapping boxes get
# removed). Uncomment and run:

for thresh in [0.1, 0.5, 0.9]:
    kept = nms(candidate_boxes, candidate_scores, iou_threshold=thresh)
    print(f"iou_threshold={thresh} -> kept {len(kept)} boxes: {kept}")

# Notice: a LOWER threshold is STRICTER (removes more boxes, because even
# small overlaps now count as "too much"). A HIGHER threshold is more
# lenient (only near-identical boxes get removed). This is a common
# source of confusion — the direction feels backwards until it clicks.


# %% ------------------------------------------------------------------
# DAY 2 CHECKPOINT
# ------------------------------------------------------------------
"""
Do not move to Day 3 until you can do these without looking at the code.

Test 1 — By hand (or on paper), compute IoU for:
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 15, 15]
  (walk through: intersection coords, intersection area, union area)

Test 2 — Explain in one sentence why IoU of two boxes that each cover half
  of the other is 0.333, not 0.5.

Test 3 — Given 4 boxes with scores [0.9, 0.8, 0.95, 0.4] where boxes 0, 1,
  2 all heavily overlap each other and box 3 is far away, trace through
  the NMS algorithm by hand: which box is picked first? What gets removed?
  What's left at the end?

Test 4 — What happens if you set iou_threshold=1.0 in NMS? What about
  iou_threshold=0.0? (Think about it, then verify by running the code.)

Test 5 — In your own words: why does a raw detector produce many
  overlapping boxes for the same object in the first place, and why is
  NMS necessary to get a clean final result?

If you can do all 5, Day 2 is done. Move to Day 3.
"""

if __name__ == "__main__":
    print("\nDay 2 script finished. Check ../assets/ for saved images.")
