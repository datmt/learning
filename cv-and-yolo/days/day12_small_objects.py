"""
DAY 12 — Small Objects (Why a Great mAP Can Still Be Useless to You)
=========================================================================

RECAP: Day 9 taught you evaluation with a single mAP number. Today's
lesson is specifically: mAP can look great overall while your detector is
nearly blind to the ONE object category you actually care about, if that
category is small. For sports analytics, this is almost always the ball.

WHY SMALL OBJECTS ARE FUNDAMENTALLY HARDER
----------------------------------------------
Recall Day 6: a badminton shuttle might be 8x8 pixels in a 1920x1080
frame. By the time that patch of the image has passed through several
downsampling steps in the backbone, its signal may be represented by a
FRACTION OF ONE PIXEL in the deepest feature maps -- it can vanish
entirely, drowned out by the surrounding background. A player, by
contrast, might be 200x400 pixels -- comfortably visible at every scale.

    image: 1920x1080
    ball:  8x8 pixels     (0.00003 of the image area)
    player: 200x400 pixels (0.077 of the image area)   <- ~2600x bigger area!

THE PRACTICAL LEVERS YOU CAN PULL
--------------------------------------
  1. Resolution (imgsz): a higher inference resolution keeps more pixels
     of the small object intact for longer through the network. Costs
     more compute.
  2. Cropping/tiling: instead of shrinking a huge image down to 640x640
     (losing detail), split it into overlapping crops and run detection
     on each crop at full resolution, then merge results.
  3. Confidence threshold: small objects are detected with LOWER
     confidence on average (less visual evidence to be sure about) --
     using your normal threshold may silently discard them.
  4. Data/augmentation: more training examples of the small object,
     especially at varied scales, help the model learn what it looks like.

Today's script demonstrates levers 1 and 3 concretely, and shows you how
to inspect PER-SIZE performance instead of trusting one aggregate number.

Run with:  python day12_small_objects.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")


# %% ------------------------------------------------------------------
# SECTION 1 — Build a test image with objects of VERY different sizes
# ------------------------------------------------------------------
def make_multiscale_image(width=1280, height=720):
    canvas = np.full((height, width, 3), (30, 90, 30), dtype=np.uint8)

    boxes = []  # (name, [x1,y1,x2,y2])

    # A big "player" (easy)
    cv2.rectangle(canvas, (200, 200), (450, 550), (90, 140, 200), -1)
    boxes.append(("big_player", [200, 200, 450, 550]))

    # A medium object
    cv2.rectangle(canvas, (700, 300), (780, 450), (140, 90, 200), -1)
    boxes.append(("medium_object", [700, 300, 780, 450]))

    # A genuinely small "ball" (hard)
    cv2.circle(canvas, (1000, 150), 6, (255, 255, 255), -1)
    boxes.append(("small_ball", [994, 144, 1006, 156]))

    return canvas, boxes


image, ground_truth_boxes = make_multiscale_image()
image_path = os.path.join(ASSETS_DIR, "day12_multiscale.jpg")
cv2.imwrite(image_path, image)

print("Ground truth objects and their pixel area:")
for name, box in ground_truth_boxes:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    print(f"  {name:15s} box={box}  area={area} px^2")


# %% ------------------------------------------------------------------
# SECTION 2 — LEVER 1: inference resolution (imgsz)
# ------------------------------------------------------------------
# YOLO resizes the input image to `imgsz` before running the network
# (remember Day 1 Section 9: this also means boxes get rescaled back to
# original coordinates on output -- Ultralytics does this for you). A
# LOWER imgsz shrinks the image more aggressively, losing more detail from
# small objects before the network even sees them.

model = YOLO("yolo11n.pt")

print("\nEffect of imgsz on detecting our tiny synthetic ball:")
for imgsz in [160, 320, 640, 1280]:
    result = model(image_path, imgsz=imgsz, conf=0.05, verbose=False)[0]
    n_detections = len(result.boxes)
    print(f"  imgsz={imgsz:5d} -> {n_detections} detections found")

print("""
(Our synthetic shapes aren't COCO classes YOLO was trained on, so absolute
detection counts here are just illustrative of the trend, not a claim
about real accuracy. The important pattern to notice: does raising imgsz
change what gets detected at all, especially near the small ball?)
""")


# %% ------------------------------------------------------------------
# SECTION 3 — LEVER 2: crop and detect at native resolution
# ------------------------------------------------------------------
# Instead of shrinking the whole 1280x720 image down to fit imgsz=640
# (losing half the ball's already-tiny detail), crop just the region
# around where we expect small objects (e.g. top area of frame) and run
# detection on that crop at FULL resolution. This is the "tiling" idea
# used in real small-object pipelines (e.g. satellite imagery, sports
# ball tracking).

def crop_and_rescale_boxes(image, crop_box, model, imgsz=640, conf=0.05):
    cx1, cy1, cx2, cy2 = crop_box
    crop = image[cy1:cy2, cx1:cx2]
    result = model(crop, imgsz=imgsz, conf=conf, verbose=False)[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    # IMPORTANT: boxes are in CROP coordinates. Translate back to the
    # original image's coordinate system by adding the crop's offset.
    # (This is a direct, practical use of Day 1's coordinate system idea.)
    boxes[:, [0, 2]] += cx1
    boxes[:, [1, 3]] += cy1
    return boxes


crop_region = [850, 0, 1280, 300]   # a region around the small ball
cropped_boxes = crop_and_rescale_boxes(image, crop_region, model)
print(f"\nDetections within the small-ball crop region, translated back to")
print(f"full-image coordinates: {len(cropped_boxes)} found")
if len(cropped_boxes) > 0:
    print(cropped_boxes)


# %% ------------------------------------------------------------------
# SECTION 4 — LEVER 3: confidence threshold vs. object size
# ------------------------------------------------------------------
# Demonstrate the core lesson with numbers: run at a normal deployment
# threshold (0.25) vs. a much lower one (0.05), and compare detection
# counts. In a real trained-on-your-data model, you'd typically see small
# objects clustered at LOWER confidence scores than large ones -- so a
# single global threshold tuned by eyeballing big, easy objects can
# silently zero out your ball detections.

for conf in [0.25, 0.10, 0.05, 0.01]:
    result = model(image_path, imgsz=1280, conf=conf, verbose=False)[0]
    print(f"conf={conf:.2f} -> {len(result.boxes)} detections")


# %% ------------------------------------------------------------------
# SECTION 5 — Per-size failure analysis (extending Day 9)
# ------------------------------------------------------------------
# The real skill: don't just report ONE mAP. Bucket your ground truth by
# size and check recall PER BUCKET. This reveals a "great overall mAP,
# terrible on small objects" situation that a single number would hide.

def area_of(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


SMALL_THRESHOLD = 32 * 32     # COCO's own definition of "small" (<32x32 px)
MEDIUM_THRESHOLD = 96 * 96    # COCO's definition of "medium" (32x32 to 96x96)

def size_bucket(box):
    a = area_of(box)
    if a < SMALL_THRESHOLD:
        return "small"
    elif a < MEDIUM_THRESHOLD:
        return "medium"
    return "large"

print("\nSize bucket for each of our synthetic ground-truth objects:")
for name, box in ground_truth_boxes:
    print(f"  {name:15s} area={area_of(box):>6} px^2  -> bucket: {size_bucket(box)}")

print("""
In a REAL evaluation (Day 9's TP/FP/FN machinery), you would compute
recall SEPARATELY for each of these buckets across your whole validation
set: recall_small, recall_medium, recall_large. It's completely normal (and
a huge trap for sports vision specifically) to see something like:
    recall_large  = 0.95   (players: great!)
    recall_medium = 0.85
    recall_small  = 0.30   (ball: unusable, even though overall mAP looks fine)
""")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
print("""
Small object detection difficulty ladder (easiest to hardest lever):
  1. Raise confidence sensitivity (lower conf threshold) -- free, but more FPs
  2. Raise imgsz -- more compute, keeps more detail
  3. Crop/tile the region of interest -- more compute+complexity, most detail
  4. Get more/better training data of the small object at multiple scales

And the evaluation habit to always apply:
  overall mAP  ->  not enough
  mAP per size bucket (small/medium/large)  ->  reveals the real story
""")


# %% ------------------------------------------------------------------
# DAY 12 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — Why does a small object's signal get "diluted" more than a large
  object's as it passes through a CNN backbone's downsampling layers?
  (Connect this to Day 6's 20x20-grid thought experiment.)

Test 2 — Name the 3 practical levers demonstrated today for improving
  small-object detection, and one cost/trade-off for each.

Test 3 — Why might a small object systematically get LOWER confidence
  scores than a large object of the same true class?

Test 4 — In your own words, explain the crop-and-translate-back technique
  from Section 3. Why do you have to ADD the crop's offset to the
  detected boxes?

Test 5 — Why is "recall broken down by object size" a more useful
  diagnostic for sports-ball detection than a single overall mAP number?

If you can do all 5, Day 12 is done. Move to Day 13.
"""

if __name__ == "__main__":
    print("\nDay 12 script finished.")
