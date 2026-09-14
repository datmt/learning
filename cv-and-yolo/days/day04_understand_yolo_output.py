"""
DAY 4 — Understand YOLO's Raw Output
=======================================

RECAP: Day 3 you called `results[0].plot()` and got a pretty picture for
free. Today you open the hood and look at exactly what data structure YOLO
actually hands you — because in real projects you rarely want the pretty
picture, you want the raw numbers (to save to a database, filter by class,
count objects, feed into tracking, etc).

By the end of today, this structure should be "second nature":

    YOLO result
      |
      +-- boxes
      |     +-- xyxy         (pixel coordinates, Day 1 format!)
      |     +-- conf         (confidence score, 0-1)
      |     +-- cls          (class ID, an integer)
      |
      +-- names               (dict: class ID -> class name string)

Run with:  python day04_understand_yolo_output.py
"""

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

# %% ------------------------------------------------------------------
# SECTION 1 — Run inference again (same as Day 3)
# ------------------------------------------------------------------
model = YOLO("yolo11n.pt")
TEST_IMAGE = os.path.join(ASSETS_DIR, "day03_bus.jpg")
if not os.path.exists(TEST_IMAGE):
    import urllib.request
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", TEST_IMAGE)

results = model(TEST_IMAGE, verbose=False)
result = results[0]


# %% ------------------------------------------------------------------
# SECTION 2 — Inspect `result.boxes`
# ------------------------------------------------------------------
# `result.boxes` is a special container (a "Boxes" object) holding ALL
# detections for this image. Think of it like a table with one row per
# detected object and columns: x1, y1, x2, y2, confidence, class.

print("result.boxes:")
print(result.boxes)
print()
print("Number of detections:", len(result.boxes))


# %% ------------------------------------------------------------------
# SECTION 3 — Pull out the three things that actually matter
# ------------------------------------------------------------------
# .xyxy  -> tensor of shape (N, 4): each row is [x1, y1, x2, y2] in pixels
#           (exactly the Day 1 format — no coincidence, this IS the
#           industry-standard representation).
# .conf  -> tensor of shape (N,): confidence score per detection (0-1)
# .cls   -> tensor of shape (N,): class ID per detection (a float that's
#           really an integer, e.g. 0.0 = person)

xyxy = result.boxes.xyxy      # PyTorch tensor
conf = result.boxes.conf
cls = result.boxes.cls

print("xyxy (boxes):\n", xyxy)
print("\nconf (confidence scores):\n", conf)
print("\ncls (class IDs):\n", cls)

# Tensors behave a lot like numpy arrays but live on PyTorch's side. To get
# plain Python numbers/numpy arrays out, use .cpu().numpy() (safe habit
# even if you're not using a GPU):
xyxy_np = xyxy.cpu().numpy()
conf_np = conf.cpu().numpy()
cls_np = cls.cpu().numpy().astype(int)


# %% ------------------------------------------------------------------
# SECTION 4 — Loop over detections yourself (don't rely on .plot())
# ------------------------------------------------------------------
print("\nDetections (readable form):")
for box, confidence, class_id in zip(xyxy_np, conf_np, cls_np):
    class_name = model.names[class_id]
    x1, y1, x2, y2 = box
    print(
        f"  {class_name:12s}  conf={confidence:.2f}  "
        f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
    )


# %% ------------------------------------------------------------------
# SECTION 5 — Build YOUR OWN visualizer (reusing Day 1's draw_box)
# ------------------------------------------------------------------
# This is the important exercise today: don't depend on `.plot()`. Prove
# you understand the data by drawing it yourself.

image_bgr = cv2.imread(TEST_IMAGE)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def draw_box(image, box, label=None, color=(255, 0, 0), thickness=2):
    out_img = image  # NOTE: draws in-place on purpose here, see loop below
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(out_img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out_img


my_visualization = image_rgb.copy()

# A different color per class makes it easier to scan the image visually.
palette = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 0, 255), (255, 165, 0)]

for box, confidence, class_id in zip(xyxy_np, conf_np, cls_np):
    class_name = model.names[class_id]
    color = palette[class_id % len(palette)]
    label = f"{class_name} {confidence:.2f}"
    draw_box(my_visualization, box, label, color=color)

plt.figure(figsize=(12, 8))
plt.imshow(my_visualization)
plt.axis("off")
plt.title("My own visualizer (not .plot())")
plt.savefig(os.path.join(ASSETS_DIR, "day04_my_visualizer.png"))
plt.close()
print("\nSaved: day04_my_visualizer.png")


# %% ------------------------------------------------------------------
# SECTION 6 — Filter detections yourself (very common real task)
# ------------------------------------------------------------------
# In a real project you almost always want to filter: "only show me
# people", "only show me confident detections", etc. Now that you have raw
# arrays, this is trivial — no special API needed.

CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASS_NAME = "person"
target_class_id = [k for k, v in model.names.items() if v == TARGET_CLASS_NAME][0]

mask = (conf_np >= CONFIDENCE_THRESHOLD) & (cls_np == target_class_id)
filtered_boxes = xyxy_np[mask]
filtered_scores = conf_np[mask]

print(f"\nFiltered to '{TARGET_CLASS_NAME}' with conf >= {CONFIDENCE_THRESHOLD}:")
print(f"  {len(filtered_boxes)} detections kept out of {len(xyxy_np)} total")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
print("""
YOLO result
  |
  +-- boxes
  |     +-- xyxy   -> [x1, y1, x2, y2] in pixels (Day 1 format)
  |     +-- conf   -> confidence score, 0.0 to 1.0
  |     +-- cls    -> class ID (int), look up name via model.names[id]
  |
  +-- names -> {0: 'person', 1: 'bicycle', 2: 'car', ...}

Everything downstream (filtering, counting, tracking, drawing) is just
plain array/list manipulation on these three things.
""")


# %% ------------------------------------------------------------------
# DAY 4 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — Without running code, what are the 3 pieces of information stored
  per detection in result.boxes?

Test 2 — Write (from memory) a one-line filter that keeps only detections
  with confidence above 0.7.

Test 3 — How do you convert a class ID integer into a human-readable class
  name string? (name the dictionary/attribute)

Test 4 — Modify the code to count how many "person" objects were detected
  in the image (just print a single number).

Test 5 — Why might you want raw .xyxy/.conf/.cls arrays instead of just
  calling result.plot() in a real application? Give one concrete example
  (e.g. saving to a database, counting objects, alerting).

If you can do all 5, Day 4 is done. Move to Day 5.
"""

if __name__ == "__main__":
    print("\nDay 4 script finished.")
