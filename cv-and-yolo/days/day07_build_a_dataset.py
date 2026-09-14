"""
DAY 7 — Build Your Own YOLO Dataset
=======================================

RECAP: Days 1-6 used a PRETRAINED model (trained on COCO's 80 generic
classes). If you want YOLO to detect something specific to YOUR problem
(e.g. "badminton shuttle", "specific player jersey numbers"), you need your
own labeled dataset, in the exact folder/file layout YOLO expects.

Today you become "dangerous": you can now produce training data, not just
consume someone else's.

THE FOLDER STRUCTURE YOLO (Ultralytics) EXPECTS
--------------------------------------------------
    sports/
        images/
            train/   <- training photos (.jpg/.png)
            val/     <- validation photos (held out, used to check progress)
        labels/
            train/   <- one .txt file PER image, same filename
            val/
        data.yaml    <- tells YOLO where the folders are + class names

THE LABEL FILE FORMAT (one line per object in that image)
-------------------------------------------------------------
    class_id  center_x  center_y  width  height

    0 0.52 0.61 0.08 0.20

means:
    class_id = 0        (look up "0" in data.yaml's class list, e.g. "ball")
    center_x = 0.52      (52% of the way across the image, horizontally)
    center_y = 0.61      (61% of the way down the image, vertically)
    width    = 0.08      (8% of the image's width)
    height   = 0.20      (20% of the image's height)

Recognize this? It's EXACTLY the "YOLO normalized" format from Day 1,
Section 7 (xyxy_to_yolo). Today is where that function becomes directly
useful, not just theoretical.

Run with:  python day07_build_a_dataset.py
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET_DIR = os.path.join(PROJECT_ROOT, "assets", "toy_sports_dataset")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

CLASS_NAMES = ["player", "ball"]


# %% ------------------------------------------------------------------
# SECTION 1 — The Day 1 conversion function, reused
# ------------------------------------------------------------------
def xyxy_to_yolo_label(class_id, box_xyxy, image_width, image_height):
    """box_xyxy = [x1,y1,x2,y2] pixels -> 'class cx cy w h' normalized string."""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2 / image_width
    cy = (y1 + y2) / 2 / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# %% ------------------------------------------------------------------
# SECTION 2 — Generate a TINY synthetic dataset
# ------------------------------------------------------------------
# In real life, you'd collect real photos and label them with a tool (e.g.
# Roboflow, LabelImg, CVAT). Since you're still learning the PIPELINE (not
# labeling itself), we synthesize simple images with known ground-truth
# boxes. This means: 1) the script runs standalone with zero setup, and
# 2) you know the "correct answer" for every box, useful for checking your
# own code.

def make_synthetic_image_and_boxes(seed, width=640, height=480):
    rng = random.Random(seed)
    canvas = np.full((height, width, 3), (30, 90, 30), dtype=np.uint8)  # green "court"

    boxes = []  # list of (class_id, [x1,y1,x2,y2])

    # 1-2 "players" (rectangles)
    for _ in range(rng.randint(1, 2)):
        w, h = rng.randint(30, 60), rng.randint(60, 100)
        x1 = rng.randint(0, width - w)
        y1 = rng.randint(0, height - h)
        x2, y2 = x1 + w, y1 + h
        color = (rng.randint(100, 255), rng.randint(0, 100), rng.randint(0, 100))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        boxes.append((0, [x1, y1, x2, y2]))   # class 0 = player

    # 1 small "ball" (circle)
    r = rng.randint(4, 8)
    cx = rng.randint(r, width - r)
    cy = rng.randint(r, height - r)
    cv2.circle(canvas, (cx, cy), r, (255, 255, 255), -1)
    boxes.append((1, [cx - r, cy - r, cx + r, cy + r]))   # class 1 = ball

    return canvas, boxes


def build_dataset(n_train=12, n_val=4):
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(n):
            seed = hash((split, i)) & 0xFFFF
            image, boxes = make_synthetic_image_and_boxes(seed)
            height, width = image.shape[:2]

            filename = f"{split}_{i:03d}"
            cv2.imwrite(os.path.join(img_dir, filename + ".jpg"), image)

            lines = [
                xyxy_to_yolo_label(class_id, box, width, height)
                for class_id, box in boxes
            ]
            with open(os.path.join(lbl_dir, filename + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")

    print(f"Built dataset at: {DATASET_DIR}")
    print(f"  train: {n_train} images, val: {n_val} images")


if not os.path.exists(DATASET_DIR):
    build_dataset()
else:
    print(f"Dataset already exists at {DATASET_DIR} (delete the folder to regenerate)")


# %% ------------------------------------------------------------------
# SECTION 3 — Write data.yaml
# ------------------------------------------------------------------
# This tiny file is how you tell Ultralytics YOLO: where are the images,
# and what do the class IDs mean.

data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
data_yaml_content = f"""\
path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
with open(data_yaml_path, "w") as f:
    f.write(data_yaml_content)

print(f"\nWrote data.yaml:\n{data_yaml_content}")


# %% ------------------------------------------------------------------
# SECTION 4 — THE DATASET SANITY CHECKER (worth keeping forever)
# ------------------------------------------------------------------
# This is the single most valuable piece of code from today. Whenever you
# build ANY YOLO dataset (synthetic or real), run something like this
# FIRST, before ever starting training. It catches: wrong class IDs,
# mismatched image/label pairs, boxes that don't make sense, wrong
# label format — all things that otherwise waste hours of "why is my
# model not learning" debugging.

def yolo_line_to_xyxy(line, image_width, image_height):
    parts = line.strip().split()
    class_id = int(parts[0])
    cx, cy, w, h = (float(v) for v in parts[1:5])
    cx *= image_width
    cy *= image_height
    w *= image_width
    h *= image_height
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return class_id, [x1, y1, x2, y2]


def sanity_check_and_visualize(split="train", n_samples=4):
    img_dir = os.path.join(DATASET_DIR, "images", split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)
    filenames = sorted(os.listdir(img_dir))[:n_samples]

    fig, axes = plt.subplots(1, len(filenames), figsize=(4 * len(filenames), 4))
    if len(filenames) == 1:
        axes = [axes]

    for ax, filename in zip(axes, filenames):
        image = cv2.imread(os.path.join(img_dir, filename))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        label_path = os.path.join(lbl_dir, filename.replace(".jpg", ".txt"))
        assert os.path.exists(label_path), f"MISSING LABEL FILE for {filename}!"

        with open(label_path) as f:
            lines = [l for l in f.readlines() if l.strip()]

        for line in lines:
            class_id, box = yolo_line_to_xyxy(line, width, height)
            x1, y1, x2, y2 = [int(v) for v in box]

            # Sanity checks (same spirit as Day 1's validate_box)
            assert 0 <= class_id < len(CLASS_NAMES), f"bad class_id {class_id}"
            assert x1 < x2 and y1 < y2, f"degenerate box in {filename}: {box}"
            assert 0 <= x1 and x2 <= width, f"box out of image bounds (x) in {filename}"
            assert 0 <= y1 and y2 <= height, f"box out of image bounds (y) in {filename}"

            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_rgb, CLASS_NAMES[class_id], (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        ax.imshow(image_rgb)
        ax.set_title(filename)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, f"day07_sanity_check_{split}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"All labels for split='{split}' passed sanity checks.")
    print(f"Saved visualization: {out_path}")


sanity_check_and_visualize("train")
sanity_check_and_visualize("val")


# %% ------------------------------------------------------------------
# SECTION 5 — Try breaking it on purpose
# ------------------------------------------------------------------
# Uncomment to see the sanity checker actually catch a bad label:
#
#   bad_label_path = os.path.join(DATASET_DIR, "labels", "train", "train_000.txt")
#   with open(bad_label_path, "a") as f:
#       f.write("5 0.5 0.5 0.1 0.1\n")   # class_id 5 doesn't exist! (only 0,1)
#   sanity_check_and_visualize("train")   # should raise an AssertionError


# %% ------------------------------------------------------------------
# DAY 7 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — Draw (on paper) the exact folder structure YOLO expects, from
  memory, including where data.yaml goes.

Test 2 — Given a label line "1 0.5 0.5 0.2 0.4" and an image that's
  800x600, calculate the pixel box [x1,y1,x2,y2] by hand.

Test 3 — What are at least 3 things your sanity checker validates? Why is
  each one worth catching BEFORE training rather than during/after?

Test 4 — What happens if an image file exists but its matching .txt label
  file is missing? What SHOULD happen (and does your checker catch it)?

Test 5 — Explain in one sentence why train/ and val/ need to be SEPARATE,
  non-overlapping sets of images.

If you can do all 5, Day 7 is done. Move to Day 8.
"""

if __name__ == "__main__":
    print("\nDay 7 script finished.")
