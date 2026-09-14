"""
DAY 3 — Run a Real YOLO Detector
===================================

RECAP: Days 1-2 you built the LEGO bricks by hand (boxes, IoU, NMS). Today
you stop building toys and drive a real, pretrained neural network that
does all of that internally, automatically, at high accuracy.

WHAT YOU'RE ACTUALLY DOING TODAY
----------------------------------
"Ultralytics" is a Python library that provides YOLO models, ready to use,
already trained on a huge dataset called COCO (80 everyday object classes:
person, car, dog, ball, chair, etc). You are NOT training anything today.
You are just running INFERENCE — feeding an image in, getting boxes out.

    image  ->  YOLO (pretrained)  ->  boxes + classes + confidence scores

Today's goal is NOT to understand every internal detail. It's:

    "I can run a modern detector on an image and see it work."

Install (already in requirements.txt, but shown here for reference):

    pip install ultralytics

The first time you run this, Ultralytics will download the model weights
file (a few MB) automatically from the internet. That's normal and only
happens once (cached locally after).
"""

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


# %% ------------------------------------------------------------------
# SECTION 1 — Load a pretrained model
# ------------------------------------------------------------------
# "yolo11n.pt" = YOLO version 11, "n" = nano (the smallest/fastest variant,
# good for learning on a laptop CPU). Bigger letters (s, m, l, x) trade
# speed for accuracy — you'll care about that choice later, not today.

model = YOLO("yolo11n.pt")
print("Model loaded. It knows these classes (first 10 of 80):")
print(list(model.names.values())[:10])


# %% ------------------------------------------------------------------
# SECTION 2 — Get a test image
# ------------------------------------------------------------------
# Use a real photo if you have one handy (drop it in the assets/ folder
# and change TEST_IMAGE below). Otherwise, we download one well-known demo
# image that Ultralytics itself uses in its docs (a street scene with
# people/buses), so this script runs out of the box.

TEST_IMAGE = os.path.join(ASSETS_DIR, "day03_bus.jpg")
if not os.path.exists(TEST_IMAGE):
    import urllib.request
    url = "https://ultralytics.com/images/bus.jpg"
    print(f"Downloading demo image from {url} ...")
    urllib.request.urlretrieve(url, TEST_IMAGE)

print("Using test image:", TEST_IMAGE)


# %% ------------------------------------------------------------------
# SECTION 3 — Run inference (the actual detection step)
# ------------------------------------------------------------------
# Calling the model like a function runs the full pipeline for you:
#   image -> preprocess -> neural network -> raw predictions -> NMS -> boxes
# Yes — the NMS you wrote by hand yesterday is happening inside here too!

results = model(TEST_IMAGE)

# `results` is a list (one entry per image you passed in — we passed one).
result = results[0]
print(f"\nFound {len(result.boxes)} objects in the image.")


# %% ------------------------------------------------------------------
# SECTION 4 — The quick built-in visualization
# ------------------------------------------------------------------
# Ultralytics gives you a one-liner that draws all boxes + labels for you.
# `.plot()` returns a numpy image (BGR, OpenCV-style) you can save/show.

annotated = result.plot()   # BGR numpy array with boxes drawn on it
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(annotated_rgb)
plt.axis("off")
plt.title("YOLO detections (built-in visualizer)")
plt.savefig(os.path.join(ASSETS_DIR, "day03_detections.png"))
plt.close()
print("Saved: day03_detections.png  <- open this and look at it!")


# %% ------------------------------------------------------------------
# SECTION 5 — The full pipeline, in plain words
# ------------------------------------------------------------------
print("""
The pipeline you just ran:

    image
      |
      v
    YOLO neural network   (this is what Day 5-6 explain: HOW it works)
      |
      v
    boxes, classes, confidence scores   (this is what Day 4 dissects)
      |
      v
    visualization  (what you just saved)

Today's win: you don't need to understand the network internals yet to
USE a state-of-the-art detector. That's the power of pretrained models.
""")


# %% ------------------------------------------------------------------
# SECTION 6 — Try your own image / try a different model size
# ------------------------------------------------------------------
# Experiment (edit and re-run):
#   1. Drop a photo of your own into assets/ and point TEST_IMAGE at it.
#   2. Try model = YOLO("yolo11s.pt") (the "small" variant) and compare
#      results/speed to the "n" (nano) variant.
#   3. Try lowering the confidence threshold to see MORE (often noisier)
#      detections:
#          results = model(TEST_IMAGE, conf=0.1)
#      vs. the default (conf=0.25) vs. a strict one (conf=0.7).


# %% ------------------------------------------------------------------
# DAY 3 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — What is the difference between TRAINING and INFERENCE? Which one
  did you do today?

Test 2 — Why did NMS from Day 2 matter here, even though you didn't call
  your own nms() function? (Hint: where did it happen?)

Test 3 — Run the model with conf=0.05 and conf=0.9. Describe, in your own
  words, what changes about the output, and why.

Test 4 — model.names is a dictionary mapping class IDs (integers) to class
  names (strings). Print it and find the ID number for "person" and for
  "car" (or two classes of your choice).

Test 5 — Explain in one sentence what `results[0].plot()` is doing
  internally (rough idea, not exact code) using words like "box", "label",
  "draw".

If you can answer all 5, Day 3 is done. Move to Day 4.
"""

if __name__ == "__main__":
    print("\nDay 3 script finished. Check ../assets/day03_detections.png")
