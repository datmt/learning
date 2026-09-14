"""
DAY 8 — Train (Fine-Tune) YOLO on Your Own Dataset
======================================================

RECAP: Day 7 you built a dataset + data.yaml + a sanity checker. Today you
actually train, meaning: start from the pretrained COCO weights (Day 3) and
keep adjusting them so the model gets good at YOUR classes ("player",
"ball") instead of (or in addition to) COCO's 80 generic classes. This is
called FINE-TUNING — much faster and needs way less data than training
from scratch.

THE PIPELINE
--------------
    dataset (Day 7)
        |
        v
      YOLO (starts from pretrained weights)
        |
        v
      training loop (repeat: predict -> compare to labels -> adjust weights)
        |
        v
      new weights file (best.pt)

THE MOST IMPORTANT LESSON TODAY (read this before running anything)
------------------------------------------------------------------------
Do NOT judge yourself by getting "amazing accuracy" on your first attempt.
The correct FIRST goal, always, is:

    1. Does the loss go DOWN over epochs? (proves the training loop works
       at all — data loading, labels, optimizer are all wired correctly)
    2. Can the model deliberately OVERFIT a tiny dataset?
       ("overfit" = memorize the training images almost perfectly)

If a model CANNOT overfit a tiny dataset (say, 12 images), something is
BROKEN in your pipeline — bad labels, images not loading, wrong class
count, etc. A model that can't even memorize 12 images will never
generalize to new ones. This is one of the best debugging habits in all of
deep learning, not just YOLO.

Run with:  python day08_train_yolo.py
(Uses the toy dataset from Day 7 — run that script first if you haven't.)
"""

import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET_DIR = os.path.join(PROJECT_ROOT, "assets", "toy_sports_dataset")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

if not os.path.exists(DATA_YAML):
    raise SystemExit(
        "Dataset not found. Run day07_build_a_dataset.py first to generate "
        f"{DATA_YAML}"
    )


# %% ------------------------------------------------------------------
# SECTION 1 — Start from pretrained weights
# ------------------------------------------------------------------
# Starting from "yolo11n.pt" (pretrained on COCO) instead of random weights
# means the backbone already knows general things like "edges", "textures",
# "shapes" — we only need to teach it OUR specific classes on top of that.
# This is why fine-tuning needs far fewer images than training from zero.

model = YOLO("yolo11n.pt")


# %% ------------------------------------------------------------------
# SECTION 2 — Train (overfit-check run)
# ------------------------------------------------------------------
# With only 12 training images and a toy synthetic task, we expect the
# model to overfit QUICKLY (get very good on train, potentially mediocre
# on val since val objects are randomly different). That's fine — the
# point today is proving the pipeline works, not achieving production
# accuracy.
#
# Key arguments explained:
#   data    -> path to data.yaml (Day 7)
#   epochs  -> how many full passes over the training data
#   imgsz   -> images get resized to this size before going into the model
#              (remember Day 1, Section 9: resizing boxes matters — YOLO's
#              training code does this for you automatically)
#   batch   -> how many images are processed together per training step
#   project/name -> where results (weights, plots) get saved

results = model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    batch=4,
    project=os.path.join(PROJECT_ROOT, "assets", "runs"),
    name="day08_overfit_check",
    exist_ok=True,
    verbose=False,
    plots=True,
)

print("\nTraining finished.")
print("Results/weights saved under: assets/runs/day08_overfit_check/")


# %% ------------------------------------------------------------------
# SECTION 3 — Check: did loss go down? Can it overfit?
# ------------------------------------------------------------------
# Ultralytics saves a results.csv with per-epoch metrics. Let's read it and
# plot the training loss curve ourselves (don't just trust a summary
# number — LOOK at the trend).

import pandas as pd
import matplotlib.pyplot as plt

results_csv = os.path.join(
    PROJECT_ROOT, "assets", "runs", "day08_overfit_check", "results.csv"
)
df = pd.read_csv(results_csv)
df.columns = [c.strip() for c in df.columns]
print("\nColumns available in results.csv:", list(df.columns))

box_loss_col = [c for c in df.columns if "box_loss" in c and "train" in c][0]

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df[box_loss_col], marker="o")
plt.xlabel("epoch")
plt.ylabel(box_loss_col)
plt.title("Training box loss over epochs (should trend DOWN)")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(PROJECT_ROOT, "assets", "day08_loss_curve.png"))
plt.close()
print("Saved: assets/day08_loss_curve.png")

first_loss = df[box_loss_col].iloc[0]
last_loss = df[box_loss_col].iloc[-1]
print(f"\nFirst epoch loss: {first_loss:.4f}")
print(f"Last epoch loss:  {last_loss:.4f}")
if last_loss < first_loss:
    print("Loss decreased. The training pipeline is working correctly.")
else:
    print("Loss did NOT decrease — something is likely wrong (check the dataset!).")


# %% ------------------------------------------------------------------
# SECTION 4 — Run the fine-tuned model on a training image
# ------------------------------------------------------------------
# Load the BEST checkpoint saved during training and run it on one of the
# training images. Since we're deliberately overfitting, it should predict
# the "player"/"ball" boxes quite well on images it has already seen.

best_weights = os.path.join(
    PROJECT_ROOT, "assets", "runs", "day08_overfit_check", "weights", "best.pt"
)
fine_tuned_model = YOLO(best_weights)

sample_image = os.path.join(DATASET_DIR, "images", "train", "train_000.jpg")
result = fine_tuned_model(sample_image, verbose=False)[0]

import cv2
annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 5))
plt.imshow(annotated)
plt.axis("off")
plt.title("Fine-tuned model on a TRAINING image (should look good if overfitting worked)")
plt.savefig(os.path.join(PROJECT_ROOT, "assets", "day08_finetuned_prediction.png"))
plt.close()
print("Saved: assets/day08_finetuned_prediction.png")
print(f"Detections found: {len(result.boxes)}")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
print("""
dataset (images + labels)
   |
   v
YOLO (pretrained backbone, fresh head for your classes)
   |
   v
training loop, repeated per epoch:
   for each batch of images:
       predict boxes/classes
       compare to ground-truth labels -> compute loss
       adjust weights to reduce loss
   |
   v
best.pt  (the checkpoint with the best validation performance)

First goal:  loss trending down + model overfits a tiny dataset.
Only AFTER that works do you scale up: more data, more epochs, evaluate
properly (Day 9) with metrics beyond "does it look right on one image".
""")


# %% ------------------------------------------------------------------
# DAY 8 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — What does "fine-tuning" mean, and why is it faster/needs less
  data than training from scratch?

Test 2 — Why did we choose to test "can the model overfit a tiny dataset"
  BEFORE worrying about validation accuracy?

Test 3 — Look at your day08_loss_curve.png. Is the trend downward? If it's
  flat or noisy, what are 2 things you'd check first? (Hint: revisit your
  Day 7 sanity checker.)

Test 4 — What's the difference between the pretrained "yolo11n.pt" you
  used on Day 3 and the "best.pt" you loaded today?

Test 5 — Explain what `imgsz=640` does to your training images, and why
  Day 1's box-resizing logic (Section 9) needs to happen automatically
  behind the scenes for training to work correctly.

If you can do all 5, Day 8 is done. Move to Day 9.
"""

if __name__ == "__main__":
    print("\nDay 8 script finished.")
