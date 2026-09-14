# 14-Day YOLO / Computer Vision Path

Beginner-friendly, code-first path from "what is a pixel" to a working
sports-analytics pipeline (detection + tracking + court geometry).

Each day is one runnable Python file in `days/`. Every file:
- explains the concept in plain English before any code
- is organized in `# %%` cells (run cell-by-cell in VS Code / Jupyter, or
  just run the whole file top to bottom)
- saves its output images/videos into `assets/`
- ends with a **checkpoint**: questions to answer from memory before
  moving to the next day. Don't skip these — the curriculum is
  cumulative, later days assume earlier ones stuck.

## Setup (do this once)

```bash
cd /home/dat/data/learning/cv-and-yolo
source .venv/bin/activate      # a Python 3.11 virtualenv, already created
# (if .venv doesn't exist yet: python3.11 -m venv .venv && pip install -r requirements.txt)
```

The venv already has everything installed: numpy, opencv-python,
matplotlib, torch, torchvision, ultralytics, scipy, pandas.

## Running a day

```bash
python days/day01_images_and_boxes.py
```

Each script is self-contained — it generates any synthetic image/video it
needs, downloads the one real demo photo/model it needs (first run only),
and writes its results to `assets/`. You never have to go find your own
data to get started; swap in real photos/video later once you're
comfortable.

## The 14 days

| Day | File | Topic |
|---|---|---|
| 1 | `day01_images_and_boxes.py` | Images as pixel arrays, bounding boxes, XYXY/XYWH/YOLO formats |
| 2 | `day02_iou_and_nms.py` | IoU from scratch, NMS from scratch |
| 3 | `day03_run_real_yolo.py` | Run a pretrained YOLO model |
| 4 | `day04_understand_yolo_output.py` | Dissect YOLO's raw output, build your own visualizer |
| 5 | `day05_cnn_fundamentals.py` | Train a tiny CNN on MNIST, visualize feature maps |
| 6 | `day06_yolo_architecture.py` | Backbone / neck / head, inspected inside a real YOLO model |
| 7 | `day07_build_a_dataset.py` | Build a YOLO dataset + a dataset sanity checker |
| 8 | `day08_train_yolo.py` | Fine-tune YOLO, prove it can overfit a tiny dataset |
| 9 | `day09_evaluation.py` | TP/FP/FN, precision/recall, mAP, visual failure analysis |
| 10 | `day10_video.py` | Run YOLO over video, measure FPS/latency |
| 11 | `day11_tracking.py` | A simplified IoU-based multi-object tracker |
| 12 | `day12_small_objects.py` | Why small objects are hard, and the levers to fix it |
| 13 | `day13_geometry.py` | Homography: pixel coordinates → real-world court coordinates |
| 14 | `day14_capstone.py` | Full pipeline: video → detect → track → geometry → analytics |

## Dependency graph (don't skip around)

```
Day 1 -> Day 2 -> Day 3 -> Day 4 -> +-- Day 5 -> Day 6
                                    +-- Day 7 -> Day 8 -> Day 9 -> Day 10 -> Day 11 -> +-- Day 12 -> Day 14
                                                                                        +-- Day 13 -> Day 14
```

## Folder layout after running everything

```
cv-and-yolo/
  days/day01...day14.py     <- the lessons
  assets/                    <- every generated image/video/dataset lands here
  data/                      <- MNIST (downloaded by Day 5)
  requirements.txt
  .venv/                     <- Python 3.11 virtualenv
  yolo11n.pt                 <- cached pretrained YOLO weights
```
