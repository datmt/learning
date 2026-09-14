Absolutely. Given your background as a backend engineer, I would **not teach this as a traditional computer-vision course**. We’ll use a code-first approach:

> **Understand the minimum theory → implement it → visualize it → use the real YOLO implementation → break it → fix it.**

The goal after Day 14 is not “I know what YOLO is.” It is:

> **I can take a new detection problem, prepare data, train YOLO, evaluate it, diagnose failures, run it on video, and add tracking/geometry.**

I’d structure every day as a Jupyter notebook with **code → experiment → observation → checkpoint**.

---

# The 14-day YOLO path

| Day | What you'll actually learn                         | Deliverable                         |
| --- | -------------------------------------------------- | ----------------------------------- |
| 1   | Images, pixels, bounding boxes, coordinate systems | Box manipulation notebook           |
| 2   | IoU + NMS                                          | NMS implementation from scratch     |
| 3   | Using YOLO                                         | First detector                      |
| 4   | Understanding YOLO outputs                         | Build your own result visualizer    |
| 5   | CNNs                                               | Tiny detector-like CNN              |
| 6   | YOLO architecture                                  | Backbone → neck → head mental model |
| 7   | Detection datasets                                 | Your own YOLO dataset               |
| 8   | Training                                           | Fine-tuned detector                 |
| 9   | Evaluation                                         | mAP + failure analysis              |
| 10  | Video                                              | Real-time detection pipeline        |
| 11  | Tracking                                           | Persistent player/ball IDs          |
| 12  | Small objects                                      | Small-ball/player detector          |
| 13  | Geometry                                           | Pixels → court coordinates          |
| 14  | Capstone                                           | Sports analytics pipeline           |

Let's walk through them.

---

# Day 1 — Images + Bounding Boxes

## Goal

By the end of today, you should be completely comfortable with:

```text
image
  ↓
pixels
  ↓
(x, y)
  ↓
bounding box
  ↓
(x1, y1, x2, y2)
```

This sounds trivial.

It isn't.

**Almost everything downstream in object detection depends on getting this representation right.**

---

## 1. Install the basic stack

In your Jupyter notebook:

```python
%pip install numpy opencv-python matplotlib
```

Then:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

---

# 2. What is an image?

Load an image.

```python
image = cv2.imread("image.jpg")

print(image.shape)
```

You'll get something like:

```text
(720, 1280, 3)
```

Meaning:

```text
height = 720
width  = 1280
channels = 3
```

OpenCV loads images as:

```text
BGR
```

while matplotlib expects:

```text
RGB
```

So:

```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off");
```

---

# 3. Understand pixel coordinates

The coordinate system is:

```text
(0,0) ───────────────→ x
  │
  │
  │
  ↓
  y
```

So:

```python
x = 100
y = 200
```

means:

> 100 pixels from the left and 200 pixels from the top.

Let's draw a point.

```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.scatter([100], [200])
plt.xlim(0, image_rgb.shape[1])
plt.ylim(image_rgb.shape[0], 0)
plt.grid()
```

This coordinate system becomes **extremely important** later when we deal with tracking and sports geometry.

---

# 4. Bounding boxes

The standard representation we'll use is:

```text
(x1, y1, x2, y2)
```

where:

```text
(x1,y1)
    ┌──────────────┐
    │              │
    │    object    │
    │              │
    └──────────────┘
                (x2,y2)
```

For example:

```python
box = np.array([100, 150, 300, 400])
```

means:

```text
left   = 100
top    = 150
right  = 300
bottom = 400
```

---

# 5. Draw a bounding box

OpenCV:

```python
image_draw = image_rgb.copy()

x1, y1, x2, y2 = box

cv2.rectangle(
    image_draw,
    (x1, y1),
    (x2, y2),
    (255, 0, 0),
    3
)

plt.figure(figsize=(10, 6))
plt.imshow(image_draw)
plt.axis("off");
```

You have now manually done one of the most fundamental things YOLO does:

> **Represent where an object is in an image.**

---

# 6. Convert between box formats

You need to know this cold.

There are several common representations.

### XYXY

```text
x1, y1, x2, y2
```

### XYWH

```text
center_x, center_y, width, height
```

### YOLO normalized format

```text
center_x / image_width
center_y / image_height
width / image_width
height / image_height
```

Let's implement conversion.

```python
def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return np.array([cx, cy, w, h])
```

Test:

```python
box = np.array([100, 150, 300, 400])

xyxy_to_xywh(box)
```

Result:

```text
[200, 275, 200, 250]
```

---

# 7. Convert to YOLO format

Suppose:

```python
image_width = 1280
image_height = 720
```

Then:

```python
def xyxy_to_yolo(box, image_width, image_height):
    cx, cy, w, h = xyxy_to_xywh(box)

    return np.array([
        cx / image_width,
        cy / image_height,
        w / image_width,
        h / image_height
    ])
```

Try:

```python
box = np.array([100, 150, 300, 400])

xyxy_to_yolo(
    box,
    image_width=1280,
    image_height=720
)
```

You'll get normalized numbers between 0 and 1.

This is the format we'll encounter when we build our own YOLO dataset.

---

# 8. Reverse the transformation

Never implement a transformation without implementing its inverse.

```python
def yolo_to_xyxy(label, image_width, image_height):
    cx, cy, w, h = label

    cx *= image_width
    cy *= image_height
    w *= image_width
    h *= image_height

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.array([x1, y1, x2, y2])
```

Test round-trip:

```python
box = np.array([100, 150, 300, 400])

label = xyxy_to_yolo(box, 1280, 720)
recovered = yolo_to_xyxy(label, 1280, 720)

print(box)
print(recovered)
```

You should get approximately:

```text
[100 150 300 400]
[100. 150. 300. 400.]
```

---

# 9. Resize an image and its boxes

This is where things become interesting.

Suppose:

```text
original: 1280 × 720
new:       640 × 360
```

The image is scaled by:

```text
0.5
```

The bounding box must also scale.

```python
def resize_box(box, old_size, new_size):
    old_w, old_h = old_size
    new_w, new_h = new_size

    sx = new_w / old_w
    sy = new_h / old_h

    x1, y1, x2, y2 = box

    return np.array([
        x1 * sx,
        y1 * sy,
        x2 * sx,
        y2 * sy
    ])
```

Test:

```python
box = np.array([100, 150, 300, 400])

resize_box(
    box,
    old_size=(1280, 720),
    new_size=(640, 360)
)
```

Result:

```text
[ 50.  75. 150. 200.]
```

---

# 10. Your first real exercise

Don't just run my code.

Create:

```python
def draw_box(image, box, label=None):
    ...
```

It should:

1. receive an image
2. receive `[x1,y1,x2,y2]`
3. draw the rectangle
4. optionally draw a label
5. return the modified image

Then:

```python
result = draw_box(
    image_rgb,
    [100, 150, 300, 400],
    "player"
)

plt.imshow(result)
plt.axis("off");
```

---

# 11. Important experiment: break it

This is one of the best ways to learn CV.

Try:

```python
box = [300, 400, 100, 150]
```

What happens?

The box is invalid because:

```text
x2 < x1
y2 < y1
```

Write:

```python
def validate_box(box, image_width, image_height):
    ...
```

It should check:

```text
x1 >= 0
y1 >= 0
x2 <= width
y2 <= height
x1 < x2
y1 < y2
```

This kind of validation becomes incredibly useful when debugging real datasets.

---

# Day 1 mental model

You should finish today understanding:

```text
IMAGE
 │
 ├── height
 ├── width
 └── pixels
       │
       ↓
OBJECT
       │
       ↓
BOUNDING BOX
       │
       ├── x1
       ├── y1
       ├── x2
       └── y2
```

And the three transformations:

```text
XYXY
  ↕
XYWH
  ↕
YOLO normalized coordinates
```

---

# Day 1 checkpoint

**Do not move to Day 2 until you can do these without looking at the answer:**

### Test 1

Given:

```python
image_width = 1920
image_height = 1080

box = [480, 270, 960, 810]
```

Calculate:

```text
center_x
center_y
width
height
```

### Test 2

Convert that box into YOLO normalized coordinates.

### Test 3

Resize the image from:

```text
1920 × 1080
```

to:

```text
640 × 360
```

and correctly transform the box.

### Test 4

Draw the transformed box.

### Test 5

Explain why a bounding box **must** be transformed when the image is resized.

If you can do those, **Day 1 is done.**

---

# Day 2 — IoU + NMS

Now we start getting into the machinery that makes object detection actually work.

You'll implement:

```text
IoU
 │
 ├── intersection
 ├── union
 └── overlap ratio
```

Then:

```text
many candidate boxes
        ↓
      NMS
        ↓
best boxes
```

You'll implement NMS yourself rather than immediately relying on YOLO.

The key code will eventually look roughly like:

```python
def iou(box_a, box_b):
    ...
```

and:

```python
def nms(boxes, scores, iou_threshold=0.5):
    ...
```

Then we'll visualize something like:

```text
       ┌────────────┐
       │     A      │
       │  ┌──────┐  │
       │  │  B   │  │
       │  └──────┘  │
       └────────────┘

A = 0.91
B = 0.87

       ↓ NMS

       ┌────────────┐
       │     A      │
       └────────────┘
```

This is the day where you'll understand **why YOLO can produce multiple overlapping predictions for the same object and how those predictions get filtered.**

---

# Day 3 — Run real YOLO

Now we stop implementing toy algorithms and use an actual detector.

We'll install Ultralytics and run:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

Then:

```python
results = model("image.jpg")
```

And visualize:

```python
results[0].show()
```

You'll see the complete pipeline:

```text
image
  ↓
YOLO
  ↓
boxes
classes
confidence
  ↓
visualization
```

The important thing on Day 3 isn't training.

It's:

> **I can run a modern detector and understand what it is doing at a high level.**

---

# Day 4 — Understand YOLO's output

This day is extremely important.

Instead of:

```python
results[0].show()
```

we'll inspect the raw-ish objects:

```python
result = results[0]

result.boxes
```

Then:

```python
result.boxes.xyxy
result.boxes.conf
result.boxes.cls
```

You'll build your own:

```python
for box, confidence, class_id in zip(...):
    print(...)
```

and then your own visualization.

By the end:

```text
YOLO result
     │
     ├── box
     │    └── x1,y1,x2,y2
     │
     ├── confidence
     │
     └── class
```

should be second nature.

---

# Day 5 — CNN fundamentals

Now we answer:

> **Where does YOLO actually get these predictions from?**

You'll build a tiny CNN in PyTorch.

Something like:

```python
class TinyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Linear(...)
```

You'll train it on MNIST/CIFAR-like data.

The important thing isn't becoming a CNN researcher.

It's understanding:

```text
image
 ↓
convolution
 ↓
feature map
 ↓
downsampling
 ↓
deeper features
 ↓
prediction
```

And especially:

> **Why spatial information survives through a CNN even though the network is learning abstract features.**

---

# Day 6 — YOLO architecture

Now we put the pieces together.

You'll learn the three major components:

```text
                YOLO
                 │
       ┌─────────┴─────────┐
       ↓                   ↓
   Backbone               Neck
       │                   │
       └──────────┬────────┘
                  ↓
                 Head
                  │
                  ↓
        boxes + classes + scores
```

Conceptually:

### Backbone

Extract features.

```text
image → low-level → mid-level → high-level features
```

### Neck

Combine information at different scales.

This matters enormously for:

```text
large objects
medium objects
small objects
```

### Head

Predict:

```text
where?
what?
how confident?
```

You'll visualize feature maps and progressively shrink the image through the network.

---

# Day 7 — Build a dataset

Now you become dangerous.

We'll create a tiny custom dataset.

For example:

```text
sports/
    images/
        train/
        val/

    labels/
        train/
        val/

    data.yaml
```

A label might be:

```text
0 0.52 0.61 0.08 0.20
```

meaning:

```text
class = 0

center_x = 0.52
center_y = 0.61
width    = 0.08
height   = 0.20
```

You'll write code that:

1. loads images
2. loads labels
3. converts YOLO coordinates to pixels
4. draws boxes
5. displays random samples

This becomes your **dataset sanity checker**.

That checker is worth keeping permanently.

---

# Day 8 — Train YOLO

Now:

```text
dataset
   ↓
YOLO
   ↓
training
   ↓
weights
```

You'll run a small fine-tuning experiment.

Something conceptually like:

```python
model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=16
)
```

The important lesson:

**Don't start by trying to achieve amazing accuracy.**

First prove:

```text
loss decreases
+
predictions improve
+
model can overfit a tiny dataset
```

A tiny dataset that the model cannot overfit is a fantastic debugging tool.

---

# Day 9 — Evaluation

Now we answer:

> **Is my detector actually good?**

You'll learn:

```text
TP
FP
FN
```

Then:

```text
precision
recall
```

Then:

```text
AP
mAP
```

But I want you to understand them visually.

For example:

```text
Ground truth:

       █████
       █████

Prediction:

       █████
       █████
```

Good IoU.

Versus:

```text
Ground truth:

       █████

Prediction:

                 █████
```

Bad localization.

You'll create a failure-analysis notebook categorizing errors:

```text
false positive
false negative
bad localization
wrong class
duplicate detection
small object missed
occlusion
```

This is much more valuable than memorizing mAP.

---

# Day 10 — Video

Images are easy.

Video introduces:

```text
frame 1
frame 2
frame 3
...
```

You'll write:

```python
cap = cv2.VideoCapture("game.mp4")

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    ...
```

Then measure:

```text
FPS
inference time
latency
```

This starts connecting your CV knowledge to your backend/systems background.

---

# Day 11 — Tracking

Detection asks:

> "What objects are in this frame?"

Tracking asks:

> "Is this the same object I saw 20 frames ago?"

You'll build:

```text
Frame 1
player → ID 7

Frame 2
player → ID 7

Frame 3
player → ID 7
```

Conceptually:

```text
YOLO
 ↓
detections
 ↓
association
 ↓
tracks
```

You'll learn the ideas behind:

```text
Kalman filtering
IoU association
appearance/features
track lifecycle
```

and implement a simplified ByteTrack-style association pipeline.

---

# Day 12 — Small objects

This is where your sports-vision curriculum gets serious.

Consider a tennis/badminton ball.

The object might occupy:

```text
8 × 8 pixels
```

while the image is:

```text
1920 × 1080
```

That's fundamentally harder than detecting a player.

You'll experiment with:

```text
resolution
crop
augmentation
confidence threshold
IoU threshold
small-object recall
```

You'll learn an important practical lesson:

> **A detector can have excellent overall mAP while being nearly useless for the object you actually care about.**

So we'll inspect per-class/per-size failures rather than trusting one number.

---

# Day 13 — Geometry

Now we leave pure detection.

Suppose YOLO gives you:

```text
player center = (843, 527)
```

Those are **image coordinates**.

But you may want:

```text
court position = (8.3m, 4.7m)
```

You'll learn:

```text
pixel coordinates
       ↓
homography
       ↓
world/court coordinates
```

Conceptually:

```text
IMAGE                     COURT

(x,y)                      (X,Y)

   ●                         ●
   │                         │
   │  homography             │
   └─────────────────────────┘
```

This is where your linear algebra background becomes extremely useful.

---

# Day 14 — Sports Vision Capstone

Finally:

```text
                  VIDEO
                    │
                    ↓
                  YOLO
                    │
             ┌──────┴──────┐
             ↓             ↓
         players          ball
             │             │
             └──────┬──────┘
                    ↓
                 TRACKING
                    │
                    ↓
              court geometry
                    │
                    ↓
             analytics
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
    heatmap      trajectories    speed
```

Your final notebook should be able to:

### Input

```text
sports video
```

### Output

```text
player bounding boxes
player IDs
ball detection
ball trajectory
court coordinates
player trajectories
heatmap
FPS
```

And ideally produce a rendered video.

---

# The most important thing: don't study these days independently

The curriculum has a deliberate dependency graph:

```text
DAY 1
Images + coordinates
       │
       ↓
DAY 2
IoU + NMS
       │
       ↓
DAY 3
YOLO inference
       │
       ↓
DAY 4
YOLO outputs
       │
       ├──────────────┐
       ↓              ↓
DAY 5             DAY 7
CNN              Dataset
       │              │
       ↓              ↓
DAY 6             DAY 8
Architecture      Training
                      │
                      ↓
                   DAY 9
                 Evaluation
                      │
                      ↓
                   DAY 10
                    Video
                      │
                      ↓
                   DAY 11
                  Tracking
                      │
               ┌──────┴──────┐
               ↓             ↓
             DAY 12        DAY 13
          Small objects   Geometry
               │             │
               └──────┬──────┘
                      ↓
                   DAY 14
                  CAPSTONE
```

That is the learning path I recommend.

## And one adjustment I'd make for you

Because you're already a **10-year backend engineer with linear algebra/calculus**, don't spend days memorizing CNN terminology or deriving every convolution equation.

Your bottleneck won't be mathematics.

It will be developing **CV intuition**:

> **What does the image look like? What does the detector actually see? Why did it predict this box? Why did it miss that object? Why does changing resolution help? Why does tracking lose an ID? Why does a model with good mAP still fail my application?**

So our notebooks should constantly alternate:

```text
CODE
 ↓
VISUALIZE
 ↓
ASK "WHY?"
 ↓
CHANGE ONE THING
 ↓
RUN AGAIN
 ↓
EXPLAIN WHAT CHANGED
```

That will get you work-ready much faster than a conventional YOLO course.

**Start with Day 1 above.** Once you've run the Day 1 notebook and exercises, bring me your output/errors (or your notebook), and we'll work through **Day 2 interactively rather than just reading ahead**.

