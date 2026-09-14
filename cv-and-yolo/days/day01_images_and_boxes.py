"""
DAY 1 — Images, Pixels, and Bounding Boxes
============================================

WHY THIS DAY MATTERS
---------------------
Every single thing YOLO (or any object detector) does, eventually boils down
to this one idea:

    "Here is a rectangle in this image, and I think it contains a dog."

That rectangle is called a BOUNDING BOX. If you don't deeply understand how
images are stored as numbers, and how boxes are represented and transformed,
nothing later (training, evaluation, tracking) will make sense — you'll just
be copy-pasting code you don't trust.

Today has NO machine learning. It's pure "data representation". Boring on
purpose. Go slow.

HOW TO RUN THIS FILE
---------------------
This file is written in "Jupyter cell" style. The comments `# %%` mark the
start of a new cell. If you open this file in VS Code (with the Python
extension) or PyCharm, you can run cell-by-cell with Shift+Enter, just like
a notebook. Or you can just run the whole file with:

    python day01_images_and_boxes.py

Install dependencies first (only need to do this once):

    pip install numpy opencv-python matplotlib
"""

# %% ------------------------------------------------------------------
# SECTION 1 — What is an image, to a computer?
# ------------------------------------------------------------------
# A color photo is just a big 3D grid (array) of numbers.
#
#   - Two of the dimensions are spatial: HEIGHT (rows) and WIDTH (columns).
#   - The third dimension is COLOR CHANNELS: usually 3 numbers per pixel,
#     one each for how much Red, Green, and Blue light is at that pixel.
#
# So a "pixel" is nothing but 3 numbers (0-255) sitting at one grid location.
# An "image" is a stack of height x width pixels.
#
# We don't have a real photo yet, so let's manufacture one: a plain gray
# rectangle with a brighter rectangle drawn on it (pretending to be an
# "object"). This means the file runs immediately with ZERO setup — no need
# to go find a .jpg first.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
DEMO_IMAGE_PATH = os.path.join(ASSETS_DIR, "day01_demo.jpg")

if not os.path.exists(DEMO_IMAGE_PATH):
    # Build a synthetic "photo" so this script never breaks for lack of a
    # real image file. Feel free to replace with your own photo later —
    # just change DEMO_IMAGE_PATH above to point at it.
    canvas = np.full((720, 1280, 3), 40, dtype=np.uint8)   # dark gray background
    cv2.rectangle(canvas, (400, 250), (750, 600), (90, 140, 200), -1)  # a "player"
    cv2.circle(canvas, (950, 200), 15, (255, 255, 255), -1)             # a "ball"
    cv2.imwrite(DEMO_IMAGE_PATH, canvas)

image = cv2.imread(DEMO_IMAGE_PATH)   # returns a numpy array

print("Type of `image`:", type(image))
print("Shape of `image`:", image.shape)

# You should see something like: (720, 1280, 3)
#   -> height = 720   (number of rows, i.e. vertical pixels)
#   -> width  = 1280  (number of columns, i.e. horizontal pixels)
#   -> channels = 3   (Blue, Green, Red — see note below!)


# %% ------------------------------------------------------------------
# SECTION 2 — The BGR vs RGB trap (a classic beginner bug)
# ------------------------------------------------------------------
# OpenCV (`cv2`) was designed decades ago and stores color channels in the
# order BLUE, GREEN, RED ("BGR"). Almost every other tool (matplotlib,
# PyTorch, PIL, your web browser) expects RED, GREEN, BLUE ("RGB").
#
# If you forget to convert, your images will look like they have a weird
# blue/orange color cast. This single bug wastes hours of beginners' time,
# so let's nail it now.

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image_rgb)
plt.title("Correctly converted to RGB for display")
plt.axis("off")
plt.savefig(os.path.join(ASSETS_DIR, "day01_step2_rgb.png"))
plt.close()
print("Saved: day01_step2_rgb.png  (open it and look)")


# %% ------------------------------------------------------------------
# SECTION 3 — Pixel coordinates: where is (x, y)?
# ------------------------------------------------------------------
# This trips people up because it's the OPPOSITE of normal math-class
# (x, y) axes. In images:
#
#     (0,0) ───────────────→ x increases to the RIGHT
#       │
#       │
#       ↓
#       y increases DOWNWARD
#
# So x=100, y=200 means "100 pixels in from the left edge, 200 pixels down
# from the top edge". Note this also means when you index the numpy array
# directly, it's `image[y, x]` (row first, i.e. height/y, then column/x) —
# backwards from how we usually say "x, y" out loud. This mismatch between
# "row,col" (numpy) and "x,y" (bounding boxes) causes tons of confusion.
# Keep it in your head permanently.

x, y = 100, 200
plt.figure()
plt.imshow(image_rgb)
plt.scatter([x], [y], c="red", s=80, label=f"point (x={x}, y={y})")
plt.xlim(0, image_rgb.shape[1])
plt.ylim(image_rgb.shape[0], 0)   # inverted on purpose: y grows downward!
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(ASSETS_DIR, "day01_step3_point.png"))
plt.close()
print("Saved: day01_step3_point.png")


# %% ------------------------------------------------------------------
# SECTION 4 — Bounding boxes: (x1, y1, x2, y2)
# ------------------------------------------------------------------
# A bounding box is just TWO points: the top-left corner and the
# bottom-right corner of a rectangle around an object.
#
#     (x1,y1)
#         ┌──────────────┐
#         │              │
#         │    object    │
#         │              │
#         └──────────────┘
#                     (x2,y2)
#
# By convention: x1 = left edge, y1 = top edge, x2 = right edge,
# y2 = bottom edge. This is called the "XYXY" format.

box = np.array([400, 250, 750, 600])   # matches the rectangle we drew above
print("box (XYXY):", box)
print("left =", box[0], " top =", box[1], " right =", box[2], " bottom =", box[3])


# %% ------------------------------------------------------------------
# SECTION 5 — Draw a bounding box on the image
# ------------------------------------------------------------------
image_draw = image_rgb.copy()
x1, y1, x2, y2 = box

cv2.rectangle(
    image_draw,
    (x1, y1),      # top-left corner
    (x2, y2),      # bottom-right corner
    (255, 0, 0),   # color in RGB (since image_draw is already RGB) — red
    3,             # line thickness in pixels
)

plt.figure(figsize=(10, 6))
plt.imshow(image_draw)
plt.axis("off")
plt.savefig(os.path.join(ASSETS_DIR, "day01_step5_box.png"))
plt.close()
print("Saved: day01_step5_box.png")
print("\nYou just manually did the #1 job of any object detector:")
print("  -> represent WHERE an object is, as a rectangle of numbers.")


# %% ------------------------------------------------------------------
# SECTION 6 — Other box formats you MUST know cold
# ------------------------------------------------------------------
# Different tools/papers use different box representations. You'll see all
# three constantly, so memorize the conversions rather than looking them up
# every time.
#
# 1) XYXY            -> (x1, y1, x2, y2)                     [corners]
# 2) XYWH            -> (center_x, center_y, width, height)  [center+size]
# 3) YOLO normalized -> same as XYWH, but each value divided by the image's
#                        width or height, so everything is between 0 and 1.
#                        This makes boxes independent of image resolution —
#                        crucial because YOLO trains on many differently
#                        sized images.

def xyxy_to_xywh(box):
    """Convert (x1,y1,x2,y2) -> (center_x, center_y, width, height)."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.array([cx, cy, w, h])


xywh = xyxy_to_xywh(box)
print("XYWH:", xywh)  # expect center=(575, 425), w=350, h=350


# %% ------------------------------------------------------------------
# SECTION 7 — Convert to YOLO normalized format
# ------------------------------------------------------------------
image_width, image_height = 1280, 720

def xyxy_to_yolo(box, image_width, image_height):
    """XYXY pixels -> YOLO normalized (all values in [0, 1])."""
    cx, cy, w, h = xyxy_to_xywh(box)
    return np.array([
        cx / image_width,
        cy / image_height,
        w / image_width,
        h / image_height,
    ])


yolo_label = xyxy_to_yolo(box, image_width, image_height)
print("YOLO normalized:", yolo_label)
# every number should be between 0 and 1


# %% ------------------------------------------------------------------
# SECTION 8 — Reverse the transformation (never write A->B without B->A!)
# ------------------------------------------------------------------
# A very useful engineering habit: whenever you write a transformation,
# immediately write its inverse and test that round-tripping gives you
# back the original value. This catches bugs immediately instead of three
# days later when your training data looks subtly wrong.

def yolo_to_xyxy(label, image_width, image_height):
    """YOLO normalized -> XYXY pixels."""
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


recovered = yolo_to_xyxy(yolo_label, image_width, image_height)
print("original box :", box)
print("recovered box:", recovered)
# should match (tiny floating point differences are fine, e.g. 400.0000001)


# %% ------------------------------------------------------------------
# SECTION 9 — Resizing images ALSO resizes boxes (this is the "gotcha")
# ------------------------------------------------------------------
# When you shrink/grow an image (very common: YOLO usually resizes every
# image to a fixed size like 640x640 before feeding it to the network), the
# pixel coordinates of your boxes are now WRONG unless you scale them by
# the exact same factor as the image. Forgetting this is one of the most
# common real bugs in object detection code — boxes end up floating in the
# wrong place, or half the size they should be.

def resize_box(box, old_size, new_size):
    """Scale a box's pixel coordinates when the image itself is resized."""
    old_w, old_h = old_size
    new_w, new_h = new_size
    sx = new_w / old_w   # scale factor for x-axis (width)
    sy = new_h / old_h   # scale factor for y-axis (height)
    x1, y1, x2, y2 = box
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy])


resized_box = resize_box(box, old_size=(1280, 720), new_size=(640, 360))
print("box at 1280x720:", box)
print("box at 640x360 :", resized_box)
# every coordinate should be exactly HALF, since 640/1280 = 0.5


# %% ------------------------------------------------------------------
# SECTION 10 — YOUR EXERCISE: write draw_box() yourself
# ------------------------------------------------------------------
# Don't just run my code — re-implement this one yourself before checking
# the reference solution below. It should:
#   1. take an image
#   2. take a box [x1, y1, x2, y2]
#   3. draw a rectangle
#   4. optionally draw a text label above the box
#   5. return the modified image (don't modify the original in place!)

def draw_box(image, box, label=None, color=(255, 0, 0), thickness=3):
    out = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(
            out, label, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
    return out


result = draw_box(image_rgb, box, "player")
plt.figure(figsize=(10, 6))
plt.imshow(result)
plt.axis("off")
plt.savefig(os.path.join(ASSETS_DIR, "day01_step10_labelled.png"))
plt.close()
print("Saved: day01_step10_labelled.png")


# %% ------------------------------------------------------------------
# SECTION 11 — Break it on purpose: invalid boxes
# ------------------------------------------------------------------
# Real datasets are messy. Annotation tools sometimes produce garbage boxes
# (flipped corners, boxes outside the image, zero-size boxes). Learning to
# validate boxes NOW will save you painful debugging sessions later when
# training mysteriously fails or a loss becomes NaN.

broken_box = [300, 400, 100, 150]   # x2 < x1 and y2 < y1 -- invalid!

def validate_box(box, image_width, image_height):
    """Return (is_valid: bool, reason: str)."""
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0:
        return False, "top-left corner is outside the image (negative)"
    if x2 > image_width or y2 > image_height:
        return False, "bottom-right corner is outside the image bounds"
    if x1 >= x2:
        return False, "x1 >= x2 (box has zero or negative width)"
    if y1 >= y2:
        return False, "y1 >= y2 (box has zero or negative height)"
    return True, "ok"


is_valid, reason = validate_box(broken_box, image_width, image_height)
print(f"broken_box valid? {is_valid}  ({reason})")

is_valid, reason = validate_box(box, image_width, image_height)
print(f"box (good) valid? {is_valid}  ({reason})")


# %% ------------------------------------------------------------------
# MENTAL MODEL TO WALK AWAY WITH
# ------------------------------------------------------------------
#   IMAGE
#    ├── height, width          <- shape of the numpy array
#    └── pixels                  <- numbers 0-255 per channel, per location
#         └── OBJECT
#              └── BOUNDING BOX
#                   ├── x1, y1 (top-left)
#                   └── x2, y2 (bottom-right)
#
#   Three interchangeable box formats, and you can convert between any of
#   them:
#       XYXY  <-->  XYWH  <-->  YOLO normalized
#
#   And: resizing an image means you MUST rescale every box that goes with
#   it, using the same width/height scale factors.


# %% ------------------------------------------------------------------
# DAY 1 CHECKPOINT — do these WITHOUT looking at the code above
# ------------------------------------------------------------------
"""
Do not move to Day 2 until you can answer/do all five confidently.
Try each one on paper or in a fresh Python shell — no peeking.

Test 1 — Given:
    image_width = 1920
    image_height = 1080
    box = [480, 270, 960, 810]
Calculate by hand: center_x, center_y, width, height.

Test 2 — Convert that same box into YOLO normalized coordinates.
    (Hint: divide center_x/width by image_width, center_y/height by
    image_height.)

Test 3 — The image gets resized from 1920x1080 down to 640x360.
    Correctly transform the box to match. What are the new pixel
    coordinates?

Test 4 — Using `draw_box()`, draw the transformed box from Test 3 on a
    640x360 image and confirm it visually looks right (roughly centered,
    reasonable size).

Test 5 — In your own words (say it out loud or write one sentence): WHY
    must a bounding box be transformed when the image is resized? What
    would go wrong if you forgot to do this?

If you can do all 5 without hesitating, Day 1 is done. Move to Day 2.
"""

if __name__ == "__main__":
    print("\nDay 1 script finished. Check the ../assets/ folder for saved images.")
