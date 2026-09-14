"""
DAY 6 — YOLO Architecture: Backbone, Neck, Head
===================================================

RECAP: Day 5 you trained a tiny CNN and saw that images shrink (28->14->7)
while gaining more channels/abstraction as they pass through convolution +
pooling layers. YOLO is built from the exact same LEGO bricks, just many
more of them, arranged in a specific 3-part structure. Today you learn that
structure and SEE it by peeking inside a real, pretrained YOLO model.

THE THREE PARTS
-----------------

                    YOLO
                     |
           +---------+---------+
           v                   v
       Backbone               Neck
           |                   |
           +---------+---------+
                     v
                    Head
                     |
                     v
           boxes + classes + scores

BACKBONE — "what patterns are in this image, at every scale?"
  A stack of convolution layers (like your TinyCNN, but deeper) that
  repeatedly downsamples the image (e.g. 640x640 -> 320x320 -> 160x160 ->
  80x80 -> 40x40 -> 20x20), producing feature maps at EACH of those sizes.
  Early/big feature maps capture fine detail (edges, small objects). Late/
  small feature maps capture coarse, abstract, whole-object information
  (but each "pixel" in that tiny grid represents a big chunk of the
  original image).

NECK — "let information flow between scales"
  A small object (like a badminton shuttle) is best detected using a
  BIG, detailed feature map (lots of pixels = lots of resolution). A big
  object (like a bus) is best detected using a SMALL, abstract feature map
  (each cell already "sees" a huge chunk of the image, so it has enough
  context to recognize the whole object). The neck mixes information
  between the backbone's different-sized feature maps so the network is
  good at BOTH small and large objects, not just one.

HEAD — "make the actual prediction, per feature map cell"
  For each cell in each feature map, the head outputs: "is there an object
  centered near here? Where exactly, and how big? What class is it, and
  how confident are we?" This is why YOLO can output multiple boxes at
  once — it's really asking that question at every grid cell, at every
  scale, all in one forward pass (that's the "You Only Look Once" idea:
  no separate step to first propose regions and then classify them).

Run with:  python day06_yolo_architecture.py
"""

import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


# %% ------------------------------------------------------------------
# SECTION 1 — Load the real model and look at its layer list
# ------------------------------------------------------------------
model = YOLO("yolo11n.pt")
torch_model = model.model   # the underlying PyTorch nn.Module

# `torch_model.model` is a flat list of all the building-block layers, in
# the order data flows through them. Let's print a SUMMARY (not every
# detail) so it doesn't overwhelm you: index, layer type, and how it
# changes the number of channels.
print(f"YOLO has {len(torch_model.model)} top-level layers/blocks.\n")
print(f"{'idx':>4}  {'type':<28}  params")
print("-" * 50)
for i, layer in enumerate(torch_model.model):
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"{i:>4}  {type(layer).__name__:<28}  {n_params:,}")


# %% ------------------------------------------------------------------
# SECTION 2 — Identify roughly where backbone ends and neck/head begin
# ------------------------------------------------------------------
# In Ultralytics' YOLO implementations, this structure is defined in a
# YAML config. As a rule of thumb (exact index varies by version):
#   - The FIRST ~half of the layers = BACKBONE (steadily downsampling,
#     channel count growing: e.g. 16 -> 32 -> 64 -> 128 -> 256).
#   - Layers with "Concat" (concatenate) that combine an early, high-res
#     feature map with a later, low-res one (after upsampling) = the NECK.
#   - The FINAL layer, usually called "Detect", = the HEAD. It's the one
#     that actually outputs box/class/confidence predictions.

last_layer = torch_model.model[-1]
print(f"\nFinal layer (the HEAD) is of type: {type(last_layer).__name__}")
print("This is the layer that converts feature maps into box/class/confidence predictions.")

concat_layers = [i for i, layer in enumerate(torch_model.model)
                  if type(layer).__name__ == "Concat"]
print(f"\n'Concat' layers (part of the NECK, merging different scales): indices {concat_layers}")


# %% ------------------------------------------------------------------
# SECTION 3 — Watch the image shrink through the backbone (with hooks)
# ------------------------------------------------------------------
# A "forward hook" lets us grab the OUTPUT of any layer during a real
# forward pass, without modifying the model. We'll record the shape of the
# feature map after each of the first several layers, and visualize a few
# feature maps — exactly like Day 5, but on a real trained network.

feature_shapes = []
saved_features = {}

def make_hook(index):
    def hook(module, input, output):
        feature_shapes.append((index, type(module).__name__, tuple(output.shape)))
        saved_features[index] = output.detach()
    return hook

hooks = []
NUM_LAYERS_TO_WATCH = 10   # backbone is roughly the first several layers
for i in range(min(NUM_LAYERS_TO_WATCH, len(torch_model.model))):
    h = torch_model.model[i].register_forward_hook(make_hook(i))
    hooks.append(h)

# Run one real forward pass using a demo image (reuse Day 3's bus photo).
demo_image_path = os.path.join(ASSETS_DIR, "day03_bus.jpg")
if not os.path.exists(demo_image_path):
    import urllib.request
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", demo_image_path)

_ = model(demo_image_path, verbose=False)   # triggers the hooks

for h in hooks:
    h.remove()

print("\nFeature map shape after each of the first layers (shape = [batch, channels, height, width]):")
for idx, layer_type, shape in feature_shapes:
    print(f"  layer {idx:>2} ({layer_type:<10}) -> {shape}")

print("""
Notice: height/width SHRINK as you go deeper (downsampling, same idea as
Day 5's MaxPool), while the channel count GROWS (more learned pattern
detectors per location). This is the backbone doing its job: trading
spatial resolution for richer per-location understanding.
""")


# %% ------------------------------------------------------------------
# SECTION 4 — Visualize a few real feature maps
# ------------------------------------------------------------------
# Pick one of the early hooked layers and show a handful of its channels,
# same technique as Day 5.

early_idx = min(saved_features.keys())
fmap = saved_features[early_idx][0]   # drop batch dimension -> [C, H, W]
n_channels_to_show = min(8, fmap.shape[0])

fig, axes = plt.subplots(1, n_channels_to_show, figsize=(2.2 * n_channels_to_show, 3))
for i, ax in enumerate(axes if n_channels_to_show > 1 else [axes]):
    ax.imshow(fmap[i].cpu(), cmap="viridis")
    ax.set_title(f"ch {i}", fontsize=8)
    ax.axis("off")
fig.suptitle(f"Real YOLO feature maps, layer {early_idx}, shape {tuple(fmap.shape)}")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "day06_real_feature_maps.png"))
plt.close()
print("Saved: day06_real_feature_maps.png")


# %% ------------------------------------------------------------------
# SECTION 5 — Why does the NECK matter? A thought experiment
# ------------------------------------------------------------------
print("""
Thought experiment (no code needed, just reasoning):

Imagine the backbone's LAST feature map is 20x20 cells, covering a 640x640
input image. Each cell therefore "represents" a 32x32 pixel patch of the
original image (640/20 = 32).

  - A bus taking up 400x300 pixels of the image: easily fits within a
    handful of 32x32-pixel cells worth of context. The 20x20 map has
    PLENTY of context per cell to recognize "bus". Good.

  - A small ball taking up 10x10 pixels: it's smaller than a SINGLE cell's
    32x32 receptive area. By the time information reaches the 20x20 map,
    the ball's signal has been blended/diluted with a lot of surrounding
    background. It might vanish entirely.

  This is exactly why YOLO also predicts from EARLIER, higher-resolution
  feature maps (e.g. 80x80, where each cell = 8x8 pixels) for small
  objects. The NECK's job is to make sure those higher-res maps ALSO get
  the benefit of the deep, abstract understanding from later layers (via
  upsampling + concatenation), not just raw shallow features.

  This directly foreshadows Day 12 (small objects) — keep this thought
  experiment in mind.
""")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
"""
                    YOLO
                     |
           +---------+---------+
           v                   v
       Backbone               Neck
   (downsample repeatedly,   (mix feature maps from
    extract features at      different depths/scales
    many scales)              so every scale has both
                               detail AND context)
           +---------+---------+
                     v
                    Head
        (per grid cell, per scale: predict
         box offset, objectness, class scores)
                     |
                     v
           boxes + classes + scores
                     |
                     v
                    NMS  (Day 2!)
                     |
                     v
              final clean detections
"""


# %% ------------------------------------------------------------------
# DAY 6 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — In one sentence each, what does the backbone do, what does the
  neck do, and what does the head do?

Test 2 — Why does a small object become harder to detect the deeper
  (further downsampled) a feature map is?

Test 3 — Looking at the `feature_shapes` printout, pick two consecutive
  layers where height/width shrank. What roughly happened to the channel
  count at the same time, and why does that trade-off make sense?

Test 4 — What is a "Concat" layer doing, conceptually, in the neck?

Test 5 — Connect this to Day 2: after the head produces MANY candidate
  boxes across all grid cells and scales, what step cleans that up into
  final detections?

If you can do all 5, Day 6 is done. Move to Day 7.
"""

if __name__ == "__main__":
    print("\nDay 6 script finished.")
