"""
DAY 5 — CNN Fundamentals (where do YOLO's predictions actually come from?)
=============================================================================

RECAP: Days 1-4 treated YOLO as a black box: image in, boxes out. Today we
open that box partway. You will train a tiny Convolutional Neural Network
(CNN) yourself, on a small/simple task, so you understand the core
mechanism every detector (including YOLO) is built from.

YOU ARE NOT BECOMING A CNN RESEARCHER TODAY. The goal is ONE piece of
intuition:

    "Why does spatial information (WHERE things are) survive as an image
     passes through a neural network, even though the network is learning
     increasingly abstract ('what is this') features?"

THE BIG IDEA, IN PLAIN ENGLISH
--------------------------------
A regular ("fully connected") neural network layer treats every input
number as unrelated to its neighbors — it flattens the image into one long
list, losing all sense of "this pixel is next to that pixel". That's
terrible for images: a cat's ear means nothing without knowing it's
attached to a cat's head nearby.

A CONVOLUTION layer instead slides a small window (e.g. 3x3 pixels) across
the image and asks the SAME small question at every location: "does this
patch look like an edge / a curve / a blob of color?" Because the question
(the "filter"/"kernel") is small and reused everywhere, two things happen:
  1. Far fewer parameters need to be learned (efficient).
  2. The OUTPUT is still a 2D grid — a "feature map" — so you still know
     WHERE each pattern was found. Spatial layout survives.

Stack several convolution layers, and each layer combines the previous
layer's simple patterns (edges) into more complex ones (corners -> shapes
-> object parts -> whole objects) — while remaining a 2D grid the whole
time. That 2D grid of "what's here, and where" is exactly what YOLO's head
(Day 6) reads to output boxes.

Run with:  python day05_cnn_fundamentals.py
(First run downloads the MNIST dataset, ~10MB, one time only.)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# %% ------------------------------------------------------------------
# SECTION 1 — The dataset: MNIST (handwritten digits 0-9)
# ------------------------------------------------------------------
# We use MNIST (not sports images) on purpose: it's tiny, trains in
# seconds/minutes on a CPU, and lets you focus 100% on the CNN mechanism
# instead of waiting around or fighting data-loading issues.

transform = transforms.Compose([transforms.ToTensor()])   # pixels -> [0,1] floats

train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Train images: {len(train_dataset)}, Test images: {len(test_dataset)}")

# Look at one example.
image, label = train_dataset[0]
print("One image tensor shape:", image.shape)  # [1, 28, 28] -> 1 channel (grayscale), 28x28 pixels
print("Its label:", label)

plt.figure()
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.savefig(os.path.join(ASSETS_DIR, "day05_sample_digit.png"))
plt.close()
print("Saved: day05_sample_digit.png")


# %% ------------------------------------------------------------------
# SECTION 2 — Build a tiny CNN
# ------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # A convolution layer: nn.Conv2d(in_channels, out_channels, kernel_size)
        #   in_channels=1  -> grayscale image has 1 channel
        #   out_channels=16 -> learn 16 different 3x3 "pattern detectors"
        #   padding=1 -> keeps the output the same H/W as the input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # MaxPool2d(2): shrink the feature map by half (28x28 -> 14x14) by
        # keeping only the strongest value in every 2x2 block. This is
        # "downsampling" — it makes the network look at a bigger effective
        # area of the original image with each layer, cheaply.
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # after conv1+pool: 14x14, after conv2+pool: 7x7, with 32 channels

        # Finally, flatten the 2D feature map into a 1D vector and use a
        # standard ("fully connected") layer to output 10 numbers — one
        # score per digit class (0-9).
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)          # [B, 16, 28, 28]
        x = F.relu(x)               # ReLU: keep positive values, zero out negatives
        x = self.pool(x)            # [B, 16, 14, 14]

        x = self.conv2(x)          # [B, 32, 14, 14]
        x = F.relu(x)
        x = self.pool(x)            # [B, 32, 7, 7]

        x = torch.flatten(x, 1)     # [B, 32*7*7] -- flatten everything except batch dim
        x = self.classifier(x)      # [B, 10]
        return x


model = TinyCNN().to(device)
print(model)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal learnable parameters: {n_params:,}")


# %% ------------------------------------------------------------------
# SECTION 3 — Train it (a few epochs is enough for MNIST)
# ------------------------------------------------------------------
# "Epoch" = one full pass through the entire training dataset.
# "Loss" = a number measuring how wrong the model's predictions are.
#          Training = repeatedly nudging the model's numbers to make loss
#          go down.

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 3   # MNIST is easy; 3 epochs already gets >97% accuracy

train_losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from last step
        predictions = model(images)    # forward pass
        loss = loss_fn(predictions, labels)
        loss.backward()                # compute gradients (backpropagation)
        optimizer.step()               # nudge the weights

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}  -  loss: {epoch_loss:.4f}")


# %% ------------------------------------------------------------------
# SECTION 4 — Evaluate: did it actually learn?
# ------------------------------------------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():   # no need to track gradients when just evaluating
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        predicted_labels = predictions.argmax(dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\nTest accuracy: {accuracy*100:.2f}%  ({correct}/{total} correct)")


# %% ------------------------------------------------------------------
# SECTION 5 — THE KEY INTUITION: visualize feature maps
# ------------------------------------------------------------------
# Let's look at what conv1 actually "sees" for one real digit. Each of the
# 16 output channels is a 2D grid the same rough shape as the input — this
# IS the "spatial information survives" idea made visible. Some channels
# will look like edge detectors; some may look almost blank (unused).

sample_image, sample_label = test_dataset[0]
sample_batch = sample_image.unsqueeze(0).to(device)  # add batch dimension

with torch.no_grad():
    feature_maps = F.relu(model.conv1(sample_batch))  # [1, 16, 28, 28]

feature_maps = feature_maps.squeeze(0).cpu()  # [16, 28, 28]

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[i], cmap="viridis")
    ax.set_title(f"ch {i}", fontsize=8)
    ax.axis("off")
fig.suptitle(f"16 feature maps from conv1, for input digit = {sample_label}")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "day05_feature_maps.png"))
plt.close()
print("Saved: day05_feature_maps.png  <- look closely, some channels highlight edges/strokes")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
print("""
image (28x28x1)
   |  conv1 (learn 16 simple pattern detectors, e.g. edges/strokes)
   v
feature map (28x28x16)
   |  pool (shrink, keep strongest signal)
   v
feature map (14x14x16)
   |  conv2 (combine simple patterns into more complex ones)
   v
feature map (14x14x32)
   |  pool
   v
feature map (7x7x32)
   |  flatten + linear layer
   v
10 class scores

At every step until the final flatten, the data is still a 2D GRID — so the
network always "knows" roughly where in the image a pattern was found, even
as WHAT it's detecting gets more abstract. This is exactly the property
YOLO's head (Day 6) exploits to output a box position, not just "there's a
dog somewhere".
""")


# %% ------------------------------------------------------------------
# DAY 5 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — In your own words, what does a convolution layer do differently
  from a normal ("fully connected") layer, and why does that matter for
  images specifically?

Test 2 — What does MaxPool2d(2) do to a feature map's height and width?
  What does it do to the number of channels?

Test 3 — Why do we call conv1's output a "feature map" instead of just
  "output"? What are the two spatial dimensions still representing?

Test 4 — If accuracy on the TRAINING set is high but accuracy on the TEST
  set is much lower, what's that phenomenon called, and what does it mean?
  (You may need to look this one up — it's called overfitting.)

Test 5 — Change conv1's out_channels from 16 to 4, retrain, and compare
  test accuracy and the feature map visualization. What changed, and why
  does that make sense given fewer "pattern detectors" are available?

If you can do all 5, Day 5 is done. Move to Day 6.
"""

if __name__ == "__main__":
    print("\nDay 5 script finished.")
