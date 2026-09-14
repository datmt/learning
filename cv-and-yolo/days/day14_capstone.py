"""
DAY 14 — Capstone: A Small Sports Analytics Pipeline
=========================================================

RECAP: You now know every piece:
  Day 1  -> boxes & coordinate systems
  Day 2  -> IoU & NMS
  Day 3-4 -> running YOLO & reading its output
  Day 5-6 -> what's happening inside the network
  Day 7-9 -> building a dataset, training, evaluating
  Day 10  -> running detection over video, measuring speed
  Day 11  -> tracking (giving objects persistent IDs across frames)
  Day 12  -> handling small objects
  Day 13  -> converting pixel positions to real-world court coordinates

Today you wire ALL of it into one pipeline:

                  VIDEO
                    |
                    v
                 DETECT (per frame)
                    |
             +------+------+
             v             v
         players          ball
             |             |
             +------+------+
                    v
                 TRACK (persistent IDs, Day 11)
                    |
                    v
             COURT GEOMETRY (Day 13: pixel -> meters)
                    |
                    v
                ANALYTICS
                    |
        +-----------+-----------+
        v           v           v
     heatmap   trajectories   speed

WHY SYNTHETIC DATA AGAIN
----------------------------
Real sports footage + a model actually trained to detect "player"/"ball"
is a full separate project (weeks, not a capstone script). To make sure
this file is something you can run RIGHT NOW and see a full pipeline work
end-to-end, we generate a synthetic video with a known-ground-truth moving
player and bouncing ball, and detect them with simple, fast color-based
detection (same technique as Day 11) instead of a trained YOLO model.

THIS IS THE KEY LESSON OF THE CAPSTONE: the detector is a SWAPPABLE
component. Every other piece of this pipeline (tracking, geometry,
analytics, rendering) works completely unchanged whether the boxes come
from color thresholding, a toy YOLO model like Day 8's, or a
production-grade fine-tuned one. If you later train a real player/ball
YOLO model (Day 7-9 for real footage), you'd swap ONLY the `detect()`
function below and keep everything downstream identical.

Run with:  python day14_capstone.py
"""

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
VIDEO_PATH = os.path.join(ASSETS_DIR, "day14_capstone_demo.mp4")

WIDTH, HEIGHT, FPS, N_FRAMES = 960, 540, 30, 150


# %% ------------------------------------------------------------------
# SECTION 1 — Generate the demo video: a player walking + a bouncing ball
# ------------------------------------------------------------------
def make_capstone_video(path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    ball_pos = np.array([100.0, 100.0])
    ball_vel = np.array([6.0, 4.5])

    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), (30, 110, 30), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (WIDTH - 1, HEIGHT - 1), (255, 255, 255), 2)  # court outline

        # Player walks in a slow zig-zag path across the court.
        t = i / N_FRAMES
        player_x = 100 + t * (WIDTH - 200)
        player_y = HEIGHT / 2 + 100 * np.sin(t * 3 * np.pi)
        px1, py1 = int(player_x - 18), int(player_y - 40)
        px2, py2 = int(player_x + 18), int(player_y + 40)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (90, 140, 200), -1)

        # Ball bounces around, faster and smaller than the player.
        ball_pos += ball_vel
        for axis, limit in [(0, WIDTH), (1, HEIGHT)]:
            if ball_pos[axis] <= 8 or ball_pos[axis] >= limit - 8:
                ball_vel[axis] *= -1
        cv2.circle(frame, tuple(ball_pos.astype(int)), 7, (255, 255, 255), -1)

        writer.write(frame)

    writer.release()


if not os.path.exists(VIDEO_PATH):
    make_capstone_video(VIDEO_PATH)
print(f"Using video: {VIDEO_PATH}")


# %% ------------------------------------------------------------------
# SECTION 2 — DETECT: the swappable component
# ------------------------------------------------------------------
# In a real system, this function's BODY would be:
#     results = model(frame, verbose=False)[0]
#     return [(box, "player" if cls==0 else "ball") for box, cls in ...]
# using a YOLO model fine-tuned like Day 8. Here, color thresholds stand
# in, but the FUNCTION SIGNATURE (frame in, list of (box, class_name) out)
# is exactly what a real detector would also provide -- that's the
# contract that keeps everything below unaffected by the swap.

def detect(frame_bgr):
    detections = []

    # NOTE: we use RETR_LIST (not RETR_EXTERNAL) below. Our frame has a
    # hollow white court-outline rectangle drawn on it, and separately, a
    # white ball blob floating inside it. RETR_EXTERNAL's "outermost
    # contours only" rule can drop a blob that sits inside another
    # shape's hole, even when the blob itself doesn't touch that shape --
    # a real gotcha worth knowing. RETR_LIST returns every contour and we
    # filter by size ourselves instead, which is simpler and reliable here.

    # "player" = the orange-ish rectangle (BGR (90,140,200) ~= HSV hue 14)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    player_mask = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
    contours, _ = cv2.findContours(player_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 200:
            detections.append(([x, y, x + w, y + h], "player"))

    # "ball" = bright white blob
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, ball_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < w * h < 400:
            detections.append(([x, y, x + w, y + h], "ball"))

    return detections


# %% ------------------------------------------------------------------
# SECTION 3 — TRACK: reuse Day 11's IoU tracker (one instance per class)
# ------------------------------------------------------------------
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next_id = 0

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.age_since_seen = 0

    def update(self, box):
        self.box = box
        self.age_since_seen = 0

    def mark_missed(self):
        self.age_since_seen += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.1, max_age=8):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, boxes):
        unmatched = list(range(len(boxes)))
        for track in self.tracks:
            best_iou, best_idx = 0.0, -1
            for i in unmatched:
                v = iou(track.box, boxes[i])
                if v > best_iou:
                    best_iou, best_idx = v, i
            if best_iou >= self.iou_threshold:
                track.update(boxes[best_idx])
                unmatched.remove(best_idx)
            else:
                track.mark_missed()
        for i in unmatched:
            self.tracks.append(Track(boxes[i]))
        self.tracks = [t for t in self.tracks if t.age_since_seen <= self.max_age]
        return [t for t in self.tracks if t.age_since_seen == 0]


trackers = {"player": SimpleTracker(iou_threshold=0.2, max_age=8),
            "ball": SimpleTracker(iou_threshold=0.05, max_age=8)}


# %% ------------------------------------------------------------------
# SECTION 4 — GEOMETRY: pixel -> court meters (Day 13)
# ------------------------------------------------------------------
# Calibrate against our known synthetic court: the video frame IS the
# court, edge to edge (we drew a white outline for exactly this reason).
# A real project would calibrate this once, manually, against real court
# markings visible in the camera's actual footage.

COURT_LENGTH_M, COURT_WIDTH_M = 20.0, 11.0   # arbitrary units for this demo court

image_corners = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype=np.float32)
court_corners = np.array([
    [0, 0], [COURT_LENGTH_M, 0], [COURT_LENGTH_M, COURT_WIDTH_M], [0, COURT_WIDTH_M]
], dtype=np.float32)
H, _ = cv2.findHomography(image_corners, court_corners)


def pixel_to_court(xy, H):
    pt = np.array([[xy]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0, 0]


def foot_point(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, y2])


# %% ------------------------------------------------------------------
# SECTION 5 — RUN THE FULL PIPELINE OVER THE VIDEO
# ------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
output_path = os.path.join(ASSETS_DIR, "day14_final_output.mp4")
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

# Analytics accumulators
court_trajectories = {"player": {}, "ball": {}}   # track_id -> list of (X, Y) meters
frame_times_ms = []
frame_idx = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    start = time.perf_counter()
    detections = detect(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000
    frame_times_ms.append(elapsed_ms)

    for class_name, tracker in trackers.items():
        boxes = [box for box, cname in detections if cname == class_name]
        active_tracks = tracker.update(boxes)

        for track in active_tracks:
            court_xy = pixel_to_court(foot_point(track.box), H)
            court_trajectories[class_name].setdefault(track.id, []).append(tuple(court_xy))

            x1, y1, x2, y2 = [int(v) for v in track.box]
            color = (0, 165, 255) if class_name == "player" else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}#{track.id} ({court_xy[0]:.1f},{court_xy[1]:.1f})m",
                        (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
avg_ms = np.mean(frame_times_ms)
print(f"\nProcessed {frame_idx} frames.")
print(f"Avg detect() time/frame: {avg_ms:.2f} ms  -> ~{1000/avg_ms:.0f} fps pipeline throughput")
print(f"Saved annotated video: {output_path}")


# %% ------------------------------------------------------------------
# SECTION 6 — ANALYTICS: trajectories, heatmap, speed
# ------------------------------------------------------------------
def track_speed_kmh(points, fps):
    """Average speed in km/h from a list of (X, Y) meter positions."""
    if len(points) < 2:
        return 0.0
    points = np.array(points)
    diffs = np.diff(points, axis=0)                 # meters moved per frame
    dists_m = np.linalg.norm(diffs, axis=1)          # distance per frame step
    total_distance_m = dists_m.sum()
    total_time_s = (len(points) - 1) / fps
    speed_mps = total_distance_m / total_time_s
    return speed_mps * 3.6   # m/s -> km/h


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Trajectories ---
ax = axes[0]
for class_name, tracks in court_trajectories.items():
    for track_id, points in tracks.items():
        if len(points) < 3:
            continue
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], marker="o", markersize=2,
                label=f"{class_name}#{track_id}")
ax.set_title("Trajectories (court meters)")
ax.set_xlim(-1, COURT_LENGTH_M + 1)
ax.set_ylim(COURT_WIDTH_M + 1, -1)
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

# --- Heatmap (player positions only -- more meaningful for occupancy) ---
ax = axes[1]
all_player_points = np.array(
    [p for pts in court_trajectories["player"].values() for p in pts]
)
if len(all_player_points) > 0:
    heatmap, xedges, yedges = np.histogram2d(
        all_player_points[:, 0], all_player_points[:, 1],
        bins=[20, 12], range=[[0, COURT_LENGTH_M], [0, COURT_WIDTH_M]],
    )
    ax.imshow(heatmap.T, origin="lower", extent=[0, COURT_LENGTH_M, 0, COURT_WIDTH_M],
              cmap="hot", aspect="auto")
ax.set_title("Player position heatmap")

# --- Speed bar chart ---
ax = axes[2]
labels, speeds = [], []
for class_name, tracks in court_trajectories.items():
    for track_id, points in tracks.items():
        if len(points) < 3:
            continue
        labels.append(f"{class_name}#{track_id}")
        speeds.append(track_speed_kmh(points, FPS))
ax.bar(labels, speeds, color=["orange" if "player" in l else "gray" for l in labels])
ax.set_ylabel("avg speed (km/h)")
ax.set_title("Average speed per track")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "day14_analytics.png"))
plt.close()
print("\nSaved: day14_analytics.png")

for label, speed in zip(labels, speeds):
    print(f"  {label:15s} avg speed = {speed:6.2f} km/h")


# %% ------------------------------------------------------------------
# SECTION 7 — Summary: what this capstone proves you can do
# ------------------------------------------------------------------
print("""
You just ran, end to end:
  video -> per-frame detection -> multi-object tracking (persistent IDs)
        -> pixel-to-court geometry -> trajectories, heatmap, speed
        -> a rendered output video with live annotations
        -> measured pipeline throughput (fps)

The ONLY thing standing between this and a real sports analytics tool is
swapping detect() for a real fine-tuned YOLO model (Day 7-9, on real
labeled footage of your actual sport) and calibrating the homography
(Day 13) against your actual camera/court. Every other stage of this
pipeline -- tracking, geometry, analytics, rendering, performance
measurement -- is exactly what you already built today.
""")


# %% ------------------------------------------------------------------
# DAY 14 CHECKPOINT (final)
# ------------------------------------------------------------------
"""
Test 1 — Draw the full pipeline (video -> ... -> analytics) from memory,
  labeling which Day introduced each stage.

Test 2 — What is the ONE function you'd need to change to plug in a real,
  trained-on-real-footage YOLO model instead of the color-based detector?
  What stays exactly the same?

Test 3 — Why do we use the FOOT point (not box center) when converting a
  detection to court coordinates? (Day 13)

Test 4 — Open day14_analytics.png. Does the heatmap match where you'd
  expect the player to spend time, given the zig-zag walking path in
  Section 1? Does the ball's speed look higher than the player's, and
  does that match what you'd expect physically?

Test 5 — If your pipeline's fps throughput is lower than the source
  video's fps, what are your options? (Day 10 -- name at least 2.)

Test 6 — Pick ONE thing from Days 1-13 that you still feel shaky on.
  Re-run that day's script, re-read its comments, and re-do its
  checkpoint before considering the whole 14-day path complete.

If you can do all 6: you've completed the path. You went from "I don't
know what YOLO is" to "I can take a new detection problem, prepare data,
train YOLO, evaluate it, diagnose failures, run it on video, and add
tracking/geometry" -- exactly the goal stated on Day 1.
"""

if __name__ == "__main__":
    print("\nDay 14 script finished. This is the end of the 14-day path.")
