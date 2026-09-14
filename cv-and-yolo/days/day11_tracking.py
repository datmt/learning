"""
DAY 11 — Tracking: "Is This the Same Object I Saw Before?"
================================================================

RECAP: Day 10 you ran YOLO on every frame of a video independently. Each
frame's detections have NO memory of previous frames — if a "player" is
detected in frame 1 and frame 2, YOLO has no idea it's the SAME player.
That's a completely separate problem called TRACKING, and it's today's
topic.

DETECTION vs TRACKING
------------------------
  Detection: "What objects are in THIS frame?"          (Day 3-9)
  Tracking:  "Is this the same object I saw 20 frames ago?" (today)

THE CORE IDEA: DETECTION + ASSOCIATION = TRACKS
----------------------------------------------------
    YOLO
      |
      v
   detections (per frame, no identity)
      |
      v
   association   <- today's new piece
      |
      v
   tracks (each with a persistent ID across frames)

"Association" means: given the boxes I see THIS frame, and the boxes (with
IDs) I was tracking LAST frame, figure out which new box corresponds to
which old track (or whether it's a brand new object, or whether an old
track has disappeared).

THE SIMPLEST POSSIBLE ASSOCIATION RULE: IoU MATCHING
----------------------------------------------------------
If frames come fast enough (e.g. 30fps), an object barely moves between
two consecutive frames. So: a box in frame N and a box in frame N+1 that
overlap a LOT (high IoU — remember Day 2!) are very likely the same
physical object. This simple idea is the seed of real trackers like
ByteTrack/SORT (which add more sophistication: predicting motion with a
Kalman filter, handling brief occlusions, using appearance features — but
IoU matching is the core you're implementing today).

Run with:  python day11_tracking.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
VIDEO_PATH = os.path.join(ASSETS_DIR, "day10_demo.mp4")

if not os.path.exists(VIDEO_PATH):
    raise SystemExit("Run day10_video.py first to create the demo video.")


# %% ------------------------------------------------------------------
# SECTION 1 — Reuse Day 2's IoU
# ------------------------------------------------------------------
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# %% ------------------------------------------------------------------
# SECTION 2 — A simplified IoU-based tracker (ByteTrack-style, minimal)
# ------------------------------------------------------------------
# Track lifecycle vocabulary (used by every real tracker):
#   NEW      -> a detection didn't match any existing track -> start a
#               brand new track with a fresh ID.
#   MATCHED  -> a detection matched an existing track (highest IoU, above
#               a threshold) -> update that track's box, keep its ID.
#   LOST     -> an existing track had no matching detection this frame
#               (object briefly occluded, or a missed detection). We keep
#               it alive for a few frames ("max_age") in case it reappears,
#               rather than immediately deleting the ID.
#   REMOVED  -> a track has been LOST for too many frames in a row -> we
#               give up and delete it.

class Track:
    _next_id = 0

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.age_since_seen = 0   # frames since last matched to a detection
        self.history = [box]      # every box this track has had (for trajectory viz)

    def update(self, box):
        self.box = box
        self.age_since_seen = 0
        self.history.append(box)

    def mark_missed(self):
        self.age_since_seen += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_age = max_age   # how many missed frames before deleting a track

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2] for the current frame."""
        unmatched_detections = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy matching: for each existing track, find its best-IoU
        # detection this frame (if above threshold and not already taken).
        for track in self.tracks:
            best_iou, best_det_idx = 0.0, -1
            for det_idx in unmatched_detections:
                current_iou = iou(track.box, detections[det_idx])
                if current_iou > best_iou:
                    best_iou, best_det_idx = current_iou, det_idx

            if best_iou >= self.iou_threshold:
                track.update(detections[best_det_idx])
                unmatched_detections.remove(best_det_idx)
                matched_track_ids.add(track.id)
            else:
                track.mark_missed()

        # Any detections nobody claimed become brand new tracks.
        for det_idx in unmatched_detections:
            self.tracks.append(Track(detections[det_idx]))

        # Remove tracks that have been missing too long.
        self.tracks = [t for t in self.tracks if t.age_since_seen <= self.max_age]

        return self.tracks


# %% ------------------------------------------------------------------
# SECTION 3 — "Detect" the synthetic ball by color (stand-in for YOLO)
# ------------------------------------------------------------------
# Day 10's demo video has a white ball on a green court — simple enough to
# detect with basic color thresholding. This keeps today's focus 100% on
# TRACKING logic rather than re-running a neural network. (In a real
# pipeline, `detections` below would just be `result.boxes.xyxy` from
# YOLO — the tracker code doesn't care where detections came from.)

def detect_white_blobs(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 20:   # ignore tiny noise specks
            boxes.append([x, y, x + w, y + h])
    return boxes


# %% ------------------------------------------------------------------
# SECTION 4 — Run detection + tracking across the whole video
# ------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracker = SimpleTracker(iou_threshold=0.2, max_age=5)

output_path = os.path.join(ASSETS_DIR, "day11_tracked.mp4")
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

colors = {}   # a stable random color per track ID, so it's visually easy to follow
rng = np.random.RandomState(0)

def color_for_id(track_id):
    if track_id not in colors:
        colors[track_id] = tuple(int(v) for v in rng.randint(60, 255, size=3))
    return colors[track_id]

frame_idx = 0
id_history_per_frame = []   # to plot ID stability later

while True:
    success, frame = cap.read()
    if not success:
        break

    detections = detect_white_blobs(frame)
    tracks = tracker.update(detections)

    for track in tracks:
        if track.age_since_seen > 0:
            continue   # don't draw tracks that weren't matched this frame
        x1, y1, x2, y2 = [int(v) for v in track.box]
        color = color_for_id(track.id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track.id}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    active_ids = sorted(t.id for t in tracks if t.age_since_seen == 0)
    id_history_per_frame.append(active_ids)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"Processed {frame_idx} frames.")
print(f"Saved tracked video: {output_path}")


# %% ------------------------------------------------------------------
# SECTION 5 — Check ID stability
# ------------------------------------------------------------------
# In a perfect world, our single ball keeps the SAME ID for the whole
# video. Let's check.

all_ids_seen = sorted(set(i for frame_ids in id_history_per_frame for i in frame_ids))
print(f"\nDistinct track IDs used across the whole video: {all_ids_seen}")
if len(all_ids_seen) == 1:
    print("The ball kept a single, stable ID the entire time. Tracking worked well.")
else:
    print(f"The ball's ID changed {len(all_ids_seen)-1} time(s)! This is a REAL failure")
    print("mode, not a bug in the demo -- our synthetic ball moves ~11px/frame, but is")
    print("only ~17px wide, so consecutive boxes barely overlap (~0.18 IoU), which is")
    print("BELOW our iou_threshold=0.2. Every frame looks like a 'new object' to the")
    print("tracker. This is exactly why real trackers use motion PREDICTION (Kalman")
    print("filtering) instead of assuming the object hasn't moved.")
    print("Try lowering iou_threshold to 0.1 in Section 4 and re-run -- ID count drops")
    print("dramatically (from ~83 down to about 2), because 0.18 > 0.1 now counts as a")
    print("match. It's still not a perfect single ID -- proving that a real tracker")
    print("needs motion prediction, not just a smaller threshold, to fully solve this.")

plt.figure(figsize=(10, 3))
for frame_num, ids in enumerate(id_history_per_frame):
    for i in ids:
        plt.scatter(frame_num, i, c="tab:blue", s=10)
plt.xlabel("frame")
plt.ylabel("track ID")
plt.title("Track ID over time (should ideally be a flat, unbroken line)")
plt.yticks(all_ids_seen)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(ASSETS_DIR, "day11_id_stability.png"))
plt.close()
print("Saved: day11_id_stability.png")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
print("""
per-frame detections (no identity, from YOLO or any detector)
   |
   v
association: match this frame's boxes to last frame's tracks via IoU
   |
   +-- matched   -> update existing track, keep its ID
   +-- unmatched detection -> start a brand NEW track
   +-- unmatched track     -> mark "missed" (LOST), delete if missed too long
   |
   v
tracks, each with a persistent ID across many frames

Real trackers (ByteTrack, SORT, DeepSORT) extend this exact skeleton with:
  - Kalman filtering: PREDICT where a track should be next frame based on
    its recent motion, instead of assuming it hasn't moved -- improves
    matching for fast-moving objects.
  - Appearance features: use what the object LOOKS like (not just
    position) to re-identify it after a longer occlusion.
  - Two-stage matching: match high-confidence detections first, then try
    to rescue low-confidence ones against remaining tracks (this is
    literally what "ByteTrack" is named for -- it doesn't throw away
    low-score boxes immediately).
""")


# %% ------------------------------------------------------------------
# DAY 11 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — In your own words: what problem does tracking solve that
  detection alone cannot?

Test 2 — Walk through, step by step, what SimpleTracker.update() does
  when: (a) a detection matches an existing track, (b) a detection
  matches nothing, (c) an existing track matches nothing.

Test 3 — What does `max_age` control, and what trade-off does raising it
  introduce? (Hint: think about what happens if TWO different objects
  cross paths while a track is "lost".)

Test 4 — Why is IoU a reasonable way to match boxes between consecutive
  VIDEO frames, when it wouldn't make sense to match boxes between two
  totally unrelated images?

Test 5 — Name one thing a real tracker (like ByteTrack) does that this
  simplified version does not, and explain why it would help.

If you can do all 5, Day 11 is done. Move to Day 12.
"""

if __name__ == "__main__":
    print("\nDay 11 script finished.")
