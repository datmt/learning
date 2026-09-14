"""
DAY 10 — Running YOLO on Video
==================================

RECAP: Days 1-9 all worked on single STILL images. Today you add the
dimension that matters for sports analytics: TIME. A video is just a
sequence of images (called "frames") played quickly one after another.

THE BIG IDEA
--------------
    video = frame 1, frame 2, frame 3, ... (e.g. 30 frames per second)

There is NO new detection theory today — you already know how to run YOLO
on ONE image (Day 3/4). Today's real content is:

  1. How to read a video frame-by-frame with OpenCV.
  2. Why running YOLO on every single frame in real time is a genuinely
     hard SYSTEMS problem (this is where your backend/systems experience
     becomes directly useful — it's a throughput/latency problem, not a
     computer vision problem).
  3. How to measure whether your pipeline is actually fast enough.

KEY VOCABULARY
-----------------
  FPS (frames per second) -> how many frames the VIDEO plays per second
     (a property of the video file itself, e.g. 30 fps).
  Inference time -> how long the MODEL takes to process one frame
     (a property of your model + hardware, e.g. 25 milliseconds).
  Processing FPS -> 1000 / inference_time_ms -> how many frames per
     second your PIPELINE can keep up with. If this is LOWER than the
     video's own FPS, you cannot process the video in real time — you'll
     either have to skip frames, use a smaller/faster model, or accept
     that you're processing it slower than real time (fine for offline
     analysis, not fine for a live broadcast overlay).

Run with:  python day10_video.py
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
VIDEO_PATH = os.path.join(ASSETS_DIR, "day10_demo.mp4")


# %% ------------------------------------------------------------------
# SECTION 1 — Get a test video (synthesize one if you don't have one)
# ------------------------------------------------------------------
# Real footage is best, but to guarantee this script runs standalone, we
# synthesize a short video: a "ball" bouncing around a green "court",
# reusing the same synthetic-drawing approach from Day 7.

def make_demo_video(path, n_frames=90, width=640, height=480, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    ball_pos = np.array([100.0, 100.0])
    ball_vel = np.array([9.0, 6.0])
    player_pos = np.array([300.0, 240.0])

    for i in range(n_frames):
        frame = np.full((height, width, 3), (30, 90, 30), dtype=np.uint8)

        # bounce the "ball" off the walls
        ball_pos += ball_vel
        for axis, limit in [(0, width), (1, height)]:
            if ball_pos[axis] <= 8 or ball_pos[axis] >= limit - 8:
                ball_vel[axis] *= -1

        cv2.circle(frame, tuple(ball_pos.astype(int)), 8, (255, 255, 255), -1)
        cv2.rectangle(
            frame,
            (int(player_pos[0] - 20), int(player_pos[1] - 40)),
            (int(player_pos[0] + 20), int(player_pos[1] + 40)),
            (200, 140, 90), -1,
        )
        writer.write(frame)

    writer.release()


if not os.path.exists(VIDEO_PATH):
    make_demo_video(VIDEO_PATH)
    print(f"Created demo video: {VIDEO_PATH}")
else:
    print(f"Using existing video: {VIDEO_PATH}")
    print("(Replace this file with real footage any time to try it on real data.)")


# %% ------------------------------------------------------------------
# SECTION 2 — Read a video frame-by-frame
# ------------------------------------------------------------------
# cv2.VideoCapture opens a video file (or a webcam, if you pass 0 instead
# of a path!). .read() gives you ONE frame at a time, like pulling one
# image off a conveyor belt. It returns (success, frame) — success is
# False once the video runs out.

cap = cv2.VideoCapture(VIDEO_PATH)

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\nVideo info: {frame_width}x{frame_height}, {video_fps:.1f} fps, {frame_count} frames "
      f"(~{frame_count/video_fps:.1f} seconds)")

success, first_frame = cap.read()
print("Read first frame successfully:", success, "shape:", first_frame.shape)
cap.release()   # always release when done (like closing a file handle)


# %% ------------------------------------------------------------------
# SECTION 3 — Run YOLO on every frame + measure speed
# ------------------------------------------------------------------
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

output_path = os.path.join(ASSETS_DIR, "day10_annotated.mp4")
writer = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (frame_width, frame_height)
)

frame_times_ms = []
frame_idx = 0

while True:
    success, frame = cap.read()
    if not success:
        break   # ran out of frames -- this is the normal way a video loop ends

    start = time.perf_counter()
    results = model(frame, verbose=False)   # note: YOLO/OpenCV both use BGR, no conversion needed here
    elapsed_ms = (time.perf_counter() - start) * 1000
    frame_times_ms.append(elapsed_ms)

    annotated_frame = results[0].plot()   # BGR, ready to write straight to video
    writer.write(annotated_frame)

    frame_idx += 1

cap.release()
writer.release()

print(f"\nProcessed {frame_idx} frames.")
print(f"Saved annotated video: {output_path}")


# %% ------------------------------------------------------------------
# SECTION 4 — Analyze speed: can this keep up with real time?
# ------------------------------------------------------------------
frame_times_ms = np.array(frame_times_ms)
avg_ms = frame_times_ms.mean()
processing_fps = 1000 / avg_ms

print(f"\nAverage inference time per frame: {avg_ms:.1f} ms")
print(f"That means processing FPS ≈ {processing_fps:.1f}")
print(f"The video itself plays at:        {video_fps:.1f} fps")

if processing_fps >= video_fps:
    print("-> Your pipeline is FASTER than real time. Could process a live stream.")
else:
    print(f"-> Your pipeline is SLOWER than real time (by {video_fps/processing_fps:.1f}x).")
    print("   Options: use a smaller model (yolo11n is already the smallest),")
    print("   use a GPU, lower imgsz, skip frames, or accept offline processing.")

plt.figure(figsize=(9, 4))
plt.plot(frame_times_ms)
plt.axhline(1000 / video_fps, color="red", linestyle="--",
            label=f"budget for real-time ({1000/video_fps:.1f} ms/frame @ {video_fps:.0f}fps)")
plt.xlabel("frame index")
plt.ylabel("inference time (ms)")
plt.title("Per-frame inference time")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(ASSETS_DIR, "day10_frame_times.png"))
plt.close()
print("Saved: day10_frame_times.png")


# %% ------------------------------------------------------------------
# SECTION 5 — A key systems technique: frame skipping
# ------------------------------------------------------------------
# If your model can't keep up, one classic technique (very much a
# "systems engineer" move) is to only run the heavy model every Nth frame,
# and reuse/interpolate results in between. Let's demonstrate the idea:

def process_with_frame_skip(video_path, skip_n=2):
    cap = cv2.VideoCapture(video_path)
    processed = 0
    total = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        total += 1
        if total % skip_n == 0:
            _ = model(frame, verbose=False)
            processed += 1
    cap.release()
    return processed, total


processed, total = process_with_frame_skip(VIDEO_PATH, skip_n=3)
print(f"\nWith skip_n=3: ran the model on {processed}/{total} frames "
      f"({processed/total*100:.0f}%) -> ~3x throughput, at the cost of "
      f"lower temporal resolution (we don't know what happened on skipped frames).")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
print("""
video file
   |
   v
VideoCapture.read() in a loop -> one frame at a time
   |
   v
YOLO(frame) -> boxes  (identical to Day 3/4, just repeated many times)
   |
   v
draw + write to output video
   |
   v
measure: inference time per frame -> compare to video's own fps
   |
   v
if too slow: smaller model / GPU / lower resolution / skip frames
""")


# %% ------------------------------------------------------------------
# DAY 10 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — What does cap.read() return, and how do you know when a video
  has ended?

Test 2 — If a video is 30 fps and your model takes 50ms per frame, is
  your pipeline fast enough for real time? Show the math.

Test 3 — Name two concrete ways to speed up a too-slow video pipeline.

Test 4 — What's the trade-off with frame skipping? What information do
  you lose?

Test 5 — Why is measuring "inference time" alone not the WHOLE story for
  a real-time system? (Hint: think about what else happens in the loop —
  reading the frame, drawing, writing output — that also takes time.)

If you can do all 5, Day 10 is done. Move to Day 11.
"""

if __name__ == "__main__":
    print("\nDay 10 script finished.")
