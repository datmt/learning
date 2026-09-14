"""
DAY 13 — Geometry: From Pixel Coordinates to Real-World Court Coordinates
==============================================================================

RECAP: All 12 previous days worked entirely in PIXEL coordinates (Day 1's
(x,y), image-space). Today you leave pure detection and answer a different
question: "given a player's pixel position, where are they ACTUALLY
standing on the court, in meters?"

WHY THIS MATTERS
-------------------
YOLO gives you:
    player center = (843, 527)   <- pixel coordinates, camera-dependent

You want:
    court position = (8.3m, 4.7m)  <- real-world coordinates, camera-independent

Pixel coordinates are USELESS for analytics on their own: "player moved
from pixel (400,300) to (420,310)" tells you nothing about real distance,
because a camera's perspective distorts distances unevenly (objects near
the camera look bigger/move more pixels per real meter than objects far
away). You need to convert into a coordinate system tied to the actual
court, not the camera.

THE TOOL: HOMOGRAPHY
------------------------
A homography is a mathematical transformation (a 3x3 matrix) that maps
points from one flat plane to another flat plane. Crucially: a sports
court IS flat, and the image sensor IS (approximately) flat, so a single
homography matrix can correctly map "pixel position on the court" to
"real-world position on the court" -- AS LONG AS the point is on the
court's ground plane (a player's FEET are on the ground; a player's HEAD
is not, and will map to the wrong court position if you're not careful --
more on this in Section 5).

    IMAGE                       COURT
    (pixel x, pixel y)          (meters X, meters Y)
       o                           o
       |                           |
       |       homography          |
       +---------------------------+

HOW YOU COMPUTE IT
----------------------
You need at least 4 point correspondences: 4 pixel locations in the image
whose real-world court coordinates you ALREADY KNOW (e.g. the 4 corners of
a badminton court, whose real dimensions are a known standard: 13.4m x
6.1m for doubles). OpenCV's `cv2.findHomography` solves for the matrix
from those 4+ pairs. Once you have it, you can map ANY point on the court
plane, not just the 4 corners you started with.

Run with:  python day13_geometry.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


# %% ------------------------------------------------------------------
# SECTION 1 — Define known correspondences: image corners <-> real court
# ------------------------------------------------------------------
# Imagine a camera looking at a badminton court from an angle (not
# straight overhead), so the court appears as a distorted quadrilateral in
# the image, NOT a perfect rectangle -- this perspective distortion is
# exactly what homography corrects for.
#
# A real badminton doubles court: 13.4m (long side) x 6.1m (short side).

COURT_LENGTH_M = 13.4
COURT_WIDTH_M = 6.1

# Real-world court corners, in meters, with (0,0) at one corner.
court_points = np.array([
    [0.0, 0.0],                          # near-left
    [COURT_LENGTH_M, 0.0],                # far-left
    [COURT_LENGTH_M, COURT_WIDTH_M],      # far-right
    [0.0, COURT_WIDTH_M],                 # near-right
], dtype=np.float32)

# Where those SAME 4 corners appear in the camera's image (pixels).
# Notice this is NOT a rectangle -- perspective makes the far side of the
# court look smaller/compressed, which is exactly the distortion we're
# correcting for.
image_points = np.array([
    [150, 600],   # near-left   (close to camera -> low in frame, spread wide)
    [550, 200],   # far-left    (far from camera -> higher in frame, closer together)
    [750, 200],   # far-right
    [1100, 600],  # near-right
], dtype=np.float32)


# %% ------------------------------------------------------------------
# SECTION 2 — Compute the homography matrix
# ------------------------------------------------------------------
# cv2.findHomography(source_points, destination_points) solves for the
# matrix H such that: destination = H @ source (in "homogeneous"
# coordinates -- OpenCV handles that detail for you).

homography_matrix, _ = cv2.findHomography(image_points, court_points)
print("Homography matrix (image pixels -> court meters):")
print(homography_matrix)


# %% ------------------------------------------------------------------
# SECTION 3 — Use it: convert a player's pixel position to court meters
# ------------------------------------------------------------------
def pixel_to_court(pixel_xy, H):
    """Map one (x, y) pixel point to (X, Y) court-meter coordinates."""
    point = np.array([[pixel_xy]], dtype=np.float32)   # shape required by cv2
    court_xy = cv2.perspectiveTransform(point, H)
    return court_xy[0, 0]


# Sanity check: the 4 corners we defined should map back to themselves
# (approximately -- tiny floating point error is fine).
print("\nSanity check -- mapping the 4 known corners should recover the")
print("known court coordinates we started with:")
for img_pt, expected_court_pt in zip(image_points, court_points):
    mapped = pixel_to_court(img_pt, homography_matrix)
    print(f"  pixel {img_pt} -> court {mapped}  (expected {expected_court_pt})")

# Now the interesting case: a player standing somewhere in the MIDDLE of
# the court (not one of our 4 known corners).
player_pixel = np.array([650.0, 350.0])
player_court_position = pixel_to_court(player_pixel, homography_matrix)
print(f"\nPlayer at pixel {player_pixel} -> court position "
      f"({player_court_position[0]:.2f}m, {player_court_position[1]:.2f}m)")


# %% ------------------------------------------------------------------
# SECTION 4 — Visualize both spaces side by side
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: image space (the distorted quadrilateral, as the camera sees it)
ax = axes[0]
poly_img = np.vstack([image_points, image_points[0]])
ax.plot(poly_img[:, 0], poly_img[:, 1], "b-o")
ax.scatter(*player_pixel, c="red", s=100, zorder=5, label="player")
ax.set_title("IMAGE space (pixels, as camera sees it)")
ax.set_xlim(0, 1280)
ax.set_ylim(720, 0)
ax.legend()
ax.grid(alpha=0.3)

# Right: court space (the TRUE rectangle, undistorted)
ax = axes[1]
poly_court = np.vstack([court_points, court_points[0]])
ax.plot(poly_court[:, 0], poly_court[:, 1], "g-o")
ax.scatter(*player_court_position, c="red", s=100, zorder=5, label="player")
ax.set_title("COURT space (real meters, top-down, undistorted)")
ax.set_xlim(-1, COURT_LENGTH_M + 1)
ax.set_ylim(COURT_WIDTH_M + 1, -1)
ax.set_xlabel("meters")
ax.set_ylabel("meters")
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "day13_image_vs_court.png"))
plt.close()
print("\nSaved: day13_image_vs_court.png")


# %% ------------------------------------------------------------------
# SECTION 5 — The classic gotcha: use the FEET, not the box center
# ------------------------------------------------------------------
# A homography is only valid for points that lie ON the plane you
# calibrated against (the court's ground plane). A player's BOUNDING BOX
# CENTER is roughly at their torso height, floating well ABOVE the ground
# plane -- mapping that point through the homography gives a WRONG court
# position (it'll look like the player is further from the camera than
# they really are, because of the same perspective effect we're trying to
# correct for).
#
# The fix: use the BOTTOM-CENTER of the bounding box (approximately where
# the feet touch the ground) as the point you feed into the homography.

def foot_point_from_box(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return np.array([(x1 + x2) / 2, y2])   # bottom-center, NOT box center


example_box = [600, 250, 700, 450]   # a player's bounding box, pixels
box_center = np.array([(example_box[0] + example_box[2]) / 2,
                        (example_box[1] + example_box[3]) / 2])
foot_point = foot_point_from_box(example_box)

court_from_center = pixel_to_court(box_center, homography_matrix)
court_from_feet = pixel_to_court(foot_point, homography_matrix)

print(f"\nUsing box CENTER   {box_center} -> court {court_from_center}")
print(f"Using box FEET      {foot_point} -> court {court_from_feet}")
print("These give DIFFERENT answers -- the feet-based one is the geometrically")
print("correct one, since only the feet actually touch the calibrated ground plane.")


# %% ------------------------------------------------------------------
# MENTAL MODEL
# ------------------------------------------------------------------
print("""
4+ known (pixel, real-world-meters) point pairs  (e.g. court corners)
   |
   v
cv2.findHomography()  -> 3x3 matrix H
   |
   v
cv2.perspectiveTransform(any pixel point, H)  -> real-world court position

Requirements / gotchas:
  - The mapped points must lie on the SAME flat plane you calibrated
    against (the ground). Use foot position, not box center or head.
  - The 4 correspondence points must be accurate -- errors there propagate
    to every point you ever map.
  - This assumes a roughly flat court (true for badminton/tennis/basketball
    courts; would need more sophisticated handling for a sloped field).
""")


# %% ------------------------------------------------------------------
# DAY 13 CHECKPOINT
# ------------------------------------------------------------------
"""
Test 1 — In your own words, what problem does a homography solve, and
  what minimum information do you need to compute one?

Test 2 — Why does the image-space quadrilateral (Section 1) look
  distorted/non-rectangular while the court-space one is a perfect
  rectangle?

Test 3 — Why must you use the bottom-center ("feet") of a player's
  bounding box rather than its center when converting to court
  coordinates? What goes wrong if you don't?

Test 4 — If you got the 4 corner pixel coordinates slightly wrong when
  calibrating, what happens to every OTHER point you map afterward?

Test 5 — Would this exact technique work if the camera were mounted on a
  drone flying at a constantly changing height/angle over the court?
  Why or why not?

If you can do all 5, Day 13 is done. Move to Day 14 (the capstone).
"""

if __name__ == "__main__":
    print("\nDay 13 script finished.")
