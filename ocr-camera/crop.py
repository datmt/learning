"""Find the 4 corner markers in a photo, flatten the panel, rotate text upright.
Returns a BGR crop of the colored panel or None."""
import cv2, numpy as np

def plausible(q, side):
    """q: 4 unordered marker centres. Panel is ~3.1:1 and markers ~0.08 of its short side."""
    c = q.mean(0); q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    q = np.roll(q, -int(np.argmin(q.sum(1))), axis=0)
    d = lambda a, b: float(np.linalg.norm(a - b))
    t, r, b, l = d(q[0], q[1]), d(q[1], q[2]), d(q[3], q[2]), d(q[0], q[3])
    W, H = (t + b) / 2, (l + r) / 2
    asp = max(W, H) / min(W, H)
    return (2.7 <= asp <= 3.5 and 0.065 <= side / min(W, H) <= 0.10
            and max(t, b) / min(t, b) < 2.2 and max(l, r) / min(l, r) < 1.5)


def find_markers(img, size=2000):
    """size = long side the photo is shrunk to before searching; thresholds below are tuned for 2000 and scale with it."""
    h, w = img.shape[:2]
    s = size / max(h, w); k = size / 2000
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (0, 0), 25 * k)
    dark = ((g.astype(np.float32) < 0.40 * bg) & (g < 90)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 45 * k * k or a > 24000 * k * k: continue
        x, y, bw, bh = cv2.boundingRect(c)
        if not 0.5 < bw / bh < 2.0 or a / (bw * bh) < 0.5: continue
        # white ring around it
        pad = max(3, int(bw * 0.35))
        ring = g[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
        if np.percentile(ring, 90) < 170: continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        cands.append((cx / s, cy / s, np.sqrt(a) / s))
    if len(cands) < 4: return None
    # choose the 4 candidates forming the largest convex quad with plausible size spread
    import itertools
    cands = sorted(cands, key=lambda t: -t[2])[:14]
    best = None
    for grp in itertools.combinations(cands, 4):
        sz = [p[2] for p in grp]
        if max(sz) / min(sz) > 3.0: continue
        pts = np.array([[p[0], p[1]] for p in grp], np.float32)
        hull = cv2.convexHull(pts)
        if len(hull) < 4: continue
        area = cv2.contourArea(hull)
        if not plausible(pts, float(np.mean(sz))): continue
        if best is None or area > best[0]: best = (area, grp)
    if best is None: return None
    grp = best[1]
    pts = np.array([[p[0], p[1]] for p in grp], np.float32)
    side = float(np.mean([p[2] for p in grp]))
    c = pts.mean(0); order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    pts = pts[order]                       # clockwise from top-left-ish (image coords)
    tl = np.argmin(pts.sum(1)); pts = np.roll(pts, -tl, axis=0)
    return pts, side                       # TL, TR, BR, BL  (image frame)

def crop_panel(img, rotate=cv2.ROTATE_90_COUNTERCLOCKWISE):
    r = find_markers(img)
    if r is None and max(img.shape[:2]) > 2000:       # small panel / far phone: markers are tiny, so search at full resolution
        r = find_markers(img, size=max(img.shape[:2]))
    if r is None: return None
    pts, side = r
    d = lambda a, b: float(np.linalg.norm(a - b))
    W = (d(pts[0], pts[1]) + d(pts[3], pts[2])) / 2
    H = (d(pts[0], pts[3]) + d(pts[1], pts[2])) / 2
    # markers sit one marker-side outside the panel corners in x and y
    out = np.array([[0, 0], [W, 0], [W, H], [W * 0 + 0, H]], np.float32)
    M = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32))
    flat = cv2.warpPerspective(img, M, (int(W), int(H)), flags=cv2.INTER_CUBIC)
    m = int(round(side * 1.0))
    if flat.shape[0] > 2 * m + 10 and flat.shape[1] > 2 * m + 10:
        flat = flat[m:-m, m:-m]
    return cv2.rotate(flat, rotate) if rotate is not None else flat
