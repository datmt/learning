"""Split a guide-box crop that holds several text lines into one crop per line (the model reads one line at a time)."""
import cv2, numpy as np

ASPECT = 3.1          # crop width / height the model was trained on


def find_bands(gray, min_h_frac=0.04, gap_frac=0.02, thr_rel=0.35):
    """Row bands that contain text: rows whose horizontal-detail energy is a fair fraction of the strongest rows."""
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (0, 0), max(1.0, w / 600)).astype(np.float32)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))          # vertical strokes -> text has lots of x-gradient
    prof = np.convolve(e.mean(1), np.ones(max(3, int(h * .008))) / max(3, int(h * .008)), "same")
    hi = np.percentile(prof, 97)
    on = prof > max(thr_rel * hi, 1e-6)
    bands, s = [], None
    for y in range(h + 1):
        v = on[y] if y < h else False
        if v and s is None: s = y
        if not v and s is not None: bands.append([s, y]); s = None
    merged = []
    for b in bands:                                               # join bands separated by a small gap
        if merged and b[0] - merged[-1][1] < gap_frac * h: merged[-1][1] = b[1]
        else: merged.append(b)
    return [b for b in merged if b[1] - b[0] >= min_h_frac * h]


def line_crops(gray, bands):
    """One crop per band at the model's aspect ratio (full width, height w/ASPECT) centred on the band;
    the other bands are painted over with the background so only one line is visible."""
    h, w = gray.shape
    bg = int(np.median(np.concatenate([gray[:, :max(2, w // 40)].ravel(), gray[:, -max(2, w // 40):].ravel()])))
    ch = int(round(w / ASPECT)); out = []
    for i, (y0, y1) in enumerate(bands):
        g = gray.copy(); pad = int((y1 - y0) * 0.35)
        for j, (a, b) in enumerate(bands):
            if j != i: g[max(0, a - pad):min(h, b + pad)] = bg
        c = (y0 + y1) // 2; top = c - ch // 2
        canvas = np.full((ch, w), bg, np.uint8)
        s0, s1 = max(0, top), min(h, top + ch)
        canvas[s0 - top:s1 - top] = g[s0:s1]
        out.append(canvas)
    return out
