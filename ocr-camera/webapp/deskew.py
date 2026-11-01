"""Guide-box crops are not straightened by corner markers, so the text can be tilted. Read the crop at several small angles and vote.
(A projection-profile tilt estimator was tried and removed: it was wrong even on straight crops.)"""
import collections
import cv2
import numpy as np


def rotate(gray, deg):
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def vote(results):
    """results: list of (text, conf_list). Majority text; ties -> highest mean confidence."""
    cnt = collections.Counter(t for t, _ in results)
    top = max(cnt.values()); cands = [t for t, c in cnt.items() if c == top]
    mc = {t: max(np.mean(c) if c else 0 for tt, c in results if tt == t) for t in cands}
    t = max(cands, key=lambda x: mc[x])
    return t, next(c for tt, c in results if tt == t), top, len(results)
