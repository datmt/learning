"""Score the model currently in deploy/crnn.onnx on the labelled phone photos saved by the 8444 app (data/reader_log).
Usage: .venv/bin/python eval_reader_log.py [--all]
Photos without a label are skipped. If train.ipynb has written data/split.json, only HELD-OUT strings (val/test) are scored, because the
notebook trains on the other phone photos. --all scores everything (training photos included, so the numbers are optimistic)."""
import glob, json, sys, collections
import cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, "."); from ocr_model import preprocess, ctc_decode_detail
from webapp.deskew import rotate, vote
ANG = (-8, -6, -4, -2, 0, 2, 4, 6, 8)
s = ort.InferenceSession("deploy/crnn.onnx", providers=["CPUExecutionProvider"])
read = lambda g: ctc_decode_detail(s.run(None, {"x": preprocess(g)[None]})[0][0])
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]
rows = [json.load(open(f)) for f in sorted(glob.glob("data/reader_log/*.json"))]; rows = [d for d in rows if d.get("truth")]
import os, hashlib
if os.path.exists("data/split.json") and "--all" not in sys.argv:
    split = json.load(open("data/split.json")); key = lambda d: "phone-" + (d.get("field_id") or "L" + hashlib.md5(d["truth"].encode()).hexdigest()[:8])
    n0 = len(rows); rows = [d for d in rows if split.get(key(d)) in ("val", "test")]
    print(f"held-out strings only (val/test per data/split.json): {len(rows)} of {n0} labelled photos; the rest were in training or are newer than the last training run")
if not rows: sys.exit("no labelled photos in data/reader_log yet")
res = collections.defaultdict(list)
for d in rows:
    g = cv2.cvtColor(cv2.imread("data/reader_log/" + d["seen"]), cv2.COLOR_BGR2GRAY)
    for name, f in (("single read", lambda: read(g)[0]), ("9-angle vote", lambda: vote([read(rotate(g, a)) for a in ANG])[0])):
        res[name].append((d["truth"], f()))
print(open("deploy/crnn.onnx.source").read().strip() if __import__("os").path.exists("deploy/crnn.onnx.source") else "model: deploy/crnn.onnx")
print(f"{len(rows)} labelled photos of {len({d['truth'] for d in rows})} different strings")
for name, r in res.items():
    ex = np.mean([t == p for t, p in r]); ce = sum(lev(p, t) for t, p in r); cl = sum(len(t) for t, _ in r)
    print(f"  {name:13} exact {ex:5.1%} | character accuracy {1 - ce / cl:5.1%} | avg wrong chars per photo {ce / len(r):.1f}")
by = collections.defaultdict(list)
for t, p in res["9-angle vote"]: by[t].append(t == p)
for t, v in by.items(): print(f"  {t!r:30} {sum(v)}/{len(v)} correct (vote)")
