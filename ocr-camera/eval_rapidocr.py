"""RapidOCR baseline, keyed by crop path (so it survives re-cropping). Reuses earlier results when they provably match."""
import json, time, cv2, os
from rapidocr_onnxruntime import RapidOCR
recs = [json.loads(l) for l in open("data/crops.jsonl")]
from ocr_model import CHARS; from phone_data import load_phone_records
recs += load_phone_records("data", CHARS)[0]                              # labelled phone photos from the 8444 app
out = {}
if os.path.exists("data/rapidocr_by_crop.json"): out = json.load(open("data/rapidocr_by_crop.json"))
elif os.path.exists("data/rapidocr_results.json"):                       # old format: list in crops.jsonl order (791 crops)
    old = json.load(open("data/rapidocr_results.json"))["det+rec"]
    n = sum(1 for r, o in zip(recs, old) if r["text"] == o["gt"])
    if n == len(old):
        out = {r["crop"]: o["pred"] for r, o in zip(recs, old)}; print(f"reused {len(out)} earlier results (ground truth matched for all)")
    else: print(f"old results do not line up ({n}/{len(old)} match) -> recomputing everything")
todo = [r for r in recs if r["crop"] not in out]; print(f"{len(todo)} crops to run, {len(out)} cached", flush=True)
eng = RapidOCR(); t0 = time.time()
for i, r in enumerate(todo, 1):
    o, _ = eng(cv2.imread("data/" + r["crop"])); out[r["crop"]] = "".join(x[1] for x in o).replace(" ", "") if o else ""
    if i % 100 == 0: print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s", flush=True); json.dump(out, open("data/rapidocr_by_crop.json", "w"))
json.dump(out, open("data/rapidocr_by_crop.json", "w")); print(f"done: {len(out)} results", flush=True)
