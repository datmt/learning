import json, cv2, sys
from multiprocessing import Pool
from pathlib import Path
from crop import crop_panel
OUT = Path("data/crops"); OUT.mkdir(exist_ok=True)

def work(r):
    c = crop_panel(cv2.imread("data/" + r["file"]))
    if c is None: return None
    name = Path(r["file"]).stem + ".png"
    cv2.imwrite(str(OUT / name), c)
    return {"crop": "crops/" + name, "item_id": r["item_id"], "text": r["text"], "font": r["style"]["family"], "setup": r.get("setup", "initial")}

if __name__ == "__main__":
    recs = [json.loads(l) for l in open("data/manifest.jsonl")]
    with Pool(12) as p: res = p.map(work, recs, chunksize=8)
    ok = [x for x in res if x]
    with open("data/crops.jsonl", "w") as f:
        for x in ok: f.write(json.dumps(x) + "\n")
    print(f"{len(ok)}/{len(recs)} cropped")
