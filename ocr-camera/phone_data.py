"""Labelled phone photos saved by the 8444 app (data/reader_log/*.json with a 'truth') as extra real crops.
The guide-box crop the app saves (*.seen.png) is already in the same 3.1:1 format as data/crops."""
import glob
import hashlib
import json
from pathlib import Path


def load_phone_records(data_dir, chars):
    out, skipped = [], 0
    for f in sorted(glob.glob(str(Path(data_dir) / "reader_log" / "*.json"))):
        d = json.load(open(f)); t = d.get("truth")
        if not t: continue
        if any(c not in chars for c in t) or not (Path(data_dir) / "reader_log" / d["seen"]).exists(): skipped += 1; continue
        key = d.get("field_id") or "L" + hashlib.md5(t.encode()).hexdigest()[:8]       # all photos of one string share an item_id -> same split
        out.append({"crop": "reader_log/" + d["seen"], "item_id": "phone-" + key, "text": t, "font": "phone", "setup": "phone-field"})
    return out, skipped
