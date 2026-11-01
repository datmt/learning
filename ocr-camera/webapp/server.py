"""Data-collection server for the password OCR project.

/display  -> open on a monitor; shows one random string at a time
/         -> open on the phone; camera capture, uploads photo labelled with the string on screen
"""
import colorsys
import collections
import json
import os
import re
import random
import string
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("OCR_DATA", ROOT.parent / "data"))   # override for testing
IMG_DIR = DATA / "images"
MANIFEST = DATA / "manifest.jsonl"
IMG_DIR.mkdir(parents=True, exist_ok=True)

LOCK = threading.Lock()
SETTINGS = {"shots_per_item": 1, "target": 1000}

# ---------------------------------------------------------------- fonts
CHARSET = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~"
CONFUSABLE = list("O0oIl1|S5sB8Z2zrnmvwVW`'.,:;gq9CcUuKkXxPp")


def discover_fonts():
    out = subprocess.run(
        ["fc-list", "--format", "%{file}|%{family[0]}|%{style[0]}|%{spacing}\n"],
        capture_output=True, text=True).stdout.splitlines()
    fonts, seen = [], set()
    need = set(ord(c) for c in string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~")
    for line in out:
        try:
            path, fam, style, spacing = line.split("|")
        except ValueError:
            continue
        if not path.lower().endswith((".ttf", ".otf")) or path in seen:
            continue
        seen.add(path)
        try:
            f = TTFont(path, lazy=True, fontNumber=0)
            cmap = f.getBestCmap() or {}
            if not need.issubset(cmap.keys()):
                continue
            # skip symbol/dingbat style fonts
            if any(k in fam.lower() for k in ("dingbat", "symbol", "emoji", "wingding", "icon")):
                continue
        except Exception:
            continue
        fonts.append({"path": path, "family": fam, "style": style, "mono": spacing in ("100", "mono")})
    return fonts


FONTS = discover_fonts()
print(f"[fonts] {len(FONTS)} usable fonts")

# ---------------------------------------------------------------- generation
def lum(rgb):
    def ch(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def rand_colors():
    """People use light mode or dark mode: plain grey backgrounds only."""
    if random.random() < 0.5:                                   # light mode
        v, f = random.randint(235, 255), random.randint(0, 60)
    else:                                                       # dark mode
        v, f = random.randint(0, 40), random.randint(205, 255)
    return (v, v, v), (f, f, f)


def rand_text():
    n = random.choice([6, 8, 8, 10, 12, 12, 14, 16, 16, 20, 24])
    mode = random.random()
    if mode < 0.2:
        pool = string.ascii_letters + string.digits
    elif mode < 0.35:
        pool = string.ascii_lowercase + string.digits
    elif mode < 0.45:
        pool = string.ascii_uppercase + string.digits
    elif mode < 0.55:
        pool = "0123456789abcdef"
    else:
        pool = CHARSET
    chars = []
    for _ in range(n):
        if random.random() < 0.3:
            c = random.choice(CONFUSABLE)
            if c not in pool:
                c = random.choice(pool)
        else:
            c = random.choice(pool)
        chars.append(c)
    return "".join(chars)


def make_item():
    bg, fg = rand_colors()
    fi = random.randrange(len(FONTS))
    return {
        "id": uuid.uuid4().hex[:10],
        "text": rand_text(),
        "style": {
            "font": fi, "family": FONTS[fi]["family"], "font_style": FONTS[fi]["style"],
            "size": round(random.uniform(0.5, 1.0), 2),
            "spacing": round(random.choice([0, 0, 0, 0.02, 0.05, 0.1]), 2),
            "rotate": 0,
            "bg": "#%02x%02x%02x" % bg, "fg": "#%02x%02x%02x" % fg,
            # panel size/position on the monitor (no tilt: change the phone angle instead). A smaller panel = fewer pixels per letter in the photo.
            "pose": {"dx": round(random.uniform(-5, 5), 1), "dy": round(random.uniform(-8, 8), 1),
                     "scale": round(random.uniform(0.4, 1.0), 2), "rz": 0, "rx": 0, "ry": 0},
        },
    }


# ---------------------------------------------------------------- state
STATE = {"current": make_item(), "shown": None, "shots": 0, "seq": 1}
ITEMS = {STATE["current"]["id"]: STATE["current"]}


def load_manifest():
    if not MANIFEST.exists():
        return []
    return [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]


RECORDS = load_manifest()
for _r in RECORDS: _r.setdefault("setup", "initial")      # photos taken before setup labels existed


def advance():
    item = make_item()
    ITEMS[item["id"]] = item
    STATE.update(current=item, shots=0, seq=STATE["seq"] + 1)


def public_state():
    return {"current": STATE["current"], "shown": STATE["shown"], "shots": STATE["shots"], "seq": STATE["seq"],
            "total": len(RECORDS), "unique": len({r["item_id"] for r in RECORDS}),
            "by_setup": dict(collections.Counter(r["setup"] for r in RECORDS)), **SETTINGS}


# ---------------------------------------------------------------- app
app = FastAPI()


@app.get("/api/state")
def get_state():
    with LOCK:
        return public_state()


@app.post("/api/shown")
def mark_shown(item_id: str = Form(...)):
    with LOCK:
        if item_id == STATE["current"]["id"]:
            STATE["shown"] = item_id
    return {"ok": True}


@app.post("/api/next")
def next_item():
    with LOCK:
        advance()
        return public_state()


@app.post("/api/settings")
def settings(shots_per_item: int = Form(None), target: int = Form(None)):
    with LOCK:
        if shots_per_item:
            SETTINGS["shots_per_item"] = max(1, min(20, shots_per_item))
        if target:
            SETTINGS["target"] = max(1, target)
        return public_state()


@app.post("/api/captured")
def captured(item_id: str = Form(...)):
    """Phone grabbed a frame for item_id: count it and advance now (the upload happens separately, in the background).
    counted=False means the display had already moved on, so the phone must drop that frame (it would be mislabelled)."""
    with LOCK:
        if item_id != STATE["current"]["id"] or STATE["shown"] != item_id:
            return {"counted": False, **public_state()}
        STATE["shots"] += 1
        if STATE["shots"] >= SETTINGS["shots_per_item"]:
            advance()
        return {"counted": True, **public_state()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), item_id: str = Form(...), meta: str = Form("{}"), setup: str = Form("")):
    setup = re.sub(r"[^A-Za-z0-9._-]+", "-", setup.strip())[:40].strip("-")
    if not setup:
        raise HTTPException(400, "setup label required")
    data = await file.read()
    if len(data) < 1000:
        raise HTTPException(400, "empty image")
    with LOCK:
        item = ITEMS.get(item_id)
        if item is None:
            raise HTTPException(409, "unknown item (server restarted?) - press Next")
        n = sum(1 for r in RECORDS if r["item_id"] == item_id) + 1
        name = f"{item_id}_{n}.jpg"
        (IMG_DIR / name).write_bytes(data)
        try:
            meta_d = json.loads(meta)
        except Exception:
            meta_d = {}
        rec = {"file": f"images/{name}", "item_id": item_id, "text": item["text"], "style": item["style"],
               "time": time.time(), "setup": setup, "meta": meta_d}
        RECORDS.append(rec)
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return {"saved": name, **public_state()}


@app.post("/api/undo")
def undo():
    """Delete the most recent photo (e.g. a blurry one)."""
    with LOCK:
        if not RECORDS:
            raise HTTPException(404, "nothing to undo")
        rec = RECORDS.pop()
        (DATA / rec["file"]).unlink(missing_ok=True)
        MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
        if rec["item_id"] == STATE["current"]["id"]:
            STATE["shots"] = max(0, STATE["shots"] - 1)
        elif rec["item_id"] in ITEMS:
            # re-show the item we just removed a photo of, so it can be retaken
            STATE.update(current=ITEMS[rec["item_id"]], shots=sum(1 for r in RECORDS if r["item_id"] == rec["item_id"]),
                         seq=STATE["seq"] + 1, shown=None)
        return {"removed": rec["file"], **public_state()}


@app.get("/fonts/{idx}")
def font_file(idx: int):
    if not 0 <= idx < len(FONTS):
        raise HTTPException(404)
    return FileResponse(FONTS[idx]["path"])


@app.get("/display")
def display_page():
    return FileResponse(ROOT / "static" / "display.html")


@app.get("/")
def phone_page():
    return FileResponse(ROOT / "static" / "capture.html")


@app.get("/api/fontcount")
def fontcount():
    return {"n": len(FONTS)}
