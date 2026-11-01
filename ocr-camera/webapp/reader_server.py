"""Test app for the trained recognizer. Phone camera -> text.
When the page's "save for review" box is ticked, the submitted image, the crop the model saw and the result are written to data/reader_log/
(so they can be checked later). Otherwise nothing is stored. Run with ./run_reader.sh  (HTTPS port 8444, so the phone camera works)."""
import base64, json, os, random, string, sys, threading, time, uuid
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from webapp.lines import find_bands, line_crops      # noqa: E402
from ocr_model import preprocess, ctc_decode_detail      # noqa: E402
from crop import crop_panel                               # noqa: E402
from webapp.deskew import rotate, vote                           # noqa: E402

ANGLES = (-8, -6, -4, -2, 0, 2, 4, 6, 8)   # read at each of these tilts, then vote

MODEL = Path(os.environ.get("OCR_ONNX", ROOT.parent / "deploy" / "crnn.onnx"))
SESS = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
app = FastAPI()
LOG = Path(os.environ.get("OCR_READER_LOG", ROOT.parent / "data" / "reader_log"))


def model_name():
    f = MODEL.parent / "crnn.onnx.source"
    return f.read_text().strip() if f.exists() else MODEL.name


def read_once(gray):
    return ctc_decode_detail(SESS.run(None, {"x": preprocess(gray)[None]})[0][0])


def recognise(gray, tta=True):
    t = time.time()
    if not tta:
        text, conf = read_once(gray); agree = (1, 1)
    else:
        text, conf, k, n = vote([read_once(rotate(gray, a)) for a in ANGLES]); agree = (k, n)
    return text, conf, (time.time() - t) * 1000, agree


def jpeg_b64(img, width=640):
    s = width / img.shape[1]
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ------------------------------------------------------------ /display: random strings shown like a real password field
CHARSET = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~"
CONFUSABLE = list("O0oIl1|S5sB8Z2zrnmvwVW`'.,:;gq9CcUuKkXxPp")
FONT_STACKS = ["system-ui, sans-serif", "Arial, Helvetica, sans-serif", "Verdana, sans-serif", "Tahoma, sans-serif", "'Trebuchet MS', sans-serif",
               "'Segoe UI', Roboto, sans-serif", "Roboto, sans-serif", "Georgia, serif", "'Courier New', monospace", "ui-monospace, Menlo, Consolas, monospace",
               "'Helvetica Neue', Helvetica, sans-serif", "'DejaVu Sans', sans-serif", "'Liberation Sans', sans-serif"]
FIELD_LOCK = threading.Lock()
FIELD = {"items": {}, "order": [], "current": None, "reads": {}, "mode": "random", "shown": None}      # mode: random | dark | light


def field_text():
    n = random.choice([6, 8, 8, 10, 12, 12, 14, 16, 16, 20, 24]); m = random.random()
    pool = (string.ascii_letters + string.digits if m < .2 else string.ascii_lowercase + string.digits if m < .35 else
            string.ascii_uppercase + string.digits if m < .45 else "0123456789abcdef" if m < .55 else CHARSET)
    out = []
    for _ in range(n):
        c = random.choice(CONFUSABLE) if random.random() < .3 else random.choice(pool)
        out.append(c if c in pool else random.choice(pool))
    return "".join(out)


def field_item():
    dark = (random.random() < .5) if FIELD["mode"] == "random" else FIELD["mode"] == "dark"
    pick = random.choice
    if dark: page, box, border, fg = pick(["#121212", "#0d1117", "#1b1d21", "#202124"]), pick(["#1e1e1e", "#161b22", "#2a2d33", "#101214"]), pick(["#3a3f47", "#30363d", "#4a4f57"]), pick(["#e8eaed", "#ffffff", "#c9d1d9", "#dcdcdc"])
    else:    page, box, border, fg = pick(["#ffffff", "#f6f7f9", "#fafafa", "#f1f3f4"]), pick(["#ffffff", "#f8f9fa", "#f1f3f4"]), pick(["#c9ced6", "#9aa0a6", "#d0d5dd", "#8a9099"]), pick(["#111111", "#202124", "#333333", "#1a1a1a"])
    return {"id": uuid.uuid4().hex[:8], "text": field_text(), "style": {
        "dark": dark, "page": page, "box": box, "border": border, "fg": fg, "font": pick(FONT_STACKS), "size": round(random.uniform(.8, 1.25), 2),
        "radius": pick(["pill", "pill", "12px", "6px", "2px"]), "badge": pick(["none", "none", "strong", "weak", "eye", "eye"]),
        "label": pick([True, False]), "x": 0, "y": 0}}


def field_advance():
    with FIELD_LOCK:
        it = field_item(); FIELD["items"][it["id"]] = it; FIELD["order"].append(it["id"]); FIELD["current"] = it["id"]; FIELD["shown"] = None
        while len(FIELD["order"]) > 200: FIELD["items"].pop(FIELD["order"].pop(0), None)
    return it


field_advance()


@app.get("/api/field/state")
def field_state(full: int = 0):
    it = FIELD["items"][FIELD["current"]]
    return {"id": it["id"], "style": it["style"], "reads": FIELD["reads"].get(it["id"], 0), "mode": FIELD["mode"], "shown": FIELD["shown"], **({"text": it["text"]} if full else {})}


@app.post("/api/field/mode")
def field_mode(mode: str = Form("random")):
    """Force light or dark mode for the strings shown on /display (takes effect at once: the current string is replaced)."""
    if mode not in ("random", "dark", "light"): return JSONResponse({"error": "mode must be random, dark or light"}, status_code=400)
    FIELD["mode"] = mode; field_advance(); return {"mode": mode}


@app.post("/api/field/shown")
def field_shown(id: str = Form(...)):
    """The /display page calls this once the string is really on screen (so the phone knows when it may shoot)."""
    if id == FIELD["current"]: FIELD["shown"] = id
    return {"ok": id == FIELD["current"]}


@app.post("/api/field/next")
def field_next():
    it = field_advance(); return {"id": it["id"]}


@app.get("/display")
def display_page(): return FileResponse(ROOT / "static" / "field_display.html", headers={"Cache-Control": "no-store"})


@app.post("/api/read")
async def read(file: UploadFile = File(...), mode: str = Form("whole"), save: str = Form("0"), tta: str = Form("1"), field_id: str = Form(""), lines: str = Form("0")):
    """mode: 'whole' = image is already just the text area (guide-box crop or a pre-cropped image);
             'markers' = find the 4 corner markers of the collection display and flatten the panel."""
    raw = await file.read()
    return await run_in_threadpool(_read, raw, mode, save, tta, field_id, lines)        # CPU work off the event loop so polling stays responsive


def _read(raw, mode, save, tta, field_id, lines="0"):
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None: return JSONResponse({"error": "could not decode image"}, status_code=400)
    t0 = time.time()
    if mode == "markers":
        img = crop_panel(img)
        if img is None: return {"error": "markers not found - fill the frame with the whole panel incl. the 4 corner squares, or use guide-box mode"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); bands = find_bands(gray) if lines == "1" else []
    if len(bands) > 1:                                    # several text lines: read each on its own, join with newlines ("\n" gets conf 1.0)
        parts = [recognise(c, tta == "1") for c in line_crops(gray, bands)]
        parts = [p for p in parts if p[0]] or parts[:1]
        text = "\n".join(p[0] for p in parts); ms = sum(p[2] for p in parts); agree = min((p[3] for p in parts), key=lambda a: a[0] / a[1])
        conf = []
        for k, p in enumerate(parts): conf += ([1.0] if k else []) + list(p[1])
    else:
        text, conf, ms, agree = recognise(gray, tta == "1")
    rid = ""; truth = None; item = FIELD["items"].get(field_id) if field_id else None
    if item:
        truth = item["text"]; FIELD["reads"][field_id] = FIELD["reads"].get(field_id, 0) + 1
    if save == "1":
        LOG.mkdir(parents=True, exist_ok=True); rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        ext = ".png" if raw[:4] == b"\x89PNG" else ".jpg"
        if mode != "whole": (LOG / f"{rid}.input{ext}").write_bytes(raw)          # in 'whole' mode the input IS the crop, so don't store it twice
        cv2.imwrite(str(LOG / f"{rid}.seen.png"), img)
        (LOG / f"{rid}.json").write_text(json.dumps({"id": rid, "mode": mode, "text": text, "conf": [round(c, 3) for c in conf], "model": model_name(),
                                                     "agree": f"{agree[0]}/{agree[1]}", "tta": tta == "1", "input": None if mode == "whole" else f"{rid}.input{ext}", "seen": f"{rid}.seen.png", "truth": truth, "field_id": field_id or None,
                                                     "field_style": item["style"] if item else None}, indent=1))
    return {"text": text, "conf": conf, "model_ms": round(ms, 1), "total_ms": round((time.time() - t0) * 1000, 1), "seen": jpeg_b64(img), "id": rid, "agree": agree, "truth": truth, "correct": (text == truth) if truth is not None else None,
            "n_reads": FIELD["reads"].get(field_id, 0) if item else None}


@app.post("/api/label")
def label(id: str = Form(...), truth: str = Form("")):
    """Attach what the text really said to a saved result."""
    f = LOG / (os.path.basename(id) + ".json")
    if not f.exists(): return JSONResponse({"error": "unknown id"}, status_code=404)
    d = json.loads(f.read_text()); d["truth"] = truth; f.write_text(json.dumps(d, indent=1)); return {"ok": True, "correct": truth == d["text"]}


@app.get("/")
def page(): return FileResponse(ROOT / "static" / "reader.html", headers={"Cache-Control": "no-store"})
