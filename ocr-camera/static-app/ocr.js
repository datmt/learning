// Browser/Node port of ocr_model.py + webapp/deskew.py + webapp/lines.py. Pure JS, no dependencies.
// A "gray image" here is {w, h, d: Uint8Array}.
(function (root) {
  const CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~";
  const IMG_H = 80, IMG_W = 384, ASPECT = 3.1, ANGLES = [-8, -6, -4, -2, 0, 2, 4, 6, 8];
  const NCLS = CHARS.length + 1;

  // ---------- resize (same idea as cv2.INTER_AREA when shrinking, bilinear when enlarging) ----------
  function axisWeights(src, dst) {
    const out = [], scale = src / dst;
    for (let i = 0; i < dst; i++) {
      if (dst < src) {                       // area average: each source pixel weighted by how much of it falls in the output pixel
        const a = i * scale, b = (i + 1) * scale, i0 = Math.floor(a), i1 = Math.min(src - 1, Math.ceil(b) - 1), idx = [], w = [];
        for (let j = i0; j <= i1; j++) { idx.push(j); w.push(Math.min(b, j + 1) - Math.max(a, j)); }
        out.push([idx, w]);
      } else {                               // bilinear with half-pixel centres
        const f = (i + 0.5) * scale - 0.5, j0 = Math.floor(f), t = f - j0;
        out.push([[Math.max(0, Math.min(src - 1, j0)), Math.max(0, Math.min(src - 1, j0 + 1))], [1 - t, t]]);
      }
    }
    return out;
  }
  function resize(img, dw, dh) {
    const { w: sw, h: sh, d } = img, wx = axisWeights(sw, dw), wy = axisWeights(sh, dh);
    const tmp = new Float32Array(dw * sh);
    for (let y = 0; y < sh; y++) for (let x = 0; x < dw; x++) {
      const [idx, w] = wx[x]; let s = 0, ws = 0;
      for (let k = 0; k < idx.length; k++) { s += d[y * sw + idx[k]] * w[k]; ws += w[k]; }
      tmp[y * dw + x] = s / ws;
    }
    const out = new Uint8Array(dw * dh);
    for (let y = 0; y < dh; y++) { const [idx, w] = wy[y]; for (let x = 0; x < dw; x++) {
      let s = 0, ws = 0; for (let k = 0; k < idx.length; k++) { s += tmp[idx[k] * dw + x] * w[k]; ws += w[k]; }
      out[y * dw + x] = Math.max(0, Math.min(255, Math.round(s / ws)));
    } }
    return { w: dw, h: dh, d: out };
  }

  // ---------- model input ----------
  const BASE_W = IMG_W * 2, BASE_H = Math.floor(BASE_W / 3.1);          // 768 x 247
  const toBase = img => resize(img, BASE_W, BASE_H);                    // step 1 (also used as the working size for tilt and line splitting)
  function toTensor(base) {                                             // step 2 + per-image standardising
    const g = resize(base, IMG_W, IMG_H).d, n = g.length, f = new Float32Array(n);
    let m = 0; for (let i = 0; i < n; i++) m += g[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) v += (g[i] - m) * (g[i] - m); const sd = Math.sqrt(v / n);
    for (let i = 0; i < n; i++) f[i] = (g[i] - m) / (sd + 1e-3);
    return f;
  }

  // ---------- CTC decoding ----------
  function ctcDecode(logits, T) {
    let s = "", conf = [], prev = 0;
    for (let t = 0; t < T; t++) {
      const o = t * NCLS; let k = 0, mx = -Infinity;
      for (let c = 0; c < NCLS; c++) if (logits[o + c] > mx) { mx = logits[o + c]; k = c; }
      if (k !== prev && k !== 0) {
        let z = 0; for (let c = 0; c < NCLS; c++) z += Math.exp(logits[o + c] - mx);
        s += CHARS[k - 1]; conf.push(1 / z);
      }
      prev = k;
    }
    return { text: s, conf };
  }

  // ---------- tilt (rotate about the centre, replicate the border) ----------
  function rotate(img, deg) {
    if (deg === 0) return img;
    const { w, h, d } = img, a = Math.cos(deg * Math.PI / 180), b = Math.sin(deg * Math.PI / 180), cx = w / 2, cy = h / 2;
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {              // inverse map of cv2.getRotationMatrix2D
      const dx = x - cx, dy = y - cy, sx = a * dx - b * dy + cx, sy = b * dx + a * dy + cy;
      const x0 = Math.floor(sx), y0 = Math.floor(sy), tx = sx - x0, ty = sy - y0;
      const X0 = Math.max(0, Math.min(w - 1, x0)), X1 = Math.max(0, Math.min(w - 1, x0 + 1)), Y0 = Math.max(0, Math.min(h - 1, y0)), Y1 = Math.max(0, Math.min(h - 1, y0 + 1));
      out[y * w + x] = Math.round((d[Y0 * w + X0] * (1 - tx) + d[Y0 * w + X1] * tx) * (1 - ty) + (d[Y1 * w + X0] * (1 - tx) + d[Y1 * w + X1] * tx) * ty);
    }
    return { w, h, d: out };
  }

  function vote(results) {                                              // most common text; ties -> highest mean confidence
    const cnt = new Map(); for (const r of results) cnt.set(r.text, (cnt.get(r.text) || 0) + 1);
    const top = Math.max(...cnt.values()); let best = null, bm = -1;
    for (const r of results) if (cnt.get(r.text) === top) { const m = r.conf.length ? r.conf.reduce((p, c) => p + c, 0) / r.conf.length : 0; if (m > bm) { bm = m; best = r; } }
    return { text: best.text, conf: best.conf, agree: [top, results.length] };
  }

  // ---------- several text lines in one crop ----------
  function gaussBlur(img, sigma) {
    const { w, h, d } = img, r = Math.max(1, Math.ceil(sigma * 3)), k = []; let ks = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-i * i / (2 * sigma * sigma)); k.push(v); ks += v; }
    const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let s = 0; for (let i = -r; i <= r; i++) s += d[y * w + Math.max(0, Math.min(w - 1, x + i))] * k[i + r]; tmp[y * w + x] = s / ks; }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let s = 0; for (let i = -r; i <= r; i++) s += tmp[Math.max(0, Math.min(h - 1, y + i)) * w + x] * k[i + r]; out[y * w + x] = s / ks; }
    return out;
  }
  function findBands(img, minH = 0.04, gap = 0.02, thr = 0.35) {
    const { w, h } = img, g = gaussBlur(img, Math.max(1, w / 600)), e = new Float32Array(h);
    for (let y = 0; y < h; y++) { let s = 0; for (let x = 0; x < w; x++) {
      const xm = Math.max(0, x - 1), xp = Math.min(w - 1, x + 1), at = yy => { const Y = Math.max(0, Math.min(h - 1, yy)); return g[Y * w + xp] - g[Y * w + xm]; };
      s += Math.abs(at(y - 1) + 2 * at(y) + at(y + 1));
    } e[y] = s / w; }
    const win = Math.max(3, Math.floor(h * 0.008)), prof = new Float32Array(h);
    for (let y = 0; y < h; y++) { let s = 0; for (let i = 0; i < win; i++) { const Y = y + i - (win >> 1); s += e[Math.max(0, Math.min(h - 1, Y))]; } prof[y] = s / win; }
    const sorted = Array.from(prof).sort((p, q) => p - q), hi = sorted[Math.min(h - 1, Math.floor(0.97 * (h - 1)))];
    const on = Array.from(prof, v => v > Math.max(thr * hi, 1e-6)), bands = []; let s = null;
    for (let y = 0; y <= h; y++) { const v = y < h ? on[y] : false; if (v && s === null) s = y; if (!v && s !== null) { bands.push([s, y]); s = null; } }
    const merged = []; for (const b of bands) { if (merged.length && b[0] - merged[merged.length - 1][1] < gap * h) merged[merged.length - 1][1] = b[1]; else merged.push([b[0], b[1]]); }
    return merged.filter(b => b[1] - b[0] >= minH * h);
  }
  function lineCrops(img, bands) {
    const { w, h, d } = img, ch = Math.round(w / ASPECT), edge = [];
    for (let y = 0; y < h; y++) for (const x of [0, 1, w - 2, w - 1]) edge.push(d[y * w + x]);
    edge.sort((p, q) => p - q); const bg = edge[edge.length >> 1], out = [];
    bands.forEach(([y0, y1], i) => {
      const g = Uint8Array.from(d), pad = Math.floor((y1 - y0) * 0.35);
      bands.forEach(([a, b], j) => { if (j !== i) g.fill(bg, Math.max(0, a - pad) * w, Math.min(h, b + pad) * w); });
      const top = Math.floor((y0 + y1) / 2) - (ch >> 1), canvas = new Uint8Array(w * ch).fill(bg);
      for (let y = Math.max(0, top); y < Math.min(h, top + ch); y++) canvas.set(g.subarray(y * w, (y + 1) * w), (y - top) * w);
      out.push({ w, h: ch, d: canvas });
    });
    return out;
  }

  // ---------- reading ----------
  // session: an onnxruntime(-web) InferenceSession; Tensor: its Tensor class
  async function readBase(session, Tensor, base, tilt) {
    const run = async im => { const out = await session.run({ x: new Tensor("float32", toTensor(im), [1, 1, IMG_H, IMG_W]) }); const o = out.logits || out[Object.keys(out)[0]]; return ctcDecode(o.data, o.dims[1]); };
    if (!tilt) { const r = await run(base); return { ...r, agree: [1, 1] }; }
    const rs = []; for (const a of ANGLES) rs.push(await run(rotate(base, a)));
    return vote(rs);
  }
  async function read(session, Tensor, gray, { tilt = true, lines = false } = {}) {
    const base = toBase(gray), bands = lines ? findBands(base) : [];
    if (bands.length < 2) return readBase(session, Tensor, base, tilt);
    const parts = []; for (const c of lineCrops(base, bands)) parts.push(await readBase(session, Tensor, c, tilt));
    const keep = parts.filter(p => p.text).length ? parts.filter(p => p.text) : parts.slice(0, 1);
    let conf = []; keep.forEach((p, k) => { conf = conf.concat(k ? [1] : [], p.conf); });
    return { text: keep.map(p => p.text).join("\n"), conf, agree: keep.map(p => p.agree).reduce((a, b) => a[0] / a[1] <= b[0] / b[1] ? a : b) };
  }

  const api = { CHARS, ANGLES, resize, toBase, toTensor, ctcDecode, rotate, vote, findBands, lineCrops, read, readBase };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.OCR = api;
})(typeof self !== "undefined" ? self : this);
