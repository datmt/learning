# Static web app (no server, no Docker)

The model runs **in the browser** with ONNX Runtime Web (WebAssembly). Photos never leave the phone. The folder is plain files, so any static host works, for example Cloudflare Pages.

| File | Size | What |
|---|---|---|
| `index.html` | 5 KB | camera page with a guide box, tilt-robust and multi-line options |
| `ocr.js` | 9 KB | JavaScript port of the Python code: resize, standardise, tilt vote, CTC decoding, line splitting |
| `model/crnn_int8.onnx` | 7.5 MB | the model with int8 weights (the fp32 file is 16.5 MB) |
| `vendor/` | 15 MB | ONNX Runtime Web (copied from the `onnxruntime-web` npm package, version 1.31.1) |

Total about 22 MB, and the biggest single file is 14.8 MB, under the 25 MiB per-file limit of Cloudflare Pages.

## Deploy on Cloudflare Pages
Dashboard: Workers & Pages, Create, Pages, upload assets, and drop the `static-app` folder. Or from a terminal:
```bash
npx wrangler pages deploy static-app --project-name ocr-camera
```
Pages serves HTTPS, which the phone camera requires. I have **not** deployed it to Cloudflare myself; I ran it from a local web server (below).

## Try it locally
```bash
cd static-app && python3 -m http.server 8000     # then open http://localhost:8000 (camera works on localhost)
```
On a phone you need HTTPS, so use the Cloudflare URL (or the Docker app).

## How I checked it
- **int8 vs fp32** (192 held-out crops of the collection page): fp32 82.3% exact / 97.0% char, int8 82.8% / 97.1%; 99.5% of reads identical. Weights of the LSTM and linear layers are int8; the convolutions stay fp32 (I did not try quantising them). These crops are the only ones I scored: the phone photos had been deleted by then.
- **JavaScript vs Python** (60 held-out crops, int8 model): single read 49/60 right in both, and 59 of 60 reads gave identical text. The 9-angle vote also gave 50/60 in both, with 59 of 60 identical.
- **In a real browser** (headless Firefox 157, 12 crops, 9-angle vote): all 12 reads equal to the Python result. One read takes about 0.65 s on a desktop CPU, single-threaded. A phone is slower, and I have not measured it.
- **Not tested:** the camera page on a phone, anything on Cloudflare, and the multi-line option on real photos. In a test with two crops shrunk and stacked into one frame, the splitter found both lines, but the small text was read badly (0 of 20 pairs fully right).

## Differences from the Python reader
- The tilt vote rotates the 768x247 working image with bilinear sampling, instead of rotating the full photo with cubic sampling as `webapp/deskew.py` does. It is faster, and gave the same text in 59 of 60 test reads.
- No saving of reads, no `/display` auto-labelling, no corner-marker mode. Those need a server; use the Docker app for collecting data.
