# Static web app (no server, no Docker)

The model runs **in the browser** with ONNX Runtime Web (WebAssembly). Photos never leave the phone. The folder is plain files, so any static host works, for example Cloudflare.

| File | Size | What |
|---|---|---|
| `index.html` | 5 KB | camera page with a guide box, tilt-robust and multi-line options |
| `ocr.js` | 9 KB | JavaScript port of the Python code: resize, standardise, tilt vote, CTC decoding, line splitting |
| `model/crnn_int8.onnx` | 7.5 MB | the model with int8 weights (the fp32 file is 16.5 MB) |
| `vendor/` | 15 MB | ONNX Runtime Web (copied from the `onnxruntime-web` npm package, version 1.31.1) |

Total about 22 MB, and the biggest single file is 14.8 MB, under the 25 MiB per-file limit of Cloudflare static assets.

## Deploy (Cloudflare Workers static assets) - live at https://ocr.datmt.com
```bash
npm i -g wrangler && wrangler login      # once
cd ocr-camera && wrangler deploy         # uploads static-app/ and attaches the domain
```
- `../wrangler.jsonc` holds the setup: assets come from `static-app`, and the route `ocr.datmt.com` with `custom_domain: true` makes Cloudflare create the DNS record and HTTPS certificate (the domain's DNS must be on Cloudflare). Change the route to use another subdomain.
- I first ran `wrangler pages project create`; wrangler auto-converted it to this Worker setup, so there is no Pages project. Use `wrangler deploy`, not `wrangler pages deploy`.
- `_headers` sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. They make the page cross-origin isolated, which lets ONNX Runtime use several WebAssembly threads (`index.html` picks `min(4, cores)`). Without them it falls back to 1 thread. Everything the page loads must come from the same origin, or `require-corp` blocks it.
- HTTPS is required for the phone camera, and the custom domain has it.
- Speed (headless Chromium, 16-core desktop, int8 model): single read 86 ms with 1 thread, 51 ms with 2, 34 ms with 4; the 9-angle vote 771 / 437 / 286 ms. Phones are slower, and I have not timed one. The tilt vote is off by default for that reason.
- Checked after deploy: the site returns 200, the `.wasm` file has type `application/wasm`, and both headers are present. Not tested: the camera and OCR on a phone.

## Try it locally
```bash
cd static-app && python3 -m http.server 8000     # then open http://localhost:8000 (camera works on localhost)
```
On a phone you need HTTPS, so use the Cloudflare URL (or the Docker app).

## How I checked it
- **int8 vs fp32** (192 held-out crops of the collection page): fp32 82.3% exact / 97.0% char, int8 82.8% / 97.1%; 99.5% of reads identical. Weights of the LSTM and linear layers are int8; the convolutions stay fp32 (I did not try quantising them). These crops are the only ones I scored: the phone photos had been deleted by then.
- **JavaScript vs Python** (60 held-out crops, int8 model): single read 49/60 right in both, and 59 of 60 reads gave identical text. The 9-angle vote also gave 50/60 in both, with 59 of 60 identical.
- **In a real browser** (headless Firefox 157, 12 crops, 9-angle vote): all 12 reads equal to the Python result. One read takes about 0.65 s on a desktop CPU, single-threaded. A phone is slower, and I have not measured it.
- **Not tested:** the camera page on a phone, and the multi-line option on real photos. In a test with two crops shrunk and stacked into one frame, the splitter found both lines, but the small text was read badly (0 of 20 pairs fully right).

## Differences from the Python reader
- The tilt vote rotates the 768x247 working image with bilinear sampling, instead of rotating the full photo with cubic sampling as `webapp/deskew.py` does. It is faster, and gave the same text in 59 of 60 test reads.
- No saving of reads, no `/display` auto-labelling, no corner-marker mode. Those need a server; use the Docker app for collecting data.
