# ocr-camera: read random passwords from a phone photo of a screen

A small OCR model (CRNN + CTC) that reads one line of password-like text (94 printable ASCII characters, no space) from a phone-camera photo of a screen. It is trained on photos of a web page I wrote, so it works for **HTML password fields shown on a screen**. It is not a general OCR engine.

All strings used for data collection and testing were random throwaway strings, never real passwords.

## Result

Scored on **held-out strings** (the test split is by string, so the model never saw those strings in training).

| Test set | n | This model, exact | char acc. | RapidOCR, exact | char acc. |
|---|---|---|---|---|---|
| All real test photos | 249 | **79.9%** | 97.4% | 37.3% | 75.3% |
| Phone photos of the field page | 153 | **78.4%** | 98.0% | 26.8% | 65.8% |
| Photos of the collection page | 96 | **82.3%** | 96.2% | 54.2% | 91.7% |
| Test crops tilted -8° / +8° | 249 | 74.3% / 75.1% | 96.7% / 96.9% | | |
| Fonts never used in synthetic training | 16 | 81.2% | 97.9% | 68.8% | 93.7% |

"Exact" means the whole string is right. The reader page can also read the photo at 9 small tilts and take a vote (`tilt-robust`): on the held-out phone reads this gives **84.1% exact / 98.6% char** (single read: 81.1% / 98.1%).

For comparison, the first version of the model scored 13% exact on my first 15 phone photos.

### What still goes wrong
- `l` vs `I` is about half of all errors (`l`, `I` and `|` look nearly the same in many sans-serif fonts). The image alone cannot settle it.
- Dropped characters (`'`, `` ` ``, `\`, `/`), and `*` read as `^`.
- Long strings: accuracy falls from 87% (8-15 characters) to 61% at 24 characters.
- `{` and `(` were confused often in my live phone test, but I could only find 2 such errors in 34 held-out cases, so I could not confirm it from the data.
- **Several text lines in one photo.** The model reads one line, so they merged into one string. The reader page now has a *multi-line* option that splits the crop into lines (`webapp/lines.py`). It is tested only on photos I stacked together, **not on real multi-line photos**.

### Limits of the evidence
- All photos come from one phone and one monitor setup.
- The model has seen about 80% of the phone strings in training; only the test split is held out. A different screen or phone has not been tested.
- Dark mode was the weak spot of the first model. I did not score dark and light separately for this one.

## Approach

1. **Collect.** `webapp/server.py` (the collection page) shows a random string with four corner markers; `crop.py` finds the markers and flattens the panel into a text crop. About 1000 photos gave 956 crops. A second page (`/display` on the reader app, `webapp/static/field_display.html`) shows a password field with random fonts, light or dark mode, and auto-capture on the phone labels each photo for free: about 1,430 photos.
2. **Split by string.** 80/10/10 train/val/test, so a test string never appears in training.
3. **Synthetic pretraining.** The generator in `train.ipynb` draws random strings in many fonts on a field or panel, with moiré, blur, glow, lifted blacks, small rotations and resolution loss. Synthetic plus real training beat real-only by about 23 points on the same split in an earlier run, so it stays.
4. **Model.** 6-layer CNN, then 2-layer bidirectional LSTM, then CTC. Input is 80x384 grey, 16.6 MB as fp32. It runs through ONNX Runtime on CPU (about 78 ms per read on one thread; an int8 version is 7.6 MB and 38 ms, see `bench_quant.py`).
5. **Compare** against RapidOCR on the same crops (`eval_rapidocr.py`).
6. **Serve.** `webapp/reader_server.py` reads a guide-box crop from the phone camera, optionally votes over 9 tilts, and can save reads for later review.

## Files

| Path | What |
|---|---|
| `train.ipynb` | whole pipeline: data, synthetic generator, training, evaluation (outputs from the final run are kept) |
| `ocr_model.py` | model, preprocessing, CTC decoding (shared by training and serving) |
| `crop.py`, `build_crops.py` | corner-marker detection and crop building |
| `phone_data.py` | turns saved phone reads into training records |
| `export_onnx.py` | checkpoint to ONNX |
| `webapp/reader_server.py`, `webapp/static/reader.html` | the reader app (phone camera) |
| `webapp/server.py`, `webapp/static/display.html` | the collection app |
| `webapp/lines.py`, `webapp/deskew.py` | line splitting, tilt voting |
| `eval_reader_log.py` | score saved phone reads with the current model |
| `model/` | best model: `crnn_final_synthetic.pt` (PyTorch) and `crnn.onnx` (what the app loads) |

The photos are not included.

## Run the web app with the model (Docker)

```bash
cd ocr-camera
docker build -t ocr-camera .
docker run --rm -p 8444:8444 -v ocr-reads:/logs ocr-camera
```

Open `https://<your-computer-ip>:8444/` **on the phone** and accept the certificate warning (it is a self-signed certificate made at start-up; phone cameras only work on HTTPS pages). Allow the camera, frame the text inside the guide box, and press the shoot button.

- `https://<ip>:8444/display` shows a test password field on a second screen to photograph.
- Ticking "save for review" stores photos and results in the `ocr-reads` volume. These can contain whatever you photographed, so delete it when done: `docker volume rm ocr-reads`.
- To use a different model: `-v /path/to/other.onnx:/app/model/crnn.onnx`.

## Train it yourself

Needs PyTorch with CUDA, the fonts the notebook lists, and your own photos from the collection page. Run `train.ipynb` top to bottom, then `python export_onnx.py data/crnn_final_synthetic.pt model/crnn.onnx`. The notebook expects `data/crops.jsonl` and `data/crops/` (and optionally `data/reader_log/`), which are not in this repo.
