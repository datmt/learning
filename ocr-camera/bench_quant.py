# Usage: /home/dat/jupyter-venv/bin/python bench_quant.py [checkpoint.pt]   (run from this folder; reuses the cells of train.ipynb)
import os, io, time, copy, json
os.environ["SYNTHETIC"] = "0"
import nbformat, numpy as np, torch
nb = nbformat.read("train.ipynb", 4)
code = [c.source for c in nb.cells if c.cell_type == "code"]
g = globals()
for i in (0, 1, 3, 5): exec(code[i], g)
import sys; ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "data/crnn_final_real_only.pt", map_location="cpu")
m = CRNN(); m.load_state_dict(ck["model"]); m.eval()
gt = [r["text"] for r in te]
def cpu_pred(model, items):
    out = []
    with torch.no_grad():
        for x, _, _, _ in loader(RealDS(items), 64, shuffle=False, workers=0): out += decode(model(x).float())
    return out
def size_mb(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return b.tell() / 1e6
x = torch.stack([to_input(te[i]["img"]) for i in range(32)])
def bench(model, xx, runs=40, warm=8):
    with torch.no_grad():
        for _ in range(warm): model(xx)
        t = time.time()
        for _ in range(runs): model(xx)
    return (time.time() - t) / runs * 1000
variants = {"fp32": m}
torch.backends.quantized.engine = "qnnpack"
q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(m), {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
variants["int8 dynamic (LSTM+Linear)"] = q
base = cpu_pred(m, te)
for name, mod in variants.items():
    p = cpu_pred(mod, te); e, c = metrics(gt, p)
    agree = np.mean([a == b for a, b in zip(p, base)])
    row = [f"{name:30} size {size_mb(mod):5.1f} MB  exact {e:.1%} char {c:.1%}  same-as-fp32 {agree:.1%}"]
    for th in (1, 4):
        torch.set_num_threads(th); row.append(f"{th}t b1 {bench(mod, x[:1]):.1f}ms")
    print(" | ".join(row), flush=True)
