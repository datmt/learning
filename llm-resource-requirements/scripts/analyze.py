"""Build tables (markdown) and figures from raw benchmark output.

Usage: python3 -I analyze.py <raw_dir> <out_dir>
Writes <out_dir>/images/*.png and <out_dir>/tables.md
"""
import csv
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW, OUT = sys.argv[1], sys.argv[2]
IMG = os.path.join(OUT, "images")
os.makedirs(IMG, exist_ok=True)

# name -> (label, kind, total params B, active params B (None = dense), quant)
MODELS = {
    "gemma-3-4b-Q4_K_M": ("Gemma 3 4B", "dense", "Q4_K_M"),
    "gemma3-12b-Q4_K_M": ("Gemma 3 12B", "dense", "Q4_K_M"),
    "gemma3-12b-Q8_0": ("Gemma 3 12B", "dense", "Q8_0"),
    "qwen3-8b-BF16": ("Qwen3 8B", "dense", "BF16"),
    "qwen3.8-27b-Q4_K_M": ("Qwen3.8 27B", "dense (hybrid)", "Q4_K_M"),
    "gpt-oss-20b-MXFP4": ("gpt-oss 20B", "MoE", "MXFP4"),
    "gemma-4-26B-A4B-Q4_K_M": ("Gemma 4 26B-A4B", "MoE", "Q4_K_M"),
    "qwen3.6-35B-A3B-Q4_K_M": ("Qwen3.6 35B-A3B", "MoE (hybrid)", "Q4_K_M"),
}
COLORS = {"dense": "#4C78A8", "dense (hybrid)": "#72B7B2", "MoE": "#F58518", "MoE (hybrid)": "#E45756"}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

# ---------------------------------------------------------------- speed
speed = {}  # name -> {(test, depth): (mean, std)} + size
for key in MODELS:
    path = os.path.join(RAW, "speed", key + ".jsonl")
    d = {}
    size = params = None
    for line in open(path):
        r = json.loads(line)
        test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        d[(test, r["n_depth"])] = (r["avg_ts"], r["stddev_ts"])
        size, params = r["model_size"] / 2**30, r["model_n_params"] / 1e9
    speed[key] = {"t": d, "size": size, "params": params}

tables = []


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


rows = []
for k, (label, kind, quant) in MODELS.items():
    s = speed[k]
    t = s["t"]
    g = lambda test, dep: f"{t[(test, dep)][0]:,.0f}" if (test, dep) in t and t[(test, dep)][0] >= 100 else (f"{t[(test, dep)][0]:.1f}" if (test, dep) in t else "–")
    rows.append([label, kind, quant, f"{s['params']:.1f}", f"{s['size']:.1f}",
                 g("pp2048", 0), g("tg128", 0), g("pp2048", 16384), g("tg128", 16384), g("pp2048", 65536), g("tg128", 65536)])
tables.append("### Speed\n\n" + md(
    ["Model", "Type", "Quant", "Params (B)", "File (GiB)", "pp @0", "tg @0", "pp @16k", "tg @16k", "pp @64k", "tg @64k"], rows))

# fig 1: generation speed bars
order = sorted(MODELS, key=lambda k: -speed[k]["t"][("tg128", 0)][0])
fig, ax = plt.subplots(figsize=(8, 4.2))
vals = [speed[k]["t"][("tg128", 0)][0] for k in order]
bars = ax.barh([f"{MODELS[k][0]} {MODELS[k][2]}" for k in order], vals, color=[COLORS[MODELS[k][1]] for k in order])
ax.invert_yaxis()
for b, v in zip(bars, vals):
    ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center")
ax.set_xlabel("generation speed, tokens/s (tg128, empty context)")
ax.set_title("Token generation speed on GB10")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], labels=list(COLORS), loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "01_generation_speed.png")); plt.close(fig)

# fig 2: prompt processing bars
order2 = sorted(MODELS, key=lambda k: -speed[k]["t"][("pp2048", 0)][0])
fig, ax = plt.subplots(figsize=(8, 4.2))
vals = [speed[k]["t"][("pp2048", 0)][0] for k in order2]
bars = ax.barh([f"{MODELS[k][0]} {MODELS[k][2]}" for k in order2], vals, color=[COLORS[MODELS[k][1]] for k in order2])
ax.invert_yaxis()
for b, v in zip(bars, vals):
    ax.text(v + 40, b.get_y() + b.get_height() / 2, f"{v:,.0f}", va="center")
ax.set_xlabel("prompt processing, tokens/s (pp2048, empty context)")
ax.set_title("Prompt processing (prefill) speed on GB10")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "02_prompt_speed.png")); plt.close(fig)

# fig 3: speed retained vs depth (relative to empty context) -- direct labels
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, test, title in ((axes[0], "tg128", "Generation speed retained"), (axes[1], "pp2048", "Prompt-processing speed retained")):
    for k, (label, kind, quant) in MODELS.items():
        t = speed[k]["t"]
        pts = sorted((dep, v[0]) for (tt, dep), v in t.items() if tt == test)
        base = pts[0][1]
        ax.plot([p[0] / 1024 for p in pts], [p[1] / base * 100 for p in pts], marker="o", color=COLORS[kind], alpha=0.9,
                ls="--" if quant in ("Q8_0", "BF16") else "-")
        ax.annotate(f"{label} {quant}", (pts[-1][0] / 1024, pts[-1][1] / base * 100), textcoords="offset points",
                    xytext=(5, -3), fontsize=6.5, color=COLORS[kind])
    ax.set_title(title + " at depth"); ax.set_xlabel("tokens already in context (k)"); ax.set_xticks([0, 16, 64]); ax.set_xlim(-2, 100)
    ax.set_ylim(55, 102)
axes[0].set_ylabel("% of empty-context speed")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "03_speed_vs_depth.png")); plt.close(fig)

# (bandwidth validation now lives in analyze_extra.py, using a measured microbenchmark instead of an inferred ceiling)

# ---------------------------------------------------------------- memory
mem = list(csv.DictReader(open(os.path.join(RAW, "memory.csv"))))
NAMES = {"gemma-3-4b-it-Q4_K_M.gguf": "Gemma 3 4B Q4", "gemma3-12b-Q4_K_M.gguf": "Gemma 3 12B Q4", "gpt-oss-20b-MXFP4.gguf": "gpt-oss 20B MXFP4",
         "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf": "Gemma 4 26B-A4B Q4", "Qwen3.8-27B-UD-Q4_K_M.gguf": "Qwen3.8 27B Q4 (hybrid)",
         "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf": "Qwen3.6 35B-A3B Q4 (hybrid)", "Qwen3-8B-BF16.gguf": "Qwen3 8B BF16"}
series = {}
for r in mem:
    if r["kv_type"] == "f16":
        series.setdefault(r["model"], {})[int(r["ctx"])] = int(r["gpu_mib"]) / 1024
fig, ax = plt.subplots(figsize=(8.5, 5))
rows = []
for m, d in series.items():
    xs = sorted(d)
    ax.plot([x / 1024 for x in xs], [d[x] for x in xs], marker="o", label=NAMES[m])
    lo, hi = xs[0], xs[-1]
    slope = (d[hi] - d[lo]) * 1024 * 1024 / (hi - lo)  # KiB per token
    rows.append([NAMES[m]] + [f"{d[c]:.1f}" if c in d else "–" for c in (4096, 32768, 131072)] + [f"{slope:.0f}"])
ax.set_xlabel("context window (k tokens)"); ax.set_ylabel("process memory, GiB (nvidia-smi)")
ax.set_xscale("log", base=2); ax.set_xticks([4, 32, 128]); ax.set_xticklabels(["4k", "32k", "128k"])
ax.set_title("Memory footprint vs. context window (1 slot, f16 KV cache)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "05_memory_vs_context.png")); plt.close(fig)
tables.append("### Process memory (GiB) vs context, measured\n\n" + md(["Model", "4k", "32k", "128k", "KV cost (KiB/token)"], rows))

# KV quantization
kvrows = []
for r in mem:
    if r["model"] == "gemma3-12b-Q4_K_M.gguf" and r["ctx"] == "131072":
        kvrows.append([r["kv_type"], f"{int(r['gpu_mib']) / 1024:.1f}"])
tables.append("### KV-cache quantization, Gemma 3 12B @128k\n\n" + md(["KV type", "Process memory (GiB)"], kvrows))

# ---------------------------------------------------------------- concurrency
conc_dir = os.path.join(RAW, "conc")
if os.path.isdir(conc_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    crow = []
    for f in sorted(glob.glob(os.path.join(conc_dir, "*.txt"))):
        name = os.path.basename(f)[:-4]
        pts = []
        for line in open(f):
            c = [x.strip() for x in line.split("|")]
            if len(c) > 10 and c[1].isdigit() and c[2].isdigit():
                # PP, TG, B, N_KV, T_PP, S_PP, T_TG, S_TG, T, S
                pp, tg, b = int(c[1]), int(c[2]), int(c[3])
                s_tg = float(c[8])
                pts.append((b, s_tg))
                crow.append([name, b, f"{s_tg:.1f}", f"{s_tg / b:.1f}"])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
    ax.set_xlabel("parallel streams"); ax.set_ylabel("aggregate generation tokens/s")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_title("Aggregate throughput vs. concurrent requests"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "06_concurrency.png")); plt.close(fig)
    tables.append("### Concurrency (aggregate tg)\n\n" + md(["Model", "Streams", "Aggregate t/s", "Per-stream t/s"], crow))

open(os.path.join(OUT, "tables.md"), "w").write("\n\n".join(tables) + "\n")
print("\n\n".join(tables))
