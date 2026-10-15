"""Tables and figures for the second round of measurements: bandwidth validation, perplexity, memory lifecycle.

Usage: python3 -I analyze_extra.py <raw1_dir> <raw2_dir> <out_dir>
  raw1 = first-round raw data (speed/, memory.csv ...), raw2 = second round (membw.jsonl, ladder/, ppl/, memlife/ ...)
Every value is labelled MEASURED, DERIVED (arithmetic on measured values) or THEORETICAL.
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

R1, R2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
IMG = os.path.join(OUT, "images")
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})
tables = []


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ bandwidth microbenchmark
bw = [json.loads(l) for l in open(os.path.join(R2, "membw.jsonl"))]
rows = []
for r in bw:
    rows.append([r["test"], f"{r['buffer_GiB']:.2f}", r["iters"], f"{r['GBps_median']:.0f}", f"{r['GBps_min']:.0f}", f"{r['GBps_max']:.0f}",
                 f"{r['GiBps_median']:.0f}", f"{r['GBps_median'] / 273 * 100:.0f}%"])
tables.append("### Memory-bandwidth microbenchmark (MEASURED)\n\n" + md(
    ["Kernel", "Buffer (GiB)", "Iters", "Median GB/s", "Min GB/s", "Max GB/s", "Median GiB/s", "% of 273 GB/s spec"], rows))

read = {r["buffer_GiB"]: r for r in bw if r["test"] == "read"}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for t, c in (("read", "#4C78A8"), ("write", "#F58518"), ("copy", "#54A24B")):
    pts = sorted((r["buffer_GiB"], r["GBps_median"], r["GBps_min"], r["GBps_max"]) for r in bw if r["test"] == t)
    ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], marker="o", label=t, color=c, capsize=3)
ax.axhline(273, color="gray", ls=":", lw=1)
ax.text(0.07, 276, "273 GB/s (published peak)", fontsize=8, color="gray")
ax.set_xscale("log", base=2)
ax.set_xlabel("buffer size (GiB)")
ax.set_ylabel("sustained GB/s (median, bars = min/max)")
ax.set_ylim(0, 300)
ax.set_title("Streaming-kernel bandwidth on GB10 (MEASURED)")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "07_bandwidth_microbenchmark.png")); plt.close(fig)

# ------------------------------------------------------------------ predicted vs observed decode speed (dense models)
streamed = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(R2, "streamed_bytes_dense.jsonl"))}


def tg_from(path):
    for l in open(path):
        d = json.loads(l)
        if d["n_gen"] == 128 and d["n_depth"] == 0:
            return d["avg_ts"], d["stddev_ts"], len(d["samples_ts"])


def nearest_read(gib):
    k = min(read, key=lambda s: abs(__import__("math").log(s / gib)))
    return k, read[k]["GiBps_median"]


cases = [
    ("Gemma 3 4B Q4_K_M", "gemma-3-4b-it-Q4_K_M.gguf", os.path.join(R1, "speed", "gemma-3-4b-Q4_K_M.jsonl")),
    ("Gemma 3 12B Q4_K_M", "gemma3-12b-Q4_K_M.gguf", os.path.join(R1, "speed", "gemma3-12b-Q4_K_M.jsonl")),
    ("Gemma 3 12B Q8_0", "gemma3-12b-Q8_0.gguf", os.path.join(R1, "speed", "gemma3-12b-Q8_0.jsonl")),
    ("Qwen3 8B Q4_K_M", "Qwen3-8B-Q4_K_M.gguf", os.path.join(R2, "ladder", "Qwen3-8B-Q4_K_M.jsonl")),
    ("Qwen3 8B Q8_0", "Qwen3-8B-Q8_0.gguf", os.path.join(R2, "ladder", "Qwen3-8B-Q8_0.jsonl")),
    ("Qwen3 8B BF16", "Qwen3-8B-BF16.gguf", os.path.join(R2, "ladder", "Qwen3-8B-BF16.jsonl")),
    ("Qwen3.8 27B Q4_K_M", "Qwen3.8-27B-UD-Q4_K_M.gguf", os.path.join(R1, "speed", "qwen3.8-27b-Q4_K_M.jsonl")),
]
rows, pts = [], []
for label, gg, sp in cases:
    s = streamed[gg]
    g = s["streamed_GiB"]
    buf, rbw = nearest_read(g)
    pred = rbw / g
    obs, sd, n = tg_from(sp)
    eff = obs * g
    rows.append([label, f"{s['tensor_GiB']:.2f}", f"{g:.2f}", "tied" if s["output_head_tied"] else "untied", f"{rbw:.0f} (@{buf:g} GiB)", f"{pred:.1f}",
                 f"{obs:.1f} ± {sd:.1f} (n={n})", f"{eff:.0f}", f"{obs / pred * 100:.0f}%"])
    pts.append((label, g, pred, obs))
tables.append("### Decode speed: theoretical prediction vs observation, dense models\n\n"
              "Columns: weights file (MEASURED), bytes streamed per token (DERIVED from the GGUF tensor table), microbenchmark read bandwidth at the nearest buffer size (MEASURED), "
              "predicted tok/s = bandwidth ÷ bytes (THEORETICAL), observed tok/s (MEASURED), effective bandwidth = observed × bytes (DERIVED, not a hardware measurement), observed ÷ predicted.\n\n"
              + md(["Model", "Tensors (GiB)", "Streamed/token (GiB)", "Output head", "Microbench read GiB/s", "Predicted tok/s", "Observed tok/s", "Effective GiB/s", "Obs ÷ pred"], rows))

fig, ax = plt.subplots(figsize=(6.4, 5.4))
lim = [8, 120]
ax.plot(lim, lim, color="gray", lw=1)
ax.fill_between(lim, [l * 0.9 for l in lim], [l * 1.1 for l in lim], color="gray", alpha=0.12, label="±10% of prediction")
for label, g, pred, obs in pts:
    ax.scatter([pred], [obs], s=50, color="#4C78A8", zorder=3)
    ax.annotate(label, (pred, obs), textcoords="offset points", xytext=(6, -4), fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted tok/s = measured read bandwidth ÷ bytes streamed per token")
ax.set_ylabel("observed tok/s (llama-bench tg128)")
ax.set_title("Bandwidth model vs. reality, dense models")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "08_prediction_vs_observed.png")); plt.close(fig)

# ------------------------------------------------------------------ MoE: INFERRED bytes per token
moe = [("gpt-oss 20B MXFP4", "gpt-oss-20b-MXFP4", 11.3, 20.9), ("Gemma 4 26B-A4B Q4_K_M", "gemma-4-26B-A4B-Q4_K_M", 15.8, 25.2),
       ("Qwen3.6 35B-A3B Q4_K_M", "qwen3.6-35B-A3B-Q4_K_M", 20.6, 34.7)]
lo_bw, hi_bw = read[16.0]["GiBps_median"], read[8.0]["GiBps_median"]
rows = []
for label, key, file_gib, params in moe:
    tg, sd, n = tg_from(os.path.join(R1, "speed", key + ".jsonl"))
    rows.append([label, f"{file_gib:.1f}", f"{tg:.1f} ± {sd:.1f} (n={n})", f"{lo_bw / tg:.1f} – {hi_bw / tg:.1f}", f"{lo_bw / tg / file_gib * 100:.0f}% – {hi_bw / tg / file_gib * 100:.0f}%"])
tables.append("### MoE models: bytes touched per token INFERRED from speed (not measured)\n\n"
              f"Inferred = measured read bandwidth ({lo_bw:.0f}–{hi_bw:.0f} GiB/s, microbenchmark at 16 and 8 GiB) ÷ observed tok/s. This is an *inference*: it assumes the kernels run at the microbenchmark's bandwidth, "
              "and it lumps together routed experts, always-read tensors (attention, shared experts, output head) and any cache effects. Treat it as an order of magnitude, not a traffic measurement.\n\n"
              + md(["Model", "File (GiB)", "Observed tok/s", "Inferred GiB/token", "Inferred share of file read per token"], rows))

# ------------------------------------------------------------------ perplexity
ppl = {}
for f in sorted(glob.glob(os.path.join(R2, "ppl", "*.log"))):
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", open(f).read())
    if m:
        ppl[os.path.basename(f)[:-4]] = (float(m.group(1)), float(m.group(2)))
groups = {"Qwen3 8B": ("qwen3-8b-BF16", [("BF16", "qwen3-8b-BF16"), ("Q8_0", "qwen3-8b-Q8_0"), ("Q4_K_M", "qwen3-8b-Q4_K_M")]),
          "Gemma 3 12B": ("gemma3-12b-Q8_0", [("Q8_0", "gemma3-12b-Q8_0"), ("Q4_K_M", "gemma3-12b-Q4_K_M")])}
sizes = {"qwen3-8b-BF16": 15.26, "qwen3-8b-Q8_0": 8.11, "qwen3-8b-Q4_K_M": 4.68, "gemma3-12b-Q8_0": 11.65, "gemma3-12b-Q4_K_M": 6.79}
rows = []
for g, (ref, items) in groups.items():
    for q, k in items:
        if k in ppl:
            p, e = ppl[k]
            d = (p / ppl[ref][0] - 1) * 100
            rows.append([g, q, f"{sizes[k]:.1f}", f"{p:.4f} ± {e:.3f}", "reference" if k == ref else f"{d:+.2f}%"])
tables.append("### Perplexity, wikitext-2 test set, 2048-token windows (MEASURED)\n\n" + md(["Model", "Quant", "Weights (GiB)", "PPL ± stat. error", "Δ vs reference"], rows))

# ------------------------------------------------------------------ memory lifecycle
ML = os.path.join(R2, "memlife")
PHASES = ["baseline", "load", "idle_after_load", "warmup", "prefill_long", "generation", "concurrent", "max_context", "idle_after"]
names = {"qwen3.6-35b-a3b-Q4_K_M": "Qwen3.6 35B-A3B Q4 (131k ctx)", "qwen3.8-27b-Q4_K_M": "Qwen3.8 27B Q4 (131k ctx)",
         "gpt-oss-20b-MXFP4": "gpt-oss 20B MXFP4 (131k ctx)", "qwen3-8b-Q4_K_M": "Qwen3 8B Q4 (32k ctx)"}
sums = {}
for f in sorted(glob.glob(os.path.join(ML, "*.summary.json"))):
    sums[os.path.basename(f)[:-13]] = json.load(open(f))
if sums:
    rows = []
    for k, s in sums.items():
        base = s["baseline_sys_used_MiB"]
        ph = s["phases"]
        load_gpu = ph["load"]["gpu_proc_MiB_peak"] / 1024
        pk_gpu = max(p["gpu_proc_MiB_peak"] or 0 for p in ph.values()) / 1024
        pk_sys = max(p["sys_used_MiB_peak"] or 0 for p in ph.values())
        rows.append([names.get(k, k), f"{load_gpu:.1f}", f"{pk_gpu:.1f}", f"{ph['idle_after']['gpu_proc_MiB_peak'] / 1024:.1f}",
                     f"{ph['idle_after_load']['rss_MiB_peak'] / 1024:.1f}", f"{max(p['rss_MiB_peak'] or 0 for p in ph.values()) / 1024:.1f}",
                     f"{(pk_sys - base) / 1024:+.1f}", f"{min(p['sys_avail_MiB_min'] or 1e9 for p in ph.values()) / 1024:.0f}",
                     f"{max(p['psi_mem_some_avg10_max'] or 0 for p in ph.values()):.1f}", "yes" if s.get("all_requests_ok") else "NO"])
    tables.append("### Memory over the process lifecycle, 1 server with 4 slots (MEASURED; GiB unless noted)\n\n"
                  "`GPU proc` = nvidia-smi per-process memory (on unified memory this is *not* isolated VRAM). `RSS` = host-resident memory of the process. "
                  "`Δ system used` = peak `free` 'used' minus the pre-launch baseline (includes everything on the machine, not only the server). "
                  "`PSI` = peak kernel memory-pressure stall, % of time (10 s average).\n\n"
                  + md(["Model", "GPU proc at load", "GPU proc peak", "GPU proc after workload", "RSS after load", "RSS peak", "Δ system used (peak)", "Min available", "Peak PSI %", "All requests OK"], rows))
    rows = []
    for k, s in sums.items():
        t = s["tokens"]
        ph = s["phases"]
        for p in ("idle_after_load", "prefill_long", "generation", "concurrent", "max_context", "idle_after"):
            if p in ph:
                d = ph[p]
                rows.append([names.get(k, k), p, d["samples"], f"{(d['gpu_proc_MiB_peak'] or 0) / 1024:.2f}", f"{(d['rss_MiB_peak'] or 0) / 1024:.2f}",
                             f"{(d['sys_used_MiB_peak'] - s['baseline_sys_used_MiB']) / 1024:+.2f}"])
    tables.append("### Per-phase peaks (MEASURED)\n\n" + md(["Model", "Phase", "Samples", "GPU proc peak (GiB)", "RSS peak (GiB)", "Δ system used peak (GiB)"], rows))

    fig, axes = plt.subplots(len(sums), 1, figsize=(10, 2.6 * len(sums)), sharex=False)
    if len(sums) == 1:
        axes = [axes]
    cols = {"baseline": "#DDDDDD", "load": "#F2CF5B", "idle_after_load": "#EEEEEE", "warmup": "#EEEEEE", "prefill_long": "#BFD7EA",
            "generation": "#C8E6C9", "concurrent": "#F8C8C8", "max_context": "#D7BDE2", "idle_after": "#EEEEEE"}
    for ax, (k, s) in zip(axes, sums.items()):
        rows_ = list(csv.DictReader(open(os.path.join(ML, k + ".samples.csv"))))
        t = [float(r["t"]) for r in rows_]
        gpu = [float(r["gpu_proc_MiB"]) / 1024 if r["gpu_proc_MiB"] else None for r in rows_]
        rss = [float(r["rss_MiB"]) / 1024 if r["rss_MiB"] else None for r in rows_]
        base = s["baseline_sys_used_MiB"]
        sysd = [(float(r["sys_used_MiB"]) - base) / 1024 for r in rows_]
        # shade phases
        start = 0
        for i in range(1, len(rows_) + 1):
            if i == len(rows_) or rows_[i]["phase"] != rows_[start]["phase"]:
                ax.axvspan(t[start], t[i - 1], color=cols.get(rows_[start]["phase"], "#EEE"), alpha=0.55, lw=0)
                if t[i - 1] - t[start] > 12:
                    ax.text((t[start] + t[i - 1]) / 2, ax.get_ylim()[1] if False else 0.02, rows_[start]["phase"].replace("_", "\n"), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
                start = i
        ax.plot(t, [g if g is not None else float("nan") for g in gpu], color="#1f4e79", lw=1.6, label="GPU proc (nvidia-smi)")
        ax.plot(t, [r if r is not None else float("nan") for r in rss], color="#C0504D", lw=1.2, label="process RSS")
        ax.plot(t, sysd, color="#555555", lw=1, ls="--", label="Δ system used vs baseline")
        ax.set_ylabel("GiB"); ax.set_title(names.get(k, k), fontsize=9, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    axes[-1].set_xlabel("seconds since sampler start")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "09_memory_lifecycle.png")); plt.close(fig)

open(os.path.join(OUT, "tables_extra.md"), "w").write("\n\n".join(tables) + "\n")
print("\n\n".join(tables))
