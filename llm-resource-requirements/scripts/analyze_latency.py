"""Tables and figures for the request-rate sweep (see bench_latency.sh / load_test.py).

Usage: python3 -I analyze_latency.py <latency_raw_dir> <out_dir>

SLOs (stated up front in bench_latency.sh, not tuned to the results):
  chat: P95 TTFT <= 2 s   AND P95 ITL <= 100 ms AND no failures
  long: P95 TTFT <= 60 s  AND P95 ITL <= 200 ms AND no failures
A rate "meets the SLO" only if percentiles are valid (>= 20 completed requests) and every condition holds.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW, OUT = sys.argv[1], sys.argv[2]
IMG = os.path.join(OUT, "images")
os.makedirs(IMG, exist_ok=True)
SLO = {"chat": {"ttft": 2.0, "itl": 0.100}, "long": {"ttft": 60.0, "itl": 0.200}, "mixed": {"ttft": 60.0, "itl": 0.200}}
NAMES = {"qwen3.6-35b-a3b": "Qwen3.6 35B-A3B (MoE)", "qwen3.8-27b": "Qwen3.8 27B (dense)"}
COL = {"qwen3.6-35b-a3b": "#E45756", "qwen3.8-27b": "#72B7B2"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

runs = []
for f in sorted(glob.glob(os.path.join(RAW, "*.summary.json"))):
    name = os.path.basename(f)[:-13]  # <model>.<profile>.r<rate>; the model name itself contains a dot, and so does the rate
    model, profile = name.split(".chat.")[0] if ".chat." in name else None, None
    for p_ in ("chat", "long", "mixed"):
        if f".{p_}.r" in name:
            model, profile = name.split(f".{p_}.r")[0], p_
    s = json.load(open(f))
    s["model"], s["name"] = model, name
    runs.append(s)


def verdict(s):
    slo = SLO[s["profile"]]
    if not s["percentiles_valid"]:
        return "n/a (n<20)"
    ok = (s["failed"] == 0 and s["ttft_s_p95"] is not None and s["ttft_s_p95"] <= slo["ttft"]
          and (s["itl_s_p95"] is None or s["itl_s_p95"] <= slo["itl"]))
    return "PASS" if ok else "FAIL"


def f(x, nd=2, scale=1.0):
    return "–" if x is None else f"{x * scale:.{nd}f}"


tables = []
for profile in ("chat", "long", "mixed"):
    rows = []
    for s in sorted((r for r in runs if r["profile"] == profile), key=lambda r: (r["model"], r["rate_req_s"])):
        rows.append([NAMES.get(s["model"], s["model"]), s["rate_req_s"], int(s["duration_s"]), s["scheduled"], s["completed"], s["failed"],
                     f(s["scheduled"] / s["duration_s"], 3), f(s["output_tok_s"], 1),
                     f(s["ttft_s_p50"], 2), f(s["ttft_s_p95"], 2), f(s["ttft_s_p99"], 2),
                     f(s["itl_s_p50"], 0, 1000), f(s["itl_s_p95"], 0, 1000),
                     f(s["e2e_s_p50"], 1), f(s["e2e_s_p95"], 1),
                     f((s.get("peak_system_used_MiB") or 0) / 1024 if s.get("peak_system_used_MiB") else None, 1), verdict(s)])
    if rows:
        hdr = ["Model", "Offered req/s", "Dur (s)", "Sent", "Done", "Failed", "Realized offered req/s", "Out tok/s", "TTFT p50 (s)", "TTFT p95 (s)", "TTFT p99 (s)",
               "ITL p50 (ms)", "ITL p95 (ms)", "E2E p50 (s)", "E2E p95 (s)", "Peak sys used (GiB)", f"SLO"]
        t = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(hdr))) + "|"]
        t += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        slo = SLO[profile]
        tables.append(f"### Load test: {profile} profile (MEASURED)\n\nSLO: P95 TTFT ≤ {slo['ttft']:g} s, P95 ITL ≤ {slo['itl'] * 1000:g} ms, zero failures. "
                      "Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.\n\n" + "\n".join(t))

# capacity summary: highest tested rate that passes, per model and profile
rows = []
for profile in ("chat", "long", "mixed"):
    for m in NAMES:
        rs = sorted((r for r in runs if r["profile"] == profile and r["model"] == m), key=lambda r: r["rate_req_s"])
        if not rs:
            continue
        passing = [r for r in rs if verdict(r) == "PASS"]
        failing = [r for r in rs if verdict(r) == "FAIL"]
        na = [r for r in rs if verdict(r).startswith("n/a")]
        best = passing[-1]["rate_req_s"] if passing else None
        first_fail = failing[0]["rate_req_s"] if failing else None
        rows.append([profile, NAMES[m], ", ".join(str(r["rate_req_s"]) for r in rs), f"{best} req/s" if best is not None else "none of the tested rates",
                     first_fail if first_fail is not None else "–", ", ".join(str(r["rate_req_s"]) for r in na) or "–"])
tables.append("### Highest tested rate meeting the SLO (MEASURED capacity point, not a theoretical maximum)\n\n"
              "| Profile | Model | Rates tested (req/s) | Highest passing | First failing | Rates with too few samples |\n|---|---|---|---:|---:|---:|\n"
              + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows))

# figures: chat + long, p95 TTFT and p95 ITL vs offered rate
for profile in ("chat", "long"):
    rs = [r for r in runs if r["profile"] == profile]
    if not rs:
        continue
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for m in NAMES:
        mr = sorted((r for r in rs if r["model"] == m), key=lambda r: r["rate_req_s"])
        if not mr:
            continue
        x = [r["rate_req_s"] for r in mr]
        v = [r for r in mr if r["percentiles_valid"]]
        axes[0].plot([r["rate_req_s"] for r in v], [r["ttft_s_p95"] for r in v], marker="o", color=COL[m], label=NAMES[m])
        axes[1].plot([r["rate_req_s"] for r in mr if r["itl_s_p95"] is not None], [r["itl_s_p95"] * 1000 for r in mr if r["itl_s_p95"] is not None], marker="o", color=COL[m], label=NAMES[m])
        axes[2].plot([r["scheduled"] / r["duration_s"] for r in mr], [r["completed"] / r["duration_s"] for r in mr], marker="o", color=COL[m], label=NAMES[m])
    slo = SLO[profile]
    axes[0].axhline(slo["ttft"], color="gray", ls="--", lw=1); axes[0].set_title("P95 time to first token (s)"); axes[0].set_yscale("log")
    axes[1].axhline(slo["itl"] * 1000, color="gray", ls="--", lw=1); axes[1].set_title("P95 inter-token latency (ms)")
    lim = max([r["rate_req_s"] for r in rs] + [0.1])
    axes[2].plot([0, lim], [0, lim], color="gray", ls=":", lw=1); axes[2].set_title("Completed vs realized offered (req/s)")
    for a in axes:
        a.set_xlabel("offered load (req/s)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{profile} profile: dashed line = SLO", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, f"10_latency_{profile}.png")); plt.close(fig)

open(os.path.join(OUT, "tables_latency.md"), "w").write("\n\n".join(tables) + "\n")
print("\n\n".join(tables))
