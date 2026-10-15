"""Tables and figures for the request-rate sweep (see bench_latency.sh / load_test.py).

Usage: python3 -I analyze_latency.py <latency_raw_dir> <out_dir> [<extra_raw_dir> ...]
  Extra dirs hold repeated runs (bench_latency_repeat.sh, file names end in .s<seed>). Runs without a seed suffix are seed 1.

SLOs (stated up front in bench_latency.sh, not tuned to the results):
  chat: P95 TTFT <= 2 s   AND P95 ITL <= 100 ms AND no failures
  long: P95 TTFT <= 60 s  AND P95 ITL <= 200 ms AND no failures
A run "meets the SLO" only if percentiles are valid (>= 20 completed requests) and every condition holds.
A RATE is reported as passing only if every trial at that rate passed; it is "mixed" if some passed and some failed.
No confidence interval is computed: with 3-5 trials per rate the honest summary is k passing of n, plus the range of P95 TTFT.
"""
import re
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW, OUT, EXTRA = sys.argv[1], sys.argv[2], sys.argv[3:]
IMG = os.path.join(OUT, "images")
os.makedirs(IMG, exist_ok=True)
SLO = {"chat": {"ttft": 2.0, "itl": 0.100}, "long": {"ttft": 60.0, "itl": 0.200}, "mixed": {"ttft": 60.0, "itl": 0.200}}
NAMES = {"qwen3.6-35b-a3b": "Qwen3.6 35B-A3B (MoE)", "qwen3.8-27b": "Qwen3.8 27B (dense)"}
COL = {"qwen3.6-35b-a3b": "#E45756", "qwen3.8-27b": "#72B7B2"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

NAME_RE = re.compile(r"^(?P<model>.+?)\.(?P<profile>chat|long|mixed)\.r(?P<rate>[0-9.]+?)(?:\.s(?P<seed>\d+))?$")
runs = []
for d_ in [RAW] + EXTRA:
    for f in sorted(glob.glob(os.path.join(d_, "*.summary.json"))):
        name = os.path.basename(f)[:-13]
        m_ = NAME_RE.match(name)
        if not m_:
            raise SystemExit(f"cannot parse run name {name!r}")
        s = json.load(open(f))
        s["model"], s["name"], s["seed"] = m_["model"], name, int(m_["seed"] or s.get("seed", 1))
        if s["profile"] != m_["profile"] or abs(s["rate_req_s"] - float(m_["rate"])) > 1e-9:
            raise SystemExit(f"{name}: file name disagrees with summary contents")
        s["realized"] = s["scheduled"] / s["duration_s"]
        s["source"] = os.path.basename(os.path.normpath(d_))
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
    for s in sorted((r for r in runs if r["profile"] == profile), key=lambda r: (r["model"], r["rate_req_s"], r["seed"])):
        rows.append([NAMES.get(s["model"], s["model"]), s["rate_req_s"], s["seed"], int(s["duration_s"]), s["scheduled"], s["completed"], s["failed"],
                     f(s["scheduled"] / s["duration_s"], 3), f(s["output_tok_s"], 1),
                     f(s["ttft_s_p50"], 2), f(s["ttft_s_p95"], 2), f(s["ttft_s_p99"], 2),
                     f(s["itl_s_p50"], 0, 1000), f(s["itl_s_p95"], 0, 1000),
                     f(s["e2e_s_p50"], 1), f(s["e2e_s_p95"], 1),
                     f((s.get("peak_system_used_MiB") or 0) / 1024 if s.get("peak_system_used_MiB") else None, 1), verdict(s)])
    if rows:
        hdr = ["Model", "Offered req/s", "Seed", "Dur (s)", "Sent", "Done", "Failed", "Realized offered req/s", "Out tok/s", "TTFT p50 (s)", "TTFT p95 (s)", "TTFT p99 (s)",
               "ITL p50 (ms)", "ITL p95 (ms)", "E2E p50 (s)", "E2E p95 (s)", "Peak sys used (GiB)", f"SLO"]
        t = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(hdr))) + "|"]
        t += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        slo = SLO[profile]
        tables.append(f"### Load test: {profile} profile (MEASURED)\n\nSLO: P95 TTFT ≤ {slo['ttft']:g} s, P95 ITL ≤ {slo['itl'] * 1000:g} ms, zero failures. "
                      "Percentiles are shown only when ≥ 20 requests completed; `n/a` means too few samples to judge.\n\n" + "\n".join(t))

# per-rate aggregate over independent trials (seeds), and the capacity bracket
def med(xs):
    xs = sorted(xs)
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)


def by_rate(profile, m):
    d = {}
    for r in runs:
        if r["profile"] == profile and r["model"] == m:
            d.setdefault(r["rate_req_s"], []).append(r)
    return dict(sorted(d.items()))


def rate_status(trials):
    v = [verdict(t) for t in trials]
    np_, nf = v.count("PASS"), v.count("FAIL")
    valid = np_ + nf  # trials with fewer than 20 completions have no verdict and are not counted either way
    if valid == 0:
        return "no verdict"
    if nf == 0:
        return "PASS (all valid trials)"
    if np_ == 0:
        return "FAIL (all valid trials)"
    return "MIXED"


agg_rows, cap_rows = [], []
for profile in ("chat", "long", "mixed"):
    for m in NAMES:
        br = by_rate(profile, m)
        if not br:
            continue
        for rate, ts in br.items():
            v = [verdict(t) for t in ts]
            p95 = [t["ttft_s_p95"] for t in ts if t["ttft_s_p95"] is not None]
            reals = [t["realized"] for t in ts]
            agg_rows.append([profile, NAMES[m], rate, len(ts), f"{min(reals):.3f}–{max(reals):.3f}", f"{v.count('PASS')}/{v.count('PASS') + v.count('FAIL')}" + (f" (+{len(ts) - v.count('PASS') - v.count('FAIL')} n<20)" if len(ts) != v.count('PASS') + v.count('FAIL') else ""),
                             sum(t["failed"] for t in ts), (f"{med(p95):.2f} ({min(p95):.2f}–{max(p95):.2f})" if p95 else "–"), rate_status(ts)])
        status = {rate: rate_status(ts) for rate, ts in br.items()}
        # highest passing rate = highest all-pass rate below the first rate that fails (or is mixed); rates without a verdict (too few samples) are skipped
        highest_pass, first_not_pass = None, None
        for rate, st in status.items():
            if st == "no verdict":
                continue
            if first_not_pass is None:
                if st.startswith("PASS"):
                    highest_pass = rate
                else:
                    first_not_pass = rate
        mixed = [rate for rate, st in status.items() if st == "MIXED"]
        nov = [rate for rate, st in status.items() if st == "no verdict"]
        cap_rows.append([profile, NAMES[m], ", ".join(f"{r} (n={len(br[r])})" for r in br),
                         f"{highest_pass} req/s" if highest_pass is not None else "none tested",
                         first_not_pass if first_not_pass is not None else "–", ", ".join(map(str, mixed)) or "–", ", ".join(map(str, nov)) or "–"])
tables.append("### Per-rate results across independent trials (MEASURED; each trial = a different random seed)\n\n"
              "A trial passes only if P95 TTFT and P95 ITL meet the SLO, nothing failed, and at least 20 requests completed; a trial with fewer than 20 completions has no verdict and is excluded from the pass count (shown as n<20). "
              "TTFT p95 is the median over trials, with the min–max range in brackets; with n of 3–5 trials the range, not a confidence interval, is the honest measure of spread.\n\n"
              "| Profile | Model | Offered req/s | Trials | Realized offered req/s (range) | Valid trials passing | Failed requests (total) | P95 TTFT (s), median (range) | Rate verdict |\n|---|---|---:|---:|---:|---:|---:|---:|---|\n"
              + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in agg_rows))
tables.append("### Capacity bracket under the SLO (MEASURED; preliminary, a bracket and not a maximum)\n\n"
              "\"Highest passing\" = the highest tested rate at which every trial met the SLO, with no lower rate failing. \"Lowest rate not passing\" = the lowest tested rate where any trial failed. "
              "Rates with mixed outcomes lie inside the uncertain region. Capacity is somewhere in this bracket; the number of trials is in brackets after each rate.\n\n"
              "| Profile | Model | Rates tested (req/s) | Highest rate passing in all trials | Lowest rate not passing | Rates with mixed outcomes | Rates with no verdict (n<20) |\n|---|---|---|---:|---:|---:|---:|\n"
              + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in cap_rows))

# outcome versus REALIZED arrival rate (post hoc, exploratory): Poisson windows of 100-400 s realize very different rates
real_rows = []
for profile in ("chat",):
    for m in NAMES:
        rs_ = [r for r in runs if r["profile"] == profile and r["model"] == m and verdict(r) in ("PASS", "FAIL")]
        if not rs_:
            continue
        pas = [r["realized"] for r in rs_ if verdict(r) == "PASS"]
        fai = [r["realized"] for r in rs_ if verdict(r) == "FAIL"]
        sep = "separated" if pas and fai and max(pas) < min(fai) else ("overlap" if pas and fai else "one outcome only")
        real_rows.append([profile, NAMES[m], len(rs_), f"{max(pas):.3f}" if pas else "–", f"{min(fai):.3f}" if fai else "–", sep,
                          sum(1 for x in fai if pas and x < max(pas)) if sep == "overlap" else 0])
tables.append("### Outcome versus realized arrival rate (EXPLORATORY, post hoc)\n\n"
              "Because each run is a short Poisson window, the realized rate (arrivals ÷ duration) differs a lot from the configured rate. "
              "This table re-sorts the same runs by realized rate. It was chosen after seeing the data, so it is a description, not a tested threshold, and it holds only for these window lengths: a longer window lets a queue build further before it ends.\n\n"
              "| Profile | Model | Runs with a verdict | Highest realized rate that passed (req/s) | Lowest realized rate that failed (req/s) | Pass/fail ranges | Failures below highest pass |\n|---|---|---:|---:|---:|---|---:|\n"
              + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in real_rows))

# figures: chat + long, p95 TTFT and p95 ITL vs offered rate; points = individual trials, line = median over trials
for profile in ("chat", "long"):
    rs = [r for r in runs if r["profile"] == profile]
    if not rs:
        continue
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for m in NAMES:
        br = by_rate(profile, m)
        if not br:
            continue
        for ax_i, (key, scale) in enumerate((("ttft_s_p95", 1), ("itl_s_p95", 1000))):
            xs, ys = [], []
            for rate, ts in br.items():
                vals = [t[key] * scale for t in ts if t[key] is not None and (key == "itl_s_p95" or t["percentiles_valid"])]
                if vals:
                    axes[ax_i].scatter([rate] * len(vals), vals, s=14, color=COL[m], alpha=0.5, zorder=2)
                    xs.append(rate); ys.append(med(vals))
            axes[ax_i].plot(xs, ys, marker="_", ms=10, color=COL[m], label=NAMES[m])
        xs, ys = [], []
        for rate, ts in br.items():
            axes[2].scatter([t["realized"] for t in ts], [t["completed"] / t["duration_s"] for t in ts], s=14, color=COL[m], alpha=0.5)
    slo = SLO[profile]
    axes[0].axhline(slo["ttft"], color="gray", ls="--", lw=1); axes[0].set_title("P95 time to first token (s), per trial"); axes[0].set_yscale("log")
    axes[1].axhline(slo["itl"] * 1000, color="gray", ls="--", lw=1); axes[1].set_title("P95 inter-token latency (ms), per trial")
    lim = max([r["realized"] for r in rs] + [0.1])
    axes[2].plot([0, lim], [0, lim], color="gray", ls=":", lw=1); axes[2].set_title("Completed vs realized offered (req/s)")
    for a_ in axes[:2]:
        a_.set_xlabel("configured arrival rate (req/s)")
    axes[2].set_xlabel("realized offered load (req/s)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{profile} profile: dashed line = SLO; dots = individual trials (seeds), bar = median", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, f"10_latency_{profile}.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
for m in NAMES:
    rr = [r for r in runs if r["profile"] == "chat" and r["model"] == m and r["percentiles_valid"]]
    ax.scatter([r["realized"] for r in rr], [r["ttft_s_p95"] for r in rr], s=22, color=COL[m], alpha=0.75, label=NAMES[m])
ax.axhline(SLO["chat"]["ttft"], color="gray", ls="--", lw=1)
ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlabel("realized offered load of the run (req/s)"); ax.set_ylabel("P95 TTFT (s)")
ax.set_title("Chat: every trial, by realized arrival rate (dashed = 2 s SLO)", fontsize=9); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "11_latency_vs_realized.png")); plt.close(fig)

open(os.path.join(OUT, "tables_latency.md"), "w").write("\n\n".join(tables) + "\n")
print("\n\n".join(tables))
