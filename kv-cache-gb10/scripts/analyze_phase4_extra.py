"""Phase 4 extras from existing raw/p4_*_f16.json: per-request TTFT spread, aggregate throughput, and the largest N meeting a TTFT budget.
Usage: python -I analyze_phase4_extra.py <blog_kv_dir>   -> tables/phase4_latency.md, tables/phase4_operating_points.md
Aggregate throughput here is DERIVED: prefill = N*prompt / max TTFT (all prompts done), total = N*(prompt+gen) / wall.
"""
import glob, json, re, statistics as st, sys
H = sys.argv[1]
NAME = {"Qwen3-8B-BF16": "Qwen3-8B (BF16)", "Qwen3.6-35B-A3B-UD-Q4_K_M": "Qwen3.6-35B-A3B", "Qwen3.8-27B-UD-Q4_K_M": "Qwen3.8-27B"}
rows = {}
for f in sorted(glob.glob(f"{H}/raw/p4_*_f16.json")):
    m = re.match(r".*/p4_(.+)_n(\d+)_p(\d+)_f16\.json", f)
    if not m: continue
    d = json.load(open(f)); q = [x for x in d["reqs"] if x.get("ttft_s") is not None and x["status"] == 200]
    t = sorted(x["ttft_s"] for x in q); n, p = int(m[2]), int(m[3])
    gen = sum(x["n_gen"] for x in q)
    rows.setdefault(m[1], []).append(dict(n=n, p=p, ok=len(q), tmin=t[0], tmed=st.median(t), tmax=t[-1], wall=d["wall_s"],
        pre=n * p / t[-1], tot=(n * p + gen) / d["wall_s"], dec=st.mean(x["server"]["predicted_tps"] for x in q)))
L = []; OP = {}
for k, R in rows.items():
    R.sort(key=lambda x: (x["p"], x["n"])); L.append(f"### {NAME[k]}\n\n| seqs | prompt/seq | TTFT min / median / max s | aggregate prefill tok/s | aggregate (prompt+gen) tok/s over wall | ok |\n|---|---|---|---|---|---|")
    for x in R:
        L.append(f"| {x['n']} | {x['p']} | {x['tmin']:.1f} / {x['tmed']:.1f} / {x['tmax']:.1f} | {x['pre']:.0f} | {x['tot']:.0f} | {x['ok']}/{x['n']} |")
    L.append("")
    OP[k] = R
open(f"{H}/tables/phase4_latency.md", "w").write("\n".join(L))
B = [10, 30, 120]
O = ["| model | prompt/seq | " + " | ".join(f"max seqs with worst TTFT ≤ {b} s" for b in B) + " |", "|---|---|" + "---|" * len(B)]
for k, R in OP.items():
    for p in sorted({x["p"] for x in R}):
        c = []
        for b in B:
            ok = [x["n"] for x in R if x["p"] == p and x["tmax"] <= b]
            c.append(str(max(ok)) if ok else "0 (one alone is slower)")
        O.append(f"| {NAME[k]} | {p//1024}K | " + " | ".join(c) + " |")
open(f"{H}/tables/phase4_operating_points.md", "w").write("\n".join(O))
print("\n".join(O))
