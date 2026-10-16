"""Repeats analysis. Part A: raw/p4_* (rep 1) + raw/p4r_*_r2/_r3 (default flags). Part B: raw/p4b_*_r1..r3 (-b 2048 -ub 2048).
Criterion fixed before running: a point is 'reproducible' if every rep's max TTFT is within +-10% of the mean over reps.
Usage: python -I analyze_phase4_repeats.py <blog_kv_dir> -> tables/phase4_repeats.md, tables/phase4_batch.md"""
import glob, json, re, statistics as st, sys
H = sys.argv[1]
NAME = {"Qwen3.6-35B-A3B-UD-Q4_K_M": "Qwen3.6-35B-A3B", "Qwen3.8-27B-UD-Q4_K_M": "Qwen3.8-27B"}
def load(f):
    d = json.load(open(f)); q = [x for x in d["reqs"] if x.get("ttft_s") is not None and x["status"] == 200]
    t = sorted(x["ttft_s"] for x in q); n = len(d["reqs"]); p = d["args"]["prompt"]
    return dict(ok=len(q) == n and not d["errors"], tmin=t[0], tmed=st.median(t), tmax=t[-1], wall=d["wall_s"], pre=n * p / t[-1],
                tot=(n * p + sum(x["n_gen"] for x in q)) / d["wall_s"])
A, B = {}, {}
for f in glob.glob(f"{H}/raw/p4_*_f16.json"):
    m = re.match(r".*/p4_(.+)_n(\d+)_p(\d+)_f16\.json", f)
    if m: A.setdefault((m[1], int(m[2]), int(m[3])), []).append(load(f))
for f in glob.glob(f"{H}/raw/p4r_*_f16_r*.json"):
    m = re.match(r".*/p4r_(.+)_n(\d+)_p(\d+)_f16_r\d\.json", f); A.setdefault((m[1], int(m[2]), int(m[3])), []).append(load(f))
for f in glob.glob(f"{H}/raw/p4b_*_f16_r*.json"):
    m = re.match(r".*/p4b_(.+)_n(\d+)_p(\d+)_f16_r\d\.json", f); B.setdefault((m[1], int(m[2]), int(m[3])), []).append(load(f))
def rng(v, f="{:.1f}"): return f"{st.mean(v):{f[2:-1]}} ({min(v):{f[2:-1]}}-{max(v):{f[2:-1]}})"
reps_keys = sorted((k for k in A if len(A[k]) >= 3), key=lambda k: (k[0], k[2], k[1]))
L = ["| model | seqs × prompt | reps | max TTFT s: mean (min-max) | median TTFT s | min TTFT s | aggregate prefill tok/s | wall s | max-TTFT within ±10% of mean |", "|---|---|---|---|---|---|---|---|---|"]
for k in reps_keys:
    R = A[k]; tm = [r["tmax"] for r in R]; mu = st.mean(tm); ok = all(abs(x - mu) / mu <= 0.10 for x in tm)
    L.append(f"| {NAME[k[0]]} | {k[1]} × {k[2]//1024}K | {len(R)}{'' if all(r['ok'] for r in R) else ' (errors!)'} | {rng(tm)} | {rng([r['tmed'] for r in R])} | {rng([r['tmin'] for r in R])} | {rng([r['pre'] for r in R], '{:.0f}')} | {rng([r['wall'] for r in R], '{:.0f}')} | {'yes' if ok else '**no**'} |")
open(f"{H}/tables/phase4_repeats.md", "w").write("\n".join(L))
C = ["| model | seqs × prompt | setting | reps | max TTFT s: mean (min-max) | median TTFT s | aggregate prefill tok/s | wall s |", "|---|---|---|---|---|---|---|---|"]
for k in sorted(B, key=lambda k: (k[0], k[2], k[1])):
    for lab, R in (("default (-b 2048 -ub 512)", A[k]), ("-b 2048 -ub 2048", B[k])):
        C.append(f"| {NAME[k[0]]} | {k[1]} × {k[2]//1024}K | {lab} | {len(R)} | {rng([r['tmax'] for r in R])} | {rng([r['tmed'] for r in R])} | {rng([r['pre'] for r in R], '{:.0f}')} | {rng([r['wall'] for r in R], '{:.0f}')} |")
open(f"{H}/tables/phase4_batch.md", "w").write("\n".join(C))
print("\n".join(L)); print(); print("\n".join(C))
