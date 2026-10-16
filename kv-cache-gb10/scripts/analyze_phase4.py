"""Phase 4 analysis: raw/p4_*_f16.json (+ logs) -> tables/phase4_*.md, tables/phase4_runs.csv, images/phase4_concurrency.png."""
import json, glob, os, re, csv, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME={"Qwen3-8B-BF16":"Qwen3-8B (BF16)","Qwen3.6-35B-A3B-UD-Q4_K_M":"Qwen3.6-35B-A3B","Qwen3.8-27B-UD-Q4_K_M":"Qwen3.8-27B"}
def logmib(log,pat):
    t=open(log).read(); m=re.findall(pat+r"\s+size =\s+([\d.]+) MiB|"+pat+r"\s+buffer size =\s+([\d.]+) MiB",t)
    return sum(float(a or b) for a,b in m) if m else None
rows=[]
for f in glob.glob(H+"/raw/p4_*_f16.json"):
    r=json.load(open(f)); a=r["args"]; log=f"{H}/logs/{r['id']}.log"
    kv=logmib(log,r"llama_kv_cache:\s+CUDA0 KV") ; rs=logmib(log,r"llama_memory_recurrent:\s+CUDA0 RS")
    q=[x for x in r["reqs"] if x.get("ttft_s") is not None]
    ttft=[x["ttft_s"] for x in q]; dec=[x["server"]["predicted_tps"] for x in q]
    mn=min(m for _,m in r["mem"])
    rows.append(dict(model=a["model"].replace(".gguf",""),n=a["n"],prompt=a["prompt"],ok=r["ok"],errors=len(r["errors"]),kv_mib=kv,rs_mib=rs,
      load_drop_mib=r["memavail_baseline_mib"]-r["memavail_loaded_mib"],peak_drop_mib=r["memavail_baseline_mib"]-mn,wall_s=r["wall_s"],
      ttft_mean=st.mean(ttft),ttft_max=max(ttft),dec_mean=st.mean(dec),
      agg_prefill_tps=a["n"]*a["prompt"]/max(ttft),agg_total_tps=a["n"]*(a["prompt"]+a["gen"])/r["wall_s"],cells=r["n_ctx_total"]))
rows.sort(key=lambda x:(list(NAME).index(x["model"]),x["prompt"],x["n"]))
for x in rows: x["cells_padded"]=-(-x["cells"]//256)*256
os.makedirs(H+"/tables",exist_ok=True)
with open(H+"/tables/phase4_runs.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def md(hdr,body): return "\n".join(["| "+" | ".join(hdr)+" |","|"+"---|"*len(hdr)]+["| "+" | ".join(map(str,b))+" |" for b in body])+"\n"
out=[]
for m in NAME:
    R=[x for x in rows if x["model"]==m]
    out.append(f"### {NAME[m]}\n\n"+md(["seqs","prompt/seq","total cells","KV GiB","recurrent MiB (per seq)","peak MemAvail drop GiB","wall s","TTFT mean / max s","decode tok/s per request","ok"],
      [[x["n"],x["prompt"],x["cells"],f"{x['kv_mib']/1024:.2f}",f"{x['rs_mib']:.1f} ({x['rs_mib']/x['n']:.1f})" if x["rs_mib"] else "-",f"{x['peak_drop_mib']/1024:.1f}",f"{x['wall_s']:.0f}",
        f"{x['ttft_mean']:.1f} / {x['ttft_max']:.1f}",f"{x['dec_mean']:.1f}",f"{x['ok']}/{x['n']}"] for x in R]))
open(H+"/tables/phase4_matrix.md","w").write("\n".join(out))
# KV per cell and recurrent per seq consistency
chk=[]
for m in NAME:
    R=[x for x in rows if x["model"]==m]
    bpc=[x["kv_mib"]*2**20/x["cells_padded"] for x in R]; rps=[x["rs_mib"]/x["n"] for x in R if x["rs_mib"]]
    chk.append([NAME[m],f"{min(bpc):.0f} - {max(bpc):.0f}",f"{min(rps):.1f} - {max(rps):.1f}" if rps else "none (no recurrent layers)"])
open(H+"/tables/phase4_kv_check.md","w").write(md(["model","KV bytes per cell, cells padded to 256 (min - max)","recurrent MiB per sequence (min - max)"],chk))
# aggregate throughput table at 4K and max context: decode-phase aggregate approximated by sum of per-request decode tok/s
fig,ax=plt.subplots(2,3,figsize=(15,8))
for j,m in enumerate(NAME):
    R=[x for x in rows if x["model"]==m]
    for P in sorted({x["prompt"] for x in R}):
        S=sorted([x for x in R if x["prompt"]==P],key=lambda x:x["n"]); ns=[x["n"] for x in S]
        ax[0][j].plot(ns,[x["peak_drop_mib"]/1024 for x in S],"o-",label=f"{P//1024}K/seq")
        ax[1][j].plot(ns,[x["ttft_mean"] for x in S],"o-",label=f"{P//1024}K/seq")
    ax[0][j].set_title(NAME[m]); ax[0][j].set_xlabel("concurrent sequences"); ax[0][j].set_ylabel("peak memory drop (GiB, MemAvailable)"); ax[0][j].set_xscale("log",base=2)
    ax[1][j].set_xlabel("concurrent sequences"); ax[1][j].set_ylabel("mean TTFT (s)"); ax[1][j].set_xscale("log",base=2); ax[1][j].set_yscale("log"); ax[0][j].legend()
plt.tight_layout(); os.makedirs(H+"/images",exist_ok=True); plt.savefig(H+"/images/phase4_concurrency.png",dpi=130)
print(open(H+"/tables/phase4_kv_check.md").read())
