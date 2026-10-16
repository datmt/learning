"""Phase 3 analysis: KV precision. raw/p2_*_{f16,q8_0,q4_0}.json, raw/ppl/*.log, raw/p3q_*.json -> tables/phase3_*.md, images/phase3_kv_precision.png."""
import json, glob, os, re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M=[("Qwen3-8B","Qwen3-8B-BF16"),("Qwen3.8-27B","Qwen3.8-27B-UD-Q4_K_M"),("Qwen3.6-35B-A3B","Qwen3.6-35B-A3B-UD-Q4_K_M")]
KV=["f16","q8_0","q4_0"]; COL={"f16":"#2a78d6","q8_0":"#1baf7a","q4_0":"#eb6834"}
def kvbuf(rid):
    t=0.0
    for l in open(f"{H}/logs/{rid}.log",errors="ignore"):
        m=re.search(r"KV buffer size =\s+([\d.]+) MiB",l)
        if m: t+=float(m.group(1))
    return t
# ---- perf
P={}
for name,f in M:
    for kv in KV:
        for fn in glob.glob(f"{H}/raw/p2_{f}_p*_{kv}.json"):
            r=json.load(open(fn)); pr=r["args"]["prompt"]
            if not r["reps"]: P[(name,pr,kv)]=dict(err=r["errors"][:1]); continue
            g=lambda k:np.array([x["server"][k] for x in r["reps"]])
            itl=[np.percentile(np.diff(x["token_times_s"])*1000,95) for x in r["reps"]]
            P[(name,pr,kv)]=dict(n=len(r["reps"]),kv_mib=kvbuf(r["id"]),ttft=np.mean([x["ttft_client_s"] for x in r["reps"]]),
                prefill=g("prompt_tps").mean(),prefill_sd=g("prompt_tps").std(ddof=1) if len(r["reps"])>1 else 0,
                decode=g("predicted_tps").mean(),decode_sd=g("predicted_tps").std(ddof=1) if len(r["reps"])>1 else 0,itl95=np.mean(itl),err=r["errors"])
L=["| model | prompt | KV type | KV buffer MiB | vs f16 | prefill tok/s | vs f16 | decode tok/s | vs f16 | TTFT s | ITL p95 ms | n |","|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name,_ in M:
    for pr in sorted({k[1] for k in P if k[0]==name}):
        b=P.get((name,pr,"f16"))
        for kv in KV:
            v=P.get((name,pr,kv))
            if not v: continue
            if "err" in v and "n" not in v: L.append(f"| {name} | {pr} | {kv} | FAILED: {v['err']} | | | | | | | | 0 |"); continue
            d=lambda k:f"{(v[k]/b[k]-1)*100:+.1f}%" if b and "n" in b and kv!="f16" else "–"
            L.append(f"| {name} | {pr} | {kv} | {v['kv_mib']:.0f} | {d('kv_mib')} | {v['prefill']:.0f}±{v['prefill_sd']:.0f} | {d('prefill')} | {v['decode']:.2f}±{v['decode_sd']:.2f} | {d('decode')} | {v['ttft']:.2f} | {v['itl95']:.1f} | {v['n']} |")
open(f"{H}/tables/phase3_perf.md","w").write("\n".join(L)+"\n"); print("\n".join(L))
# ---- perplexity
PP={}
for name,f in M:
    for kv in KV:
        fn=f"{H}/raw/ppl/{f}_{kv}.log"
        if os.path.exists(fn):
            m=re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)",open(fn,errors="ignore").read())
            if m: PP[(name,kv)]=(float(m[1]),float(m[2]))
L=["| model | KV type | PPL ± se | Δ vs f16 | Δ % |","|---|---|---|---|---|"]
for name,_ in M:
    for kv in KV:
        if (name,kv) in PP:
            p,s=PP[(name,kv)]; b=PP.get((name,"f16"))
            L.append(f"| {name} | {kv} | {p:.4f}±{s:.4f} | "+(f"{p-b[0]:+.4f} | {(p/b[0]-1)*100:+.2f}% |" if b and kv!="f16" else "– | – |"))
open(f"{H}/tables/phase3_ppl.md","w").write("\n".join(L)+"\n"); print("\n".join(L))
# ---- greedy agreement vs f16
G={}
for name,f in M:
    for kv in KV:
        fn=f"{H}/raw/p3q_greedy_{f}_{kv}.json"
        if os.path.exists(fn): G[(name,kv)]={x["i"]:x["tokens"] for x in json.load(open(fn))["results"]}
L=["| model | KV type | prompts identical to f16 | mean first-divergence token (of 256) | mean tokens matching before first divergence |","|---|---|---|---|---|"]
for name,_ in M:
    b=G.get((name,"f16"))
    for kv in ["q8_0","q4_0"]:
        v=G.get((name,kv))
        if not b or not v: continue
        div=[]
        for i in b:
            if i not in v: continue
            x,y=b[i],v[i]; k=next((j for j in range(min(len(x),len(y))) if x[j]!=y[j]),min(len(x),len(y))); div.append((k,len(x)==len(y) and k==len(x)))
        L.append(f"| {name} | {kv} | {sum(d[1] for d in div)}/{len(div)} | {np.mean([d[0] for d in div]):.0f} | {np.mean([d[0] for d in div])/256*100:.0f}% |")
open(f"{H}/tables/phase3_greedy.md","w").write("\n".join(L)+"\n"); print("\n".join(L))
# ---- needle
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); w=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-w)/d,(c+w)/d)
N={}
for fn in glob.glob(f"{H}/raw/p3q_needle_*.json"):
    r=json.load(open(fn)); a=r["args"]; name=[n for n,f in M if f==a["model"].replace(".gguf","")][0]
    N[(name,a["kv"],a["ctx"])]=r
L=["| model | context | KV type | overall | 95% CI | by depth 0/25/50/75/100% | errors |","|---|---|---|---|---|---|---|"]
for name,_ in M:
    for c in (8192,32768):
        for kv in KV:
            r=N.get((name,kv,c))
            if not r: continue
            res=r["results"]; k=sum(x["ok"] for x in res); lo,hi=wilson(k,len(res))
            bd=" / ".join(f"{sum(x['ok'] for x in res if x['depth']==d)}/{sum(1 for x in res if x['depth']==d)}" for d in (0.0,.25,.5,.75,1.0))
            L.append(f"| {name} | {c} | {kv} | {k}/{len(res)} ({k/max(len(res),1)*100:.0f}%) | {lo*100:.0f}–{hi*100:.0f}% | {bd} | {len(r['errors'])} |")
open(f"{H}/tables/phase3_needle.md","w").write("\n".join(L)+"\n"); print("\n".join(L))
# ---- plot
fig,ax=plt.subplots(1,3,figsize=(14,4.4),facecolor="#fcfcfb")
for a_ in ax:
    a_.set_facecolor("#fcfcfb"); [a_.spines[s].set_visible(False) for s in("top","right")]; a_.grid(axis="y",color="#e6e5e1"); a_.set_axisbelow(True)
names=[n for n,_ in M]; x=np.arange(len(names)); w=.27
big=lambda n:32768 if n=="Qwen3-8B" else 131072
for i,kv in enumerate(KV):
    ax[0].bar(x+(i-1)*w,[P.get((n,big(n),kv),{}).get("kv_mib",0)/1024 for n in names],w,color=COL[kv],label=kv)
    ax[1].bar(x+(i-1)*w,[P.get((n,big(n),kv),{}).get("decode",0) for n in names],w,color=COL[kv])
    ax[2].bar(x+(i-1)*w,[(PP[(n,kv)][0]/PP[(n,"f16")][0]-1)*100 if (n,kv) in PP and (n,"f16") in PP else 0 for n in names],w,color=COL[kv])
ax[0].set_title("KV buffer at max tested context (GiB)",loc="left",fontsize=11); ax[1].set_title("Decode tok/s at max tested context",loc="left",fontsize=11); ax[2].set_title("Perplexity change vs f16 KV (%)",loc="left",fontsize=11)
for a_ in ax: a_.set_xticks(x); a_.set_xticklabels(names,fontsize=9)
ax[0].legend(frameon=False); plt.tight_layout(); plt.savefig(f"{H}/images/phase3_kv_precision.png",dpi=140,facecolor="#fcfcfb")
