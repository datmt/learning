"""Phase 2 analysis: speed vs context from raw/p2_*.json (+ llama-bench depth runs) -> tables/, images/."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
th={x["arch"]:x for x in json.load(open(f"{H}/theory.json"))}
MAP={"Qwen3-8B":("Qwen3-8B","qwen3"),"Qwen3.6-35B":("Qwen3.6-35B-A3B","qwen35moe"),"Qwen3.8-27B":("Qwen3.8-27B","qwen35"),"gemma3-12b":("Gemma3-12B","gemma3"),"gpt-oss-20b":("gpt-oss-20b","gpt-oss")}
GB={"Qwen3-8B":16.39e9,"Qwen3.6-35B-A3B":None,"Qwen3.8-27B":None}
data={}
for f in sorted(glob.glob(f"{H}/raw/p2_*.json")):
    r=json.load(open(f)); k=next(k for k in MAP if os.path.basename(f)[3:].startswith(k)); name=MAP[k][0]
    for x in r["reps"]:
        ts=np.array(x["token_times_s"]); itl=np.diff(ts)*1000
        data.setdefault(name,{}).setdefault(r["args"]["prompt"],[]).append(dict(
            ttft=ts[0],prefill=x["server"]["prompt_tps"],decode_srv=x["server"]["predicted_tps"],
            decode_cli=(len(ts)-1)/(ts[-1]-ts[0]),itl_mean=itl.mean(),itl_p50=np.percentile(itl,50),itl_p95=np.percentile(itl,95),itl_max=itl.max()))
order=["Qwen3-8B","Qwen3.8-27B","Qwen3.6-35B-A3B","Gemma3-12B","gpt-oss-20b"]
COL=dict(zip(order,["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"]))
lab=lambda p:f"{p//1024}K" if p%1024==0 else f"{round(p/1024)}K"
L=["| model | prompt tokens | n | TTFT s (mean±sd) | prefill tok/s | decode tok/s (server) | decode tok/s (client) | ITL mean ms | ITL p50 | ITL p95 | decode vs 4K |","|---|---|---|---|---|---|---|---|---|---|---|"]
S={}
for m in order:
    base=None
    for p in sorted(data.get(m,{})):
        v=data[m][p]; g=lambda k:np.array([z[k] for z in v])
        d=g("decode_srv").mean(); base=base or d; S[(m,p)]={k:g(k).mean() for k in v[0]}|{k+"_sd":g(k).std(ddof=1) for k in v[0]}
        L.append(f"| {m} | {p} | {len(v)} | {g('ttft').mean():.2f}±{g('ttft').std(ddof=1):.2f} | {g('prefill').mean():.0f}±{g('prefill').std(ddof=1):.0f} | {d:.2f}±{g('decode_srv').std(ddof=1):.2f} | {g('decode_cli').mean():.2f} | {g('itl_mean').mean():.1f} | {g('itl_p50').mean():.1f} | {g('itl_p95').mean():.1f} | {d/base*100:.0f}% |")
open(f"{H}/tables/phase2_speed.md","w").write("\n".join(L)+"\n"); print("\n".join(L))
# bench depth
B={}
for f in glob.glob(f"{H}/raw/p2bench_*.json"):
    try: j=json.load(open(f))
    except Exception: continue
    n=os.path.basename(f)[8:-5]; name=[MAP[k][0] for k in MAP if n.startswith(k)][0]
    for e in j:
        if e.get("n_gen",0)>0: B.setdefault(name,{})[e["n_depth"]]=(e["avg_ts"],e["stddev_ts"])
if B:
    T=["| model | depth | llama-bench decode tok/s | server decode tok/s (same depth) |","|---|---|---|---|"]
    for m in order:
        for d,(a,s) in sorted(B.get(m,{}).items()):
            sv=[S[(m,p)]["decode_srv"] for (mm,p) in S if mm==m and p==d]
            T.append(f"| {m} | {d} | {a:.2f}±{s:.2f} | {sv[0]:.2f} |" if sv else f"| {m} | {d} | {a:.2f}±{s:.2f} | – |")
    open(f"{H}/tables/phase2_bench_depth.md","w").write("\n".join(T)+"\n"); print("\n".join(T))
# plots: 4 panels
fig,ax=plt.subplots(2,2,figsize=(12,8.5),facecolor="#fcfcfb")
panels=[("prefill","Prefill throughput (tok/s)",False),("ttft","Time to first token (s, log)",True),("decode_srv","Decode throughput (tok/s)",False),("itl_p95","Inter-token latency p95 (ms)",False)]
for a_,(k,t,lg) in zip(ax.flat,panels):
    a_.set_facecolor("#fcfcfb"); [a_.spines[s].set_visible(False) for s in("top","right")]
    a_.grid(color="#e6e5e1",lw=.8); a_.set_axisbelow(True); a_.set_title(t,loc="left",fontsize=11); a_.tick_params(colors="#52514e")
    for m in order:
        ps=sorted(p for (mm,p) in S if mm==m)
        if not ps: continue
        y=np.array([S[(m,p)][k] for p in ps]); sd=np.array([S[(m,p)][k+"_sd"] for p in ps])
        a_.errorbar(ps,y,yerr=sd,fmt="o-",color=COL[m],lw=2,ms=6,mfc=COL[m],mec="#fcfcfb",label=m,capsize=2)
    a_.set_xscale("log",base=2); a_.set_xticks([4096,8192,16384,32768,65536,131072]); a_.set_xticklabels(["4K","8K","16K","32K","64K","128K"])
    if lg: a_.set_yscale("log")
    elif k!="itl_p95": a_.set_ylim(bottom=0)
ax[0,0].legend(frameon=False,fontsize=9); [a_.set_xlabel("prompt tokens",color="#52514e") for a_ in ax[1]]
plt.tight_layout(); plt.savefig(f"{H}/images/phase2_speed_vs_context.png",dpi=140,facecolor="#fcfcfb")
# --- tie decode slowdown to KV bytes read (inferred): ms/token = a + b*ctx ; theory b = (global B/tok + swa const share) / BW
BW=273e9
Q=["| model | fitted slope (µs per 1K ctx tokens) | intercept ms/token | theory KV read (µs per 1K tokens @273 GB/s) | measured/theory |","|---|---|---|---|---|"]
for m in order:
    ps=sorted(p for (mm,p) in S if mm==m); arch=MAP[[k for k in MAP if MAP[k][0]==m][0]][1]
    x=np.array(ps,float); y=np.array([1000/S[(m,p)]["decode_srv"] for p in ps]); b,a=np.polyfit(x,y,1)
    t=th[arch]["global_bytes_per_token"]*1024/BW*1e6
    Q.append(f"| {m} | {b*1024*1000:.1f} | {a:.1f} | {t:.1f} | {b*1024*1000/t:.2f}x |")
open(f"{H}/tables/phase2_decode_slope.md","w").write("\n".join(Q)+"\n"); print("\n".join(Q))
