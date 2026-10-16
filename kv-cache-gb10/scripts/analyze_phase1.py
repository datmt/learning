"""Phase 1 analysis: parse KV/RS/breakdown lines from logs, join with theory, write tables/ and images/."""
import json, glob, re, os, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
th={x["arch"]:x for x in json.load(open(f"{H}/theory.json"))}
ARCH={"Qwen3-8B":"qwen3","Qwen3.6-35B":"qwen35moe","Qwen3.8-27B":"qwen35","gemma3-12b":"gemma3","gpt-oss-20b":"gpt-oss"}
NAME={"qwen3":"Qwen3-8B","qwen35moe":"Qwen3.6-35B-A3B","qwen35":"Qwen3.8-27B","gemma3":"Gemma3-12B","gpt-oss":"gpt-oss-20b"}
rows=[]
for f in sorted(glob.glob(f"{H}/raw/p1*.json")):
    r=json.load(open(f)); a=r["args"]; log=open(f"{H}/logs/{r['id']}.log").read()
    arch=next(v for k,v in ARCH.items() if os.path.basename(a["model"]).startswith(k))
    kv=re.findall(r"llama_kv_cache: size =\s+([\d.]+) MiB \(\s*(\d+) cells,\s+(\d+) layers",log)
    rs=re.search(r"llama_memory_recurrent: size =\s+([\d.]+) MiB",log)
    bd=re.search(r"CUDA0 \(GB10\)\s+\| \d+ = (\d+) \+ \((\d+) =\s+(\d+) \+\s+(\d+) \+\s+(\d+)\)",log)
    s={x["stage"]:x for x in r["snaps"]}; b=s["baseline"]["mem"]["MemAvailable"]
    t=r.get("timings") or {}
    rows.append(dict(id=r["id"],cfg=r["id"][2],model=NAME[arch],arch=arch,ctx_arg=a["ctx"],prompt=a["prompt"],rep=a["rep"],
      ok=int(not r["errors"] or all(e.startswith("prompt tokens") for e in r["errors"])),error=(r["errors"] or [""])[0][:80],
      kv_main_mib=float(kv[0][0]) if kv else None,cells_main=int(kv[0][1]) if kv else None,layers_main=int(kv[0][2]) if kv else None,
      kv_swa_mib=float(kv[1][0]) if len(kv)>1 else 0.0,cells_swa=int(kv[1][1]) if len(kv)>1 else 0,
      rs_mib=float(rs.group(1)) if rs else 0.0,
      gpu_self_mib=int(bd.group(2)) if bd else None,gpu_model_mib=int(bd.group(3)) if bd else None,
      gpu_context_mib=int(bd.group(4)) if bd else None,gpu_compute_mib=int(bd.group(5)) if bd else None,
      sys_delta_load_gib=(b-s["post_load"]["mem"]["MemAvailable"])/2**30 if "post_load" in s else None,
      sys_delta_req_gib=(b-s["post_request"]["mem"]["MemAvailable"])/2**30 if "post_request" in s else None,
      rss_gib=(s["post_request"]["rss"] or 0)/2**30 if "post_request" in s else None,
      prefill_tps=t.get("prompt_per_second"),decode_tps=t.get("predicted_per_second")))
os.makedirs(f"{H}/tables",exist_ok=True); os.makedirs(f"{H}/images",exist_ok=True)
with open(f"{H}/tables/phase1_runs.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
good=[x for x in rows if x["ok"] and x["kv_main_mib"]]
# --- bytes/token: slope of main (global) KV vs cells over config b + a, per model
out=["| model | KV layers (log) | theory global B/tok | measured global B/tok | match | SWA cache cells | SWA MiB | recurrent state MiB (1 seq) | runs ok/total |","|---|---|---|---|---|---|---|---|---|"]
slopes={}
for arch in NAME:
    g=[x for x in good if x["arch"]==arch]
    cells=np.array([x["cells_main"] for x in g],float); mib=np.array([x["kv_main_mib"] for x in g])
    sl,ic=np.polyfit(cells,mib*2**20,1); slopes[arch]=(sl,ic)
    t=th[arch]["global_bytes_per_token"]; n=[x for x in rows if x["arch"]==arch]
    out.append(f"| {NAME[arch]} | {g[0]['layers_main']} | {t} | {sl:.0f} (intercept {ic:.0f} B) | {'yes' if abs(sl-t)<1 else 'NO (%+.1f%%)'%((sl/t-1)*100)} | {sorted({x['cells_swa'] for x in g})} | {sorted({x['kv_swa_mib'] for x in g})} | {g[0]['rs_mib']:.1f} | {sum(x['ok'] for x in n)}/{len(n)} |")
open(f"{H}/tables/phase1_theory_vs_measured.md","w").write("\n".join(out)+"\n"); print("\n".join(out))
# --- allocation table cfg b, mean over reps
agg=collections.defaultdict(list)
for x in good:
    if x["cfg"]=="b": agg[(x["model"],x["prompt"])].append(x)
tab=["| model | prompt tokens | KV main MiB | KV SWA MiB | recurrent MiB | GPU self MiB (model+ctx+compute) | peak ctx MiB | sys MemAvailable delta after request GiB (mean of reps) |","|---|---|---|---|---|---|---|---|"]
for (m,p),v in sorted(agg.items(),key=lambda kv:(list(NAME.values()).index(kv[0][0]),kv[0][1])):
    f=lambda k:np.mean([y[k] for y in v if y[k] is not None])
    tab.append(f"| {m} | {p} | {f('kv_main_mib'):.0f} | {f('kv_swa_mib'):.0f} | {f('rs_mib'):.0f} | {f('gpu_self_mib'):.0f} | {f('gpu_context_mib'):.0f} | {f('sys_delta_req_gib'):.1f} (n={len(v)}) |")
open(f"{H}/tables/phase1_alloc_cfg_b.md","w").write("\n".join(tab)+"\n")
# --- plot
COL=dict(zip(NAME.values(),["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"]))
order=["Qwen3-8B","Qwen3.8-27B","Qwen3.6-35B-A3B","Gemma3-12B","gpt-oss-20b"]
COL={"Qwen3-8B":"#2a78d6","Qwen3.8-27B":"#eb6834","Qwen3.6-35B-A3B":"#1baf7a","Gemma3-12B":"#eda100","gpt-oss-20b":"#e87ba4"}
fig,ax=plt.subplots(1,2,figsize=(12,4.6),facecolor="#fcfcfb")
for a_ in ax:
    a_.set_facecolor("#fcfcfb"); [a_.spines[s].set_visible(False) for s in("top","right")]
    a_.grid(color="#e6e5e1",lw=.8); a_.set_axisbelow(True); a_.tick_params(colors="#52514e")
for m in order:
    arch=[k for k,v in NAME.items() if v==m][0]; T=th[arch]
    ks=sorted({p for (mm,p) in agg if mm==m}); 
    y=[np.mean([z["kv_main_mib"]+z["kv_swa_mib"] for z in agg[(m,p)]])/1024 for p in ks]
    ax[0].plot(ks,y,"o-",color=COL[m],lw=2,ms=6,mfc=COL[m],mec="#fcfcfb",label=m)
    xs=np.array(ks,float); thy=(T["global_bytes_per_token"]*xs+T["swa_const_bytes_at_window"])/2**30
    ax[0].plot(xs,thy,"--",color=COL[m],lw=1,alpha=.7)
    y2=[np.mean([z["gpu_self_mib"] for z in agg[(m,p)]])/1024 for p in ks]
    ax[1].plot(ks,y2,"o-",color=COL[m],lw=2,ms=6,mfc=COL[m],mec="#fcfcfb",label=m)
ax[0].set_title("KV cache allocated vs prompt tokens (x axis log)\nsolid = measured, dashed = Phase 0 theory",loc="left",color="#0b0b0b",fontsize=11)
ax[1].set_title("GPU memory of the server process (model + context + compute)",loc="left",color="#0b0b0b",fontsize=11)
for a_,yl in zip(ax,("GiB","GiB")):
    a_.set_xscale("log",base=2); a_.set_xlabel("prompt tokens",color="#52514e"); a_.set_ylabel(yl,color="#52514e")
    a_.set_xticks([4096,8192,16384,32768,65536,131072]); a_.set_xticklabels(["4K","8K","16K","32K","64K","128K"])
ax[0].legend(frameon=False,fontsize=9,loc="upper left"); plt.tight_layout()
plt.savefig(f"{H}/images/phase1_memory_vs_context.png",dpi=140,facecolor="#fcfcfb")
