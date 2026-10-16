"""Theoretical KV bytes/token from GGUF metadata + tensor inventory (no model load)."""
import sys, re, json
from gguf import GGUFReader
M="/data/engines/models/"
MODELS={"Qwen3-8B (dense, GQA)":M+"Qwen3-8B-BF16.gguf",
 "Qwen3.6-35B-A3B (MoE, hybrid)":M+"Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
 "Qwen3.8-27B (dense, hybrid)":M+"Qwen3.8-27B-UD-Q4_K_M.gguf",
 "Gemma3-12B (dense, SWA 5:1)":M+"gemma3-12b-Q4_K_M.gguf",
 "Gemma4-26B-A4B (MoE, SWA 5:1)":M+"gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
 "gpt-oss-20b (MoE, SWA 1:1)":M+"gpt-oss-20b-MXFP4.gguf"}
def scal(r,k):
    f=r.fields.get(k); 
    if f is None: return None
    v=f.parts[f.data[0]]; return v.tolist()[0] if v.size==1 else v.tolist()
rows=[]
for name,p in MODELS.items():
    r=GGUFReader(p); a=scal(r,"general.architecture") if False else bytes(r.fields["general.architecture"].parts[-1]).decode()
    L=int(scal(r,f"{a}.block_count"))
    kvl={int(m.group(1)) for t in r.tensors if (m:=re.match(r"blk\.(\d+)\.attn_k\.weight",t.name))}
    nx=scal(r,f"{a}.nextn_predict_layers") or 0   # trailing MTP block(s): tensors exist, llama.cpp ignores them (n_layer = block_count - nextn)
    kvl={i for i in kvl if i<L-int(nx)}
    hk=r.fields[f"{a}.attention.head_count_kv"]; hkv=[int(hk.parts[i][0]) for i in hk.data]
    hkv=hkv*L if len(hkv)==1 else hkv
    kl=int(scal(r,f"{a}.attention.key_length")); vl=int(scal(r,f"{a}.attention.value_length"))
    klw=scal(r,f"{a}.attention.key_length_swa") or kl; vlw=scal(r,f"{a}.attention.value_length_swa") or vl
    win=scal(r,f"{a}.attention.sliding_window")
    sp=r.fields.get(f"{a}.attention.sliding_window_pattern")
    if sp: swa={i for i,x in enumerate(sp.data) if bool(sp.parts[x][0])}
    elif a=="gemma3": swa={i for i in range(L) if (i+1)%6}
    elif a=="gpt-oss": swa={i for i in range(L) if i%2==0}   # assumption: verify with runtime
    else: swa=set()
    glob=[i for i in sorted(kvl) if i not in swa]; sw=[i for i in sorted(kvl) if i in swa]
    b=2  # f16
    g_bpt=sum(hkv[i]*(kl+vl)*b for i in glob)
    s_per=sum(hkv[i]*(klw+vlw)*b for i in sw)   # bytes per token, but capped at window
    rows.append(dict(model=name,arch=a,layers=L,kv_layers=len(kvl),global_layers=len(glob),swa_layers=len(sw),
      kv_heads=sorted(set(hkv[i] for i in kvl)),key_len=kl,val_len=vl,window=win,
      global_bytes_per_token=g_bpt,swa_bytes_per_token_until_window=s_per,
      swa_const_bytes_at_window=s_per*(win or 0)))
json.dump(rows,open("theory.json","w"),indent=1)
print("| model | layers | layers w/ KV | global | SWA | KV heads | K/V dim | f16 B/token (global) | KiB/tok | +SWA const (MiB) | 32K | 128K |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    g=x["global_bytes_per_token"]; c=x["swa_const_bytes_at_window"]
    f=lambda n:(g*n+c)/2**30
    print(f"| {x['model']} | {x['layers']} | {x['kv_layers']} | {x['global_layers']} | {x['swa_layers']} | {x['kv_heads']} | {x['key_len']}/{x['val_len']} | {g} | {g/1024:.1f} | {c/2**20:.0f} | {f(32768):.2f} GiB | {f(131072):.2f} GiB |")
