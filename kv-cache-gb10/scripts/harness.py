"""Run one (model, ctx, prompt_tokens) point: fresh llama-server, memory snapshots, one request. Writes raw/<id>.json."""
import json, os, subprocess, sys, time, re, argparse, requests
BIN="/data/engines/llama.cpp/build/bin/llama-server"
MODELS="/data/engines/models/"
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT=8191
TEXT=("The quick brown fox jumps over the lazy dog while the rain falls softly on the quiet town. "
      "Engineers measure memory, latency and throughput to understand how systems behave under load. ")

def meminfo():
    d={}
    for l in open("/proc/meminfo"):
        k,v=l.split(":"); d[k]=int(v.split()[0])*1024
    return {k:d[k] for k in ("MemTotal","MemFree","MemAvailable","Cached","Buffers","Shmem")}
def rss(pid):
    for l in open(f"/proc/{pid}/status"):
        if l.startswith("VmRSS"): return int(l.split()[1])*1024
def gpu():
    out=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
    return out
def snap(pid,label):
    return dict(stage=label,t=time.time(),mem=meminfo(),rss=rss(pid),nvidia_smi=gpu())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--ctx",type=int,required=True)
    ap.add_argument("--prompt",type=int,required=True); ap.add_argument("--gen",type=int,default=64)
    ap.add_argument("--kv",default="f16"); ap.add_argument("--np",type=int,default=1)
    ap.add_argument("--rep",type=int,default=1); ap.add_argument("--tag",default="pilot")
    ap.add_argument("--extra",default="")
    a=ap.parse_args()
    rid=f"{a.tag}_{os.path.basename(a.model).replace('.gguf','')}_c{a.ctx}_p{a.prompt}_{a.kv}_r{a.rep}"
    assert subprocess.run(["pgrep","-f",f"port {PORT}"],capture_output=True).returncode!=0,"port in use"
    cmd=[BIN,"-m",MODELS+a.model,"-c",str(a.ctx),"-np",str(a.np),"-ngl","999","-fa","on",
         "--cache-type-k",a.kv,"--cache-type-v",a.kv,"--metrics","--port",str(PORT),"--host","127.0.0.1",
         "--no-cache-prompt","--cache-ram","0","-lv","4"]+a.extra.split()
    rec=dict(id=rid,args=vars(a),cmd=cmd,errors=[],snaps=[],session=time.strftime("%Y%m%dT%H%M%S"))
    base=meminfo(); logf=open(f"{HERE}/logs/{rid}.log","w")
    p=subprocess.Popen(cmd,stdout=logf,stderr=subprocess.STDOUT)
    try:
        rec["snaps"].append(dict(stage="baseline",t=time.time(),mem=base,rss=None,nvidia_smi=gpu()))
        base_url=f"http://127.0.0.1:{PORT}"; t0=time.time()
        while time.time()-t0<600:
            if p.poll() is not None: raise RuntimeError(f"server exited rc={p.returncode}")
            try:
                if requests.get(base_url+"/health",timeout=2).status_code==200: break
            except requests.RequestException: pass
            time.sleep(1)
        else: raise RuntimeError("load timeout")
        rec["load_s"]=time.time()-t0; time.sleep(3)
        rec["snaps"].append(snap(p.pid,"post_load"))
        # build prompt of exactly N tokens
        n=a.prompt; txt=TEXT*(n//10+10)
        toks=requests.post(base_url+"/tokenize",json={"content":txt,"add_special":False}).json()["tokens"][:n]
        prompt=requests.post(base_url+"/detokenize",json={"tokens":toks}).json()["content"]
        rec["props"]=requests.get(base_url+"/props").json()
        # request: stream so we can sample memory during generation
        body=dict(prompt=prompt,n_predict=a.gen,temperature=0,stream=False,cache_prompt=False,ignore_eos=True)
        r=requests.post(base_url+"/completion",json=body,timeout=1800)
        rec["http_status"]=r.status_code
        if r.status_code!=200: rec["errors"].append(r.text[:500])
        else:
            j=r.json(); rec["timings"]=j.get("timings"); rec["tokens_evaluated"]=j.get("tokens_evaluated")
            rec["tokens_predicted"]=j.get("tokens_predicted"); rec["text_head"]=j.get("content","")[:80]
            if j.get("tokens_evaluated")!=a.prompt: rec["errors"].append(f"prompt tokens {j.get('tokens_evaluated')} != {a.prompt}")
            if j.get("tokens_predicted")!=a.gen: rec["errors"].append(f"gen tokens {j.get('tokens_predicted')} != {a.gen}")
        rec["snaps"].append(snap(p.pid,"post_request"))
        try: rec["slots"]=requests.get(base_url+"/slots",timeout=5).json()
        except Exception as e: rec["errors"].append(f"slots: {e}")
        rec["metrics"]=requests.get(base_url+"/metrics",timeout=5).text
    except Exception as e:
        rec["errors"].append(repr(e))
    finally:
        p.terminate()
        try: p.wait(60)
        except subprocess.TimeoutExpired: p.kill()
        logf.close(); time.sleep(3)
        rec["snaps"].append(dict(stage="post_exit",t=time.time(),mem=meminfo(),rss=None,nvidia_smi=gpu()))
        pat=re.compile(r"KV buffer|llama_kv|llama_memory|recurrent|compute buffer|model buffer|n_ctx_slot|flash|swa",re.I)
        rec["log_lines"]=[l.rstrip() for l in open(f"{HERE}/logs/{rid}.log") if pat.search(l)][:60]
        json.dump(rec,open(f"{HERE}/raw/{rid}.json","w"),indent=1)
    print(rid,"errors:",rec["errors"])
    if "timings" in rec: print({k:rec["timings"][k] for k in ("prompt_n","prompt_per_second","predicted_n","predicted_per_second")})
    s={x["stage"]:x for x in rec["snaps"]}
    b=s["baseline"]["mem"]["MemAvailable"]
    for k in ("post_load","post_request","post_exit"):
        if k in s: print(k,"MemAvailable delta GiB: %.2f"%((b-s[k]["mem"]["MemAvailable"])/2**30),"rss GiB:",(s[k]["rss"] or 0)/2**30)
main()
