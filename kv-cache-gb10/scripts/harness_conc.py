"""Phase 4: one fresh server per (model, n_seq, ctx_per_seq). N simultaneous distinct prompts, streamed. Memory sampled at 0.5 s. Writes raw/p4_*.json."""
import json, os, subprocess, sys, time, argparse, random, threading, requests
BIN="/data/engines/llama.cpp/build/bin/llama-server"; MODELS="/data/engines/models/"
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PORT=8191
WORDS="river stone quiet engine garden window copper violet market thunder paper lantern harbor meadow signal ribbon canyon orchard pencil velvet anchor breeze cobble drift ember falcon glacier hollow island jungle kettle lemon marble nectar oasis pebble quartz rocket saddle timber umbrella valley willow yarn zephyr".split()
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--n",type=int,required=True); ap.add_argument("--prompt",type=int,required=True)
ap.add_argument("--gen",type=int,default=64); ap.add_argument("--kv",default="f16"); ap.add_argument("--tag",default="")
ap.add_argument("--ctx",type=int,default=0,help="override total -c (cells); default n*(prompt+gen+8)")
ap.add_argument("--np",type=int,default=0,help="override slot count (default n)")
ap.add_argument("--prefix",default="p4",help="raw/log file prefix (p4 = original matrix, p4r = repeats, p4b = batch variant)")
ap.add_argument("--extra",default="",help="extra llama-server flags, space separated (e.g. \"-b 2048 -ub 2048\")")
a=ap.parse_args()
ctx=a.ctx or a.n*(a.prompt+a.gen+8); npar=a.np or a.n
rid=f"{a.prefix}_{a.model.replace('.gguf','')}_n{a.n}_p{a.prompt}_{a.kv}{a.tag}"
cmd=[BIN,"-m",MODELS+a.model,"-c",str(ctx),"-np",str(npar),"--kv-unified","-ngl","999","-fa","on","--cache-type-k",a.kv,"--cache-type-v",a.kv,
     "--port",str(PORT),"--host","127.0.0.1","--no-cache-prompt","--cache-ram","0","-lv","4"]+a.extra.split()
rec=dict(id=rid,args=vars(a),cmd=cmd,n_ctx_total=ctx,errors=[],reqs=[],mem=[],session=time.strftime("%Y%m%dT%H%M%S"))
def memavail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1])//1024
stop=threading.Event(); T0=[None]
def sampler():
    while not stop.is_set():
        if T0[0] is not None: rec["mem"].append((round(time.perf_counter()-T0[0],2),memavail()))
        time.sleep(0.5)
rec["memavail_baseline_mib"]=memavail()
logf=open(f"{HERE}/logs/{rid}.log","w"); p=subprocess.Popen(cmd,stdout=logf,stderr=subprocess.STDOUT); U=f"http://127.0.0.1:{PORT}"
def one(i,prompt,out):
    body=dict(prompt=prompt,n_predict=a.gen,temperature=0,stream=True,cache_prompt=False,ignore_eos=True)
    r_=dict(i=i,status=None,ts=[]); t_send=time.perf_counter(); r_["t_send"]=t_send-T0[0]
    try:
        with requests.post(U+"/completion",json=body,stream=True,timeout=7200) as r:
            r_["status"]=r.status_code
            if r.status_code!=200: r_["err"]=r.text[:300]
            else:
                for line in r.iter_lines():
                    if not line.startswith(b"data: "): continue
                    j=json.loads(line[6:]); now=time.perf_counter()
                    if j.get("stop"): r_["final"]=j
                    elif j.get("content") is not None: r_["ts"].append(now-t_send)
        r_["t_done"]=time.perf_counter()-T0[0]
    except Exception as e: r_["err"]=repr(e)
    out[i]=r_
try:
    t0=time.time()
    while time.time()-t0<900:
        if p.poll() is not None: raise RuntimeError(f"server exited {p.returncode}")
        try:
            if requests.get(U+"/health",timeout=2).status_code==200: break
        except requests.RequestException: pass
        time.sleep(1)
    else: raise RuntimeError("load timeout")
    rec["load_s"]=time.time()-t0; time.sleep(3)
    rec["memavail_loaded_mib"]=memavail()
    prompts=[]
    for i in range(a.n):
        rng=random.Random(1000+i); txt=" ".join(rng.choice(WORDS) for _ in range(a.prompt+50))
        toks=requests.post(U+"/tokenize",json={"content":txt,"add_special":False}).json()["tokens"][:a.prompt]
        prompts.append(requests.post(U+"/detokenize",json={"tokens":toks}).json()["content"])
    requests.post(U+"/completion",json=dict(prompt="Hello",n_predict=8,temperature=0,cache_prompt=False),timeout=300)
    th=threading.Thread(target=sampler,daemon=True); th.start(); T0[0]=time.perf_counter(); time.sleep(1)
    out=[None]*a.n; ths=[threading.Thread(target=one,args=(i,prompts[i],out)) for i in range(a.n)]
    for t in ths: t.start()
    for t in ths: t.join()
    time.sleep(1); rec["wall_s"]=time.perf_counter()-T0[0]
    try: rec["metrics"]=requests.get(U+"/metrics",timeout=5).text[:4000]
    except Exception as e: rec["metrics"]=repr(e)
    ok=0; tot_gen=0
    for r_ in out:
        f=r_.pop("final",None)
        if r_.get("status")!=200 or f is None: rec["errors"].append(f"req{r_['i']}: status {r_.get('status')} {r_.get('err')}")
        else:
            t=f["timings"]; n_eval=f.get("tokens_evaluated"); n_pred=f.get("tokens_predicted")
            if n_eval not in (a.prompt,a.prompt+1): rec["errors"].append(f"req{r_['i']}: prompt tokens {n_eval}")
            if n_pred!=a.gen: rec["errors"].append(f"req{r_['i']}: gen tokens {n_pred}")
            r_.update(n_prompt=n_eval,n_gen=n_pred,ttft_s=r_["ts"][0] if r_["ts"] else None,
                server=dict(prompt_ms=t["prompt_ms"],prompt_tps=t["prompt_per_second"],predicted_ms=t["predicted_ms"],predicted_tps=t["predicted_per_second"]))
            ok+=1; tot_gen+=n_pred
        rec["reqs"].append(r_)
    rec["ok"]=ok
    print(rid,f"ok {ok}/{a.n} wall {rec['wall_s']:.1f}s min_avail {min(m for _,m in rec['mem']) if rec['mem'] else None} MiB",flush=True)
except Exception as e:
    rec["errors"].append(repr(e)); print(rid,"ERROR",e)
finally:
    stop.set(); p.terminate()
    try: p.wait(60)
    except subprocess.TimeoutExpired: p.kill()
    logf.close(); time.sleep(3)
    json.dump(rec,open(f"{HERE}/raw/{rid}.json","w"),indent=1)
