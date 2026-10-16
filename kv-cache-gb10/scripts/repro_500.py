"""Phase 4: try to reproduce HTTP 500 'context size exceeded' with concurrent requests after a pool-filling request. Prod-like flags. Writes raw/p4_repro500.json."""
import json, os, subprocess, sys, time, random, threading, requests
BIN="/data/engines/llama.cpp/build/bin/llama-server"; M="/data/engines/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PORT=8191; U=f"http://127.0.0.1:{PORT}"
WORDS="river stone quiet engine garden window copper violet market thunder paper lantern harbor meadow signal ribbon canyon orchard pencil velvet anchor breeze cobble drift ember falcon glacier hollow island jungle kettle lemon marble".split()
cmd=[BIN,"-m",M,"--ctx-size","131072","--parallel","4","--kv-unified","--flash-attn","on","--cache-reuse","256","-ngl","99","--jinja",
     "--port",str(PORT),"--host","127.0.0.1","-lv","4"]
rec=dict(cmd=cmd,trials=[],session=time.strftime("%Y%m%dT%H%M%S"))
logf=open(f"{HERE}/logs/p4_repro500.log","w"); p=subprocess.Popen(cmd,stdout=logf,stderr=subprocess.STDOUT)
def mk(seed,n):
    rng=random.Random(seed); txt=" ".join(rng.choice(WORDS) for _ in range(n+50))
    t=requests.post(U+"/tokenize",json={"content":txt,"add_special":False}).json()["tokens"][:n]
    return requests.post(U+"/detokenize",json={"tokens":t}).json()["content"]
def req(tag,prompt,gen,out,delay=0):
    time.sleep(delay); t=time.perf_counter(); r_=dict(tag=tag,status=None)
    try:
        r=requests.post(U+"/completion",json=dict(prompt=prompt,n_predict=gen,temperature=0,ignore_eos=True),timeout=3600)
        r_.update(status=r.status_code,body=r.text[:200] if r.status_code!=200 else None,n_prompt=(r.json().get("tokens_evaluated") if r.status_code==200 else None))
    except Exception as e: r_["err"]=repr(e)
    r_["s"]=round(time.perf_counter()-t,1); out.append(r_)
def trial(name,big,small_n,small_len,small_delay,seed):
    """big: prompt tokens of the pool-filling request (0=none); it is sent first; smalls start small_delay s later (-1 = after big finishes)."""
    out=[]; ths=[]
    if big:
        bp=mk(seed,big); tb=threading.Thread(target=req,args=("big",bp,32,out)); tb.start(); ths.append(tb)
        if small_delay<0: tb.join()
        else: time.sleep(small_delay)
    for i in range(small_n):
        sp=mk(seed*10+i+1,small_len); t=threading.Thread(target=req,args=(f"small{i}",sp,64,out)); t.start(); ths.append(t)
    for t in ths: t.join()
    rec["trials"].append(dict(name=name,big=big,small_n=small_n,small_len=small_len,small_delay=small_delay,results=out))
    print(name,[(r["tag"],r["status"],r["s"]) for r in out],flush=True)
try:
    t0=time.time()
    while time.time()-t0<900:
        if p.poll() is not None: raise RuntimeError(f"server exited {p.returncode}")
        try:
            if requests.get(U+"/health",timeout=2).status_code==200: break
        except requests.RequestException: pass
        time.sleep(1)
    else: raise RuntimeError("load timeout")
    time.sleep(3)
    # A: big (fills pool) finishes, then 4 concurrent smalls
    trial("A_big130k_then_4x8k",130000,4,8000,-1,1)
    # B: smalls arrive while big is still prefilling
    trial("B_big130k_during_4x8k",130000,4,8000,20,2)
    # C: no big request, 4 concurrent 32k (fills 128k of 131k pool)
    trial("C_4x32k",0,4,32000,0,3)
    # D: 4 concurrent 40k (exceeds pool: 160k > 131k)
    trial("D_4x40k_exceeds_pool",0,4,40000,0,4)
    # E: repeat A once more
    trial("E_repeat_A",130000,4,8000,-1,5)
except Exception as e:
    rec["error"]=repr(e); print("ERROR",e)
finally:
    p.terminate()
    try: p.wait(60)
    except subprocess.TimeoutExpired: p.kill()
    logf.close(); json.dump(rec,open(f"{HERE}/raw/p4_repro500.json","w"),indent=1)
