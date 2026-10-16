"""Phase 2: one fresh server per (model, prompt); warmup, then N streamed repetitions. Client-side per-token timestamps + server timings. Writes raw/p2_*.json."""
import json, os, subprocess, sys, time, argparse, requests
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
BIN="/data/engines/llama.cpp/build/bin/llama-server"; MODELS="/data/engines/models/"
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PORT=8191
TEXT=("The quick brown fox jumps over the lazy dog while the rain falls softly on the quiet town. "
      "Engineers measure memory, latency and throughput to understand how systems behave under load. ")
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--prompt",type=int,required=True)
ap.add_argument("--gen",type=int,default=128); ap.add_argument("--reps",type=int,default=5); ap.add_argument("--kv",default="f16")
a=ap.parse_args()
ctx=a.prompt+a.gen+8
rid=f"p2_{a.model.replace('.gguf','')}_p{a.prompt}_{a.kv}"
cmd=[BIN,"-m",MODELS+a.model,"-c",str(ctx),"-np","1","-ngl","999","-fa","on","--cache-type-k",a.kv,"--cache-type-v",a.kv,
     "--port",str(PORT),"--host","127.0.0.1","--no-cache-prompt","--cache-ram","0","-lv","4"]
rec=dict(id=rid,args=vars(a),cmd=cmd,errors=[],reps=[],session=time.strftime("%Y%m%dT%H%M%S"))
logf=open(f"{HERE}/logs/{rid}.log","w"); p=subprocess.Popen(cmd,stdout=logf,stderr=subprocess.STDOUT); U=f"http://127.0.0.1:{PORT}"
try:
    t0=time.time()
    while time.time()-t0<600:
        if p.poll() is not None: raise RuntimeError(f"server exited {p.returncode}")
        try:
            if requests.get(U+"/health",timeout=2).status_code==200: break
        except requests.RequestException: pass
        time.sleep(1)
    else: raise RuntimeError("load timeout")
    rec["load_s"]=time.time()-t0; time.sleep(2)
    txt=TEXT*(a.prompt//10+10)
    toks=requests.post(U+"/tokenize",json={"content":txt,"add_special":False}).json()["tokens"][:a.prompt]
    prompt=requests.post(U+"/detokenize",json={"tokens":toks}).json()["content"]
    # warmup (short, discarded)
    requests.post(U+"/completion",json=dict(prompt="Hello",n_predict=8,temperature=0,cache_prompt=False),timeout=300)
    for i in range(a.reps):
        body=dict(prompt=prompt,n_predict=a.gen,temperature=0,stream=True,cache_prompt=False,ignore_eos=True)
        ts=[]; final=None; t_send=time.perf_counter()
        with requests.post(U+"/completion",json=body,stream=True,timeout=3600) as r:
            if r.status_code!=200:
                rec["errors"].append(f"rep{i}: HTTP {r.status_code} {r.text[:200]}"); continue
            for line in r.iter_lines():
                if not line.startswith(b"data: "): continue
                j=json.loads(line[6:]); now=time.perf_counter()
                if j.get("stop"): final=j
                elif j.get("content") is not None: ts.append(now-t_send)
        if final is None: rec["errors"].append(f"rep{i}: no final chunk"); continue
        t=final["timings"]; n_pred=final.get("tokens_predicted"); n_eval=final.get("tokens_evaluated")
        if n_eval!=a.prompt and n_eval!=a.prompt+1: rec["errors"].append(f"rep{i}: prompt tokens {n_eval}")
        if n_pred!=a.gen: rec["errors"].append(f"rep{i}: gen tokens {n_pred}")
        rec["reps"].append(dict(rep=i+1,n_prompt=n_eval,n_gen=n_pred,n_chunks=len(ts),ttft_client_s=ts[0] if ts else None,
            token_times_s=ts,server=dict(prompt_ms=t["prompt_ms"],prompt_tps=t["prompt_per_second"],predicted_ms=t["predicted_ms"],predicted_tps=t["predicted_per_second"])))
        print(rid,"rep",i+1,"ttft %.2fs"%ts[0],"prefill %.0f t/s"%t["prompt_per_second"],"decode %.1f t/s"%t["predicted_per_second"],flush=True)
except Exception as e:
    rec["errors"].append(repr(e)); print(rid,"ERROR",e)
finally:
    p.terminate()
    try: p.wait(60)
    except subprocess.TimeoutExpired: p.kill()
    logf.close(); time.sleep(3)
    json.dump(rec,open(f"{HERE}/raw/{rid}.json","w"),indent=1)
