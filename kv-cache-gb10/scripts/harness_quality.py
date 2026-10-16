"""Phase 3 quality: one fresh llama-server per (model, kv); runs greedy-agreement prompts and/or needle retrieval.
Writes raw/p3q_<mode>_<model>_<kv>[_c<ctx>].json. Needle: random 6-digit code hidden at a depth in non-repeating filler."""
import json, os, random, subprocess, sys, time, argparse, requests
BIN="/data/engines/llama.cpp/build/bin/llama-server"; MODELS="/data/engines/models/"
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PORT=8192
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--kv",required=True); ap.add_argument("--mode",choices=["greedy","needle"],required=True)
ap.add_argument("--ctx",type=int,default=8192); ap.add_argument("--trials",type=int,default=20)
a=ap.parse_args()
rid=f"p3q_{a.mode}_{a.model.replace('.gguf','')}_{a.kv}"+(f"_c{a.ctx}" if a.mode=="needle" else "")
GREEDY_PROMPTS=[
 "Explain in detail how a hash table handles collisions, then give a Python implementation with open addressing.",
 "Write a short story about a lighthouse keeper who discovers that the light has been signalling to something in the sea.",
 "Summarise the causes of the First World War, listing the main alliances and the immediate trigger, in about 300 words.",
 "Derive the formula for the sum of the first n squares step by step, and verify it for n = 10.",
 "List ten practical tips for reducing memory use in a large Python data-processing job, with one sentence of reasoning each.",
 "Translate the following into French and then explain two grammar points: 'The engineers measured how much memory each request needed.'",
 "Describe how the TCP three-way handshake works and what happens when a SYN packet is lost.",
 "Write a bash script that watches a directory for new files and compresses each one, then explain every line."]
WORDS="river stone market engine garden window bridge copper violin harbor lantern meadow orbit pencil quartz ribbon saddle timber velvet winter yellow anchor basket candle dragon ember falcon glacier hammer island jungle kettle ladder mirror needle oyster pepper rabbit shadow tunnel umbrella voyage willow".split()
def filler(rng,ntok_words):
    out=[]
    for _ in range(ntok_words//9):
        out.append(f"On day {rng.randint(1,365)} the {rng.choice(WORDS)} team moved {rng.randint(2,99)} {rng.choice(WORDS)}s past the {rng.choice(WORDS)}.")
    return " ".join(out)
cmd=[BIN,"-m",MODELS+a.model,"-c",str(a.ctx if a.mode=="needle" else 4096),"-np","1","-ngl","999","-fa","on","--cache-type-k",a.kv,"--cache-type-v",a.kv,
     "--port",str(PORT),"--host","127.0.0.1","--no-cache-prompt","--cache-ram","0","-lv","4"]
rec=dict(id=rid,args=vars(a),cmd=cmd,errors=[],results=[],session=time.strftime("%Y%m%dT%H%M%S"))
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
    time.sleep(2)
    if a.mode=="greedy":
        for i,pr in enumerate(GREEDY_PROMPTS):
            r=requests.post(U+"/completion",json=dict(prompt=pr,n_predict=256,temperature=0,top_k=1,seed=1,cache_prompt=False,ignore_eos=True,return_tokens=True),timeout=900)
            if r.status_code!=200: rec["errors"].append(f"prompt{i}: HTTP {r.status_code}"); continue
            j=r.json(); rec["results"].append(dict(i=i,tokens=j["tokens"],text=j["content"]))
            print(rid,"prompt",i,len(j["tokens"]),flush=True)
    else:
        rng=random.Random(12345); depths=[0.0,0.25,0.5,0.75,1.0]
        for d in depths:
            for t in range(a.trials):
                code=str(rng.randint(100000,999999)); needle=f" Remember this: the secret code for vault {rng.randint(10,99)} is {code}. "
                budget=a.ctx-400
                words=int(budget*0.72)   # ~tokens per word calibration below
                fill=filler(rng,words)
                # trim filler to budget tokens
                toks=requests.post(U+"/tokenize",json={"content":fill,"add_special":False}).json()["tokens"][:budget]
                cut=int(len(toks)*d)
                pre=requests.post(U+"/detokenize",json={"tokens":toks[:cut]}).json()["content"]
                post=requests.post(U+"/detokenize",json={"tokens":toks[cut:]}).json()["content"]
                q="\n\nQuestion: What is the secret code mentioned above? Answer with the six digits only.\nAnswer:"
                prompt=pre+needle+post+q
                r=requests.post(U+"/completion",json=dict(prompt=prompt,n_predict=64,temperature=0,top_k=1,cache_prompt=False),timeout=3600)
                if r.status_code!=200: rec["errors"].append(f"d{d} t{t}: HTTP {r.status_code} {r.text[:120]}"); continue
                j=r.json(); ok=code in j["content"].replace(" ","")
                rec["results"].append(dict(depth=d,trial=t,code=code,answer=j["content"],ok=ok,n_prompt=j.get("tokens_evaluated")))
            n=[x for x in rec["results"] if x["depth"]==d]; print(rid,"depth",d,sum(x["ok"] for x in n),"/",len(n),flush=True)
except Exception as e:
    rec["errors"].append(repr(e)); print(rid,"ERROR",e)
finally:
    p.terminate()
    try: p.wait(60)
    except subprocess.TimeoutExpired: p.kill()
    logf.close(); time.sleep(3)
    json.dump(rec,open(f"{HERE}/raw/{rid}.json","w"),indent=1)
