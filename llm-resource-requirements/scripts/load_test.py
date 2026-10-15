"""Open-loop (Poisson-arrival) load test for llama-server.

Measures what a user feels, per request: time to first token (TTFT, includes queueing),
inter-token latency (ITL) and end-to-end latency, plus throughput, failures and peak memory.

Standard library only. Run with:  python3 -I load_test.py --help

Design notes
  * Open loop: requests arrive on a pre-drawn Poisson schedule regardless of whether earlier requests
    finished. A closed loop ("send the next when the last returns") hides overload.
  * Prompts are token-id arrays cut from a text corpus at random offsets (tokenised by the server
    itself), with cache_prompt=false, so there is no prompt-cache help and lengths are exact.
  * Output length is forced (ignore_eos) so the workload is identical across models.
  * Each streamed chunk from llama-server carries one generated token; ITL = gaps between chunks.
"""
import argparse
import http.client
import json
import math
import os
import random
import subprocess
import threading
import time
from urllib.parse import urlparse

PROFILES = {
    # (prompt_tokens (lo, median, hi), output_tokens (lo, median, hi)) drawn from a clipped log-normal
    "chat": {"prompt": (100, 400, 1500), "output": (50, 200, 500)},
    "long": {"prompt": (8000, 16000, 24000), "output": (100, 300, 600)},
}
PROFILES["mixed"] = {"mix": [("chat", 0.8), ("long", 0.2)]}


def draw(rng, lo_med_hi):
    lo, med, hi = lo_med_hi
    v = rng.lognormvariate(math.log(med), 0.5)
    return int(min(max(v, lo), hi))


def sample_shape(rng, profile):
    p = PROFILES[profile]
    if "mix" in p:
        r, acc, name = rng.random(), 0.0, p["mix"][-1][0]
        for cand, w in p["mix"]:
            acc += w
            if r <= acc:
                name = cand
                break
        return sample_shape(rng, name)
    return profile, draw(rng, p["prompt"]), draw(rng, p["output"])


def post_json(host, port, path, body, timeout=120):
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return json.loads(data)


def load_corpus_tokens(host, port, path, max_chars=900_000):
    text = open(path, encoding="utf-8", errors="replace").read()[:max_chars]
    out = post_json(host, port, "/tokenize", {"content": text}, timeout=300)
    return out["tokens"]


def one_request(host, port, tokens, rng_seed, plen, olen, timeout, results, lock, t0, kind):
    rng = random.Random(rng_seed)
    off = rng.randrange(0, max(1, len(tokens) - plen))
    body = {"prompt": tokens[off:off + plen], "n_predict": olen, "ignore_eos": True, "stream": True,
            "cache_prompt": False, "temperature": 0.0}
    rec = {"kind": kind, "prompt_tokens": plen, "output_tokens_requested": olen, "t_send": time.time() - t0,
           "ok": False, "error": None, "ttft": None, "e2e": None, "chunks": 0, "itl": []}
    t_send = time.time()
    try:
        c = http.client.HTTPConnection(host, port, timeout=timeout)
        c.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            rec["error"] = f"http_{r.status}"
            r.read()
        else:
            last = None
            stop_seen = False
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                now = time.time()
                try:
                    ev = json.loads(line[5:])
                except Exception:
                    continue
                if ev.get("content"):
                    rec["chunks"] += 1
                    if rec["ttft"] is None:
                        rec["ttft"] = now - t_send
                    else:
                        rec["itl"].append(now - last)
                    last = now
                if ev.get("stop"):
                    stop_seen = True
                    break
            rec["e2e"] = time.time() - t_send
            rec["ok"] = stop_seen and rec["chunks"] > 0
            if not rec["ok"]:
                rec["error"] = "incomplete_stream"
        c.close()
    except TimeoutError:
        rec["error"] = "timeout"
    except Exception as e:  # connection reset, refused, ...
        rec["error"] = type(e).__name__
    with lock:
        results.append(rec)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def mem_sampler(pid, stop, samples):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return ""
    while not stop.is_set():
        used = None
        for ln in sh(["free", "-m"]).splitlines():
            if ln.startswith("Mem:"):
                used = int(ln.split()[2])
        gpu = None
        for ln in sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
            a = [x.strip() for x in ln.split(",")]
            if len(a) == 2 and a[0] == str(pid):
                gpu = int(a[1])
        samples.append((time.time(), used, gpu))
        stop.wait(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--profile", choices=list(PROFILES), required=True)
    ap.add_argument("--rate", type=float, required=True, help="mean arrival rate, requests/s")
    ap.add_argument("--duration", type=float, required=True, help="seconds during which arrivals are generated")
    ap.add_argument("--timeout", type=float, default=300, help="per-request client timeout, seconds")
    ap.add_argument("--drain", type=float, default=120, help="extra seconds to wait for in-flight requests")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pid", type=int, help="server pid, for memory sampling")
    ap.add_argument("--out", required=True, help="prefix for output files (.requests.jsonl, .summary.json)")
    ap.add_argument("--tokens-cache", help="json file caching corpus tokens")
    a = ap.parse_args()

    u = urlparse(a.url)
    host, port = u.hostname, u.port
    if a.tokens_cache and os.path.exists(a.tokens_cache):
        tokens = json.load(open(a.tokens_cache))
    else:
        tokens = load_corpus_tokens(host, port, a.corpus)
        if a.tokens_cache:
            json.dump(tokens, open(a.tokens_cache, "w"))

    rng = random.Random(a.seed)
    sched, t = [], 0.0
    while True:
        t += rng.expovariate(a.rate)
        if t >= a.duration:
            break
        kind, plen, olen = sample_shape(rng, a.profile)
        sched.append((t, kind, plen, olen, rng.randrange(1 << 30)))

    results, lock = [], threading.Lock()
    stop, samples = threading.Event(), []
    st = None
    if a.pid:
        st = threading.Thread(target=mem_sampler, args=(a.pid, stop, samples), daemon=True)
        st.start()
    t0 = time.time()
    threads = []
    for (ts, kind, plen, olen, seed) in sched:
        d = ts - (time.time() - t0)
        if d > 0:
            time.sleep(d)
        th = threading.Thread(target=one_request, args=(host, port, tokens, seed, plen, olen, a.timeout, results, lock, t0, kind), daemon=True)
        th.start()
        threads.append(th)
    arrivals_end = time.time()
    deadline = arrivals_end + a.drain
    for th in threads:
        th.join(max(0, deadline - time.time()))
    wall = time.time() - t0
    unfinished = sum(1 for th in threads if th.is_alive())
    stop.set()
    if st:
        st.join(timeout=3)

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    errs = {}
    for r in bad:
        errs[r["error"]] = errs.get(r["error"], 0) + 1
    if unfinished:
        errs["unfinished_at_end"] = unfinished
    itl_all = [x for r in ok for x in r["itl"]]
    n = len(ok)
    enough = n >= 20  # percentiles over fewer than 20 samples are not meaningful
    sm = {
        "profile": a.profile, "rate_req_s": a.rate, "duration_s": a.duration, "seed": a.seed,
        "scheduled": len(sched), "completed": n, "failed": len(bad) + unfinished, "errors": errs,
        "wall_s": round(wall, 1),
        "completed_req_s": round(n / a.duration, 4),
        "input_tok_s": round(sum(r["prompt_tokens"] for r in ok) / a.duration, 1),
        "output_tok_s": round(sum(r["chunks"] for r in ok) / a.duration, 1),
        "percentiles_valid": enough,
    }
    for name, xs in (("ttft_s", [r["ttft"] for r in ok]), ("itl_s", itl_all), ("e2e_s", [r["e2e"] for r in ok])):
        for q in (0.5, 0.95, 0.99):
            v = pct(xs, q) if (enough or name == "itl_s") else None
            if name == "itl_s" and len(itl_all) < 100:
                v = None
            sm[f"{name}_p{int(q * 100)}"] = None if v is None else round(v, 4)
    if samples:
        us = [s[1] for s in samples if s[1] is not None]
        gs = [s[2] for s in samples if s[2] is not None]
        sm["peak_system_used_MiB"] = max(us) if us else None
        sm["peak_server_gpu_MiB"] = max(gs) if gs else None
    json.dump(sm, open(a.out + ".summary.json", "w"), indent=1)
    with open(a.out + ".requests.jsonl", "w") as f:
        for r in sorted(results, key=lambda r: r["t_send"]):
            r2 = dict(r)
            r2["itl_n"] = len(r2.pop("itl"))
            f.write(json.dumps(r2) + "\n")
    print(json.dumps(sm))


if __name__ == "__main__":
    main()
