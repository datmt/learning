"""Memory over the life of a llama-server process, sampled every 0.5 s.

Phases: baseline (before launch) -> load -> idle_after_load -> warmup -> prefill_long -> generation
        -> max_context -> concurrent -> idle_after

For each sample we record, with the phase label:
  sys_used_MiB       `free -m` "used" (anonymous + kernel memory, excludes reclaimable page cache)
  sys_avail_MiB      `free -m` "available"
  swap_used_MiB      swap in use
  gpu_proc_MiB       nvidia-smi per-process memory for the server pid (unified memory: this is NOT isolated VRAM)
  rss_MiB            /proc/<pid>/status VmRSS (host-resident part of the process)
  psi_mem_some_avg10 kernel memory-pressure stall (% of time some task stalled on memory, 10 s average)

Standard library only. Usage:
  python3 -I memory_lifecycle.py --model X.gguf --ctx 131072 --np 4 --corpus wiki.test.raw --out prefix
"""
import argparse
import http.client
import json
import subprocess
import threading
import time

S = "/data/engines/llama.cpp/build/bin/llama-server"
PORT = 8099


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def snapshot(pid):
    used = avail = swap = None
    for ln in sh(["free", "-m"]).splitlines():
        p = ln.split()
        if ln.startswith("Mem:"):
            used, avail = int(p[2]), int(p[6])
        elif ln.startswith("Swap:"):
            swap = int(p[2])
    gpu = None
    for ln in sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        a = [x.strip() for x in ln.split(",")]
        if pid and len(a) == 2 and a[0] == str(pid):
            gpu = int(a[1])
    rss = None
    if pid:
        try:
            for ln in open(f"/proc/{pid}/status"):
                if ln.startswith("VmRSS"):
                    rss = int(ln.split()[1]) // 1024
        except OSError:
            pass
    psi = None
    try:
        for ln in open("/proc/pressure/memory"):
            if ln.startswith("some"):
                psi = float(ln.split("avg10=")[1].split()[0])
    except OSError:
        pass
    return dict(sys_used_MiB=used, sys_avail_MiB=avail, swap_used_MiB=swap, gpu_proc_MiB=gpu, rss_MiB=rss, psi_mem_some_avg10=psi)


def req(path, body, timeout=1800):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    d = r.read()
    c.close()
    return json.loads(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, required=True, help="total context (shared KV pool across slots)")
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    state = {"phase": "baseline", "pid": None, "stop": False}
    rows = []

    def sampler():
        t0 = time.time()
        while not state["stop"]:
            r = snapshot(state["pid"])
            r.update(t=round(time.time() - t0, 2), phase=state["phase"])
            rows.append(r)
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    time.sleep(6)  # baseline: nothing of ours running

    state["phase"] = "load"
    proc = subprocess.Popen([S, "-m", a.model, "--host", "127.0.0.1", "--port", str(PORT), "-c", str(a.ctx), "-np", str(a.np),
                             "-fa", "on", "-ngl", "99", "--load-mode", "none", "--kv-unified", "-ctk", a.kv, "-ctv", a.kv, "--jinja"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state["pid"] = proc.pid
    ok = False
    for _ in range(300):
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                ok = True
                break
        except OSError:
            pass
        time.sleep(2)
    if not ok:
        state["stop"] = True
        proc.kill()
        raise SystemExit("server did not become healthy")

    text = open(a.corpus, encoding="utf-8", errors="replace").read()
    toks = req("/tokenize", {"content": text})["tokens"]

    results = []  # one record per request: proves the workload really ran (a fast failure would otherwise look like a fast success)

    def run(phase, prompt_toks, n_predict, parallel=1):
        state["phase"] = phase
        def one(i):
            off = i * 7919 % max(1, len(toks) - prompt_toks)
            rec = {"phase": phase, "prompt_tokens_sent": prompt_toks, "n_predict": n_predict, "ok": False}
            try:
                r = req("/completion", {"prompt": toks[off:off + prompt_toks], "n_predict": n_predict, "ignore_eos": True,
                                        "cache_prompt": False, "temperature": 0.0})
                if "error" in r:
                    rec["error"] = str(r["error"])[:300]
                else:
                    rec["tokens_evaluated"] = r.get("tokens_evaluated")
                    rec["tokens_predicted"] = r.get("tokens_predicted")
                    # ok only if the whole prompt was evaluated and all requested tokens were generated
                    rec["ok"] = rec["tokens_evaluated"] == prompt_toks and rec["tokens_predicted"] == n_predict
                    if not rec["ok"]:
                        rec["error"] = "unexpected token counts"
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            results.append(rec)
        ts = [threading.Thread(target=one, args=(i,)) for i in range(parallel)]
        t = time.time()
        [x.start() for x in ts]
        [x.join() for x in ts]
        return time.time() - t

    timings = {}
    state["phase"] = "idle_after_load"; time.sleep(10)
    # Order matters: the concurrent phase runs BEFORE the pool-filling max_context request. After a request that fills the whole
    # shared KV pool, later requests were intermittently rejected with "Context size has been exceeded" (see memlife_v2_* data).
    timings["warmup"] = run("warmup", 32, 16)
    long_p = min(len(toks) - 1, 16384, a.ctx // 4)
    timings["prefill_long"] = run("prefill_long", long_p, 1)
    timings["generation"] = run("generation", 64, 256)
    conc_p = min(len(toks) - 1, a.ctx // a.np // 2 - 300)  # all slots together use ~half the shared pool
    timings["concurrent"] = run("concurrent", conc_p, 128, parallel=a.np)
    per_slot_max = a.ctx - 400
    timings["max_context"] = run("max_context", min(len(toks) - 1, per_slot_max), 64)
    state["phase"] = "idle_after"; time.sleep(15)

    state["stop"] = True
    th.join(timeout=3)
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()

    # summarise
    phases = []
    for r in rows:
        if r["phase"] not in phases:
            phases.append(r["phase"])
    base = [r for r in rows if r["phase"] == "baseline"]
    b_used = sorted(r["sys_used_MiB"] for r in base if r["sys_used_MiB"] is not None)[len(base) // 2]
    summary = {"model": a.model, "ctx_total": a.ctx, "slots": a.np, "kv_type": a.kv, "baseline_sys_used_MiB": b_used,
               "phase_seconds": {k: round(v, 1) for k, v in timings.items()}, "tokens": {"prefill_long": long_p, "max_context": min(len(toks) - 1, per_slot_max), "concurrent_each": conc_p}, "requests": [], "phases": {}}
    summary["requests"] = results
    summary["all_requests_ok"] = all(r["ok"] for r in results)
    for ph in phases:
        rr = [r for r in rows if r["phase"] == ph]
        d = {"samples": len(rr)}
        for k in ("sys_used_MiB", "gpu_proc_MiB", "rss_MiB", "swap_used_MiB"):
            v = [r[k] for r in rr if r[k] is not None]
            d[k + "_peak"] = max(v) if v else None
        v = [r["sys_avail_MiB"] for r in rr if r["sys_avail_MiB"] is not None]
        d["sys_avail_MiB_min"] = min(v) if v else None
        v = [r["psi_mem_some_avg10"] for r in rr if r["psi_mem_some_avg10"] is not None]
        d["psi_mem_some_avg10_max"] = max(v) if v else None
        summary["phases"][ph] = d
    json.dump(summary, open(a.out + ".summary.json", "w"), indent=1)
    with open(a.out + ".samples.csv", "w") as f:
        keys = ["t", "phase", "sys_used_MiB", "sys_avail_MiB", "swap_used_MiB", "gpu_proc_MiB", "rss_MiB", "psi_mem_some_avg10"]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
