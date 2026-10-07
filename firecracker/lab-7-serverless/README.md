# Lab 7 — Mini Serverless Platform (Capstone)

> Beginner-friendly. You combine Labs 1–6 into a primitive FaaS:
> **one short-lived microVM per HTTP request.** The architecture is
> tiny; the ideas are production-grade.

```text
                 API  (POST /run {"cmd": "echo hi"})
                  │
             VM scheduler (Python, stdlib, threaded)
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      VM-1       VM-2      VM-3
      job A      job B     job C
```

Per request:

```text
POST /run {"cmd": "..."}
   ↓
create microVM (fresh rootfs COPY — never share one ext4)
   ↓
wait for boot + configure guest net (SSH retry loop)
   ↓
inject workload (ssh root@<job-ip> "<cmd>"), capture stdout+stderr
   ↓
destroy VM (SendCtrlAltDel → del TAP → rm socket+rootfs copy)
   ↓
return JSON {vm: "job-N", ms: 2100, output: "..."}
```

## 0. Concepts (the 6 ideas this server teaches)

| Idea | Meaning | How `server.py` does it |
|---|---|---|
| **FaaS (Function-as-a-Service)** | User sends code, platform runs it isolated, returns output. AWS Lambda + Firecracker made this famous. | `POST /run` = the whole product. |
| **Cold start** | Time from request to first output: spawn VMM + boot kernel + systemd + SSH ready ≈ 1–3 s here. Containers ≈ 100 ms; warm pools / snapshots fix it (stretch goals). | `ms` field in every response — watch it. `client.sh` times 3 colds in a row. |
| **Scheduler + isolation per job** | Each job gets a FRESH VM: own socket (`/tmp/fc-job-N.socket`), TAP (`tap-jN`), IP (`10.0.0.10+N`), MAC, rootfs copy. Job N can never see job M's files/procs/net. | `boot_vm()` builds the tuple; `destroy_vm()` always runs in `finally`. |
| **Why copy the rootfs?** | Two kernels journaling one ext4 = corruption. Copy-per-job is the v1 strategy. Filesystems with reflink (btrfs/xfs) make the copy near-free; production uses overlays/snapshots. | `shutil.copyfile(base, /tmp/fc-job-N.ext4)`. Deleted after each job. |
| **SSH as the v1 agent** | Real platforms inject work via vsock/guest-agent (no TCP, no keys). We reuse SSH: zero new deps, and you already debugged it in Labs 1–4. Trade-off: SSH-ready wait dominates cold start. | `run_in_vm()`: retry loop configures guest net, then execs `cmd` with timeout. |
| **Threads + `finally` cleanup** | `ThreadingHTTPServer`: each request runs concurrently (up to `--max-vms` sensibly). `try/finally` guarantees TAP/socket/rootfs die even when the job throws. Leaked TAPs are the #1 capstone bug — this pattern prevents them. | `H.do_POST` → `boot_vm` → `run_in_vm` → `destroy_vm` in `finally`. |

Design limits (intentional, stated so you can extend): no auth, no queue
(bursts spawn unbounded until `--max-vms` discipline is added), no snapshot
cache, fixed 256 MB/1 vCPU. Each is a stretch goal below.

## 1. Files

- `commands.sh` — runbook (start server → client → stats).
- `server.py` — stdlib-only scheduler, must run as **root**:
  `POST /run`, `GET /health`, `GET /stats` (`jobs_total`, `avg_ms`).
  Flags: `--port`, `--max-vms` (advisory), `--boot-wait` (SSH patience).
- `client.sh` — health → single run → 3 timed sequential runs → stats.
  Your cold-start benchmark.
- Compare with Docker (optional, no file needed):
  `time sudo docker run --rm alpine uname -a` vs `time curl POST /run`.
  Expect ~10× gap — that gap is the isolation you're buying (Lab 5).

## 2. Quick start

```bash
cd lab-7-serverless
sudo python3 server.py --port 8080 --max-vms 3 &
bash client.sh
```

Manual:

```bash
curl -s -X POST localhost:8080/run -d '{"cmd":"uname -a; echo hello"}'
# {"vm":"job-3","ms":2100,"output":"Linux ...\nhello\n"}
curl -s localhost:8080/stats
# {"jobs_total": 3, "total_ms": ..., "avg_ms": ...}
```

Read the code path for one request: `do_POST` → `boot_vm` (Lab 4
miniaturized) → `run_in_vm` (Lab 3 SSH) → `destroy_vm` (Lab 2 cleanup).
You already know every step — the server just sequences them.

## 3. Verify

- 3 sequential `POST /run` return 200 with distinct `vm` ids (`job-1..3`).
- `bridge link` shows no `tap-j*` after jobs finish (cleanup in `finally` works).
- `GET /stats` shows `jobs_total` = 3 and a sane `avg_ms` (~1000–3000).
- Bonus: run 3 `curl`s in parallel — all succeed, outputs never interleave
  (per-job VM isolation doing its job).

## 4. Stretch goals (in difficulty order)

1. **Timeout/kill**: wrap `run_in_vm` in `timeout 10s`; always `destroy_vm`
   on expiry. (Robustness before speed.)
2. **Snapshot/resume**: boot once, `PUT /vm-state` to save, restore per job.
   Re-run `client.sh` — cold-start `ms` should collapse. This is how real
   Lambda gets sub-second microVMs.
3. **Jailer**: spawn the VMM under `jailer` (uid/gid + cgroup + chroot) like
   production. Compare `/proc/<pid>/status` before/after (Lab 2 lens).

## Troubleshooting

- Must run as root (`sudo python3 server.py`) — TAP + KVM + bridge need it.
- Port in use → `--port 8081` (and `PORT=8081 bash client.sh`).
- SSH flake / `run_in_vm` timeout → slow disk: raise `--boot-wait` to 20,
  or `ls -lh /tmp/fc-job-*.ext4` to confirm the copy isn't stalled.
- Leaked `tap-j*` after Ctrl-C → `ip -o link | grep tap-j` then
  `sudo ip link del <name>`; the `finally` path only covers in-request failures.
