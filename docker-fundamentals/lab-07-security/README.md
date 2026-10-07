# Lab 7 — Security: Why Container Root Is Not Real Root

**Question:** `whoami` says `root` inside the container — can it reformat the host, load kernel
modules, sniff all traffic? (Usually: no.)

**Answer:** three filters stand between container root and the kernel: **capabilities**
(root chopped into ~40 fine-grained privileges, most dangerous ones dropped),
**seccomp** (a syscall allow/deny filter, ~300+ syscalls permitted by default), and the
namespaces/cgroups from earlier labs. `--privileged` throws almost all of that away.

> ⚠️ **Only run the `--privileged` steps on an isolated lab machine/VM.** They intentionally
> remove the guardrails.

## 0. Beginner primer

- **User**: `root` (uid 0) can do anything — *unless* the kernel is told otherwise.
- **Capabilities**: split "root power" into pieces: `CAP_NET_ADMIN` (configure network),
  `CAP_SYS_ADMIN` (mount, swiss-army-knife — basically full admin), `CAP_SYS_PTRACE` (debug other
  processes), `CAP_DAC_OVERRIDE` (ignore file permissions), ... `capsh --print` lists them.
  Docker's default keeps ~14 useful ones (e.g. `CHOWN`, `NET_BIND_SERVICE`, `SETUID`) and drops the rest.
- **seccomp** (secure computing): a per-process BPF filter on *which syscalls* may run.
  `docker info | grep -i seccomp` → `Profile: builtin`. Blocked examples: `mount` in odd contexts,
  `reboot`, `swapon`, `ptrace` variants. Check denials via `dmesg`/audit log.
- **`--cap-add` / `--cap-drop`**: fine-tune. `--cap-add=NET_ADMIN` lets a container run `tcpdump`;
  `--cap-drop=ALL` + add-back-minimal is best practice for real services.
- **`--privileged`**: gives ALL capabilities + full device access + disables seccomp/AppArmor —
  container root ≈ host root. For hardware access / nested Docker / debugging ONLY, never production.
- **user namespaces** (bonus): map container uid 0 → unprivileged host uid. Rootless Docker uses this.

```text
container root
   │  capabilities (which powers kept?)
   ▼
   │  seccomp (which syscalls allowed?)
   ▼
namespaces + cgroups + read-only mounts
   ▼
kernel
```

## 1. Run it

```bash
./demo.sh      # default caps vs host caps, live cap-add/cap-drop tests
./priv.sh      # DANGEROUS: default vs --privileged contrast (lab VM only!)
./inspect.sh   # decode CapEff hex, seccomp profile, docker-bench style checklist
./cleanup.sh
```

## 2. Manual walkthrough

```bash
# 1. You are "root" — but which powers? (demo.sh builds `lab7-tools:local` once for `ip`/`ping`/`capsh`)
docker run --rm lab7-tools:local whoami            # root
docker run --rm lab7-tools:local capsh --print | head -20
# lighter, no extra image needed — the kernel's own record:
docker run --rm ubuntu:22.04 grep Cap /proc/self/status
# CapEff: 00000000a80425fb  (compare with host init below)

grep CapEff /proc/1/status                         # host init (real root): 000001ffffffffff — far more
grep CapEff /proc/self/status                      # your shell (unprivileged user): ~0000000000000000
capsh --decode=00000000a80425fb                    # human-readable kept set (or run capsh inside lab7-tools)

# 2. Feel a missing capability (lab7-tools has `ip`; plain ubuntu does not):
docker run --rm lab7-tools:local ip link add dummy0 type dummy 2>&1  # Operation not permitted (needs NET_ADMIN)
docker run --rm --cap-add=NET_ADMIN lab7-tools:local ip link add dummy0 type dummy && echo "with NET_ADMIN it works ✔"

# 3. Drop everything (least privilege):
docker run --rm lab7-tools:local id
docker run --rm --cap-drop=ALL lab7-tools:local ping -c1 8.8.8.8 2>&1 | head -3  # ping needs NET_RAW → fails

# 4. seccomp: default profile blocks exotic syscalls:
docker info | grep -A2 -i seccomp
docker run --rm --security-opt seccomp=unconfined ubuntu:22.04 whoami  # disables filter (lab only)

# 5. PRIVILEGED contrast (lab VM ONLY):
docker run --rm ubuntu:22.04 grep Cap /proc/self/status
docker run --rm --privileged ubuntu:22.04 grep Cap /proc/self/status   # CapEff = ...3ffffffffff (ALL)
docker run --rm --privileged ubuntu:22.04 ls /dev | head              # host devices visible!
```

Decode helper: `capsh --decode=<hex>` or python `capabilities` — see `inspect.sh`.

## 3. Rules of thumb for real services

1. Never `--privileged` in production. Ever.
2. Start from `--cap-drop=ALL`, add back only what breaks (`--cap-add=NET_BIND_SERVICE`, ...).
3. Keep the default seccomp profile unless you ship a custom JSON with a reason per added syscall.
4. Prefer `--read-only` + tmpfs + `-v` for data, `--pids-limit`, `--memory`, non-root `USER` in Dockerfile.
5. Rootless Docker / user namespaces for multi-tenant hosts; microVMs (Firecracker) when you need a guest kernel boundary.

## 4. Check your understanding

1. `whoami` → root but `ip link add` → denied. Explain in one sentence. (Uid 0 without CAP_NET_ADMIN in its bounding set.)
2. What does `--privileged` change across all three filters? (All caps + all devices + no seccomp/AppArmor + wider mounts.)
3. Why is `CAP_SYS_ADMIN` called "the new root"? (It unlocks mount, namespaces, cgroup, bpf-adjacent paths — ~near-full control.)

Next: **Lab 8 (bonus)** — assemble Labs 2–6 by hand: a mini-container from `unshare` + overlay + cgroup, no Docker.
