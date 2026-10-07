# Lab 6 — Cgroups: Who Enforces `--memory` and `--cpus`?

**Question:** `docker run --memory=100m --cpus=0.5 nginx` — who counts the bytes and throttles the CPU?
Docker polling in userspace?

**Answer:** the **kernel**, via **cgroups** (control groups). Docker just *writes limit files*;
the kernel enforces them on every allocation/schedule. Namespaces answer *"what can it see?"*;
cgroups answer *"how much can it use?"*.

## 0. Beginner primer

- **cgroup**: a set of processes with shared resource limits + accounting. Hierarchy of groups,
  each with files like `memory.max`, `cpu.max`, `pids.max`, `memory.current`.
- **cgroup v1 vs v2**: v1 had one hierarchy per resource (`/sys/fs/cgroup/memory/...`);
  **v2 has ONE unified tree** (`/sys/fs/cgroup/...`) — what Ubuntu 24.04 / Arch / modern Docker use.
  This lab teaches v2 only.
- **Controller**: a resource type (`cpu`, `memory`, `io`, `pids`). Enabled per subtree via `cgroup.subtree_control`.
- **Docker mapping**: `--memory=100m` → `memory.max=104857600`; `--cpus=0.5` → `cpu.max="50000 100000"`
  (quota 50ms per 100ms period); `--pids-limit` → `pids.max`. OOM inside the limit kills the container
  process (check `docker inspect` → `OOMKilled`).
- Docker uses the **systemd driver**: container cgroups live under
  `/sys/fs/cgroup/system.slice/docker-<id>.scope/` (not the old `docker/` path).

```text
docker run --memory=100m --cpus=0.5
        │  (writes files once)
        ▼
/sys/fs/cgroup/system.slice/docker-<id>.scope/
   ├── memory.max = 104857600
   ├── cpu.max = 50000 100000
   └── cgroup.procs = <container PIDs>
        │  (kernel enforces continuously)
        ▼
   allocations throttled / OOM-killed
```

## 1. Run it

```bash
./demo.sh      # starts `limited` (100m, 0.5 cpu), shows its cgroup files
./stress.sh    # optional: OOM demo — allocs 200MB in a 100MB container, watches it die
./inspect.sh   # full cgroup walk: current usage, stats, systemd path
./cleanup.sh
```

## 2. Manual walkthrough

```bash
docker run -d --rm --name limited --memory=100m --cpus=0.5 nginx:alpine

CID=$(docker inspect limited --format '{{.Id}}')
echo "ID: $CID"

# Where is its cgroup?
cat /proc/$(docker inspect limited --format '{{.State.Pid}}')/cgroup
# 0::/system.slice/docker-<id>.scope   <-- v2: single line, systemd path

SCOPE="/sys/fs/cgroup/system.slice/docker-${CID}.scope"
ls "$SCOPE"
cat "$SCOPE/memory.max"     # 104857600
cat "$SCOPE/cpu.max"        # 50000 100000
cat "$SCOPE/pids.max"       # max (unless --pids-limit)
cat "$SCOPE/memory.current" # live usage
cat "$SCOPE/cpu.stat"       # throttling counters

# Watch memory move:
docker exec limited cat /sys/fs/cgroup/memory.current   # inside view (cgroup ns)
cat "$SCOPE/memory.current"                              # host view — same number!

docker stats limited --no-stream
```

OOM proof:

```bash
docker run --rm --memory=50m ubuntu:22.04 bash -c 'x=a; while true; do x="$x$x"; done'
echo $?   # 137 = SIGKILL by the kernel OOM killer (inside the cgroup)
# keep the corpse to inspect: omit --rm, then:
docker inspect <dead-container> --format 'OOMKilled={{.State.OOMKilled}}'  # true
```

CPU proof:

```bash
docker run -d --rm --name hog --cpus=0.5 ubuntu:22.04 bash -c 'while true; do :; done'
top -p $(docker inspect hog --format '{{.State.Pid}}')   # ~50% of one CPU
cat /sys/fs/cgroup/system.slice/docker-$(docker inspect hog --format '{{.Id}}').scope/cpu.stat
docker rm -f hog
```

## 3. Check your understanding

1. Who enforces the limit — `dockerd` or the kernel? (Kernel; dockerd only wrote `memory.max` at setup. Kill dockerd and limits persist.)
2. `memory.current` is identical inside and on host — why? (Same kernel counter; cgroup ns only affects *view*, not the value.)
3. Container gets OOM-killed at 100 MB although host has 30 GB free — contradiction? (No — the cgroup ceiling is what matters, not host capacity.)

Next: **Lab 7** — the last lock on the cage: capabilities + seccomp + why root isn't root.
