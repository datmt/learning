# Lab 8 (Bonus) — Build a Mini-Container by Hand: `unshare` + Overlay + Cgroup

**Goal:** assemble Labs 2–6 into one artifact — a ~60-line shell "runtime" that does what `runc`
does in miniature: new **pid/mnt/uts/net namespaces** + **overlay rootfs** + **memory limit** +
`exec` your command as PID 1 of its little world. No Docker daemon, no containerd.

You will understand Docker *and* be ready for Firecracker (same idea, plus a guest kernel).

## 0. What `mini.sh` does (read it — it's commented for beginners)

```text
1. Build overlay: lowerdir=<host Ubuntu rootfs>  upperdir=$WORK/upper  workdir=$WORK/work
                 merged=$WORK/merged   (needs sudo: mount -t overlay)
2. Write cgroup: /sys/fs/cgroup/mini-<id>/  { memory.max, pids.max, cgroup.procs }
3. unshare --pid --fork --mount --uts --net --map-root-user   # new namespaces
     + mount -t proc proc $MERGED/proc ; mount --bind /dev, /sys (minimal)
     + chroot/pivot into $MERGED ; hostname mini-ctr ; exec "$@" as PID 1
4. Cleanup unmounts overlay + removes cgroup (trap EXIT)
```

Flags: `--map-root-user` = user namespace mapping (container root → your unprivileged host uid,
safer for a hand-rolled lab). Networking is intentionally `none`-like (isolated net ns, only `lo`) —
wiring veth+bridge+NAT by hand is a full Lab 5 exercise in itself (see `net-tips.txt` in this folder).

## 1. Run it

```bash
./mini.sh --mem 50m /bin/bash -c 'echo "I am PID $$ on $(hostname)"; ps aux 2>/dev/null || ls /proc; echo ---; cat /etc/os-release | head -2; echo ---; uname -r'
./mini.sh --help
./demo.sh    # guided tour: isolation proofs + cgroup limit proof
sudo ./cleanup-stale.sh   # only if a previous run died mid-mount (usually unneeded; mini.sh self-cleans)
```

Requirements: `sudo`, `unshare` (util-linux), an Ubuntu rootfs. `mini.sh` auto-creates one via
`docker export` if Docker exists, else prints how to use `debootstrap`/download.

## 2. Proofs to try inside

```bash
# PID 1 illusion (Lab 2):
sudo ./mini.sh /bin/bash -c 'echo $$'                       # 1

# Own hostname (UTS ns):
sudo ./mini.sh --hostname tiny /bin/bash -c 'hostname'      # tiny

# Own mounts (Lab 3) + overlay image (Lab 4):
sudo ./mini.sh /bin/bash -c 'mount | head; echo ---; touch /made-inside; ls /made-inside'
ls /made-inside 2>&1 || echo "not on host ✔ (went to upperdir)"

# Memory cap (Lab 6):
sudo ./mini.sh --mem 50m /bin/bash -c 'cat /sys/fs/cgroup/memory.max'

# Net isolation (Lab 5):
sudo ./mini.sh /bin/bash -c 'ip addr'                       # only lo
```

## 3. runc parallel (why this matters)

| `mini.sh` step | `runc` / Docker equivalent |
|---|---|
| `unshare --pid --uts --mount --net` | `clone3` with `CLONE_NEWPID\|NEWUTS\|NEWNS\|NEWNET` from OCI `config.json` |
| `mount -t overlay ... $MERGED` | snapshotters (overlayfs) + `pivot_root` to merged dir |
| `echo 50M > memory.max` | cgroup path writes from HostConfig |
| capability drops (commented example) | bounding-set + seccomp BPF from daemon defaults |
| `exec "$@"` | container init = your `CMD` as PID 1 |

Read `mini.sh` top-to-bottom once — every Docker concept from Labs 1–7 appears in order.

## 4. Then Firecracker clicks

```text
mini-container:  process → namespaces/cgroups → HOST kernel (shared)
Docker:          same, orchestrated by dockerd/containerd/runc
Firecracker:     guest process → GUEST kernel → virtual devices → VMM → host kernel
```

Same orchestration instinct, stronger boundary (own kernel + virtual HW). That's the whole course arc.
