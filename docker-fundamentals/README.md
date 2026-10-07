# Docker Fundamentals — From Linux Process to Container

A hands-on mini-course for **absolute beginners to Linux + Docker** who want to understand
**Docker runtime internals**: how a container is just a Linux process with a clever disguise.

> **Core idea:** A container is not a lightweight VM. It is a normal Linux process on the host,
> but the kernel makes that process see an alternate world (via namespaces, cgroups,
> capabilities, seccomp, overlay filesystems, and virtual networking).

Tested on: Arch Linux, kernel 7.x, Docker 29.x (`overlay2`, cgroup v2, `runc`, `containerd`).
Also works on Ubuntu 22.04/24.04 with Docker installed. Most labs need `sudo` for host inspection
(`ip`, `iptables`/`nft`, `/sys/fs/cgroup`, `/proc`).

## The stack you will learn

```text
docker CLI
   ▼
dockerd (Docker daemon)
   ▼
containerd
   ▼
containerd-shim
   ▼
runc (OCI runtime — does clone/unshare/mount/execve)
   ▼
Linux kernel: namespaces | cgroups | capabilities | seccomp | overlayfs | net
```

Docker configures the environment **once** up front. After that the container process talks
**directly to the kernel** via syscalls — Docker is not in the hot path.

## Lab map

| Lab | Folder | Question it answers |
|-----|--------|---------------------|
| 1 | `lab-01-container-is-a-process/` | Is a container really just a host process? |
| 2 | `lab-02-pid-namespaces/` | Why is my container PID 1 but host PID 18234? |
| 3 | `lab-03-mount-namespace/` | Why does the container see a different filesystem/`mount` table? |
| 4 | `lab-04-filesystem-overlayfs/` | Where does `/bin/bash` in `ubuntu` come from? What are image layers? |
| 5 | `lab-05-networking/` | How does `eth0` + `veth` + `docker0` + `-p 8080:80` work? |
| 6 | `lab-06-cgroups/` | How do `--memory` / `--cpus` limits actually get enforced? |
| 7 | `lab-07-security/` | Why is container `root` not real root? What do `--cap-add`, `--privileged`, seccomp do? |
| 8 (bonus) | `lab-08-mini-container/` | Can I build a tiny "Docker" with `unshare` + `chroot`/overlay + `cgroup`? |

Do them in order. Each lab is self-contained: `README.md` + runnable `*.sh` scripts + cleanup.

## How to use each lab

```bash
cd lab-01-container-is-a-process
cat README.md          # read the concepts first (5–10 min)
./demo.sh              # runs the experiment
./inspect.sh           # extra host-side inspection (needs sudo for some parts)
./cleanup.sh           # stops/removes lab containers
```

Conventions:

- `demo.sh` — creates containers, prints host-vs-container views side by side.
- `inspect.sh` — deeper host inspection (`/proc`, `nsenter`, `ip`, cgroup, capabilities).
- `cleanup.sh` — idempotent teardown (`docker rm -f ...`).
- All scripts are `set -euo pipefail` and commented for beginners.

## Prerequisites

```bash
docker --version
docker info | grep -E 'Storage Driver|Cgroup Version|Runtimes'
uname -a              # compare with: docker run --rm ubuntu uname -a
id; groups            # ideally your user is in the `docker` group, else prefix docker with sudo
```

Pull images once (saves time in class):

```bash
docker pull ubuntu:22.04
docker pull nginx:alpine
```

## The one 60-second experiment

If you only do one thing, do this — it destroys the "container has its own Linux" myth:

```bash
uname -a
docker run --rm ubuntu:22.04 uname -a     # same kernel!
docker run --rm ubuntu:22.04 cat /etc/os-release  # Ubuntu *filesystem*, host kernel
```

Same kernel, different root filesystem. That gap *is* Docker.

## Docker vs microVM (Firecracker) — the punchline

```text
Docker container:   app → shared host kernel
Firecracker microVM: app → guest kernel → virtual HW → Firecracker VMM → host kernel
```

- Docker boundary = `process → host kernel` (namespaces + cgroups + seccomp + caps).
- Firecracker boundary = `guest kernel → VMM → host kernel` (hardware virtualization + own kernel).

That is why microVMs give stronger isolation at higher cost, and why understanding
Docker internals first makes Firecracker click later.

## Safety

- Labs 1–6 are safe on any dev machine.
- Lab 7 uses `--privileged` **only on an isolated lab machine/VM**. Never on production.
- Lab 8 needs `sudo` + `unshare`/`mount`. Use a disposable VM if you are cautious.

## Cleanup everything

```bash
for d in lab-*/; do (cd "$d" && ./cleanup.sh 2>/dev/null || true); done
docker ps -a
```
